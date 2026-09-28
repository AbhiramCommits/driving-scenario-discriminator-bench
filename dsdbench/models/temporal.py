"""PyTorch temporal discriminators on raw (T, 6) trajectory tensors.

Two architectures behind one interface (both take a ``TemporalConfig`` dataclass
and return logits of shape (B, 1)):

* ``TemporalCNN`` -- 4 dilated 1D convolution blocks (dilations 1/2/4/8) with
  GroupNorm + GELU, global average pooling, linear head.
* ``TrajTransformer`` -- 4-layer encoder, 4 heads, d_model=128, learned
  positional embeddings, CLS-token pooling.

The training loop uses AdamW, a cosine LR schedule, gradient clipping, AMP when
CUDA is available, and deterministic seeding (including
``torch.use_deterministic_algorithms`` where possible). It runs on CPU so CI
can execute a 2-epoch smoke train.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dsdbench.models.common import (
    auc,
    config_from_dict,
    load_dataset,
    set_seed,
    write_metrics,
    write_predictions,
)
from dsdbench.models.registry import register


@dataclass
class TemporalConfig:
    """Model + training configuration for the temporal nets."""

    arch: str = "cnn"  # "cnn" or "transformer"
    seed: int = 0
    # data
    seq_len: int = 61
    in_channels: int = 6
    # cnn
    hidden_dim: int = 64
    kernel_size: int = 3
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    num_groups: int = 8
    # transformer
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    dim_feedforward: int = 256
    # shared
    dropout: float = 0.1
    # training
    epochs: int = 20
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    use_amp: bool = True


class TemporalCNN(nn.Module):
    """Dilated 1D conv stack + GroupNorm + GELU + global avg pool + linear head."""

    def __init__(self, config: TemporalConfig) -> None:
        super().__init__()
        if config.hidden_dim % config.num_groups != 0:
            raise ValueError("hidden_dim must be divisible by num_groups")
        blocks: list[nn.Module] = []
        in_ch = config.in_channels
        for dilation in config.dilations:
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_ch,
                        config.hidden_dim,
                        config.kernel_size,
                        dilation=dilation,
                        padding=dilation * (config.kernel_size - 1) // 2,
                    ),
                    nn.GroupNorm(config.num_groups, config.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                )
            )
            in_ch = config.hidden_dim
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(config.hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) -> logits (B, 1)."""
        h = x.transpose(1, 2)  # (B, C, T)
        for block in self.blocks:
            h = block(h)
        h = h.mean(dim=2)  # global average pooling
        return cast("torch.Tensor", self.head(h))


class TrajTransformer(nn.Module):
    """Transformer encoder over per-timestep channel embeddings with CLS pooling."""

    def __init__(self, config: TemporalConfig) -> None:
        super().__init__()
        self.input_proj = nn.Linear(config.in_channels, config.d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.seq_len + 1, config.d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        layer = nn.TransformerEncoderLayer(
            config.d_model,
            config.n_heads,
            config.dim_feedforward,
            config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=config.n_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, 1)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) -> logits (B, 1)."""
        batch, seq, _ = x.shape
        h = self.input_proj(x)  # (B, T, D)
        cls = self.cls_token.expand(batch, -1, -1)  # (B, 1, D)
        h = torch.cat([cls, h], dim=1)  # (B, T+1, D)
        h = h + self.pos_embed[:, : seq + 1]
        h = self.encoder(h)
        h = self.norm(h[:, 0])  # CLS token
        return cast("torch.Tensor", self.head(h))


def build_model(config: TemporalConfig) -> nn.Module:
    if config.arch == "cnn":
        return TemporalCNN(config)
    if config.arch == "transformer":
        return TrajTransformer(config)
    raise ValueError(f"unknown arch {config.arch!r}; expected 'cnn' or 'transformer'")


def compute_normalization(
    trajectories: np.ndarray, train_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std computed on the train split only (shape (6,))."""
    train = trajectories[train_mask]
    mean = train.mean(axis=(0, 1))
    std = train.std(axis=(0, 1))
    return mean.astype(np.float32), std.astype(np.float32)


def normalize(trajectories: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    # numpy's unparameterized ndarray arithmetic is inferred as Any.
    return (trajectories - mean) / (std + 1e-5)  # type: ignore[no-any-return]


def fit(
    model: nn.Module,
    config: TemporalConfig,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray | None = None,
    val_y: np.ndarray | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Train the model and return epoch/val metrics (no artifact IO)."""
    device = torch.device(device)
    model = model.to(device)
    generator = torch.Generator().manual_seed(config.seed)
    train_ds = TensorDataset(
        torch.from_numpy(np.asarray(train_x, np.float32)),
        torch.from_numpy(np.asarray(train_y, np.float32)),
    )
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True, generator=generator, drop_last=False
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    criterion = nn.BCEWithLogitsLoss()
    use_amp = config.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    train_losses: list[float] = []
    for _epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=use_amp):
                loss = criterion(model(xb).squeeze(-1), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.detach().cpu())
            n_batches += 1
        scheduler.step()
        train_losses.append(epoch_loss / max(n_batches, 1))

    model.eval()
    with torch.no_grad():
        final_train_loss = float(
            criterion(
                model(_to_tensor(train_x, device)).squeeze(-1), _to_tensor(train_y, device)
            ).cpu()
        )
    metrics: dict[str, Any] = {
        "train_losses": train_losses,
        "final_train_loss": final_train_loss,
        "epochs_run": config.epochs,
    }
    if val_x is not None and val_y is not None and len(val_x) > 0:
        val_probs = predict_probs(model, val_x, device)
        val_loss = _bce(val_y, val_probs)
        metrics["val_loss"] = val_loss
        metrics["val_auc"] = auc(np.asarray(val_y), val_probs)
    return metrics


def predict_probs(
    model: nn.Module, x: np.ndarray, device: torch.device | str = "cpu", batch_size: int = 256
) -> np.ndarray:
    """Sigmoid probabilities for a (N, T, 6) array, batched, no grad."""
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = torch.from_numpy(np.asarray(x[i : i + batch_size], np.float32)).to(device)
            out.append(np.asarray(torch.sigmoid(model(xb).squeeze(-1)).cpu().numpy()))
    return np.concatenate(out) if out else np.empty(0, dtype=np.float32)


def _to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(x, np.float32)).to(device)


def _bce(y: np.ndarray, probs: np.ndarray) -> float:
    p = np.clip(np.asarray(probs, np.float64), 1e-7, 1.0 - 1e-7)
    yv = np.asarray(y, np.float64)
    return float(-np.mean(yv * np.log(p) + (1.0 - yv) * np.log(1.0 - p)))


class TemporalTrainer:
    """Shared trainer for the temporal nets (subclassed per arch)."""

    ARCH = "cnn"

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = {**config, "arch": self.ARCH}

    def train(
        self, data_dir: str | Path, config: dict[str, Any], run_dir: str | Path
    ) -> dict[str, Any]:
        cfg = config_from_dict(TemporalConfig, {**config, "arch": self.ARCH})
        start = time.monotonic()
        set_seed(cfg.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dataset = load_dataset(data_dir, seq_len=cfg.seq_len)
        split = np.asarray(dataset.labels["split"])
        train_mask = split == "train"
        val_mask = split == "val"
        test_mask = split == "test"
        mean, std = compute_normalization(dataset.trajectories, train_mask)
        x = normalize(dataset.trajectories, mean, std)

        model = build_model(cfg)
        metrics = fit(
            model,
            cfg,
            x[train_mask],
            dataset.y[train_mask],
            x[val_mask] if val_mask.any() else None,
            dataset.y[val_mask] if val_mask.any() else None,
            device=device,
        )
        if test_mask.any():
            metrics["test_auc"] = auc(
                dataset.y[test_mask], predict_probs(model, x[test_mask], device)
            )

        probs = predict_probs(model, x, device)
        write_predictions(run_dir, dataset.labels, probs)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": dataclasses.asdict(cfg),
                "channel_mean": mean.tolist(),
                "channel_std": std.tolist(),
            },
            Path(run_dir) / "checkpoint.pt",
        )
        metrics["elapsed_s"] = round(time.monotonic() - start, 3)
        write_metrics(run_dir, metrics)
        return metrics


@register("cnn")
class CNNTrainer(TemporalTrainer):
    ARCH = "cnn"


@register("transformer")
class TransformerTrainer(TemporalTrainer):
    ARCH = "transformer"
