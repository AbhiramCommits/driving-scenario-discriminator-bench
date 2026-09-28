"""Tests for the ML discriminator models."""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import pytest

torch = pytest.importorskip("torch")

# LightGBM runs in a spawned subprocess (see dsdbench.models.gbt), so only its
# availability is probed here -- importing it would load a second OpenMP
# runtime into the pytest process and crash torch.
HAS_LIGHTGBM = importlib.util.find_spec("lightgbm") is not None

from dsdbench.data.pipeline import run_pipeline  # noqa: E402
from dsdbench.models.common import set_seed  # noqa: E402
from dsdbench.models.registry import available, create  # noqa: E402
from dsdbench.models.temporal import (  # noqa: E402
    TemporalCNN,
    TemporalConfig,
    TrajTransformer,
    build_model,
    fit,
)
from dsdbench.train import main as train_main  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs"

SMOKE_PIPELINE_KWARGS: dict[str, Any] = {
    "synthetic_fallback": True,
    "n_scenes": 16,
    "agents_per_scene": 4,
    "duration_s": 12.0,
    "seed": 11,
}


@pytest.fixture(scope="session")
def smoke_data(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("ml-smoke") / "bench"
    run_pipeline(out, **SMOKE_PIPELINE_KWARGS)
    return out


# ---------------------------------------------------------------------------
# Shapes and gradient flow
# ---------------------------------------------------------------------------


def _base_temporal_config(**overrides: Any) -> TemporalConfig:
    values: dict[str, Any] = dict(seq_len=61, in_channels=6)
    values.update(overrides)
    return TemporalConfig(**values)


@pytest.mark.parametrize("arch", ["cnn", "transformer"])
def test_model_shapes(arch: str):
    set_seed(0)
    model = build_model(_base_temporal_config(arch=arch))
    x = torch.randn(4, 61, 6)
    out = model(x)
    assert out.shape == (4, 1)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("arch", ["cnn", "transformer"])
def test_model_gradient_flow(arch: str):
    set_seed(0)
    model = build_model(_base_temporal_config(arch=arch))
    x = torch.randn(4, 61, 6, requires_grad=False)
    y = torch.randint(0, 2, (4,)).float()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x).squeeze(-1), y)
    loss.backward()
    grad_norms = []
    for name, param in model.named_parameters():
        assert param.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite gradient for {name}"
        grad_norms.append(float(param.grad.norm()))
    # Gradients reach the input layers of both architectures.
    assert max(grad_norms) > 0.0


def test_transformer_uses_cls_token_and_positional_embeddings():
    config = _base_temporal_config(arch="transformer", seq_len=31)
    model = build_model(config)
    assert isinstance(model, TrajTransformer)
    assert model.pos_embed.shape == (1, 32, config.d_model)
    assert model.cls_token.shape == (1, 1, config.d_model)
    # Shorter sequences than seq_len are fine (positional embedding slicing).
    assert model(torch.randn(2, 17, 6)).shape == (2, 1)


def test_cnn_has_four_dilated_blocks():
    config = _base_temporal_config(arch="cnn", dilations=(1, 2, 4, 8))
    model = build_model(config)
    assert isinstance(model, TemporalCNN)
    assert len(model.blocks) == 4
    assert [b[0].dilation[0] for b in model.blocks] == [1, 2, 4, 8]


# ---------------------------------------------------------------------------
# 1-batch overfit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arch", ["cnn", "transformer"])
def test_overfit_single_batch(arch: str):
    set_seed(0)
    config = _base_temporal_config(
        arch=arch,
        seq_len=31,
        hidden_dim=32,
        num_groups=4,
        d_model=64,
        n_layers=2,
        n_heads=4,
        dim_feedforward=128,
        dropout=0.0,
        epochs=150,
        lr=2e-2,
        weight_decay=0.0,
        batch_size=8,
        max_grad_norm=5.0,
    )
    model = build_model(config)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(8, 31, 6)).astype(np.float32)
    y = rng.integers(0, 2, size=8).astype(np.float32)
    metrics = fit(model, config, x, y, device="cpu")
    assert metrics["final_train_loss"] < 0.1, metrics


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_contains_expected_models():
    assert set(available()) >= {"gbt", "cnn", "transformer"}


def test_registry_round_trip():
    from dsdbench.models.gbt import GBTTrainer
    from dsdbench.models.temporal import CNNTrainer, TemporalTrainer, TransformerTrainer

    first = create("cnn", {"epochs": 2})
    second = create("cnn", {"epochs": 2})
    assert isinstance(first, CNNTrainer)
    assert isinstance(first, TemporalTrainer)
    assert first is not second
    assert first.config["arch"] == "cnn"

    assert isinstance(create("transformer", {}), TransformerTrainer)
    assert create("transformer", {}).config["arch"] == "transformer"
    assert isinstance(create("gbt", {}), GBTTrainer)


def test_registry_unknown_model_raises():
    with pytest.raises(ValueError, match="unknown model"):
        create("does_not_exist", {})


# ---------------------------------------------------------------------------
# End-to-end smoke runs (CPU, < 2 min each)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["gbt", "cnn", "transformer"])
def test_end_to_end_smoke_run(model: str, smoke_data: Path, tmp_path: Path):
    if model == "gbt":
        if not HAS_LIGHTGBM:
            pytest.skip("lightgbm not installed")
        if importlib.util.find_spec("shap") is None:
            pytest.skip("shap not installed")
    artifacts = tmp_path / "artifacts"
    start = time.monotonic()
    rc = train_main(
        [
            "--model",
            model,
            "--config",
            str(CONFIGS / f"{model}.yaml"),
            "--data-dir",
            str(smoke_data),
            "--artifacts-dir",
            str(artifacts),
            "--version",
            "smoke",
            "--epochs",
            "2",
        ]
    )
    elapsed = time.monotonic() - start
    assert rc == 0
    assert elapsed < 120.0, f"{model} smoke run took {elapsed:.1f}s"

    run_dir = artifacts / model / "smoke"
    assert (run_dir / "config.yaml").exists()
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert "elapsed_s" in metrics
    if model in ("cnn", "transformer"):
        assert (run_dir / "checkpoint.pt").exists()
        assert metrics["epochs_run"] == 2
        assert len(metrics["train_losses"]) == 2
    else:
        assert (run_dir / "model.txt").exists()
        assert (run_dir / "gbt_shap.json").exists()
        assert (artifacts / "gbt_shap.json").exists()

    preds = pq.read_table(run_dir / "predictions.parquet").to_pydict()
    assert len(preds["segment_id"]) > 0
    assert len(preds["segment_id"]) == len(set(preds["segment_id"]))
    probs = np.asarray(preds["prob_sim"])
    assert np.all(np.isfinite(probs))
    assert probs.min() >= 0.0 and probs.max() <= 1.0
    # The dataset has 16 scenes -> 12/1/3 train/val/test: all splits present.
    assert set(preds["split"]) == {"train", "val", "test"}
