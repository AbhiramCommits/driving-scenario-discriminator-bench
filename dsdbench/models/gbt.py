"""LightGBM binary classifier on the tabular segment features.

Training pipeline:

* grouped CV by scene (``GroupKFold`` on ``scene_id``) with per-fold AUC;
* a final model trained on the ``train`` split with early stopping on the
  ``val`` split (fixed seed throughout);
* the final booster saved as a model artifact;
* SHAP-based global feature importance (mean |SHAP| per feature, from a
  deterministic sample of the training data) saved to ``gbt_shap.json``.

Isolation note: LightGBM, PyTorch, scikit-learn and Homebrew all bundle their
own copies of the Intel OpenMP runtime; loading several of them into one
process crashes OpenMP initialization (observed on macOS). Training therefore
runs in a spawned subprocess so the gbt path never shares a process with
PyTorch, making the trainer safe to mix with the temporal models.
"""

from __future__ import annotations

import importlib.util
import json
import multiprocessing
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dsdbench.data.features import FEATURE_NAMES
from dsdbench.models.common import (
    Dataset,
    auc,
    config_from_dict,
    load_dataset,
    write_metrics,
    write_predictions,
)
from dsdbench.models.registry import register


@dataclass
class GBTConfig:
    seed: int = 0
    n_estimators: int = 400
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 20
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    early_stopping_rounds: int = 50
    n_cv_folds: int = 5
    shap_sample_size: int = 1500


def _require_lightgbm() -> None:
    if importlib.util.find_spec("lightgbm") is None:
        raise ImportError(
            "LightGBM is required for the gbt model; install with `pip install 'dsdbench[ml]'`"
        )


def _classifier(cfg: GBTConfig) -> Any:
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        num_leaves=cfg.num_leaves,
        min_child_samples=cfg.min_child_samples,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        random_state=cfg.seed,
        n_jobs=-1,
        verbosity=-1,
    )


def _grouped_cv_aucs(dataset: Dataset, cfg: GBTConfig) -> list[float]:
    from sklearn.model_selection import GroupKFold

    n_scenes = len(set(dataset.scenes))
    n_splits = max(2, min(cfg.n_cv_folds, n_scenes))
    folds = GroupKFold(n_splits=n_splits)
    fold_aucs: list[float] = []
    for train_idx, test_idx in folds.split(
        dataset.feature_matrix, dataset.y, groups=dataset.scenes
    ):
        model = _classifier(cfg)
        model.fit(dataset.feature_matrix[train_idx], dataset.y[train_idx])
        value = auc(
            dataset.y[test_idx], model.predict_proba(dataset.feature_matrix[test_idx])[:, 1]
        )
        if value is not None:
            fold_aucs.append(value)
    return fold_aucs


def _shap_importance(model: Any, dataset: Dataset, cfg: GBTConfig) -> dict[str, float]:
    import shap

    train_mask = np.asarray(dataset.labels["split"]) == "train"
    indices = np.flatnonzero(train_mask)
    if indices.size > cfg.shap_sample_size:
        rng = np.random.default_rng(cfg.seed)
        indices = np.sort(rng.choice(indices, size=cfg.shap_sample_size, replace=False))
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(dataset.feature_matrix[indices])
    if isinstance(values, list):
        values = values[1]  # positive-class SHAP values for binary LightGBM
    values = np.asarray(values)
    importance = {
        FEATURE_NAMES[i]: float(np.abs(values[:, i]).mean()) for i in range(values.shape[1])
    }
    return dict(sorted(importance.items(), key=lambda kv: kv[1], reverse=True))


def _train_impl(data_dir: str, config: dict[str, Any], run_dir: str) -> dict[str, Any]:
    """Full gbt training (module-level so it can be spawned into a subprocess)."""
    import lightgbm as lgb

    cfg = config_from_dict(GBTConfig, config)
    start = time.monotonic()
    run_dir_path = Path(run_dir)
    dataset = load_dataset(data_dir)

    split = np.asarray(dataset.labels["split"])
    train_mask = split == "train"
    val_mask = split == "val"
    test_mask = split == "test"

    fold_aucs = _grouped_cv_aucs(dataset, cfg)

    final = _classifier(cfg)
    if val_mask.any() and len(np.unique(dataset.y[val_mask])) > 1:
        final.fit(
            dataset.feature_matrix[train_mask],
            dataset.y[train_mask],
            eval_set=[(dataset.feature_matrix[val_mask], dataset.y[val_mask])],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)],
        )
    else:
        final.fit(dataset.feature_matrix[train_mask], dataset.y[train_mask])

    probs = final.predict_proba(dataset.feature_matrix)[:, 1]
    write_predictions(run_dir_path, dataset.labels, probs)
    final.booster_.save_model(str(run_dir_path / "model.txt"))

    importance = _shap_importance(final, dataset, cfg)
    (run_dir_path / "gbt_shap.json").write_text(json.dumps(importance, indent=2, sort_keys=True))

    metrics: dict[str, Any] = {
        "cv_fold_aucs": fold_aucs,
        "cv_mean_auc": float(np.mean(fold_aucs)) if fold_aucs else None,
        "n_estimators_fitted": int(final.best_iteration_ or cfg.n_estimators),
        "val_auc": auc(dataset.y[val_mask], probs[val_mask]) if val_mask.any() else None,
        "test_auc": auc(dataset.y[test_mask], probs[test_mask]) if test_mask.any() else None,
        "top_shap_features": list(importance)[:10],
        "elapsed_s": round(time.monotonic() - start, 3),
    }
    write_metrics(run_dir_path, metrics)
    return metrics


@register("gbt")
class GBTTrainer:
    """LightGBM discriminator trainer (see module docstring)."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = dict(config)

    def train(
        self, data_dir: str | Path, config: dict[str, Any], run_dir: str | Path
    ) -> dict[str, Any]:
        _require_lightgbm()
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        # Run in a spawned subprocess: keep LightGBM's OpenMP runtime out of
        # the calling process so it can coexist with PyTorch (see module note).
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(1) as pool:
            return pool.apply(_train_impl, (str(data_dir), config, str(run_dir)))
