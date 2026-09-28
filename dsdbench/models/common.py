"""Shared helpers for the ML models: dataset loading, seeding, artifact IO."""

from __future__ import annotations

import dataclasses
import json
import random
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from dsdbench.data.features import FEATURE_NAMES

T = TypeVar("T")

TRAJECTORY_CHANNELS = ("t", "x", "y", "heading", "speed", "yaw_rate")


@dataclass
class Dataset:
    """In-memory view of a built benchmark dataset (see dsdbench.data.pipeline)."""

    labels: dict[str, list[Any]]  # one entry per segment (incl. "split")
    feature_matrix: np.ndarray  # (N, 30) float32, ordered by segment_id
    trajectories: np.ndarray  # (N, seq_len, 6) float32, ordered by segment_id
    y: np.ndarray  # (N,) float32, real=0 / sim=1
    segment_ids: np.ndarray  # (N,) int64
    scenes: np.ndarray  # (N,) str


def config_from_dict(cls: type[T], data: dict[str, Any]) -> T:
    """Build a dataclass from a (possibly superset) config dict, ignoring extra keys."""
    fields = {f.name for f in dataclasses.fields(cast(Any, cls))}
    return cls(**{k: v for k, v in data.items() if k in fields})


def load_dataset(data_dir: str | Path, seq_len: int = 61) -> Dataset:
    """Load segments/features/labels Parquet files and pivot them per segment."""
    data_dir = Path(data_dir)
    labels_raw = pq.read_table(data_dir / "labels.parquet").to_pydict()
    features_raw = pq.read_table(data_dir / "features.parquet").to_pydict()
    segments_raw = pq.read_table(data_dir / "segments.parquet").to_pydict()

    # Order labels and features by segment_id so row i == segment i everywhere.
    label_order = np.argsort(labels_raw["segment_id"])
    labels = {k: [v[i] for i in label_order] for k, v in labels_raw.items()}
    feature_order = np.argsort(features_raw["segment_id"])
    features = {k: [v[i] for i in feature_order] for k, v in features_raw.items()}

    trajectories, segment_ids = pivot_trajectories(segments_raw, seq_len)
    feature_matrix = np.column_stack(
        [np.asarray(features[name], dtype=np.float32) for name in FEATURE_NAMES]
    )
    y = np.asarray([1.0 if s == "sim" else 0.0 for s in labels["source"]], dtype=np.float32)
    scenes = np.asarray(labels["scene_id"], dtype=object)
    return Dataset(
        labels=labels,
        feature_matrix=feature_matrix,
        trajectories=trajectories,
        y=y,
        segment_ids=segment_ids,
        scenes=scenes,
    )


def pivot_trajectories(
    segments: dict[str, list[Any]], seq_len: int
) -> tuple[np.ndarray, np.ndarray]:
    """Pivot long-format segments into an (N, seq_len, 6) float32 tensor."""
    segment_ids = np.asarray(segments["segment_id"], dtype=np.int64)
    t = np.asarray(segments["t"], dtype=np.float64)
    order = np.lexsort((t, segment_ids))
    ids_sorted = segment_ids[order]
    unique_ids, counts = np.unique(ids_sorted, return_counts=True)
    if not np.all(counts == seq_len):
        bad = unique_ids[counts != seq_len][:5]
        raise ValueError(
            f"expected {seq_len} rows per segment; got anomalies at segment_id(s) {list(bad)}"
        )
    cols = [np.asarray(segments[name], dtype=np.float64)[order] for name in TRAJECTORY_CHANNELS]
    stacked = np.stack(cols, axis=1)  # (M, 6)
    return stacked.reshape(len(unique_ids), seq_len, 6).astype(np.float32), unique_ids


def set_seed(seed: int) -> None:
    """Deterministic seeding for numpy/python/torch (+ torch deterministic mode)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        with suppress(RuntimeError):
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def write_metrics(run_dir: str | Path, metrics: dict[str, Any]) -> None:
    (Path(run_dir) / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))


def write_predictions(run_dir: str | Path, labels: dict[str, list[Any]], probs: np.ndarray) -> None:
    """Persist per-segment predicted probabilities as Parquet (one row per segment)."""
    table = pa.table(
        {
            "segment_id": labels["segment_id"],
            "scene_id": labels["scene_id"],
            "source": labels["source"],
            "maneuver": labels["maneuver"],
            "split": labels["split"],
            "prob_sim": np.asarray(probs, dtype=np.float64),
        }
    )
    pq.write_table(table, Path(run_dir) / "predictions.parquet")


def auc(y_true: np.ndarray, probs: np.ndarray) -> float | None:
    """ROC AUC, or None when the labels are single-class."""
    y = np.asarray(y_true)
    p = np.asarray(probs)
    if y.size < 2 or len(np.unique(y)) < 2:
        return None
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, p))
