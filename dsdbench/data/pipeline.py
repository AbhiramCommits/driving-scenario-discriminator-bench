"""End-to-end benchmark dataset build: ingest -> simulate -> label -> features -> persist.

Produces, in ``out_dir``:

* ``segments.parquet`` -- raw trajectories, one row per timestep
  (segment_id, scene_id, source, agent_token, t, x, y, heading, speed, yaw_rate)
* ``labels.parquet`` -- one row per segment (maneuver, split, config_json)
* ``features.parquet`` -- one row per segment, 30 scalar features
* ``benchmark.duckdb`` -- DuckDB database with the same tables plus the split
  tables and checks from ``queries.sql``
* ``queries.sql`` -- copy of the bundled SQL queries
* ``manifest.json`` -- build summary (counts, skipped tracks, sampling ranges)

The train/val/test split is assigned per scene (deterministic md5 buckets, 70/15/15
by default), so every segment of a scene lands in exactly one split and no scene
leaks across splits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from dsdbench.data.features import FEATURE_NAMES, extract_features
from dsdbench.data.maneuver_labels import ManeuverLabeler
from dsdbench.data.nuscenes_ingest import RawSegment, ingest_nuscenes, ingest_synthetic
from dsdbench.data.sim_generator import SAMPLING_RANGES, generate_matched

logger = logging.getLogger(__name__)

QUERIES_SQL = Path(__file__).with_name("queries.sql")
SWEEP_MARKER = "-- SWEEP-SECTION:"

# Deterministic sensor ("recorder") noise applied to both real and simulated
# recordings before anything is persisted. Real logged data always carries
# sensor noise, and the simulator's output is exact ground truth; without a
# shared recorder model the two sides would be trivially separable by
# numerical roughness alone, masking the realism knobs under study.
RECORDER_NOISE = {
    "pos_std": 0.02,  # m, per axis
    "heading_std": 0.005,  # rad
    "speed_std": 0.05,  # m/s
    "yaw_std": 0.005,  # rad/s
}


def assign_splits(
    scene_ids: Iterable[str], *, train_frac: float = 0.7, val_frac: float = 0.15
) -> dict[str, str]:
    """Deterministic grouped split assignment: one split per scene.

    Buckets ``md5(scene_id) % 100`` into train/val/test so the assignment is
    stable across runs and processes (no RNG, no ordering dependence).
    """
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1")
    train_bucket = int(round(100.0 * train_frac))
    val_bucket = train_bucket + int(round(100.0 * val_frac))
    result: dict[str, str] = {}
    for scene_id in sorted(set(scene_ids)):
        bucket = int(hashlib.md5(scene_id.encode("utf-8")).hexdigest(), 16) % 100
        result[scene_id] = (
            "train" if bucket < train_bucket else ("val" if bucket < val_bucket else "test")
        )
    return result


def iter_sql_statements(sql_text: str) -> list[str]:
    """Split a SQL file into executable statements (skips blank/comment-only)."""
    statements: list[str] = []
    for raw in sql_text.split(";"):
        stmt = raw.strip()
        if not stmt:
            continue
        if all(line.strip().startswith("--") for line in stmt.splitlines()):
            continue
        statements.append(stmt)
    return statements


def pipeline_queries(sql_text: str) -> list[str]:
    """Statements of queries.sql that the dataset pipeline executes (before
    the ``SWEEP_MARKER`` banner; sweep queries need the sweep_results table
    which only exists after experiments/run_benchmark.py runs)."""
    lines = sql_text.splitlines()
    cut = next((i for i, line in enumerate(lines) if line.startswith(SWEEP_MARKER)), len(lines))
    return iter_sql_statements("\n".join(lines[:cut]))


def sweep_queries(sql_text: str) -> list[str]:
    """Statements of queries.sql that run over the sweep results table
    (from the ``SWEEP_MARKER`` banner onward; executed by
    experiments/run_benchmark.py against results.duckdb)."""
    lines = sql_text.splitlines()
    cut = next((i for i, line in enumerate(lines) if line.startswith(SWEEP_MARKER)), len(lines))
    return iter_sql_statements("\n".join(lines[cut:]))


def _recorder_seed(key: str) -> int:
    return int(hashlib.md5(key.encode("utf-8")).hexdigest()[:16], 16) % 2**32


def _apply_recorder_noise(segments: list[RawSegment], *, tag: str) -> None:
    """Apply a deterministic recording model (sensor noise + bandwidth limit)
    to a set of recordings.

    Real sensors add Gaussian noise and low-pass the signal; the simulator
    output is exact ground truth. Recording both sides with the same model
    keeps them comparable and prevents numerical roughness alone from being a
    trivially detectable sim fingerprint.
    """
    from scipy.ndimage import uniform_filter1d

    for seg in segments:
        rng = np.random.default_rng(_recorder_seed(f"{seg.scene_id}:{seg.agent_token}:{tag}"))
        seg.x = seg.x + rng.normal(0.0, RECORDER_NOISE["pos_std"], seg.x.shape)
        seg.y = seg.y + rng.normal(0.0, RECORDER_NOISE["pos_std"], seg.y.shape)
        seg.heading = seg.heading + rng.normal(
            0.0, RECORDER_NOISE["heading_std"], seg.heading.shape
        )
        seg.speed = np.clip(
            seg.speed + rng.normal(0.0, RECORDER_NOISE["speed_std"], seg.speed.shape), 0.0, None
        )
        seg.yaw_rate = seg.yaw_rate + rng.normal(0.0, RECORDER_NOISE["yaw_std"], seg.yaw_rate.shape)
        # Sensor bandwidth: 3-tap low-pass (0.3 s window at 10 Hz).
        seg.heading = uniform_filter1d(seg.heading, size=3, mode="nearest")
        seg.speed = uniform_filter1d(seg.speed, size=3, mode="nearest")
        seg.yaw_rate = uniform_filter1d(seg.yaw_rate, size=3, mode="nearest")


def run_pipeline(
    out_dir: str | Path,
    *,
    synthetic_fallback: bool = False,
    nuscenes_root: str | Path | None = None,
    version: str = "v1.0-mini",
    n_scenes: int = 10,
    agents_per_scene: int = 8,
    duration_s: float = 24.0,
    seed: int = 0,
    n_threads: int | None = None,
    db_name: str = "benchmark.duckdb",
    knob_ranges: dict[str, tuple[float, float] | list[float]] | None = None,
    knob_fixed: dict[str, float | int] | None = None,
) -> dict[str, Any]:
    """Run the full pipeline and return a build summary dict.

    ``knob_ranges`` / ``knob_fixed`` override the simulator realism knobs for
    benchmark sweeps (see ``dsdbench.data.sim_generator.sample_config``).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if synthetic_fallback:
        real_segments, ingest_stats = ingest_synthetic(n_scenes, agents_per_scene, duration_s, seed)
        # The synthetic fallback generates noiseless "recordings"; add the
        # shared recorder noise so they resemble logged sensor data.
        _apply_recorder_noise(real_segments, tag="real")
    else:
        if nuscenes_root is None:
            raise ValueError("--nuscenes-root is required unless --synthetic-fallback is set")
        real_segments, ingest_stats = ingest_nuscenes(nuscenes_root, version)
    if not real_segments:
        raise ValueError("no real segments produced; increase the run duration or dataset size")

    ranges_override = {
        key: (float(values[0]), float(values[1])) for key, values in (knob_ranges or {}).items()
    }
    sim_segments = generate_matched(
        real_segments,
        n_threads=n_threads,
        ranges_override=ranges_override or None,
        fixed_override=knob_fixed,
    )
    # The simulator output is exact ground truth; record it with the same
    # sensor model as the real side.
    _apply_recorder_noise([sim_seg.segment for sim_seg in sim_segments], tag="sim")
    labeler = ManeuverLabeler()
    splits = assign_splits(seg.scene_id for seg in real_segments)

    seg_rows: dict[str, list[np.ndarray]] = {name: [] for name in SEGMENT_COLUMNS}
    label_rows: dict[str, list[Any]] = {name: [] for name in LABEL_COLUMNS}
    feat_rows: dict[str, list[Any]] = {"segment_id": [], "scene_id": [], "source": []}
    feat_cols: dict[str, list[float]] = {name: [] for name in FEATURE_NAMES}

    for i, seg in enumerate(real_segments):
        _append_segment(
            seg_rows, label_rows, feat_rows, feat_cols, seg, i, "real", None, splits, labeler
        )
    for j, sim_seg in enumerate(sim_segments):
        _append_segment(
            seg_rows,
            label_rows,
            feat_rows,
            feat_cols,
            sim_seg.segment,
            len(real_segments) + j,
            "sim",
            sim_seg.config,
            splits,
            labeler,
        )

    seg_table = pa.table({name: np.concatenate(cols) for name, cols in seg_rows.items()})
    pq.write_table(seg_table, out / "segments.parquet")
    pq.write_table(pa.table(label_rows), out / "labels.parquet")
    pq.write_table(pa.table({**feat_rows, **feat_cols}), out / "features.parquet")

    _load_duckdb(out, db_name)
    (out / "queries.sql").write_text(QUERIES_SQL.read_text())

    maneuver_counts = Counter(label_rows["maneuver"])
    split_counts = Counter(label_rows["split"])
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "n_scenes": len(set(label_rows["scene_id"])),
        "n_real_segments": len(real_segments),
        "n_sim_segments": len(sim_segments),
        "ingest_stats": ingest_stats,
        "maneuver_counts": dict(sorted(maneuver_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "sim_config_keys": sorted(sim_segments[0].config) if sim_segments else [],
        "sim_config_sampling_ranges": {
            key: list(values) for key, values in SAMPLING_RANGES.items()
        },
        "pipeline_params": {
            "synthetic_fallback": synthetic_fallback,
            "nuscenes_root": str(nuscenes_root) if nuscenes_root is not None else None,
            "version": version,
            "n_scenes": n_scenes,
            "agents_per_scene": agents_per_scene,
            "duration_s": duration_s,
            "seed": seed,
            "n_threads": n_threads,
            "knob_ranges": {k: list(v) for k, v in ranges_override.items()} or None,
            "knob_fixed": knob_fixed,
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    logger.info(
        "pipeline done: %d real + %d sim segments in %d scenes -> %s",
        len(real_segments),
        len(sim_segments),
        manifest["n_scenes"],
        out,
    )
    return manifest


SEGMENT_COLUMNS = (
    "segment_id",
    "scene_id",
    "source",
    "agent_token",
    "t",
    "x",
    "y",
    "heading",
    "speed",
    "yaw_rate",
)
LABEL_COLUMNS = (
    "segment_id",
    "scene_id",
    "source",
    "agent_token",
    "maneuver",
    "split",
    "config_json",
)


def _append_segment(
    seg_rows: dict[str, list[np.ndarray]],
    label_rows: dict[str, list[Any]],
    feat_rows: dict[str, list[Any]],
    feat_cols: dict[str, list[float]],
    seg: RawSegment,
    segment_id: int,
    source: str,
    config: dict[str, Any] | None,
    splits: dict[str, str],
    labeler: ManeuverLabeler,
) -> None:
    n = int(seg.t.size)
    seg_rows["segment_id"].append(np.full(n, segment_id, dtype=np.int64))
    seg_rows["scene_id"].append(np.array([seg.scene_id] * n, dtype=object))
    seg_rows["source"].append(np.array([source] * n, dtype=object))
    seg_rows["agent_token"].append(np.array([seg.agent_token] * n, dtype=object))
    for name in ("t", "x", "y", "heading", "speed", "yaw_rate"):
        seg_rows[name].append(np.asarray(getattr(seg, name), dtype=np.float64))

    maneuver = labeler.label(seg.t, seg.x, seg.y, seg.heading, seg.speed, seg.yaw_rate)
    label_rows["segment_id"].append(segment_id)
    label_rows["scene_id"].append(seg.scene_id)
    label_rows["source"].append(source)
    label_rows["agent_token"].append(seg.agent_token)
    label_rows["maneuver"].append(maneuver)
    label_rows["split"].append(splits[seg.scene_id])
    label_rows["config_json"].append(
        json.dumps(config, sort_keys=True) if config is not None else None
    )

    feat_rows["segment_id"].append(segment_id)
    feat_rows["scene_id"].append(seg.scene_id)
    feat_rows["source"].append(source)
    feats = extract_features(seg.t, seg.x, seg.y, seg.heading, seg.speed, seg.yaw_rate)
    for name in FEATURE_NAMES:
        feat_cols[name].append(feats[name])


def _load_duckdb(out: Path, db_name: str) -> None:
    db_path = out / db_name
    if db_path.exists():
        db_path.unlink()
    con = duckdb.connect(str(db_path))
    for table in ("segments", "labels", "features"):
        parquet_path = str((out / f"{table}.parquet").resolve())
        con.execute(
            f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM read_parquet(?)",
            [parquet_path],
        )
    for stmt in pipeline_queries(QUERIES_SQL.read_text()):
        con.execute(stmt)
    leaked = con.execute(
        "SELECT scene_id, count(DISTINCT split) AS n_splits FROM labels "
        "GROUP BY scene_id HAVING n_splits > 1"
    ).fetchall()
    if leaked:
        raise RuntimeError(f"scene leakage across splits: {leaked}")
    con.close()
    logger.info("wrote DuckDB database %s", db_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m dsdbench.data.pipeline",
        description="Build the dsdbench benchmark dataset (Parquet + DuckDB).",
    )
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory.")
    parser.add_argument(
        "--nuscenes-root",
        type=Path,
        default=None,
        help="nuScenes dataroot (real ingest; requires dsdbench[nuscenes]).",
    )
    parser.add_argument("--version", default="v1.0-mini", help="nuScenes dataset version.")
    parser.add_argument(
        "--synthetic-fallback",
        action="store_true",
        help="Use the synthetic stand-in dataset instead of nuScenes (no dataset "
        "download required; CI uses this flag).",
    )
    parser.add_argument("--n-scenes", type=int, default=10, help="Synthetic scene count.")
    parser.add_argument(
        "--agents-per-scene", type=int, default=8, help="Synthetic agents per scene."
    )
    parser.add_argument(
        "--duration-s", type=float, default=24.0, help="Synthetic run duration [s]."
    )
    parser.add_argument("--seed", type=int, default=0, help="Synthetic dataset seed.")
    parser.add_argument(
        "--n-threads",
        type=int,
        default=None,
        help="Threads for batch_rollout (default: min(8, n_cpu)).",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    summary = run_pipeline(
        args.out_dir,
        synthetic_fallback=args.synthetic_fallback,
        nuscenes_root=args.nuscenes_root,
        version=args.version,
        n_scenes=args.n_scenes,
        agents_per_scene=args.agents_per_scene,
        duration_s=args.duration_s,
        seed=args.seed,
        n_threads=args.n_threads,
    )
    logger.info("build summary:\n%s", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
