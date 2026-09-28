"""Tests for the dsdbench data layer."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import duckdb
import numpy as np
import pyarrow.parquet as pq
import pytest

from dsdbench.data.features import FEATURE_NAMES, extract_features
from dsdbench.data.maneuver_labels import ManeuverLabeler
from dsdbench.data.nuscenes_ingest import cut_windows
from dsdbench.data.pipeline import assign_splits, iter_sql_statements, run_pipeline

DT = 0.1
N_SAMPLES = 61  # 6 s at 10 Hz

PIPELINE_KWARGS = {
    "synthetic_fallback": True,
    "n_scenes": 24,
    "agents_per_scene": 4,
    "duration_s": 15.0,
    "seed": 3,
}


@pytest.fixture(scope="session")
def built_dataset(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict]:
    out = tmp_path_factory.mktemp("dsdbench") / "bench"
    summary = run_pipeline(out, **PIPELINE_KWARGS)
    return out, summary


# ---------------------------------------------------------------------------
# Window cutting
# ---------------------------------------------------------------------------


def _track(n_points: int, fs: float = 10.0) -> tuple[np.ndarray, ...]:
    t = np.arange(n_points) / fs
    x = t
    y = np.zeros_like(t)
    heading = np.zeros_like(t)
    speed = np.full_like(t, 5.0)
    yaw_rate = np.zeros_like(t)
    return t, x, y, heading, speed, yaw_rate


def test_window_cutting_exact_window_length():
    segments = cut_windows(*_track(61), scene_id="s", agent_token="a")
    assert len(segments) == 1
    assert segments[0].t.size == N_SAMPLES
    np.testing.assert_allclose(segments[0].t, np.arange(N_SAMPLES) / 10.0)


def test_window_cutting_shorter_than_window_skipped():
    assert cut_windows(*_track(60)) == []
    assert cut_windows(*_track(1)) == []
    assert cut_windows(*_track(0)) == []


def test_window_cutting_overlap_is_fifty_percent():
    segments = cut_windows(*_track(91))  # 9.0 s
    assert len(segments) == 2
    assert segments[0].t[0] == 0.0 and segments[0].t[-1] == 6.0
    assert segments[1].t[0] == 3.0 and segments[1].t[-1] == 9.0
    # 50 % overlap: the second half of window 0 is the first half of window 1.
    np.testing.assert_allclose(segments[0].t[30:], segments[1].t[:31])
    np.testing.assert_allclose(segments[0].x[30:], segments[1].x[:31])


def test_window_cutting_boundary_just_below_second_window():
    segments = cut_windows(*_track(90))  # 8.9 s: one full step short of 2 windows
    assert len(segments) == 1


def test_window_cutting_multiple_windows_and_content():
    segments = cut_windows(*_track(121))  # 12.0 s
    assert len(segments) == 3
    assert [float(seg.t[0]) for seg in segments] == [0.0, 3.0, 6.0]
    # Window k contains exactly samples [k*30 : k*30+61] of the source track.
    t_src = np.arange(121) / 10.0
    for k, seg in enumerate(segments):
        np.testing.assert_allclose(seg.t, t_src[k * 30 : k * 30 + 61])


# ---------------------------------------------------------------------------
# Maneuver labeler on hand-constructed synthetic trajectories
# ---------------------------------------------------------------------------


def _build_trajectory(heading_fn, speed_fn) -> tuple[np.ndarray, ...]:
    t = np.arange(N_SAMPLES) * DT
    heading = heading_fn(t)
    speed = speed_fn(t)
    x = np.zeros(N_SAMPLES)
    y = np.zeros(N_SAMPLES)
    for i in range(1, N_SAMPLES):
        x[i] = x[i - 1] + speed[i - 1] * np.cos(heading[i - 1]) * DT
        y[i] = y[i - 1] + speed[i - 1] * np.sin(heading[i - 1]) * DT
    yaw_rate = np.gradient(np.unwrap(heading), DT)
    return t, x, y, heading, speed, yaw_rate


def test_labeler_lane_keep():
    traj = _build_trajectory(lambda t: np.zeros_like(t), lambda t: np.full_like(t, 10.0))
    assert ManeuverLabeler().label(*traj) == "lane_keep"


def test_labeler_cut_in():
    # Lateral shift (~3.8 m) completing at t = 3.5 s (fraction 0.58, i.e. the
    # central part of the window), zero net heading change.
    heading = lambda t: np.where(  # noqa: E731
        (t >= 0.5) & (t <= 3.5), 0.2 * np.sin(np.pi * (t - 0.5) / 3.0), 0.0
    )
    traj = _build_trajectory(heading, lambda t: np.full_like(t, 10.0))
    assert ManeuverLabeler().label(*traj) == "cut_in"


def test_labeler_merge():
    # Lateral shift peaking at t = 1.5 s (fraction 0.25, the window edge).
    heading = lambda t: np.where(t <= 1.5, 0.25 * np.sin(np.pi * t / 1.5), 0.0)  # noqa: E731
    traj = _build_trajectory(heading, lambda t: np.full_like(t, 10.0))
    assert ManeuverLabeler().label(*traj) == "merge"


def test_labeler_stop_and_go():
    t_ctrl = [0.0, 2.0, 3.0, 3.5, 4.5, 6.0]
    v_ctrl = [10.0, 10.0, 0.0, 0.0, 10.0, 10.0]
    speed = lambda t: np.interp(t, t_ctrl, v_ctrl)  # noqa: E731
    traj = _build_trajectory(lambda t: np.zeros_like(t), speed)
    assert ManeuverLabeler().label(*traj) == "stop_and_go"


def test_labeler_unprotected_left():
    heading = lambda t: (np.pi / 2.0) * (t / 6.0)  # noqa: E731
    traj = _build_trajectory(heading, lambda t: np.full_like(t, 5.0))
    assert ManeuverLabeler().label(*traj) == "unprotected_left"


def test_labeler_thresholds_configurable(tmp_path: Path):
    cfg_path = tmp_path / "maneuver_config.yaml"
    cfg_path.write_text("lateral_shift:\n  min_lateral_m: 5.0\n")
    strict = ManeuverLabeler(config_path=cfg_path)
    heading = lambda t: np.where(  # noqa: E731
        (t >= 0.5) & (t <= 3.5), 0.2 * np.sin(np.pi * (t - 0.5) / 3.0), 0.0
    )
    traj = _build_trajectory(heading, lambda t: np.full_like(t, 10.0))
    assert ManeuverLabeler().label(*traj) == "cut_in"
    # With the raised threshold the same trajectory is no longer a cut-in.
    assert strict.label(*traj) == "lane_keep"


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


def test_assign_splits_deterministic_and_grouped():
    scenes = [f"scene_{i:03d}" for i in range(60)]
    first = assign_splits(scenes)
    second = assign_splits(scenes)
    assert first == second
    # Order independence: reversed input gives the same per-scene assignment.
    assert assign_splits(list(reversed(scenes)))["scene_000"] == first["scene_000"]
    # Duplicate scene ids collapse to a single assignment entry.
    assert set(assign_splits(["a", "a", "b"])) == {"a", "b"}
    counts = Counter(first.values())
    assert counts["train"] > counts["val"] and counts["val"] >= counts["test"]
    assert all(split in {"train", "val", "test"} for split in first.values())


def test_no_scene_leakage_across_splits(built_dataset: tuple[Path, dict]):
    out, _ = built_dataset
    con = duckdb.connect(str(out / "benchmark.duckdb"), read_only=True)
    leaked = con.execute(
        "SELECT scene_id, count(DISTINCT split) AS n_splits FROM labels "
        "GROUP BY scene_id HAVING n_splits > 1"
    ).fetchall()
    con.close()
    assert leaked == []


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


def test_all_features_finite(built_dataset: tuple[Path, dict]):
    out, _ = built_dataset
    table = pq.read_table(out / "features.parquet").to_pydict()
    assert set(FEATURE_NAMES) <= set(table)
    for name in FEATURE_NAMES:
        assert np.all(np.isfinite(np.asarray(table[name]))), name


def test_features_finite_on_degenerate_segments():
    t = np.arange(N_SAMPLES) * DT
    rng = np.random.default_rng(0)

    # Circular constant-rate turn.
    heading_circle = 2.0 * np.pi * t / 6.0
    circle = (
        5.0 * np.sin(heading_circle),
        -5.0 * np.cos(heading_circle),
        heading_circle,
        np.full_like(t, 5.0),
        np.full_like(t, 2.0 * np.pi / 6.0),
    )

    cases = {
        "stopped": (
            np.zeros_like(t),
            np.zeros_like(t),
            np.zeros_like(t),
            np.zeros_like(t),
            np.zeros_like(t),
        ),
        "constant_speed_straight": (
            10.0 * t,
            np.zeros_like(t),
            np.zeros_like(t),
            np.full_like(t, 10.0),
            np.zeros_like(t),
        ),
        "circle": circle,
        "random_walk": (
            np.cumsum(rng.normal(0, 0.1, N_SAMPLES)),
            np.cumsum(rng.normal(0, 0.1, N_SAMPLES)),
            np.cumsum(rng.normal(0, 0.05, N_SAMPLES)),
            np.abs(np.cumsum(rng.normal(0, 0.1, N_SAMPLES))) + 1.0,
            rng.normal(0, 0.1, N_SAMPLES),
        ),
    }
    for name, (x, y, heading, speed, yaw_rate) in cases.items():
        feats = extract_features(t, x, y, heading, speed, yaw_rate)
        assert all(np.isfinite(value) for value in feats.values()), name
        assert len(feats) == len(FEATURE_NAMES)


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------


def test_pipeline_artifacts_and_schema(built_dataset: tuple[Path, dict]):
    out, summary = built_dataset
    for name in (
        "segments.parquet",
        "labels.parquet",
        "features.parquet",
        "benchmark.duckdb",
        "queries.sql",
        "manifest.json",
    ):
        assert (out / name).exists()

    labels = pq.read_table(out / "labels.parquet").to_pydict()
    n_labels = len(labels["segment_id"])
    assert n_labels == 2 * summary["n_real_segments"]
    assert labels["source"].count("real") == labels["source"].count("sim")
    assert set(labels["maneuver"]) <= {
        "cut_in",
        "merge",
        "unprotected_left",
        "lane_keep",
        "stop_and_go",
    }
    assert set(labels["split"]) <= {"train", "val", "test"}
    # Every simulated segment carries its sampled config as ground truth.
    for i, source in enumerate(labels["source"]):
        if source == "sim":
            assert labels["config_json"][i] is not None

    segments = pq.read_table(out / "segments.parquet").to_pydict()
    assert set(segments["segment_id"]) == set(labels["segment_id"])
    per_segment = Counter(segments["segment_id"])
    assert set(per_segment.values()) == {N_SAMPLES}

    features = pq.read_table(out / "features.parquet").to_pydict()
    assert set(features["segment_id"]) == set(labels["segment_id"])
    assert set(FEATURE_NAMES) <= set(features)


def test_pipeline_rerun_deterministic(built_dataset: tuple[Path, dict], tmp_path: Path):
    out, _ = built_dataset
    out2 = tmp_path / "bench2"
    run_pipeline(out2, **PIPELINE_KWARGS)
    assert (
        pq.read_table(out / "labels.parquet").to_pydict()
        == pq.read_table(out2 / "labels.parquet").to_pydict()
    )
    assert (
        pq.read_table(out / "features.parquet").to_pydict()
        == pq.read_table(out2 / "features.parquet").to_pydict()
    )


def test_queries_sql_executes_and_split_tables_populated(built_dataset: tuple[Path, dict]):
    out, _ = built_dataset
    sql_text = (out / "queries.sql").read_text()
    con = duckdb.connect(str(out / "benchmark.duckdb"))
    for stmt in iter_sql_statements(sql_text):
        con.execute(stmt)
    leaked = con.execute(
        "SELECT scene_id, count(DISTINCT split) AS n_splits FROM labels "
        "GROUP BY scene_id HAVING n_splits > 1"
    ).fetchall()
    assert leaked == []
    for table in ("train_set", "val_set", "test_set", "split_features"):
        assert con.execute(f"SELECT count(*) FROM {table}").fetchone()[0] > 0
    # Every row of train/val/test belongs to exactly one split.
    total = con.execute(
        "SELECT (SELECT count(*) FROM train_set) + (SELECT count(*) FROM val_set) "
        "+ (SELECT count(*) FROM test_set)"
    ).fetchone()[0]
    assert total == con.execute("SELECT count(*) FROM segments").fetchone()[0]
    con.close()
