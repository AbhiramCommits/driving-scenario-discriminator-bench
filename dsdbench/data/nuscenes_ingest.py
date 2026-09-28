"""Ingest driving trajectories from nuScenes (or a synthetic fallback) and cut them
into fixed-rate, fixed-length per-agent segments.

The primary input is the nuScenes v1.0-mini split read through the
``nuscenes-devkit`` package. Per-agent pose tracks are resampled to a fixed
10 Hz grid, lightly smoothed, differentiated into speed / yaw-rate, and cut
into 6 s windows with 50 % overlap. Tracks shorter than one window (or with
fewer observations than a window needs) are skipped and counted in the logged
stats. All coordinates are expressed in a map-relative frame (the nuScenes
global frame, translated by the scene origin so the scene starts near the
origin).

Because downloading nuScenes requires credentials, the ``--synthetic-fallback``
flag generates a deterministic stand-in dataset from map-free spline paths so
the full pipeline runs end-to-end without the dataset. CI uses this flag.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike
from scipy.ndimage import uniform_filter1d

from dsdbench._sim import SimConfig, State, rollout

logger = logging.getLogger(__name__)

FS_HZ = 10.0
WINDOW_S = 6.0
OVERLAP = 0.5

_SYNTHETIC_TEMPLATES = ("cruise", "cut_in", "merge", "turn_left")
# Note: "stop_go" is intentionally not a synthetic template: the simulator's
# longitudinal control is a constant acceleration, so it cannot reproduce a
# stop-and-go speed profile and its sim counterpart would be trivially
# detectable (see README Limitations).


@dataclass
class RawSegment:
    """One fixed-length segment of a single agent's trajectory.

    All arrays are float64 with shape (61,) for the default 6 s / 10 Hz
    settings. ``heading`` is continuous (unwrapped), in radians.
    """

    scene_id: str
    agent_token: str
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    heading: np.ndarray
    speed: np.ndarray
    yaw_rate: np.ndarray


def cut_windows(
    t: ArrayLike,
    x: ArrayLike,
    y: ArrayLike,
    heading: ArrayLike,
    speed: ArrayLike,
    yaw_rate: ArrayLike,
    *,
    scene_id: str = "",
    agent_token: str = "",
    window_s: float = WINDOW_S,
    overlap: float = OVERLAP,
    fs: float = FS_HZ,
) -> list[RawSegment]:
    """Cut a uniformly sampled track into fixed-length overlapping windows.

    ``overlap`` is the fraction of shared duration between consecutive windows
    (0.5 = 50 %). A track shorter than one window yields no segments; the last
    window may extend past the exact end only when the track is long enough to
    contain it fully.
    """
    t_arr = np.asarray(t, dtype=np.float64)
    if t_arr.size == 0:
        return []
    window_n = int(round(fs * window_s)) + 1
    step_n = int(round(fs * window_s * (1.0 - overlap)))
    n_windows = (t_arr.size - window_n) // step_n + 1 if t_arr.size >= window_n else 0

    arrays = {
        name: np.asarray(arr, dtype=np.float64)
        for name, arr in (
            ("x", x),
            ("y", y),
            ("heading", heading),
            ("speed", speed),
            ("yaw_rate", yaw_rate),
        )
    }
    segments: list[RawSegment] = []
    for k in range(n_windows):
        i0 = k * step_n
        i1 = i0 + window_n
        segments.append(
            RawSegment(
                scene_id=scene_id,
                agent_token=agent_token,
                t=t_arr[i0:i1].copy(),
                x=arrays["x"][i0:i1].copy(),
                y=arrays["y"][i0:i1].copy(),
                heading=arrays["heading"][i0:i1].copy(),
                speed=arrays["speed"][i0:i1].copy(),
                yaw_rate=arrays["yaw_rate"][i0:i1].copy(),
            )
        )
    return segments


# ---------------------------------------------------------------------------
# nuScenes ingestion
# ---------------------------------------------------------------------------


def ingest_nuscenes(
    root: str | Path, version: str = "v1.0-mini"
) -> tuple[list[RawSegment], dict[str, int]]:  # pragma: no cover
    """Ingest the nuScenes dataset at ``root`` and return per-agent segments.

    Requires the ``nuscenes-devkit`` package (install ``dsdbench[nuscenes]``).
    """
    try:
        from nuscenes.nuscenes import NuScenes
    except ImportError as exc:
        raise ImportError(
            "ingesting nuScenes requires the nuscenes-devkit package; install "
            "with `pip install 'dsdbench[nuscenes]'` or pass --synthetic-fallback"
        ) from exc

    nusc = NuScenes(version=version, dataroot=str(root), verbose=False)
    stats = {
        "n_scenes": 0,
        "n_instances": 0,
        "n_runs": 0,
        "n_segments": 0,
        "skipped_short_track": 0,
    }
    segments: list[RawSegment] = []
    window_n = int(round(FS_HZ * WINDOW_S)) + 1

    for scene in nusc.scene:
        stats["n_scenes"] += 1
        instances = _collect_scene_instances(nusc, scene)
        origin = _scene_origin(nusc, scene)
        for token, (t_obs, xs, ys, hs) in sorted(instances.items()):
            stats["n_instances"] += 1
            xs = xs - origin[0]
            ys = ys - origin[1]
            for run_t, run_x, run_y, run_h in _split_runs(t_obs, xs, ys, hs):
                if run_t[-1] - run_t[0] < WINDOW_S - 1e-9:
                    stats["skipped_short_track"] += 1
                    continue
                t_grid = np.arange(run_t[0], run_t[-1] + 1e-9, 1.0 / FS_HZ)
                if t_grid.size < window_n:
                    stats["skipped_short_track"] += 1
                    continue
                stats["n_runs"] += 1
                x_r = _resample(run_t, run_x, t_grid)
                y_r = _resample(run_t, run_y, t_grid)
                h_r = _resample(run_t, np.unwrap(run_h), t_grid)
                speed, yaw_rate = _derive_kinematics(x_r, y_r, h_r)
                for seg in cut_windows(
                    t_grid - t_grid[0],
                    x_r,
                    y_r,
                    h_r,
                    speed,
                    yaw_rate,
                    scene_id=scene["name"],
                    agent_token=token,
                ):
                    segments.append(seg)
                    stats["n_segments"] += 1

    logger.info(
        "nuScenes ingest: %d scenes, %d instances, %d runs, %d segments; "
        "skipped %d tracks shorter than %.1f s",
        stats["n_scenes"],
        stats["n_instances"],
        stats["n_runs"],
        stats["n_segments"],
        stats["skipped_short_track"],
        WINDOW_S,
    )
    return segments, stats


def _collect_scene_instances(
    nusc: Any, scene: dict[str, Any]
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:  # pragma: no cover
    """Walk the sample chain of a scene and collect per-instance (t, x, y, yaw)."""
    collected: dict[str, list[tuple[float, float, float, float]]] = {}
    sample_token: str = scene["first_sample_token"]
    while sample_token != "":
        sample = nusc.get("sample", sample_token)
        ts = sample["timestamp"] / 1e6
        for ann_token in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_token)
            yaw = _quat_to_yaw(ann["rotation"])
            collected.setdefault(ann["instance_token"], []).append(
                (ts, float(ann["translation"][0]), float(ann["translation"][1]), yaw)
            )
        sample_token = sample["next"]
    out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for token, rows in collected.items():
        rows.sort(key=lambda r: r[0])
        arr = np.asarray(rows, dtype=np.float64)
        out[token] = (arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3])
    return out


def _scene_origin(nusc: Any, scene: dict[str, Any]) -> np.ndarray:  # pragma: no cover
    sample = nusc.get("sample", scene["first_sample_token"])
    ego = nusc.get("ego_pose", sample["ego_pose_token"])
    return np.asarray(ego["translation"][:2], dtype=np.float64)


def _quat_to_yaw(rotation: dict[str, float]) -> float:  # pragma: no cover
    w, x, y, z = rotation["w"], rotation["x"], rotation["y"], rotation["z"]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _split_runs(
    t: np.ndarray, x: np.ndarray, y: np.ndarray, heading: np.ndarray, max_gap_s: float = 1.0
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:  # pragma: no cover
    """Split an observation track at gaps larger than ``max_gap_s`` seconds."""
    if t.size < 2:
        return []
    split_idx = np.flatnonzero(np.diff(t) > max_gap_s) + 1
    runs = []
    for start, stop in zip(np.r_[0, split_idx], np.r_[split_idx, t.size], strict=True):
        if stop - start >= 2:
            runs.append(
                (
                    t[start:stop].copy(),
                    x[start:stop].copy(),
                    y[start:stop].copy(),
                    heading[start:stop].copy(),
                )
            )
    return runs


def _resample(
    t_obs: np.ndarray, values: np.ndarray, t_grid: np.ndarray
) -> np.ndarray:  # pragma: no cover
    """Linear resample + light 5-tap smoothing to remove keyframe artifacts."""
    interp = np.interp(t_grid, t_obs, values)
    return cast("np.ndarray", uniform_filter1d(interp, size=5, mode="nearest"))


def _derive_kinematics(
    x: np.ndarray, y: np.ndarray, heading: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:  # pragma: no cover
    dt = 1.0 / FS_HZ
    speed = np.hypot(np.gradient(x, dt), np.gradient(y, dt))
    yaw_rate = np.gradient(heading, dt)
    return speed, yaw_rate


# ---------------------------------------------------------------------------
# Synthetic fallback
# ---------------------------------------------------------------------------


def _rollout_reference_windows(
    x: np.ndarray,
    y: np.ndarray,
    heading: np.ndarray,
    speed: np.ndarray,
    scene_id: str,
    agent_token: str,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Roll out the reference controller per 6 s window, exactly like the
    simulated side (same path fitting, same horizon, no future knowledge)."""
    window_n = int(round(FS_HZ * WINDOW_S)) + 1
    step_n = int(round(FS_HZ * WINDOW_S * (1.0 - OVERLAP)))
    runs = []
    for i0 in range(0, max(1, x.size - window_n + 1), step_n):
        i1 = i0 + window_n
        config = SimConfig()
        config.dt = 0.1
        config.lookahead_gain = 0.9
        config.min_lookahead = 0.3
        config.wheelbase = 2.7
        config.max_steer = 0.6
        config.max_accel = 3.0
        config.max_decel = 5.0
        config.accel = 0.0
        config.pos_noise_std = 0.002
        config.heading_noise_std = 0.0003
        config.steer_noise_std = 0.0003
        config.latency_steps = 0
        config.steer_bias = 0.0

        from dsdbench.data.sim_generator import fit_reference_path

        path = fit_reference_path(x[i0:i1], y[i0:i1])
        seed = (
            int(hashlib.md5(f"{scene_id}:{agent_token}:{i0}".encode()).hexdigest()[:16], 16) % 2**32
        )
        trajectory = rollout(
            path,
            State(
                x=float(x[i0]),
                y=float(y[i0]),
                heading=float(heading[i0]),
                speed=float(speed[i0]),
            ),
            window_n - 1,
            config,
            seed,
        )
        runs.append(
            (
                np.asarray(trajectory[:, 0]),
                np.asarray(trajectory[:, 1]),
                np.asarray(trajectory[:, 2]),
                np.asarray(trajectory[:, 3]),
                np.asarray(trajectory[:, 4]),
                np.asarray(trajectory[:, 5]),
            )
        )
    return runs


def ingest_synthetic(
    n_scenes: int = 10,
    agents_per_scene: int = 8,
    duration_s: float = 24.0,
    seed: int = 0,
) -> tuple[list[RawSegment], dict[str, int]]:
    """Generate a deterministic stand-in dataset from map-free spline paths.

    Each agent's template defines a reference path (spline-integrated heading
    profile) and a constant reference speed. The "real" trajectory is a
    roll-out of the *same* controller used for simulation, with a benign
    reference config (small noise, no latency, no bias), so the real side has
    the same discrete-control texture as the simulated side. The simulator's
    realism knobs are then the only systematic difference the benchmark
    measures. Runs are cut into 6 s / 50 %-overlap windows exactly like the
    nuScenes path.
    """
    stats = {
        "n_scenes": 0,
        "n_instances": 0,
        "n_runs": 0,
        "n_segments": 0,
        "skipped_short_track": 0,
    }
    segments: list[RawSegment] = []
    master = np.random.default_rng(seed)
    scene_seeds = master.integers(0, 2**31, size=n_scenes)
    for si in range(n_scenes):
        scene_id = f"syn_{si:03d}"
        rng = np.random.default_rng(int(scene_seeds[si]))
        stats["n_scenes"] += 1
        for ai in range(agents_per_scene):
            agent_token = f"{scene_id}_agent_{ai:02d}"
            template = _SYNTHETIC_TEMPLATES[int(rng.integers(0, len(_SYNTHETIC_TEMPLATES)))]
            t, x, y, heading, speed, yaw_rate = _synthesize_run(rng, template, duration_s)
            stats["n_instances"] += 1
            stats["n_runs"] += 1
            agent_segments = 0
            for run in _rollout_reference_windows(x, y, heading, speed, scene_id, agent_token):
                run_segments = cut_windows(*run, scene_id=scene_id, agent_token=agent_token)
                segments.extend(run_segments)
                stats["n_segments"] += len(run_segments)
                agent_segments += len(run_segments)
            if agent_segments == 0:
                stats["skipped_short_track"] += 1

    logger.info(
        "synthetic ingest: %d scenes, %d agents, %d segments",
        stats["n_scenes"],
        stats["n_instances"],
        stats["n_segments"],
    )
    return segments, stats


def _synthesize_run(
    rng: np.random.Generator, template: str, duration_s: float, fs: float = FS_HZ
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build one 10 Hz (t, x, y, heading, speed, yaw_rate) run for a template."""
    n = int(round(duration_s * fs)) + 1
    t = np.arange(n) / fs
    speed = _speed_profile(rng, template, t)
    s = np.concatenate(([0.0], np.cumsum(0.5 * (speed[1:] + speed[:-1]) / fs)))
    s_max = float(s[-1])

    ds_fine = 0.05
    s_fine = np.arange(0.0, s_max + ds_fine, ds_fine)
    if s_fine.size < 3:
        s_fine = np.linspace(0.0, max(s_max, 1.0), 5)
    heading_fine = _heading_profile(rng, template, s_fine, s_max)
    dx = np.diff(s_fine)
    x_fine = np.concatenate(([0.0], np.cumsum(np.cos(heading_fine[:-1]) * dx)))
    y_fine = np.concatenate(([0.0], np.cumsum(np.sin(heading_fine[:-1]) * dx)))

    x = np.interp(s, s_fine, x_fine)
    y = np.interp(s, s_fine, y_fine)
    heading = np.interp(s, s_fine, heading_fine)
    curvature = np.interp(s, s_fine, np.gradient(heading_fine, s_fine))
    yaw_rate = speed * curvature
    return t, x, y, heading, speed, yaw_rate


def _speed_profile(rng: np.random.Generator, template: str, t: np.ndarray) -> np.ndarray:
    if template == "turn_left":
        base = rng.uniform(4.0, 7.0)
    elif template in ("cut_in", "merge"):
        base = rng.uniform(5.0, 9.0)
    else:
        base = rng.uniform(8.0, 12.0)
    # Constant speed per run: the simulator's longitudinal control is a
    # constant acceleration, so injecting speed variation here would create a
    # trivially detectable sim-real gap unrelated to the realism knobs.
    # Sensor noise is added later by the pipeline's recorder noise model.
    return np.full(t.shape, base)


def _heading_profile(
    rng: np.random.Generator, template: str, s: np.ndarray, s_max: float
) -> np.ndarray:
    """Smooth heading profile per arc length s (rad).

    Maneuvers use raised-cosine ramps with realistic durations (~20 m lane
    changes, ~35-45 % of the run for a quarter turn) so heading rates and
    lateral accelerations stay in physically plausible ranges.
    """
    if template == "cruise":
        return np.full(s.shape, rng.uniform(-0.03, 0.03))
    if template == "stop_go":
        return np.zeros(s.shape)
    if template == "turn_left":
        # Smooth quarter turn (CCW) over 35-45% of the run.
        s0 = rng.uniform(0.25, 0.3) * s_max
        s1 = s0 + rng.uniform(0.35, 0.45) * s_max
        total = np.pi / 2.0
        profile = np.zeros(s.shape)
        turning = (s > s0) & (s <= s1)
        profile[turning] = total * (1.0 - np.cos(np.pi * (s[turning] - s0) / (s1 - s0))) / 2.0
        profile[s > s1] = total
        return profile

    # Lateral-shift templates: a smooth raised-cosine heading bump with zero
    # net heading change. The lateral displacement is A * L / 2, so the
    # amplitude A = 4 / L yields a ~2 m lane change.
    if template == "merge":
        # Short bump anchored at the run start (peaks near the window edge).
        ramp_len = rng.uniform(9.0, 12.0)
        s0 = 0.0
    else:  # cut_in: a longer bump in the middle of the run
        ramp_len = rng.uniform(18.0, 26.0)
        s0 = rng.uniform(0.3, 0.5) * s_max
        s0 = min(s0, max(s_max - ramp_len - 1.0, 0.0))
    amplitude = 4.0 / ramp_len
    profile = np.zeros(s.shape)
    bump = (s > s0) & (s <= s0 + ramp_len)
    profile[bump] = amplitude * (1.0 - np.cos(2.0 * np.pi * (s[bump] - s0) / ramp_len)) / 2.0
    return profile


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m dsdbench.data.nuscenes_ingest",
        description="Ingest nuScenes (or synthetic fallback) trajectory segments.",
    )
    parser.add_argument(
        "--nuscenes-root",
        type=Path,
        default=None,
        help="Path to the nuScenes dataroot (e.g. .../v1.0-mini).",
    )
    parser.add_argument("--version", default="v1.0-mini", help="nuScenes dataset version.")
    parser.add_argument(
        "--synthetic-fallback",
        action="store_true",
        help="Generate a deterministic synthetic stand-in dataset instead of reading "
        "nuScenes (no dataset download required; used in CI).",
    )
    parser.add_argument("--n-scenes", type=int, default=10, help="Synthetic scene count.")
    parser.add_argument("--agents-per-scene", type=int, default=8, help="Synthetic agent count.")
    parser.add_argument("--duration-s", type=float, default=24.0, help="Synthetic run duration.")
    parser.add_argument("--seed", type=int, default=0, help="Synthetic dataset seed.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.synthetic_fallback:
        segments, stats = ingest_synthetic(
            args.n_scenes, args.agents_per_scene, args.duration_s, args.seed
        )
    else:
        if args.nuscenes_root is None:
            parser.error("--nuscenes-root is required unless --synthetic-fallback is set")
        segments, stats = ingest_nuscenes(args.nuscenes_root, args.version)

    logger.info("ingest done: %d segments (stats: %s)", len(segments), stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
