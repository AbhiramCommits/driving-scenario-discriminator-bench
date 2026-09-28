"""Generate matched simulated counterpart segments for every real segment.

For each real segment a smooth reference path is fitted through the observed
(x, y) points (arc-length-parameterized smoothing spline, extended a few metres
past the end), a ``SimConfig`` is sampled from the documented realism knobs,
and ``dsdbench._sim.batch_rollout`` tracks that path from the observed initial
state with the same duration. The sampled config is recorded per segment so the
realism knobs are known ground truth for the discriminator benchmark.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import splev, splprep

from dsdbench._sim import SimConfig, State, batch_rollout
from dsdbench.data.nuscenes_ingest import RawSegment

logger = logging.getLogger(__name__)

# Must match dsd_sim::kDefaultSeedBase; per-scenario rollout seeds are derived
# from it and the scenario index, so generation is deterministic.
DEFAULT_SEED_BASE = 0x243F6A8885A308D3

# Realism knobs sampled per segment. (lo, hi) ranges; stddev knobs are sampled
# log-uniformly, the rest uniformly. dt is fixed at 10 Hz and max_steer is fixed.
SAMPLING_RANGES: dict[str, tuple[float, float]] = {
    "wheelbase": (2.4, 3.2),
    "max_accel": (2.5, 4.0),
    "max_decel": (4.0, 6.0),
    "accel": (-0.5, 0.5),
    "lookahead_gain": (2.0, 5.0),
    "min_lookahead": (0.2, 1.0),
    "pos_noise_std": (1e-3, 0.15),
    "heading_noise_std": (1e-4, 0.03),
    "steer_bias": (-0.1, 0.1),
    "steer_noise_std": (1e-4, 0.05),
}

CONFIG_KEYS = (
    "dt",
    "wheelbase",
    "max_steer",
    "max_accel",
    "max_decel",
    "accel",
    "lookahead_gain",
    "min_lookahead",
    "pos_noise_std",
    "heading_noise_std",
    "latency_steps",
    "steer_bias",
    "steer_noise_std",
)


@dataclass
class SimSegment:
    """A simulated segment paired with the SimConfig that produced it."""

    segment: RawSegment
    config: dict[str, Any]


def _seed_from_str(s: str) -> int:
    """Deterministic 32-bit seed from a string (stable across processes)."""
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:16], 16) % 2**32


def sample_config(
    rng: np.random.Generator, *, dt: float = 0.1, latency_max: int = 4
) -> tuple[SimConfig, dict[str, float | int]]:
    """Sample one SimConfig; returns the config and its plain-dict record."""
    cfg = SimConfig()
    cfg.dt = dt
    cfg.wheelbase = float(rng.uniform(*SAMPLING_RANGES["wheelbase"]))
    cfg.max_steer = 0.6  # fixed
    cfg.max_accel = float(rng.uniform(*SAMPLING_RANGES["max_accel"]))
    cfg.max_decel = float(rng.uniform(*SAMPLING_RANGES["max_decel"]))
    cfg.accel = float(rng.uniform(*SAMPLING_RANGES["accel"]))
    cfg.lookahead_gain = float(rng.uniform(*SAMPLING_RANGES["lookahead_gain"]))
    cfg.min_lookahead = float(rng.uniform(*SAMPLING_RANGES["min_lookahead"]))
    cfg.pos_noise_std = _loguniform(rng, "pos_noise_std")
    cfg.heading_noise_std = _loguniform(rng, "heading_noise_std")
    cfg.latency_steps = int(rng.integers(0, latency_max + 1))
    cfg.steer_bias = float(rng.uniform(*SAMPLING_RANGES["steer_bias"]))
    cfg.steer_noise_std = _loguniform(rng, "steer_noise_std")
    record = {
        "dt": float(cfg.dt),
        "wheelbase": float(cfg.wheelbase),
        "max_steer": float(cfg.max_steer),
        "max_accel": float(cfg.max_accel),
        "max_decel": float(cfg.max_decel),
        "accel": float(cfg.accel),
        "lookahead_gain": float(cfg.lookahead_gain),
        "min_lookahead": float(cfg.min_lookahead),
        "pos_noise_std": float(cfg.pos_noise_std),
        "heading_noise_std": float(cfg.heading_noise_std),
        "latency_steps": int(cfg.latency_steps),
        "steer_bias": float(cfg.steer_bias),
        "steer_noise_std": float(cfg.steer_noise_std),
    }
    return cfg, record


def _loguniform(rng: np.random.Generator, key: str) -> float:
    lo, hi = SAMPLING_RANGES[key]
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def fit_reference_path(
    x: ArrayLike,
    y: ArrayLike,
    *,
    n_points: int = 200,
    tail_m: float = 10.0,
    smoothing_rms: float = 0.05,
) -> np.ndarray:
    """Fit a smoothing spline through (x, y) and return a (N, 2) polyline.

    The spline is parameterized by arc length, smoothed to roughly
    ``smoothing_rms`` metres of residual, resampled to ``n_points`` points and
    extended ``tail_m`` metres along the final tangent so the pure-pursuit
    controller has lookahead beyond the segment end.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.size < 4:
        raise ValueError("reference path needs at least 4 points")
    arc = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(x_arr), np.diff(y_arr)))))
    # Drop points that do not advance the path: splprep needs strictly
    # increasing parameter values (stopped stretches repeat coordinates).
    keep = np.concatenate(([True], np.diff(arc) > 1e-6))
    xs, ys, arcs = x_arr[keep], y_arr[keep], arc[keep]
    if xs.size < 4 or arcs[-1] < 1e-3:
        # Stationary segment: a degenerate path the controller never leaves.
        return np.tile(np.array([x_arr[0], y_arr[0]]), (n_points, 1))

    u = arcs / arcs[-1]
    k = min(3, xs.size - 1)
    try:
        tck, _ = splprep([xs, ys], u=u, s=smoothing_rms**2 * xs.size, k=k)
    except ValueError:
        # Some tracks are too short/regular for the requested smoothing budget.
        tck, _ = splprep([xs, ys], u=u, s=0.0, k=k)
    uu = np.linspace(0.0, 1.0, n_points)
    points = np.column_stack(splev(uu, tck))

    tangent = np.asarray(splev([1.0], tck, der=1), dtype=np.float64).reshape(2)
    tangent_norm = float(np.hypot(tangent[0], tangent[1]))
    if tangent_norm < 1e-12:
        return points
    tangent /= tangent_norm
    tail_offsets = 0.5 * np.arange(1, int(tail_m / 0.5) + 1)[:, None]
    tail = points[-1] + tangent * tail_offsets
    return np.vstack([points, tail])


def generate_matched(
    real_segments: Sequence[RawSegment],
    n_threads: int | None = None,
    seed_base: int = DEFAULT_SEED_BASE,
) -> list[SimSegment]:
    """Generate one simulated counterpart per real segment via batch_rollout.

    Deterministic: config sampling is seeded from (scene, agent) ids and the
    rollout seeds come from ``seed_base`` + scenario index, independent of
    thread count.
    """
    if not real_segments:
        return []
    n_threads = n_threads or max(1, min(8, os.cpu_count() or 4))

    configs: list[SimConfig] = []
    paths: list[np.ndarray] = []
    states: list[State] = []
    horizons: list[int] = []
    records: list[dict[str, float | int]] = []

    for seg in real_segments:
        config_seed = _seed_from_str(f"{seg.scene_id}:{seg.agent_token}")
        rng = np.random.default_rng(config_seed)
        cfg, record = sample_config(rng)
        if float(np.max(seg.speed)) < 0.5:
            # A stationary real vehicle should not be simulated as creeping away.
            cfg.accel = min(cfg.accel, 0.0)
            record["accel"] = float(cfg.accel)
        record["config_seed"] = config_seed
        configs.append(cfg)
        paths.append(fit_reference_path(seg.x, seg.y))
        states.append(
            State(
                x=float(seg.x[0]),
                y=float(seg.y[0]),
                heading=float(seg.heading[0]),
                speed=float(seg.speed[0]),
            )
        )
        horizons.append(int(seg.t.size - 1))
        records.append(record)

    trajectories = batch_rollout(
        configs, paths, states, horizons, n_threads=n_threads, seed_base=seed_base
    )
    logger.info("generated %d simulated segments (n_threads=%d)", len(trajectories), n_threads)

    sims: list[SimSegment] = []
    for real, traj, record in zip(real_segments, trajectories, records, strict=True):
        sims.append(
            SimSegment(
                RawSegment(
                    scene_id=real.scene_id,
                    agent_token=real.agent_token,
                    t=traj[:, 0].copy(),
                    x=traj[:, 1].copy(),
                    y=traj[:, 2].copy(),
                    heading=traj[:, 3].copy(),
                    speed=traj[:, 4].copy(),
                    yaw_rate=traj[:, 5].copy(),
                ),
                config=record,
            )
        )
    return sims
