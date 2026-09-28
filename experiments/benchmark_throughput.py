"""Measure ``dsdbench._sim.batch_rollout`` throughput at 1 vs 8 threads.

Numbers are machine-dependent (they are reported in the README as such), but
the measurement itself is a committed, deterministic script so the figures can
be regenerated anywhere with ``make reproduce``.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from dsdbench._sim import SimConfig, State, batch_rollout


def make_workload(
    n_rollouts: int = 256, horizon: int = 60
) -> tuple[list[SimConfig], list[np.ndarray], list[State], list[int]]:
    """One curved reference path shared by all rollouts (zero-noise configs)."""
    s = np.linspace(0.0, 80.0, 400)
    path = np.column_stack([s, 2.0 * np.sin(0.05 * s)])
    configs: list[SimConfig] = []
    paths: list[np.ndarray] = []
    states: list[State] = []
    horizons: list[int] = []
    for i in range(n_rollouts):
        config = SimConfig()
        config.dt = 0.1
        config.pos_noise_std = 0.0
        config.heading_noise_std = 0.0
        config.steer_noise_std = 0.0
        config.latency_steps = 0
        configs.append(config)
        paths.append(path)
        states.append(State(x=0.0, y=0.0, heading=0.0, speed=6.0 + 0.01 * i))
        horizons.append(horizon)
    return configs, paths, states, horizons


def measure(n_threads: int, n_rollouts: int = 256, horizon: int = 60, repeats: int = 3) -> float:
    """Median rollouts/second for batch_rollout with ``n_threads`` workers."""
    configs, paths, states, horizons = make_workload(n_rollouts, horizon)
    times: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        batch_rollout(configs, paths, states, horizons, n_threads=n_threads)
        times.append(time.perf_counter() - start)
    return n_rollouts / float(np.median(times))


def write_throughput(
    out_path: str | Path, n_rollouts: int = 256, horizon: int = 60, repeats: int = 3
) -> dict[str, Any]:
    """Measure and persist throughput JSON; returns the measurement dict."""
    rate_1 = measure(1, n_rollouts, horizon, repeats)
    rate_8 = measure(8, n_rollouts, horizon, repeats)
    payload = {
        "rollouts_per_sec_n_threads_1": round(rate_1, 1),
        "rollouts_per_sec_n_threads_8": round(rate_8, 1),
        "speedup_8_vs_1": round(rate_8 / rate_1, 2),
        "n_rollouts": n_rollouts,
        "horizon": horizon,
        "repeats": repeats,
        "machine": platform.platform(),
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m experiments.benchmark_throughput",
        description="Measure batch_rollout throughput at 1 vs 8 threads.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "experiments" / "throughput.json",
    )
    parser.add_argument("--n-rollouts", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)

    payload = write_throughput(args.out, args.n_rollouts, args.horizon, args.repeats)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
