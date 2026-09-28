"""pytest-benchmark benchmarks for ``dsdbench._sim.batch_rollout``.

Not collected by the normal test suite (``testpaths = ["tests"]``); run with:

    pytest benchmarks --benchmark-only --benchmark-json docs/bench.json
"""

from __future__ import annotations

import numpy as np

from dsdbench._sim import SimConfig, State, batch_rollout


def make_workload(n_rollouts: int = 512, horizon: int = 60) -> tuple:
    s = np.linspace(0.0, 80.0, 400)
    path = np.column_stack([s, 2.0 * np.sin(0.05 * s)])
    configs = []
    paths = []
    states = []
    horizons = []
    for i in range(n_rollouts):
        config = SimConfig()
        config.dt = 0.1
        config.pos_noise_std = 0.01
        config.heading_noise_std = 0.001
        config.steer_noise_std = 0.001
        config.latency_steps = 2
        configs.append(config)
        paths.append(path)
        states.append(State(x=0.0, y=0.0, heading=0.0, speed=6.0 + 0.01 * i))
        horizons.append(horizon)
    return configs, paths, states, horizons


def test_batch_rollout_1_thread(benchmark) -> None:
    workload = make_workload()

    def run() -> None:
        batch_rollout(*workload, n_threads=1)

    benchmark(run)


def test_batch_rollout_8_threads(benchmark) -> None:
    workload = make_workload()

    def run() -> None:
        batch_rollout(*workload, n_threads=8)

    benchmark(run)


def test_batch_rollout_1_thread_tiny_buffer(benchmark) -> None:
    """Horizon-1 rollouts: measures per-rollout overhead with minimal
    trajectory-buffer writes (used for the SoA write-cost estimate in
    docs/PERF.md)."""
    configs, paths, states, horizons = make_workload(horizon=60)
    configs = configs[:64]
    paths = paths[:64]
    states = states[:64]
    horizons = [1] * 64

    def run() -> None:
        batch_rollout(configs, paths, states, horizons, n_threads=1)

    benchmark(run)
