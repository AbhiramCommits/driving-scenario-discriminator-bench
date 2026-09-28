"""Profile determinism: identical output across repeats and thread counts.

Runs the same seeded batch_rollout workload 20 times (5 repeats at 1, 2, 4
and 8 threads) and asserts every run produces bit-identical trajectories.
Also verifies that the scene-level BCa bootstrap CI is identical across
joblib worker counts. Exits non-zero on any mismatch; wired into CI.

    python scripts/profile_determinism.py
"""

from __future__ import annotations

import hashlib
import sys

import numpy as np

from dsdbench._sim import SimConfig, State, batch_rollout
from dsdbench.eval.bootstrap import bootstrap_auc

REPEATS_PER_THREAD_COUNT = 5
THREAD_COUNTS = (1, 2, 4, 8)


def build_workload(n_scenarios: int = 48, horizon: int = 60, seed: int = 0) -> tuple:
    """Seeded curved-path workload with noisy configs and per-scenario seeds."""
    rng = np.random.default_rng(seed)
    configs = []
    paths = []
    states = []
    horizons = []
    for i in range(n_scenarios):
        s = np.linspace(0.0, 60.0, 300)
        path = np.column_stack([s, (1.0 + 0.5 * (i % 3)) * np.sin(0.04 * s + 0.1 * i)])
        config = SimConfig()
        config.dt = 0.1
        config.pos_noise_std = 0.05
        config.heading_noise_std = 0.01
        config.steer_noise_std = 0.02
        config.latency_steps = 3
        config.steer_bias = 0.1
        configs.append(config)
        paths.append(path)
        states.append(
            State(
                x=float(rng.uniform(-2, 2)),
                y=float(rng.uniform(-2, 2)),
                heading=float(rng.uniform(-0.3, 0.3)),
                speed=float(rng.uniform(4, 10)),
            )
        )
        horizons.append(horizon)
    return configs, paths, states, horizons


def trajectory_hash(trajectories: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for traj in trajectories:
        digest.update(np.ascontiguousarray(traj).tobytes())
    return digest.hexdigest()


def main() -> int:
    configs, paths, states, horizons = build_workload()

    print(
        f"batch_rollout determinism: {REPEATS_PER_THREAD_COUNT} repeats at "
        f"{THREAD_COUNTS} thread count(s)"
    )
    hashes: dict[int, str] = {}
    for n_threads in THREAD_COUNTS:
        seen: set[str] = set()
        for _rep in range(REPEATS_PER_THREAD_COUNT):
            trajectories = batch_rollout(configs, paths, states, horizons, n_threads=n_threads)
            seen.add(trajectory_hash(trajectories))
        assert len(seen) == 1, f"non-identical outputs across repeats at {n_threads} threads"
        hashes[n_threads] = seen.pop()
        print(
            f"  {n_threads} threads: {hashes[n_threads][:16]}... (all "
            f"{REPEATS_PER_THREAD_COUNT} repeats identical)"
        )
    if len(set(hashes.values())) != 1:
        print("FAIL: outputs differ across thread counts", file=sys.stderr)
        return 1
    print("  batch_rollout: bit-identical across all thread counts")

    print("BCa bootstrap determinism: n_jobs 1 vs 4")
    rng = np.random.default_rng(7)
    scenes = np.repeat(np.arange(12), 12)
    probs = 1.0 / (1.0 + np.exp(-(rng.normal(0, 1, scenes.size))))
    y = (rng.random(scenes.size) < probs).astype(float)
    ci_serial = bootstrap_auc(y, probs, scenes, n_resamples=400, seed=1, n_jobs=1)
    ci_parallel = bootstrap_auc(y, probs, scenes, n_resamples=400, seed=1, n_jobs=4)
    if (ci_serial.ci_low, ci_serial.ci_high) != (ci_parallel.ci_low, ci_parallel.ci_high):
        print("FAIL: bootstrap CI depends on n_jobs", file=sys.stderr)
        return 1
    print(
        f"  bootstrap CI: [{ci_serial.ci_low:.6f}, {ci_serial.ci_high:.6f}] "
        f"identical for n_jobs 1 and 4"
    )

    print("PASS: deterministic across repeats, thread counts, and worker counts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
