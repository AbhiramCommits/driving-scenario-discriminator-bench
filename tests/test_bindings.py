import numpy as np

from dsdbench import SimConfig, State, batch_rollout, rollout


def straight_path(n: int = 200) -> np.ndarray:
    return np.column_stack([np.arange(n, dtype=np.float64), np.zeros(n, dtype=np.float64)])


def make_config() -> SimConfig:
    config = SimConfig()
    config.dt = 0.1
    config.accel = 0.0
    config.pos_noise_std = 0.0
    config.heading_noise_std = 0.0
    config.steer_noise_std = 0.0
    config.steer_bias = 0.0
    config.latency_steps = 0
    return config


def test_rollout_shape_and_dtype():
    traj = rollout(straight_path(), State(speed=10.0), 100, make_config(), seed=7)
    assert traj.shape == (101, 6)
    assert traj.dtype == np.float64
    assert traj.flags["C_CONTIGUOUS"]
    assert traj[:, 0][0] == 0.0
    assert traj[:, 0][-1] == 100 * 0.1


def test_rollout_matches_closed_form():
    v0, horizon = 10.0, 100
    traj = rollout(straight_path(), State(speed=v0), horizon, make_config(), seed=1)
    t = traj[:, 0]
    np.testing.assert_allclose(traj[:, 1], v0 * t, atol=1e-9)
    np.testing.assert_allclose(traj[:, 2], 0.0, atol=1e-9)
    np.testing.assert_allclose(traj[:, 3], 0.0, atol=1e-9)
    np.testing.assert_allclose(traj[:, 4], v0, atol=1e-12)
    np.testing.assert_allclose(traj[:, 5], 0.0, atol=1e-12)


def test_batch_rollout_shape_and_dtype():
    configs = [make_config() for _ in range(5)]
    paths = [straight_path() for _ in range(5)]
    states = [State(speed=5.0 + i) for i in range(5)]
    horizons = [10 + i for i in range(5)]
    result = batch_rollout(configs, paths, states, horizons, n_threads=4)
    assert len(result) == 5
    for traj, horizon in zip(result, horizons, strict=True):
        assert traj.shape == (horizon + 1, 6)
        assert traj.dtype == np.float64


def test_batch_rollout_identical_for_1_vs_8_threads():
    configs = [make_config() for _ in range(8)]
    paths = [straight_path() for _ in range(8)]
    states = [State(speed=5.0 + i) for i in range(8)]
    horizons = [10 + i for i in range(8)]
    single = batch_rollout(configs, paths, states, horizons, n_threads=1)
    multi = batch_rollout(configs, paths, states, horizons, n_threads=8)
    assert len(single) == len(multi) == 8
    for a, b in zip(single, multi, strict=True):
        assert a.shape == b.shape
        assert a.dtype == np.float64
        assert np.array_equal(a, b)


def test_batch_rollout_repeated_calls_identical():
    configs = [make_config() for _ in range(4)]
    paths = [straight_path() for _ in range(4)]
    states = [State(speed=4.0) for _ in range(4)]
    horizons = [20, 21, 22, 23]
    first = batch_rollout(configs, paths, states, horizons, n_threads=4)
    second = batch_rollout(configs, paths, states, horizons, n_threads=4)
    for a, b in zip(first, second, strict=True):
        assert np.array_equal(a, b)
