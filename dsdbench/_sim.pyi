from collections.abc import Sequence
from typing import Any

import numpy as np

__version__: str

class State:
    x: float
    y: float
    heading: float
    speed: float
    steer_angle: float

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        heading: float = 0.0,
        speed: float = 0.0,
        steer_angle: float = 0.0,
    ) -> None: ...

class Control:
    steer: float
    accel: float

    def __init__(self, steer: float = 0.0, accel: float = 0.0) -> None: ...

class ControllerErrorModel:
    steer_bias: float
    steer_noise_std: float
    latency_steps: int

    def __init__(
        self,
        steer_bias: float = 0.0,
        steer_noise_std: float = 0.0,
        latency_steps: int = 0,
    ) -> None: ...

class SimConfig:
    dt: float
    wheelbase: float
    max_steer: float
    max_accel: float
    max_decel: float
    accel: float
    lookahead_gain: float
    min_lookahead: float
    pos_noise_std: float
    heading_noise_std: float
    latency_steps: int
    steer_bias: float
    steer_noise_std: float

    def __init__(self) -> None: ...

class KinematicBicycleModel:
    def __init__(
        self,
        wheelbase: float,
        max_steer: float,
        max_accel: float,
        max_decel: float,
        dt: float,
    ) -> None: ...
    def step(self, state: State, control: Control) -> State: ...
    @property
    def wheelbase(self) -> float: ...
    @property
    def max_steer(self) -> float: ...
    @property
    def max_accel(self) -> float: ...
    @property
    def max_decel(self) -> float: ...
    @property
    def dt(self) -> float: ...

class PurePursuitController:
    def __init__(
        self,
        lookahead_gain: float = 3.0,
        min_lookahead: float = 0.5,
        error_model: ControllerErrorModel = ...,
    ) -> None: ...
    def reset(self, seed: int) -> None: ...
    def compute_steer(
        self, state: State, path: np.ndarray[Any, Any], wheelbase: float
    ) -> float: ...

class NoiseModel:
    def __init__(self, pos_std: float = 0.0, heading_std: float = 0.0) -> None: ...
    def reset(self, seed: int) -> None: ...
    def observe(self, state: State) -> State: ...

def rollout(
    path: np.ndarray[Any, Any],
    initial_state: State,
    horizon: int,
    config: SimConfig,
    seed: int = 0,
) -> np.ndarray[Any, np.dtype[np.float64]]: ...
def batch_rollout(
    configs: Sequence[SimConfig],
    paths: Sequence[np.ndarray[Any, Any]],
    initial_states: Sequence[State],
    horizons: Sequence[int],
    n_threads: int = 1,
    seed_base: int = ...,
) -> list[np.ndarray[Any, np.dtype[np.float64]]]: ...
