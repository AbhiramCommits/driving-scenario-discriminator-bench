# driving-scenario-discriminator-bench

Benchmark for machine-learning discriminators that separate **real logged driving
trajectories** from **simulated ones**.

## Goal

Given a dataset of driving trajectories — some recorded from real vehicles, some
produced by a simulator — the task is to train and evaluate binary discriminators
that classify each trajectory as real or simulated. The simulator in this repo
(`dsdbench._sim`) is a controllable, seeded source of synthetic trajectories
(kinematic bicycle model + pure-pursuit tracking with injectable error models),
so the real-vs-simulated gap can be tuned and studied systematically.

## Layout

- `src/sim/` — C++17 simulator core: kinematic bicycle model, pure-pursuit
  controller, observation noise models, deterministic batch rollout engine
- `src/bindings/` — pybind11 bindings exposed as `dsdbench._sim`
- `dsdbench/` — Python package (Python 3.11+)
- `tests/cpp/` — Catch2 unit tests for the C++ core
- `tests/` — pytest tests for the Python bindings

## Install

Requires Python 3.11+ and CMake 3.22+.

```sh
pip install -e ".[dev]"
```

## Test

```sh
pytest                          # Python bindings
cmake -S . -B build -DDSD_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build          # C++ unit tests
```

## Development

`ruff`, `mypy`, `clang-format` and `pytest` are wired into pre-commit:

```sh
pre-commit install
pre-commit run --all-files
```
