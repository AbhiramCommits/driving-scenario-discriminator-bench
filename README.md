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
- `dsdbench/data/` — data layer: ingestion, simulation, labeling, features,
  persistence (see below)
- `tests/cpp/` — Catch2 unit tests for the C++ core
- `tests/` — pytest tests for the Python bindings and data layer

## Data layer

`dsdbench/data/` builds the benchmark dataset end-to-end:

1. `nuscenes_ingest.py` — ingests the nuScenes v1.0-mini split (via
   `nuscenes-devkit`, install with `pip install "dsdbench[nuscenes]"`), resamples
   per-agent tracks to 10 Hz, cuts 6 s / 50 %-overlap windows in a map-relative
   frame, and skips short tracks with logged counts. `--synthetic-fallback`
   generates a deterministic stand-in dataset from map-free spline paths so the
   pipeline runs without the dataset download (CI uses this flag).
2. `sim_generator.py` — fits a smoothing-spline reference path per real segment
   and rolls out a matched simulated counterpart with `dsdbench._sim`, sampling
   `SimConfig` realism knobs that are recorded per segment as ground truth.
3. `maneuver_labels.py` — rule-based labeling (`cut_in`, `merge`,
   `unprotected_left`, `lane_keep`, `stop_and_go`), thresholds documented in the
   docstring and configurable via `maneuver_config.yaml`.
4. `features.py` — 30 scalar features per segment (jerk/lateral-accel stats,
   yaw-rate spectral energy via Welch, curvature-speed correlation,
   heading-change smoothness, TTC proxies).

```sh
# Real ingest (requires nuScenes credentials + dataroot)
python -m dsdbench.data.pipeline --out-dir data/bench --nuscenes-root /path/to/v1.0-mini

# Synthetic fallback (no dataset download)
python -m dsdbench.data.pipeline --synthetic-fallback --out-dir data/bench
```

Outputs in the out-dir: `segments.parquet` (long format, one row per timestep),
`labels.parquet`, `features.parquet`, `benchmark.duckdb` (DuckDB database with
the split tables from `queries.sql`), `queries.sql`, `manifest.json`. Splits are
assigned per scene (deterministic md5 buckets, 70/15/15) so no scene leaks
across train/val/test.

## ML discriminators

`dsdbench/models/` implements the discriminator models behind a name→constructor
registry (`dsdbench.models.registry`):

- `gbt` — LightGBM on the 30 tabular features: grouped CV by scene, early
  stopping on the val split, fixed seed, saved booster, and SHAP-based global
  feature importance (`gbt_shap.json`). Training runs in a spawned subprocess
  so LightGBM's OpenMP runtime never clashes with PyTorch's.
- `cnn` — TemporalCNN: 4 dilated 1D conv blocks (1/2/4/8), GroupNorm, GELU,
  global average pooling, linear head.
- `transformer` — TrajTransformer: 4-layer encoder, 4 heads, d_model=128,
  learned positional embeddings, CLS-token pooling.

Both temporal nets share a config dataclass and training loop (AdamW, cosine
schedule, gradient clipping, AMP on CUDA, deterministic seeding incl.
`torch.use_deterministic_algorithms` where possible) and run on CPU for CI.

```sh
pip install -e ".[dev,ml]"          # ml extras: torch, lightgbm, shap, scikit-learn
python -m dsdbench.train --model gbt --config configs/gbt.yaml
python -m dsdbench.train --model cnn --config configs/cnn.yaml --epochs 2
python -m dsdbench.train --model transformer --config configs/transformer.yaml
```

Each run writes a versioned directory under `artifacts/<model>/<version>/`
containing `config.yaml`, `metrics.json`, per-segment predicted probabilities
(`predictions.parquet`), and the model checkpoint (`checkpoint.pt` /
`model.txt` + `gbt_shap.json`).

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
