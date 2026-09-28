# Performance notes

All measurements below are reproducible with committed scripts (this machine:
macOS arm64, Python 3.13, CMake Release build). Absolute numbers are
machine-dependent; ratios are the meaningful quantities.

## batch_rollout throughput

Measured two ways, both committed:

```sh
python -m experiments.benchmark_throughput     # -> experiments/throughput.json
pytest benchmarks --benchmark-only             # pytest-benchmark, 512 rollouts x horizon 60
```

| measurement | 1 thread | 8 threads | speedup |
| --- | --- | --- | --- |
| `benchmark_throughput` (256 rollouts, median of 3) | 10.6k rollouts/s | 39.5k rollouts/s | 3.71x |
| pytest-benchmark (512 rollouts, noisy configs, latency=2) | 8.1k rollouts/s | 27.5k rollouts/s | 3.4x |

The `experiments/throughput.json` numbers are the ones quoted in the README;
`make reproduce` regenerates them.

## Where the time goes

`cProfile` attributes ~100% of the call to the C++ `batch_rollout` builtin
(~102 µs/rollout at 1 thread for a 61-step rollout), so Python-side overhead
outside the binding is negligible. The horizon scaling identifies the
bottleneck:

- horizon 1 rollouts run at ~116k rollouts/s (~8.6 µs each);
- horizon 60 rollouts run at ~8.1k rollouts/s (~102 µs each);

i.e. per-rollout cost scales with the number of control steps (the pure-pursuit
segment scan with `sqrt`/`atan2` per step), not with the trajectory buffer
size. The trajectory is ~2.9 KB per rollout (61 x 6 float64); at 10k
rollouts/s that is ~29 MB/s of write traffic, far below memory bandwidth.

## Struct-of-arrays evaluation (verdict: not adopted)

A SIMD-friendly struct-of-arrays (SoA) trajectory buffer — six contiguous
per-channel arrays instead of interleaved (t, x, y, heading, speed, yaw_rate)
rows — was evaluated because per-step channels would stream contiguously.
It is **not adopted**, because profiling does not justify it:

1. The write path is already sequential (48-byte contiguous rows) and totals
   <1% of achievable bandwidth at current throughput.
2. The bottleneck is per-step pursuit compute (see horizon scaling above), not
   buffer layout.
3. Consumers (NumPy bindings, features) want the AoS (T, 6) layout, so an SoA
   core would add an exact 6-way interleave (a full extra pass over every
   trajectory) at the boundary, costing more than it saves.

If per-step compute were ever optimized to the point that writes dominate
(~10x faster than today), the SoA change should be revisited with the
same benchmark before/after.

## CUDA note

No GPU is available on the development machine
(`torch.cuda.is_available() == False`), so only CPU numbers are documented
here. The temporal trainer already selects CUDA when present and enables AMP
with a `GradScaler` (see `dsdbench/models/temporal.py:fit`). To record
GPU-vs-CPU batch inference timings, run on a CUDA host:

```sh
python - << 'EOF'
import time, torch
from dsdbench.models.temporal import TemporalConfig, build_model, predict_probs
import numpy as np
x = np.random.default_rng(0).normal(size=(256, 61, 6)).astype("float32")
for device in ("cpu", "cuda"):
    model = build_model(TemporalConfig(arch="transformer")).to(device)
    for _ in range(3):
        predict_probs(model, x, device)
    t0 = time.perf_counter()
    for _ in range(20):
        predict_probs(model, x, device)
    print(device, f"{20 * 256 / (time.perf_counter() - t0):.0f} trajectories/s")
EOF
```
