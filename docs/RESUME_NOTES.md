# Resume notes: measured claims and how to reproduce them

Every claim below is backed by a committed artifact or script. The number is
stated, the file that contains it, and the exact command that regenerates it.

## 1. Simulator throughput and threading speedup

**Claim.** `batch_rollout` sustains 10,645 rollouts/s on 1 thread and 39,537
rollouts/s on 8 threads — a **3.71x** speedup — on the committed 256-rollout
x 60-step workload (macOS arm64; absolute rates are machine-dependent, the
speedup is the claim).

- File: `experiments/throughput.json`
- Command: `python -m experiments.benchmark_throughput`
- Cross-check: `pytest benchmarks --benchmark-only` (see `docs/PERF.md`)

## 2. Discriminator AUCs with scene-clustered BCa confidence intervals

**Claim.** Mean AUC-ROC across the 18 sweep configurations (test split,
95% scene-level BCa bootstrap CI bounds likewise averaged):

| model | mean AUC | mean CI |
| --- | --- | --- |
| transformer | 0.536 | [0.520, 0.551] |
| cnn | 0.655 | [0.620, 0.679] |
| gbt | 0.873 | [0.795, 0.912] |

- File: `experiments/results.parquet` (rows with `scope = 'overall'`)
- Command: `python -m experiments.write_readme` (regenerates the README
  table); raw query against `experiments/results.duckdb`:
  `SELECT model, avg(auc), avg(ci_low), avg(ci_high) FROM sweep_results
  WHERE scope = 'overall' GROUP BY 1`
- Method: `dsdbench/eval/bootstrap.py` (Efron 1987; Field & Welsh 2007);
  coverage of the CI procedure is itself tested in
  `tests/test_eval.py::test_bootstrap_ci_coverage_nominal` (200 trials).

## 3. Calibration improvement from recalibration

**Claim.** At the mid-knob configuration, isotonic recalibration (fit on
val, applied to test) moves ECE from 0.4234 to 0.0694 for gbt. For the
temporal models on this tiny val split it degrades ECE (cnn 0.1451 to
0.4173, transformer 0.1015 to 0.4722) — reported honestly in the README.

- File: `experiments/results.parquet` (columns `ece`, `ece_after_isotonic` at
  `knob_noise='mid' AND knob_latency=2 AND knob_bias=0`)
- Command:
  `SELECT model, ece, ece_after_isotonic FROM sweep_results WHERE
  scope='overall' AND knob_noise='mid' AND knob_latency=2 AND knob_bias=0`
- Method: `dsdbench/eval/calibration.py` (Platt 1999; Zadrozny & Elkan 2002;
  ECE per Naeini et al. 2015).

## 4. Power-analysis sample size

**Claim.** Detecting a discriminator AUC of 0.55 against the chance null
(AUC = 0.5) at 80% power with α = 0.05 (one-sided, balanced classes,
Hanley-McNeil variance) requires **816 segments**.

- Command: `python -c "from dsdbench.eval.power import required_samples;
  print(required_samples(0.55, alpha=0.05, power=0.8))"`
- Method: `dsdbench/eval/power.py` (Hanley & McNeil 1982); the variance and
  power formulas are verified against a hand-computed reference in
  `tests/test_eval.py::test_power_matches_hand_computed_reference`.

## 5. Slices under FDR-corrected regression gating

**Claim.** The committed baseline tracks **7 slices** (4 maneuver slices:
`cut_in`, `lane_keep`, `merge`, `unprotected_left`; 3 knob slices:
`noise_low`, `noise_high`, `latency_gt0`), each with its AUC and CI recorded;
`dsdbench.evaluate --baseline` exits non-zero when any slice's CI lower bound
falls more than 0.05 below its baseline AUC or a slice disappears
(criterion derived in `docs/METHODOLOGY.md`).

- File: `artifacts/baseline.json`
- Command:
  `python -m dsdbench.evaluate --run artifacts/cnn/<version>
  --baseline artifacts/baseline.json` (exit code 1 = regression)
- Method: `dsdbench/eval/slices.py` (Benjamini & Hochberg 1995 FDR across
  slice tests; regression criterion per `check_regression`).

## 6. Determinism across repeats and thread counts

**Claim.** The same seeded workload produces **bit-identical** trajectories
across 5 repeats at each of 1/2/4/8 threads, and the BCa bootstrap CI is
identical for 1 vs 4 joblib workers.

- Command: `python scripts/profile_determinism.py` (wired into CI)
- Supporting C++ tests (via `ctest`): straight-line rollout matches the
  closed-form constant-velocity solution to 1e-9; zero-noise batch rollouts
  are bit-identical for 1 vs 8 threads; steering saturates at max_steer.

## 7. Which knob degrades realism most (sweep finding)

**Claim.** Observation **noise** is the knob whose levels spread the mean AUC
the most (0.578 at `noise_low` to 0.785 at `noise_high`, spread 0.207),
followed by actuation latency (0.640 to 0.766); controller bias barely moves
it (0.685 to 0.691). The maneuver slice hardest to simulate faithfully is
**cut_in** (mean AUC 0.780 across configurations and models).

- File: `experiments/results.duckdb` (table `sweep_results`)
- Command: the sweep section of `dsdbench/data/queries.sql` (executed by
  `python -m experiments.run_benchmark`); regenerate everything with
  `make reproduce`
- Caveats: see README "Limitations" — the sweep pins secondary knobs and the
  synthetic "real" side uses a benign reference rollout of the same
  controller, so knob effects are associative, not causal.
