"""Tests for the dsdbench statistical evaluation harness."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.stats import kstest, norm

from dsdbench.eval.bootstrap import bca_bootstrap
from dsdbench.eval.calibration import recalibrate
from dsdbench.eval.metrics import (
    auc_roc,
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
    youden_accuracy,
)
from dsdbench.eval.permutation import paired_permutation_test, permutation_test
from dsdbench.eval.power import achieved_power, hanley_mcneil_se, required_samples
from dsdbench.eval.slices import (
    SliceReport,
    SliceResult,
    baseline_from_report,
    check_regression,
)

RNG = np.random.default_rng(0)


def _logistic(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _scene_scores(
    n_scenes: int, per_scene: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scores with within-scene correlation: y | score via logistic model."""
    rng = np.random.default_rng(seed)
    scenes = np.repeat(np.arange(n_scenes), per_scene)
    scene_effect = rng.normal(0.0, 0.5, size=n_scenes)
    score = scene_effect[scenes // per_scene] + rng.normal(0.0, 1.0, size=scenes.size)
    probs = _logistic(score)
    y = (rng.random(scenes.size) < probs).astype(np.float64)
    return y, probs, scenes


# ---------------------------------------------------------------------------
# Bootstrap CI coverage
# ---------------------------------------------------------------------------


def _fast_auc(y: np.ndarray, probs: np.ndarray) -> float:
    """Vectorized Mann-Whitney AUC (keeps the 200-trial coverage test fast)."""
    pos = probs[y == 1]
    neg = probs[y == 0]
    if pos.size == 0 or neg.size == 0:
        return 0.5
    return float(np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :]))


def test_bootstrap_ci_coverage_nominal():
    n_trials = 200
    n_scenes, per_scene = 20, 10
    # Population AUC reference via a large Monte Carlo sample of the same DGP
    # (single O(n log n) AUC call -- do not use _fast_auc here, it is O(n^2)).
    big = np.random.default_rng(1234)
    n_big = 400_000
    big_scene_effect = big.normal(0.0, 0.5, size=n_big // 20)
    big_score = np.repeat(big_scene_effect, 20) + big.normal(0.0, 1.0, size=n_big)
    big_y = (big.random(n_big) < _logistic(big_score)).astype(np.float64)
    population_auc = auc_roc(big_y, _logistic(big_score))

    covered = 0
    for trial in range(n_trials):
        y, probs, scenes = _scene_scores(n_scenes, per_scene, seed=1000 + trial)
        ci = bca_bootstrap(
            y,
            probs,
            scenes,
            _fast_auc,
            n_resamples=600,
            seed=2000 + trial,
            n_jobs=1,
        )
        covered += int(ci.ci_low <= population_auc <= ci.ci_high)
    coverage = covered / n_trials
    # Nominal 95% coverage; clustered-data BCa runs slightly conservative and
    # binomial sampling noise around 200 trials is ~0.03.
    assert 0.90 <= coverage <= 0.99, f"coverage = {coverage:.3f}"


def test_bootstrap_deterministic_across_jobs():
    y, probs, scenes = _scene_scores(10, 20, seed=7)
    ci_serial = bca_bootstrap(
        y, probs, scenes, lambda yy, pp: auc_roc(yy, pp), n_resamples=100, seed=1, n_jobs=1
    )
    ci_parallel = bca_bootstrap(
        y, probs, scenes, lambda yy, pp: auc_roc(yy, pp), n_resamples=100, seed=1, n_jobs=2
    )
    assert ci_serial.ci_low == pytest.approx(ci_parallel.ci_low, abs=1e-12)
    assert ci_serial.ci_high == pytest.approx(ci_parallel.ci_high, abs=1e-12)


# ---------------------------------------------------------------------------
# Permutation p-values
# ---------------------------------------------------------------------------


def test_permutation_pvalue_uniform_under_null():
    # Under the null, y is independent of probs: p-values must be uniform on
    # the exact-permutation grid (Phipson & Smyth 2010).
    rng = np.random.default_rng(0)
    p_values = []
    n_trials, n_permutations = 150, 99
    for trial in range(n_trials):
        y = rng.integers(0, 2, size=60).astype(np.float64)
        probs = rng.random(60)
        result = permutation_test(y, probs, n_permutations=n_permutations, seed=trial, n_jobs=1)
        p_values.append(result.p_value)
    p_values = np.asarray(p_values)
    # One-sample KS test against uniform; the grid is discrete so the KS
    # statistic is small but not zero.
    ks = kstest(p_values, "uniform")
    assert ks.statistic < 0.15, f"KS statistic {ks.statistic:.3f} (p={ks.pvalue:.3f})"
    # The exact p-value never reports zero.
    assert (p_values > 0.0).all()
    assert (p_values <= 1.0).all()


def test_permutation_detects_signal():
    y, probs, _ = _scene_scores(12, 20, seed=5)
    result = permutation_test(y, probs, n_permutations=200, seed=3, n_jobs=1)
    assert result.p_value <= 1.0 / (1.0 + 200)


def test_paired_permutation_symmetric_when_exchangeable():
    y, probs, _ = _scene_scores(12, 20, seed=9)
    permutation_test(y, probs, n_permutations=100, seed=1, n_jobs=1)
    b = paired_permutation_test(y, probs, probs, n_permutations=100, seed=1, n_jobs=1)
    assert b.observed_statistic == pytest.approx(0.0, abs=1e-12)
    assert b.p_value == 1.0


# ---------------------------------------------------------------------------
# Power analysis
# ---------------------------------------------------------------------------


def test_power_matches_hand_computed_reference():
    auc, n_pos, n_neg = 0.7, 100, 100
    # Hand-computed Hanley-McNeil variance (Hanley & McNeil 1982, eq. 1-2).
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc**2 / (1.0 + auc)
    var = (auc * (1.0 - auc) + (n_pos - 1) * (q1 - auc**2) + (n_neg - 1) * (q2 - auc**2)) / (
        n_pos * n_neg
    )
    se_hand = np.sqrt(var)
    assert hanley_mcneil_se(auc, n_pos, n_neg) == pytest.approx(se_hand, rel=1e-12)

    # Hand-computed power of the one-sided test against AUC_null = 0.5.
    z_crit = norm.ppf(0.95)
    power_hand = norm.cdf((auc - 0.5) / se_hand - z_crit)
    assert achieved_power(auc, n_pos + n_neg) == pytest.approx(power_hand, rel=1e-12)


def test_required_samples_reaches_target_power():
    n = required_samples(0.65, power=0.8)
    assert achieved_power(0.65, n) >= 0.8
    assert achieved_power(0.65, n - 2) < 0.8
    assert n % 2 == 0


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------


def test_ece_zero_for_perfectly_calibrated():
    # Every bin is exactly calibrated by construction: two predicted
    # probabilities whose empirical positive fractions match exactly.
    y = np.concatenate([np.ones(150), np.zeros(350), np.ones(400), np.zeros(100)]).astype(float)
    probs = np.concatenate([np.full(500, 0.3), np.full(500, 0.8)])
    assert expected_calibration_error(y, probs, n_bins=2) == pytest.approx(0.0, abs=1e-12)
    assert maximum_calibration_error(y, probs, n_bins=2) == pytest.approx(0.0, abs=1e-12)


def test_ece_positive_when_miscalibrated():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=1000).astype(float)
    probs = np.full(1000, 0.9)  # wildly overconfident
    assert expected_calibration_error(y, probs, n_bins=10) > 0.3
    assert maximum_calibration_error(y, probs, n_bins=10) > 0.3


def test_recalibration_reduces_ece_on_overconfident_scores():
    rng = np.random.default_rng(0)
    n = 4000
    z = rng.normal(0.0, 1.0, size=n)
    y = (rng.random(n) < _logistic(z)).astype(float)
    probs = _logistic(4.0 * z)  # temperature-scaled: overconfident
    order = rng.permutation(n)
    y_val, p_val = y[order[: n // 2]], probs[order[: n // 2]]
    y_test, p_test = y[order[n // 2 :]], probs[order[n // 2 :]]
    result = recalibrate(y_val, p_val, y_test, p_test, n_bins=10)
    assert result.ece_after_isotonic < result.ece_before
    assert result.ece_after_platt < result.ece_before
    assert result.probs_isotonic.shape == p_test.shape


# ---------------------------------------------------------------------------
# Regression flag
# ---------------------------------------------------------------------------


def _slice_report(entries: list[tuple[str, float, float]]) -> SliceReport:
    report = SliceReport()
    for name, auc, ci_low in entries:
        report.results.append(
            SliceResult(
                name=name,
                n=100,
                n_pos=50,
                n_neg=50,
                auc=auc,
                ci_low=ci_low,
                ci_high=ci_low + 0.1,
                p_perm=0.01,
                p_adj=0.02,
            )
        )
    return report


def test_regression_flag_fires_on_degraded_baseline():
    baseline = baseline_from_report(
        _slice_report([("maneuver:cut_in", 0.90, 0.85), ("maneuver:lane_keep", 0.75, 0.70)])
    )
    degraded = _slice_report([("maneuver:cut_in", 0.65, 0.60), ("maneuver:lane_keep", 0.76, 0.73)])
    result = check_regression(degraded, baseline, threshold=0.05)
    assert result.flag
    assert len(result.violations) == 1
    assert result.violations[0]["slice"] == "maneuver:cut_in"
    assert result.violations[0]["reason"] == "auc_ci_low_below_baseline"


def test_regression_flag_missing_slice():
    baseline = baseline_from_report(_slice_report([("maneuver:merge", 0.80, 0.75)]))
    current = _slice_report([])
    result = check_regression(current, baseline, threshold=0.05)
    assert result.flag
    assert result.violations[0]["reason"] == "missing"


def test_regression_flag_clean_when_within_tolerance():
    baseline = baseline_from_report(_slice_report([("maneuver:lane_keep", 0.75, 0.70)]))
    current = _slice_report([("maneuver:lane_keep", 0.76, 0.72)])
    result = check_regression(current, baseline, threshold=0.05)
    assert not result.flag
    assert result.violations == []


# ---------------------------------------------------------------------------
# Additional metric sanity
# ---------------------------------------------------------------------------


def test_metrics_on_perfect_separation():
    y = np.array([0.0] * 50 + [1.0] * 50)
    probs = np.concatenate([np.full(50, 0.1), np.full(50, 0.9)])
    assert auc_roc(y, probs) == pytest.approx(1.0)
    assert brier_score(y, probs) == pytest.approx(0.01, abs=1e-12)
    result = youden_accuracy(y, probs)
    assert result.accuracy == pytest.approx(1.0)
    assert result.threshold == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# End-to-end evaluate + replay CLI (light settings)
# ---------------------------------------------------------------------------

SMOKE_PIPELINE_KWARGS: dict[str, Any] = {
    "synthetic_fallback": True,
    "n_scenes": 16,
    "agents_per_scene": 4,
    "duration_s": 12.0,
    "seed": 11,
}

EVAL_ARGS = ["--n-resamples", "60", "--n-permutations", "60", "--n-jobs", "1"]


@pytest.fixture(scope="session")
def smoke_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    pytest.importorskip("torch")
    from dsdbench.data.pipeline import run_pipeline
    from dsdbench.train import main as train_main

    base = tmp_path_factory.mktemp("eval-smoke")
    data = base / "bench"
    run_pipeline(data, **SMOKE_PIPELINE_KWARGS)
    artifacts = base / "artifacts"
    rc = train_main(
        [
            "--model",
            "cnn",
            "--config",
            str(Path(__file__).parents[1] / "configs" / "cnn.yaml"),
            "--data-dir",
            str(data),
            "--artifacts-dir",
            str(artifacts),
            "--version",
            "eval-smoke",
            "--epochs",
            "2",
        ]
    )
    assert rc == 0
    return artifacts / "cnn" / "eval-smoke"


def test_evaluate_cli_writes_report_and_exits_clean(smoke_run: Path):
    pytest.importorskip("matplotlib")
    from dsdbench.evaluate import main as evaluate_main

    rc = evaluate_main(["--run", str(smoke_run), *EVAL_ARGS])
    assert rc == 0
    eval_dir = smoke_run / "eval"
    for name in ("report.md", "metrics.json", "baseline.json", "config.json"):
        assert (eval_dir / name).exists()
    for name in ("roc_slices.png", "reliability.png", "forest.png", "power_curve.png"):
        assert (eval_dir / "figures" / name).exists()
    metrics = json.loads((eval_dir / "metrics.json").read_text())
    assert "auc_roc" in metrics["metrics"]
    assert (eval_dir / "report.md").read_text().startswith("# dsdbench evaluation report")


def test_evaluate_cli_regression_flag_gates(smoke_run: Path):
    pytest.importorskip("matplotlib")
    from dsdbench.evaluate import main as evaluate_main

    # Self-sufficient: produce a baseline, inflate it, and re-check.
    assert evaluate_main(["--run", str(smoke_run), *EVAL_ARGS]) == 0
    baseline = json.loads((smoke_run / "eval" / "baseline.json").read_text())
    for name in baseline["slices"]:
        baseline["slices"][name]["auc"] += 0.2
    degraded_path = smoke_run / "degraded-baseline.json"
    degraded_path.write_text(json.dumps(baseline))
    rc = evaluate_main(["--run", str(smoke_run), "--baseline", str(degraded_path), *EVAL_ARGS])
    assert rc == 1
    metrics = json.loads((smoke_run / "eval" / "metrics.json").read_text())
    assert metrics["regression"]["flag"] is True


def test_replay_matches_reported_metrics(smoke_run: Path, tmp_path: Path):
    pytest.importorskip("torch")
    from dsdbench.evaluate import main as evaluate_main
    from dsdbench.replay import main as replay_main

    # Replay re-executes eval with the stored eval config; ensure it exists.
    assert evaluate_main(["--run", str(smoke_run), *EVAL_ARGS]) == 0

    start = time.monotonic()
    rc = replay_main(["--run", str(smoke_run), "--tmp-root", str(tmp_path / "replay-tmp")])
    elapsed = time.monotonic() - start
    assert rc == 0, "replay failed (see output)"
    assert elapsed < 240.0
