"""Per-slice evaluation with FDR control and regression gating.

The full metric suite is run per *slice*: each maneuver class
(``cut_in``, ``merge``, ``unprotected_left``, ``lane_keep``, ``stop_and_go``)
and each simulator-knob bucket (observation noise low/high, actuation latency
0 / >0 steps). Because many slices are tested simultaneously, per-slice
"better than chance" p-values are adjusted with the Benjamini-Hochberg
procedure to control the false discovery rate. A *regression flag* fires when
a slice's AUC CI lower bound drops more than a configured threshold below the
AUC recorded in a stored baseline JSON, allowing CI gating of model quality.

References
----------
* Benjamini, Y., & Hochberg, Y. (1995). "Controlling the false discovery
  rate: a practical and powerful approach to multiple testing." JRSS-B,
  57(1), 289-300.
* Slice-level CIs: scene-clustered BCa bootstrap (see ``bootstrap.py``;
  Efron 1987; Efron & Tibshirani 1993).
* Slice p-values: Monte Carlo permutation tests with exact p-values
  (see ``permutation.py``; Phipson & Smyth 2010).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from dsdbench.eval.bootstrap import bca_bootstrap
from dsdbench.eval.metrics import auc_roc
from dsdbench.eval.permutation import permutation_test

BASELINE_SCHEMA = "dsdbench-slice-baseline-v1"


@dataclass
class SliceConfig:
    alpha: float = 0.05
    n_bins: int = 15
    n_resamples: int = 1000
    n_permutations: int = 500
    seed: int = 0
    n_jobs: int = -1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SliceResult:
    name: str
    n: int
    n_pos: int
    n_neg: int
    auc: float
    ci_low: float
    ci_high: float
    p_perm: float
    p_adj: float = float("nan")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SliceReport:
    results: list[SliceResult] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": [r.to_dict() for r in self.results],
            "skipped": self.skipped,
        }


@dataclass
class RegressionResult:
    flag: bool
    threshold: float
    violations: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "flag": self.flag,
            "threshold": self.threshold,
            "violations": self.violations,
        }


def build_slices(
    predictions: dict[str, list[Any]], labels: dict[str, list[Any]]
) -> dict[str, np.ndarray]:
    """Build boolean row masks (over prediction rows) for every slice.

    Maneuver slices select rows by their maneuver label. Knob buckets are
    defined on *simulated* rows (their config is stored in ``labels`` as
    ``config_json``) and always include all real rows as negatives:

    * ``knob:noise_low`` / ``knob:noise_high`` -- position noise stddev at or
      below / above the sim median;
    * ``knob:latency_0`` / ``knob:latency_gt0`` -- zero / nonzero actuation
      latency steps.
    """
    source = np.asarray(predictions["source"], dtype=object)
    real = source == "real"
    sim = source == "sim"
    masks: dict[str, np.ndarray] = {}

    maneuver = np.asarray(predictions["maneuver"], dtype=object)
    for name in sorted(set(maneuver)):
        masks[f"maneuver:{name}"] = maneuver == name

    config_by_segment: dict[Any, dict[str, Any]] = {}
    for segment_id, config_json in zip(labels["segment_id"], labels["config_json"], strict=True):
        if config_json is not None:
            config_by_segment[segment_id] = json.loads(config_json)

    pred_ids = predictions["segment_id"]
    noise = np.asarray(
        [config_by_segment.get(seg, {}).get("pos_noise_std", np.nan) for seg in pred_ids],
        dtype=np.float64,
    )
    latency = np.asarray(
        [config_by_segment.get(seg, {}).get("latency_steps", np.nan) for seg in pred_ids],
        dtype=np.float64,
    )
    sim_noise = noise[sim]
    sim_noise = sim_noise[np.isfinite(sim_noise)]
    if sim_noise.size > 0:
        median_noise = float(np.median(sim_noise))
        masks["knob:noise_low"] = real | (sim & (noise <= median_noise))
        masks["knob:noise_high"] = real | (sim & (noise > median_noise))
    sim_latency = latency[sim]
    if np.isfinite(sim_latency).any():
        masks["knob:latency_0"] = real | (sim & (latency == 0))
        masks["knob:latency_gt0"] = real | (sim & (latency > 0))
    return masks


def run_slices(
    predictions: dict[str, list[Any]],
    labels: dict[str, list[Any]],
    config: SliceConfig,
    *,
    min_per_class: int = 2,
) -> SliceReport:
    """Compute AUC + scene-level BCa CI + permutation p for every slice,
    then apply Benjamini-Hochberg FDR adjustment across slice p-values."""
    y = np.asarray([1.0 if s == "sim" else 0.0 for s in predictions["source"]], dtype=np.float64)
    probs = np.asarray(predictions["prob_sim"], dtype=np.float64)
    scenes = np.asarray(predictions["scene_id"])
    masks = build_slices(predictions, labels)

    report = SliceReport()
    for name in sorted(masks):
        mask = masks[name]
        y_s, p_s, c_s = y[mask], probs[mask], scenes[mask]
        n_pos = int(np.sum(y_s == 1.0))
        n_neg = int(np.sum(y_s == 0.0))
        if n_pos < min_per_class or n_neg < min_per_class or y_s.size < 4:
            report.skipped.append(name)
            continue
        auc = float(auc_roc(y_s, p_s))
        ci = bca_bootstrap(
            y_s,
            p_s,
            c_s,
            _auc_stat,
            statistic_name="auc_roc",
            n_resamples=config.n_resamples,
            seed=config.seed,
            alpha=config.alpha,
            n_jobs=config.n_jobs,
        )
        perm = permutation_test(
            y_s,
            p_s,
            _auc_stat,
            n_permutations=config.n_permutations,
            seed=config.seed,
            n_jobs=config.n_jobs,
        )
        report.results.append(
            SliceResult(
                name=name,
                n=int(y_s.size),
                n_pos=n_pos,
                n_neg=n_neg,
                auc=auc,
                ci_low=ci.ci_low,
                ci_high=ci.ci_high,
                p_perm=perm.p_value,
            )
        )

    if report.results:
        pvalues = np.asarray([r.p_perm for r in report.results])
        adjusted = _benjamini_hochberg(pvalues)
        for result, p_adj in zip(report.results, adjusted, strict=True):
            result.p_adj = float(p_adj)
    return report


def _auc_stat(y: np.ndarray, probs: np.ndarray) -> float:
    value = auc_roc(y, probs)
    return 0.5 if not np.isfinite(value) else float(value)


def _benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg (1995) linear step-up FDR adjustment."""
    from scipy.stats import false_discovery_control

    return np.asarray(false_discovery_control(pvalues, method="bh"), dtype=np.float64)


# Below this many segments a slice's bootstrap AUC CI spans most of [0.5, 1],
# so comparing it against a baseline is noise rather than signal.
DEFAULT_MIN_SLICE_N = 30


def baseline_from_report(report: SliceReport) -> dict[str, Any]:
    """Serialize a slice report into the stored baseline JSON schema."""
    return {
        "schema": BASELINE_SCHEMA,
        "slices": {
            r.name: {"auc": r.auc, "ci_low": r.ci_low, "ci_high": r.ci_high, "n": r.n}
            for r in report.results
        },
    }


def check_regression(
    report: SliceReport,
    baseline: dict[str, Any],
    threshold: float = 0.05,
    min_slice_n: int = DEFAULT_MIN_SLICE_N,
) -> RegressionResult:
    """Flag slices that got detectably and materially worse than the baseline.

    A slice is violated when its current AUC CI lies entirely below the
    baseline's CI (``current_ci_high < baseline_ci_low``) *and* its point AUC
    dropped by more than ``threshold``, or when the baseline contains a slice
    that is missing from the current report (e.g. the model lost the ability to
    even score a maneuver). Slices with fewer than ``min_slice_n`` segments in
    the baseline or the current run are not compared: their bootstrap CIs are
    too wide for an AUC difference to mean anything."""
    if baseline.get("schema") != BASELINE_SCHEMA:
        raise ValueError("baseline is not a dsdbench slice baseline JSON")
    baseline_slices = baseline.get("slices", {})
    current = {r.name: r for r in report.results}
    violations: list[dict[str, Any]] = []
    for name, stored in sorted(baseline_slices.items()):
        result = current.get(name)
        if result is None:
            violations.append(
                {"slice": name, "reason": "missing", "baseline_auc": stored["auc"], "ci_low": None}
            )
            continue
        if int(stored["n"]) < min_slice_n or result.n < min_slice_n:
            continue
        auc_drop = float(stored["auc"]) - result.auc
        if result.ci_high < float(stored["ci_low"]) and auc_drop > threshold:
            violations.append(
                {
                    "slice": name,
                    "reason": "auc_ci_below_baseline_ci",
                    "baseline_auc": stored["auc"],
                    "baseline_ci_low": stored["ci_low"],
                    "current_auc": result.auc,
                    "ci_high": result.ci_high,
                    "delta": auc_drop,
                }
            )
    return RegressionResult(flag=len(violations) > 0, threshold=threshold, violations=violations)
