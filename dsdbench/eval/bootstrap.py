"""Scene-clustered BCa bootstrap confidence intervals.

Trajectory segments from the same scene are correlated (shared road geometry,
traffic, and for sim segments shared rollout machinery), so naive
segment-level resampling underestimates uncertainty. Resampling here is done
at the *scene* level: scenes are drawn with replacement and all segments of a
drawn scene are included. The bias-corrected and accelerated (BCa) method
adjusts the percentile endpoints for skew and for the rate of change of the
statistic with respect to the data.

References
----------
* Efron, B. (1987). "Better bootstrap confidence intervals." JASA, 82(397),
  171-185.
* Efron, B., & Tibshirani, R. J. (1993). "An Introduction to the Bootstrap."
  Chapman & Hall. (BCa intervals, Sec. 14.3)
* DiCiccio, T. J., & Efron, B. (1996). "Bootstrap confidence intervals."
  Statistical Science, 11(3), 189-228.
* Clustered bootstrap: Field, C. A., & Welsh, A. H. (2007). "Bootstrapping
  clustered data." JRSS-B, 69(3), 369-390.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import norm

from dsdbench.eval.metrics import auc_roc, expected_calibration_error

Statistic = Callable[[np.ndarray, np.ndarray], float]


@dataclass
class BCaResult:
    """Scene-level BCa bootstrap confidence interval for one statistic."""

    statistic_name: str
    estimate: float
    ci_low: float
    ci_high: float
    bias_correction: float  # z0
    acceleration: float  # a
    alpha: float
    n_resamples: int
    n_scenes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def auc_statistic(y: np.ndarray, probs: np.ndarray) -> float:
    """AUC-ROC statistic, mapped to chance (0.5) for single-class resamples."""
    value = auc_roc(y, probs)
    return 0.5 if not np.isfinite(value) else float(value)


def ece_statistic(n_bins: int) -> Statistic:
    """Factory returning an ECE statistic with fixed bin count."""

    def statistic(y: np.ndarray, probs: np.ndarray) -> float:
        return expected_calibration_error(y, probs, n_bins=n_bins)

    return statistic


def bca_bootstrap(
    y: Any,
    probs: Any,
    scenes: Any,
    statistic: Statistic,
    *,
    statistic_name: str = "statistic",
    n_resamples: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
    n_jobs: int = -1,
) -> BCaResult:
    """Scene-level BCa bootstrap CI for ``statistic(y, probs)``.

    ``scenes`` gives the scene id of every segment; scenes are the resampling
    unit. Deterministic given ``seed`` (each resample gets its own derived
    seed, so results do not depend on ``n_jobs``).
    """
    if n_resamples < 1:
        raise ValueError("n_resamples must be >= 1")
    y_arr = np.asarray(y, dtype=np.float64)
    probs_arr = np.asarray(probs, dtype=np.float64)
    scenes_arr = np.asarray(scenes)
    if not (y_arr.shape == probs_arr.shape == scenes_arr.shape):
        raise ValueError("y, probs, scenes must share a shape")
    if y_arr.size < 4:
        raise ValueError("need at least 4 segments for a bootstrap")

    scene_list = np.unique(scenes_arr)
    members = [np.flatnonzero(scenes_arr == s) for s in scene_list]
    estimate = float(statistic(y_arr, probs_arr))

    # Jackknife over scenes for the BCa acceleration a (Efron 1987).
    jackknife = np.empty(len(scene_list), dtype=np.float64)
    for i in range(len(scene_list)):
        keep = np.concatenate([m for j, m in enumerate(members) if j != i])
        jackknife[i] = statistic(y_arr[keep], probs_arr[keep])
    centered = float(jackknife.mean()) - jackknife
    denominator = 6.0 * float(np.sum(centered**2)) ** 1.5
    acceleration = float(np.sum(centered**3) / denominator) if denominator > 0.0 else 0.0

    # Parallel bootstrap resamples, each with its own derived generator
    # (independent children of a single base generator: deterministic and
    # independent of job scheduling).
    children = np.random.default_rng(seed).spawn(n_resamples)

    def _one_resample(b: int) -> float:
        rng = children[b]
        draw = rng.integers(0, len(scene_list), size=len(scene_list))
        idx = np.concatenate([members[d] for d in draw])
        return float(statistic(y_arr[idx], probs_arr[idx]))

    if n_jobs == 1:
        boot = np.asarray([_one_resample(b) for b in range(n_resamples)], dtype=np.float64)
    else:
        boot = np.asarray(
            Parallel(n_jobs=n_jobs)(delayed(_one_resample)(b) for b in range(n_resamples)),
            dtype=np.float64,
        )

    # Bias correction z0 = Phi^-1(P(boot < estimate)), clipped to keep
    # Phi well-defined for degenerate samples (Efron & Tibshirani 1993).
    p_less = float(np.mean(boot < estimate))
    p_less = float(np.clip(p_less, 1.0 / (2.0 * n_resamples), 1.0 - 1.0 / (2.0 * n_resamples)))
    z0 = float(norm.ppf(p_less))

    z_lo = float(norm.ppf(alpha / 2.0))
    z_hi = float(norm.ppf(1.0 - alpha / 2.0))
    alpha_lo = float(norm.cdf(z0 + (z0 + z_lo) / (1.0 - acceleration * (z0 + z_lo))))
    alpha_hi = float(norm.cdf(z0 + (z0 + z_hi) / (1.0 - acceleration * (z0 + z_hi))))
    ci_low, ci_high = (float(v) for v in np.quantile(boot, [alpha_lo, alpha_hi]))

    return BCaResult(
        statistic_name=statistic_name,
        estimate=estimate,
        ci_low=ci_low,
        ci_high=ci_high,
        bias_correction=z0,
        acceleration=acceleration,
        alpha=alpha,
        n_resamples=n_resamples,
        n_scenes=len(scene_list),
    )


def bootstrap_auc(
    y: Any,
    probs: Any,
    scenes: Any,
    *,
    n_resamples: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
    n_jobs: int = -1,
) -> BCaResult:
    """Scene-level BCa bootstrap CI on the AUC-ROC."""
    return bca_bootstrap(
        y,
        probs,
        scenes,
        auc_statistic,
        statistic_name="auc_roc",
        n_resamples=n_resamples,
        seed=seed,
        alpha=alpha,
        n_jobs=n_jobs,
    )


def bootstrap_ece(
    y: Any,
    probs: Any,
    scenes: Any,
    *,
    n_bins: int = 15,
    n_resamples: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
    n_jobs: int = -1,
) -> BCaResult:
    """Scene-level BCa bootstrap CI on the ECE."""
    return bca_bootstrap(
        y,
        probs,
        scenes,
        ece_statistic(n_bins),
        statistic_name="ece",
        n_resamples=n_resamples,
        seed=seed,
        alpha=alpha,
        n_jobs=n_jobs,
    )
