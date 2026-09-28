"""Statistical power analysis for AUC-ROC.

For a balanced sample of ``n`` segments the standard error of the empirical
AUC against a null AUC of 0.5 is approximated with the Hanley-McNeil formula.
The one-sided test "H0: AUC <= auc_null" rejects at level ``alpha`` when
``(auc_hat - auc_null) / SE > z_{1-alpha}``; plugging in a target AUC gives
the achieved power, and inverting over ``n`` gives the sample size required
to reach a desired power.

References
----------
* Hanley, J. A., & McNeil, B. J. (1982). "The meaning and use of the area
  under a receiver operating characteristic (ROC) curve." Radiology,
  143(1), 29-36.
* Obuchowski, N. A. (2005). "ROC analysis." American Journal of
  Roentgenology, 184(2), 364-372. (power/sample size discussion)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import norm


def hanley_mcneil_se(auc: float, n_pos: int, n_neg: int) -> float:
    """Standard error of the empirical AUC (Hanley & McNeil 1982, eq. 1-2)."""
    if not 0.0 <= auc <= 1.0:
        raise ValueError(f"auc must be in [0, 1], got {auc}")
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc**2 / (1.0 + auc)
    variance = (auc * (1.0 - auc) + (n_pos - 1) * (q1 - auc**2) + (n_neg - 1) * (q2 - auc**2)) / (
        n_pos * n_neg
    )
    if variance < 0.0:
        # Tiny negative values can arise from the approximation; clamp.
        variance = 0.0
    return float(np.sqrt(variance))


def achieved_power(auc: float, n: int, *, alpha: float = 0.05, auc_null: float = 0.5) -> float:
    """Power of the one-sided AUC test for a balanced sample of ``n`` segments."""
    if n < 4:
        raise ValueError("n must be at least 4")
    n_pos = n // 2
    n_neg = n - n_pos
    se = hanley_mcneil_se(auc, n_pos, n_neg)
    z = (auc - auc_null) / se - float(norm.ppf(1.0 - alpha))
    return float(norm.cdf(z))


def required_samples(
    auc: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    auc_null: float = 0.5,
    max_n: int = 10_000_000,
) -> int:
    """Smallest even segment count ``n`` reaching ``power`` for the target AUC.

    Assumes balanced classes (n/2 positives, n/2 negatives) and a one-sided
    test of AUC against ``auc_null`` at level ``alpha``.
    """
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")
    for n in range(4, max_n, 2):
        if achieved_power(auc, n, alpha=alpha, auc_null=auc_null) >= power:
            return n
    raise ValueError(f"no n < {max_n} reaches power {power} for AUC {auc}")


def power_curve(
    aucs: tuple[float, ...] = (0.6, 0.65, 0.7, 0.8),
    n_max: int = 2000,
    *,
    alpha: float = 0.05,
    auc_null: float = 0.5,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Power vs. sample size grid for several target AUCs.

    Returns ``(n_grid, [power_auc1, power_auc2, ...])``.
    """
    n_grid = np.arange(4, n_max + 1, 2, dtype=np.int64)
    curves = [
        np.asarray([achieved_power(a, int(n), alpha=alpha, auc_null=auc_null) for n in n_grid])
        for a in aucs
    ]
    return n_grid, curves


def plot_power_curve(
    aucs: tuple[float, ...] = (0.6, 0.65, 0.7, 0.8),
    n_max: int = 2000,
    *,
    alpha: float = 0.05,
    power_target: float = 0.8,
    ax: Any = None,
) -> Any:
    """Plot achieved power vs. sample size for each target AUC.

    Returns the matplotlib axis; pass ``ax`` to draw onto an existing figure.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    n_grid, curves = power_curve(aucs, n_max, alpha=alpha)
    for auc, curve in zip(aucs, curves, strict=True):
        ax.plot(n_grid, curve, label=f"AUC = {auc:g}")
    ax.axhline(power_target, color="0.5", linestyle="--", linewidth=1)
    ax.set_xlabel("Number of segments (balanced classes)")
    ax.set_ylabel(f"Power (one-sided test, $\\alpha$ = {alpha})")
    ax.set_title("AUC power analysis (Hanley-McNeil)")
    ax.legend(title="Target AUC")
    ax.grid(alpha=0.3)
    return ax
