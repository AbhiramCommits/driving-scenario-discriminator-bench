"""Permutation tests for discriminator quality.

Under the null hypothesis the labels carry no information about the scores, so
randomly permuting the labels leaves the distribution of any statistic
unchanged. Monte Carlo permutation tests therefore estimate the p-value as the
fraction of permuted statistics at least as extreme as the observed one. The
p-value uses the +1 correction (``(1 + count) / (1 + n_permutations)``), which
makes the Monte Carlo p-value *exact* in the sense that under the null it is
stochastically uniform on the discrete grid of attainable values and can never
be reported as zero.

References
----------
* Phipson, B., & Smyth, G. K. (2010). "Permutation P-values Should Never Be
  Zero: Calculating Exact P-values When Permutations Are Randomly Drawn."
  Statistical Applications in Genetics and Molecular Biology, 9(1).
* Paired comparisons: Good, P. (2013). "Permutation, Parametric, and
  Bootstrap Tests of Hypotheses." Springer, Ch. 2.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from joblib import Parallel, delayed

from dsdbench.eval.bootstrap import auc_statistic

Statistic = Callable[[np.ndarray, np.ndarray], float]


@dataclass
class PermutationResult:
    """Result of a (paired) permutation test."""

    statistic_name: str
    observed_statistic: float
    p_value: float
    n_permutations: int
    alternative: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _exact_p(count: int, n_permutations: int) -> float:
    """Exact Monte Carlo p-value (Phipson & Smyth 2010, eq. 1)."""
    return (1.0 + count) / (1.0 + n_permutations)


def permutation_test(
    y: Any,
    probs: Any,
    statistic: Statistic = auc_statistic,
    *,
    n_permutations: int = 1000,
    seed: int = 0,
    n_jobs: int = -1,
    alternative: str = "greater",
) -> PermutationResult:
    """Test whether ``statistic(y, probs)`` beats chance by label permutation.

    ``alternative="greater"`` (default) tests "better than chance" against a
    symmetric null; ``"two-sided"`` tests deviation in either direction.
    Deterministic given ``seed``; per-permutation derived seeds make the
    result independent of ``n_jobs``.
    """
    if alternative not in ("greater", "two-sided"):
        raise ValueError("alternative must be 'greater' or 'two-sided'")
    y_arr = np.asarray(y, dtype=np.float64)
    probs_arr = np.asarray(probs, dtype=np.float64)
    if y_arr.shape != probs_arr.shape or y_arr.size < 4:
        raise ValueError("y and probs must share a shape with at least 4 rows")

    observed = float(statistic(y_arr, probs_arr))
    children = np.random.default_rng(seed).spawn(n_permutations)

    def _one(b: int) -> float:
        rng = children[b]
        perm = rng.permutation(y_arr.size)
        return float(statistic(y_arr[perm], probs_arr))

    if n_jobs == 1:
        permuted = np.asarray([_one(b) for b in range(n_permutations)], dtype=np.float64)
    else:
        permuted = np.asarray(
            Parallel(n_jobs=n_jobs)(delayed(_one)(b) for b in range(n_permutations)),
            dtype=np.float64,
        )
    if alternative == "greater":
        count = int(np.sum(permuted >= observed))
    else:
        count = int(np.sum(np.abs(permuted) >= abs(observed)))

    return PermutationResult(
        statistic_name=getattr(statistic, "__name__", "statistic"),
        observed_statistic=observed,
        p_value=_exact_p(count, n_permutations),
        n_permutations=n_permutations,
        alternative=alternative,
    )


def paired_permutation_test(
    y: Any,
    probs_a: Any,
    probs_b: Any,
    statistic: Statistic = auc_statistic,
    *,
    n_permutations: int = 1000,
    seed: int = 0,
    n_jobs: int = -1,
) -> PermutationResult:
    """Paired permutation test comparing two models on the *same* segments.

    Under the null the two models are exchangeable, so randomly swapping the
    two scores within each segment (with probability 1/2) produces the null
    distribution of the difference in ``statistic``. The p-value is two-sided.
    """
    y_arr = np.asarray(y, dtype=np.float64)
    a_arr = np.asarray(probs_a, dtype=np.float64)
    b_arr = np.asarray(probs_b, dtype=np.float64)
    if not (y_arr.shape == a_arr.shape == b_arr.shape) or y_arr.size < 4:
        raise ValueError("y, probs_a, probs_b must share a shape with at least 4 rows")

    observed = float(statistic(y_arr, a_arr) - statistic(y_arr, b_arr))
    children = np.random.default_rng(seed).spawn(n_permutations)

    def _one(b: int) -> float:
        rng = children[b]
        swap = rng.random(y_arr.size) < 0.5
        pa = np.where(swap, b_arr, a_arr)
        pb = np.where(swap, a_arr, b_arr)
        return float(statistic(y_arr, pa) - statistic(y_arr, pb))

    if n_jobs == 1:
        permuted = np.asarray([_one(b) for b in range(n_permutations)], dtype=np.float64)
    else:
        permuted = np.asarray(
            Parallel(n_jobs=n_jobs)(delayed(_one)(b) for b in range(n_permutations)),
            dtype=np.float64,
        )
    count = int(np.sum(np.abs(permuted) >= abs(observed)))

    return PermutationResult(
        statistic_name=getattr(statistic, "__name__", "statistic_diff"),
        observed_statistic=observed,
        p_value=_exact_p(count, n_permutations),
        n_permutations=n_permutations,
        alternative="two-sided",
    )
