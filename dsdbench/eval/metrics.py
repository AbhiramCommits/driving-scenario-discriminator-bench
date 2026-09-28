"""Point metrics for scoring real-vs-simulated discriminators.

Labels are 0/1 (real/simulated) with predicted probabilities ``probs`` of the
positive class. All functions return plain floats (``nan`` where a metric is
undefined, e.g. single-class inputs for ranking metrics).

References
----------
* AUC-ROC: Fawcett, T. (2006). "An introduction to ROC analysis." Pattern
  Recognition Letters, 27(8), 861-874.
* AUC-PR: Davis, J., & Goadrich, M. (2006). "The relationship between
  Precision-Recall and ROC curves." ICML.
* Youden's index: Youden, W. J. (1950). "Index for rating diagnostic tests."
  Cancer, 3(1), 32-35.
* Brier score: Brier, G. W. (1950). "Verification of forecasts expressed in
  terms of probability." Monthly Weather Review, 78(1), 1-3.
* Expected/Maximum Calibration Error with equal-mass (quantile) binning:
  Naeini, M. P., Cooper, G., & Hauskrecht, M. (2015). "Obtaining well
  calibrated probabilities using Bayesian binning." AAAI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _as_arrays(y: Any, probs: Any) -> tuple[np.ndarray, np.ndarray]:
    y_arr = np.asarray(y, dtype=np.float64)
    p_arr = np.asarray(probs, dtype=np.float64)
    if y_arr.shape != p_arr.shape:
        raise ValueError(
            f"y and probs must have the same shape, got {y_arr.shape} vs {p_arr.shape}"
        )
    return y_arr, p_arr


def auc_roc(y: Any, probs: Any) -> float:
    """Area under the ROC curve (Fawcett 2006); NaN for single-class labels."""
    from sklearn.metrics import roc_auc_score

    y_arr, p_arr = _as_arrays(y, probs)
    if y_arr.size < 2 or len(np.unique(y_arr)) < 2:
        return float("nan")
    return float(roc_auc_score(y_arr, p_arr))


def auc_pr(y: Any, probs: Any) -> float:
    """Area under the precision-recall curve (Davis & Goadrich 2006); NaN if undefined."""
    from sklearn.metrics import average_precision_score

    y_arr, p_arr = _as_arrays(y, probs)
    if y_arr.size < 2 or len(np.unique(y_arr)) < 2:
        return float("nan")
    return float(average_precision_score(y_arr, p_arr))


@dataclass
class YoudenResult:
    """Classification at the threshold maximizing Youden's index (Youden 1950)."""

    threshold: float
    accuracy: float
    sensitivity: float
    specificity: float

    def to_dict(self) -> dict[str, float]:
        return {
            "threshold": self.threshold,
            "accuracy": self.accuracy,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
        }


def youden_accuracy(y: Any, probs: Any) -> YoudenResult:
    """Accuracy at the Youden-optimal threshold (argmax tpr - fpr)."""
    from sklearn.metrics import roc_curve

    y_arr, p_arr = _as_arrays(y, probs)
    if y_arr.size < 2 or len(np.unique(y_arr)) < 2:
        return YoudenResult(float("nan"), float("nan"), float("nan"), float("nan"))
    fpr, tpr, thresholds = roc_curve(y_arr, p_arr)
    j = tpr - fpr
    best = int(np.argmax(j))
    threshold = float(thresholds[best]) if best < len(thresholds) else float(p_arr.max())
    pred = (p_arr >= threshold).astype(np.float64)
    correct = float(np.mean(pred == y_arr))
    sensitivity = float(tpr[best])
    specificity = float(1.0 - fpr[best])
    return YoudenResult(threshold, correct, sensitivity, specificity)


def brier_score(y: Any, probs: Any) -> float:
    """Mean squared error between predicted probabilities and labels (Brier 1950)."""
    y_arr, p_arr = _as_arrays(y, probs)
    return float(np.mean((p_arr - y_arr) ** 2))


def _calibration_bins(
    y: np.ndarray, probs: np.ndarray, n_bins: int
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Equal-mass binning: sort by predicted probability, split into n_bins
    bins of (nearly) equal size; return bin sizes, mean predictions, and
    empirical positive fractions (Naeini et al. 2015)."""
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    order = np.argsort(probs)
    indices = np.array_split(order, n_bins)
    sizes = [int(len(idx)) for idx in indices]
    mean_pred = np.asarray([float(probs[idx].mean()) for idx in indices])
    frac_pos = np.asarray([float(y[idx].mean()) for idx in indices])
    return sizes, mean_pred, frac_pos


def expected_calibration_error(y: Any, probs: Any, n_bins: int = 15) -> float:
    """Expected Calibration Error with equal-mass binning (Naeini et al. 2015)."""
    y_arr, p_arr = _as_arrays(y, probs)
    sizes, mean_pred, frac_pos = _calibration_bins(y_arr, p_arr, n_bins)
    weights = np.asarray(sizes, dtype=np.float64) / y_arr.size
    return float(np.sum(weights * np.abs(mean_pred - frac_pos)))


def maximum_calibration_error(y: Any, probs: Any, n_bins: int = 15) -> float:
    """Maximum Calibration Error: the largest per-bin |confidence - accuracy| gap."""
    y_arr, p_arr = _as_arrays(y, probs)
    _, mean_pred, frac_pos = _calibration_bins(y_arr, p_arr, n_bins)
    return float(np.max(np.abs(mean_pred - frac_pos)))
