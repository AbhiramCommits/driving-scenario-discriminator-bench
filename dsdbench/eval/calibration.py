"""Recalibration: isotonic regression and Platt scaling.

A discriminator's sigmoid outputs are not guaranteed to be calibrated
probabilities. Both recalibrators are fit on the *val* split only and applied
to the *test* split, with ECE/MCE reported before and after so calibration
gains are never tuned on the data they are reported on.

References
----------
* Platt, J. (1999). "Probabilistic outputs for support vector machines and
  comparisons to regularized likelihood methods." Advances in Large Margin
  Classifiers.
* Zadrozny, B., & Elkan, C. (2002). "Transforming classifier scores into
  accurate multiclass probability estimates." KDD.
* Calibration error: Naeini, M. P., Cooper, G., & Hauskrecht, M. (2015).
  "Obtaining well calibrated probabilities using Bayesian binning." AAAI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from dsdbench.eval.metrics import expected_calibration_error, maximum_calibration_error

_EPS = 1e-7


def _clip(probs: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(probs, dtype=np.float64), _EPS, 1.0 - _EPS)


class PlattCalibrator:
    """Logistic regression on logit-transformed scores (Platt 1999)."""

    def __init__(self) -> None:
        self._model: Any = None

    def fit(self, y: Any, probs: Any) -> PlattCalibrator:
        from sklearn.linear_model import LogisticRegression

        p = _clip(np.asarray(probs, dtype=np.float64))
        logits = np.log(p / (1.0 - p))
        self._model = LogisticRegression()
        self._model.fit(logits.reshape(-1, 1), np.asarray(y))
        return self

    def predict(self, probs: Any) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("PlattCalibrator must be fit before predict")
        p = _clip(np.asarray(probs, dtype=np.float64))
        logits = np.log(p / (1.0 - p))
        return np.asarray(self._model.predict_proba(logits.reshape(-1, 1))[:, 1], dtype=np.float64)


class IsotonicCalibrator:
    """Isotonic regression calibrator (Zadrozny & Elkan 2002)."""

    def __init__(self) -> None:
        self._model: Any = None

    def fit(self, y: Any, probs: Any) -> IsotonicCalibrator:
        from sklearn.isotonic import IsotonicRegression

        self._model = IsotonicRegression(out_of_bounds="clip")
        self._model.fit(np.asarray(probs, dtype=np.float64), np.asarray(y, dtype=np.float64))
        return self

    def predict(self, probs: Any) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("IsotonicCalibrator must be fit before predict")
        return np.asarray(
            self._model.predict(np.asarray(probs, dtype=np.float64)), dtype=np.float64
        )


@dataclass
class RecalibrationResult:
    """ECE/MCE on test before and after recalibration (calibrators fit on val)."""

    ece_before: float
    ece_after_platt: float
    ece_after_isotonic: float
    mce_before: float
    mce_after_platt: float
    mce_after_isotonic: float
    n_val: int
    n_test: int
    probs_platt: np.ndarray
    probs_isotonic: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "ece_before": self.ece_before,
            "ece_after_platt": self.ece_after_platt,
            "ece_after_isotonic": self.ece_after_isotonic,
            "mce_before": self.mce_before,
            "mce_after_platt": self.mce_after_platt,
            "mce_after_isotonic": self.mce_after_isotonic,
            "n_val": self.n_val,
            "n_test": self.n_test,
        }


def recalibrate(
    y_val: Any,
    probs_val: Any,
    y_test: Any,
    probs_test: Any,
    *,
    n_bins: int = 15,
) -> RecalibrationResult:
    """Fit Platt + isotonic calibrators on val, evaluate ECE/MCE on test."""
    y_val_arr = np.asarray(y_val, dtype=np.float64)
    probs_val_arr = np.asarray(probs_val, dtype=np.float64)
    y_test_arr = np.asarray(y_test, dtype=np.float64)
    probs_test_arr = np.asarray(probs_test, dtype=np.float64)
    if y_val_arr.size < 2:
        raise ValueError("recalibration needs at least 2 val samples")

    platt = PlattCalibrator().fit(y_val_arr, probs_val_arr)
    isotonic = IsotonicCalibrator().fit(y_val_arr, probs_val_arr)
    probs_platt = platt.predict(probs_test_arr)
    probs_isotonic = isotonic.predict(probs_test_arr)

    return RecalibrationResult(
        ece_before=expected_calibration_error(y_test_arr, probs_test_arr, n_bins=n_bins),
        ece_after_platt=expected_calibration_error(y_test_arr, probs_platt, n_bins=n_bins),
        ece_after_isotonic=expected_calibration_error(y_test_arr, probs_isotonic, n_bins=n_bins),
        mce_before=maximum_calibration_error(y_test_arr, probs_test_arr, n_bins=n_bins),
        mce_after_platt=maximum_calibration_error(y_test_arr, probs_platt, n_bins=n_bins),
        mce_after_isotonic=maximum_calibration_error(y_test_arr, probs_isotonic, n_bins=n_bins),
        n_val=int(y_val_arr.size),
        n_test=int(y_test_arr.size),
        probs_platt=probs_platt,
        probs_isotonic=probs_isotonic,
    )
