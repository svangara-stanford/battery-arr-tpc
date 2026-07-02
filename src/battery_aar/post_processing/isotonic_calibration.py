"""Post-hoc isotonic calibration of candidate predictions.

Learns a monotone correction ``g`` such that ``g(y_pred) ~= y_true`` from
(prediction, target) pairs, then applies it to new predictions. Because the
correction is non-decreasing it never reorders predictions — rank metrics
(spearman/kendall) are unchanged; only magnitudes move, which is what reduces
RMSE/MAE bias.

The calibrator must be fit on pairs the reported split never sees (e.g. train
predictions vs train targets); never fit on the split being scored.

All space-aware helpers operate on raw-cycle-space inputs and handle the
raw/log10 calibration-space mapping internally, so callers (the compiled
candidate template and the offline CLI) do not duplicate that logic.
"""

from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

SUPPORTED_CALIBRATION_SPACES = ("raw", "log10")

_LOG_FLOOR = 1e-9


def _validate_vector(name: str, values) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return arr


class IsotonicCalibrator:
    """Monotone post-hoc correction of regression predictions.

    Wraps :class:`sklearn.isotonic.IsotonicRegression` with
    ``increasing=True`` (never reorders predictions) and
    ``out_of_bounds="clip"`` (inputs outside the fitted range map to the
    boundary calibrated values).
    """

    def __init__(self) -> None:
        self._iso: IsotonicRegression | None = None
        self.x_min_: float | None = None
        self.x_max_: float | None = None

    @property
    def is_fitted(self) -> bool:
        return self._iso is not None

    def fit(self, y_pred, y_true) -> "IsotonicCalibrator":
        y_pred = _validate_vector("y_pred", y_pred)
        y_true = _validate_vector("y_true", y_true)
        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"shape mismatch: y_pred {y_pred.shape} vs y_true {y_true.shape}"
            )
        if y_pred.size < 2:
            raise ValueError(f"need at least 2 samples to fit, got {y_pred.size}")
        self._iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        self._iso.fit(y_pred, y_true)
        self.x_min_ = float(y_pred.min())
        self.x_max_ = float(y_pred.max())
        return self

    def transform(self, y_pred) -> np.ndarray:
        if self._iso is None:
            raise RuntimeError("IsotonicCalibrator.transform called before fit")
        return self._iso.predict(_validate_vector("y_pred", y_pred))

    def fit_transform(self, y_pred, y_true) -> np.ndarray:
        return self.fit(y_pred, y_true).transform(y_pred)


def fit_isotonic_in_space(
    y_pred_raw, y_true_raw, calibration_space: str
) -> Tuple[IsotonicCalibrator, str]:
    """Fit a calibrator on raw-space pairs in the requested space.

    Returns ``(calibrator, effective_space)``. The effective space can differ
    from the requested one: fitting in log10 space requires strictly positive
    predictions and targets, and when that fails we fall back to raw space
    with a warning rather than aborting the candidate. Callers must persist
    the effective space and pass it to :func:`apply_isotonic_in_space`.
    """
    if calibration_space not in SUPPORTED_CALIBRATION_SPACES:
        raise ValueError(
            f"unsupported calibration_space {calibration_space!r}; "
            f"expected one of {SUPPORTED_CALIBRATION_SPACES}"
        )
    y_pred_raw = _validate_vector("y_pred_raw", y_pred_raw)
    y_true_raw = _validate_vector("y_true_raw", y_true_raw)
    effective_space = calibration_space
    if calibration_space == "log10" and (
        np.any(y_pred_raw <= 0) or np.any(y_true_raw <= 0)
    ):
        warnings.warn(
            "isotonic_calibration_log10_fallback: nonpositive predictions or "
            "targets; falling back to raw calibration space",
            RuntimeWarning,
            stacklevel=2,
        )
        effective_space = "raw"
    calibrator = IsotonicCalibrator()
    if effective_space == "log10":
        calibrator.fit(np.log10(y_pred_raw), np.log10(y_true_raw))
    else:
        calibrator.fit(y_pred_raw, y_true_raw)
    return calibrator, effective_space


def apply_isotonic_in_space(
    calibrator: IsotonicCalibrator, y_pred_raw, calibration_space: str
) -> np.ndarray:
    """Apply a fitted calibrator to raw-space predictions.

    ``calibration_space`` must be the effective space returned by
    :func:`fit_isotonic_in_space`. In log10 space, nonpositive predictions are
    clipped to a positive floor before the log, so they map deterministically
    to the lowest calibrated value.
    """
    if calibration_space not in SUPPORTED_CALIBRATION_SPACES:
        raise ValueError(
            f"unsupported calibration_space {calibration_space!r}; "
            f"expected one of {SUPPORTED_CALIBRATION_SPACES}"
        )
    y_pred_raw = _validate_vector("y_pred_raw", y_pred_raw)
    if calibration_space == "log10":
        y_pred_log = np.log10(np.clip(y_pred_raw, _LOG_FLOOR, None))
        return np.power(10.0, calibrator.transform(y_pred_log))
    return calibrator.transform(y_pred_raw)


def calibrate_prediction_frames(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    calibration_space: str = "raw",
) -> Tuple[pd.DataFrame, str]:
    """Fit on a train predictions frame, calibrate a test predictions frame.

    ``train_frame`` needs ``y_pred`` and ``y_true`` (``y`` accepted as an
    alias) columns; ``test_frame`` needs ``y_pred``. Returns a copy of
    ``test_frame`` with ``y_pred_calibrated`` added (and
    ``residual_calibrated`` when a truth column is present), plus the
    effective calibration space.
    """
    truth_col = next((c for c in ("y_true", "y") if c in train_frame.columns), None)
    if truth_col is None:
        raise ValueError("train_frame must contain a 'y_true' (or 'y') column")
    if "y_pred" not in train_frame.columns:
        raise ValueError("train_frame must contain a 'y_pred' column")
    if "y_pred" not in test_frame.columns:
        raise ValueError("test_frame must contain a 'y_pred' column")

    calibrator, effective_space = fit_isotonic_in_space(
        train_frame["y_pred"].to_numpy(dtype=float),
        train_frame[truth_col].to_numpy(dtype=float),
        calibration_space,
    )
    result = test_frame.copy()
    result["y_pred_calibrated"] = apply_isotonic_in_space(
        calibrator, result["y_pred"].to_numpy(dtype=float), effective_space
    )
    test_truth_col = next((c for c in ("y_true", "y") if c in result.columns), None)
    if test_truth_col is not None:
        result["residual_calibrated"] = (
            result["y_pred_calibrated"] - result[test_truth_col]
        )
    return result, effective_space
