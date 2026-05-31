from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .bms_features import BatteryCell, build_feature_matrix
from .mat_model_loader import load_oed_mat_model

TARGET_TRANSFORMS = ("auto", "log10_cycle_life", "log10_inverse_cycle_life")


def cutoff_for_batch_name(batch_name: str | None) -> int:
    if batch_name == "oed1":
        return 98
    if batch_name == "oed4":
        return 95
    return 100


def model_path_for_batch_name(bms_dir: str | Path, batch_name: str | None) -> Path:
    base = Path(bms_dir)
    if batch_name == "oed1":
        return base / "oed_model_batch1.mat"
    return base / "oed_model.mat"


def apply_oed_model(
    cells: list[BatteryCell],
    model_path: str | Path,
    cutoff_cycle: int,
    batch_name: str | None = None,
    target_transform: str = "auto",
) -> pd.DataFrame:
    if target_transform not in TARGET_TRANSFORMS:
        raise ValueError(f"target_transform must be one of {TARGET_TRANSFORMS}, got {target_transform}")
    model = load_oed_mat_model(model_path)
    feat, status = build_feature_matrix(cells, cutoff_cycle)
    selected = model.feat_ind_python
    inv_des = np.linalg.inv(model.des_mat)
    intermediate: list[dict[str, Any]] = []

    for idx, cell in enumerate(cells):
        row_status = status.iloc[idx].to_dict() if idx < len(status) else {"status": "unavailable", "error": "missing status"}
        raw_output = se = np.nan
        if row_status["status"] != "unavailable":
            try:
                feat_scaled_full = (feat[idx] - model.mu) / model.sigma
                feat_scaled = feat_scaled_full[selected]
                if not np.all(np.isfinite(feat_scaled)):
                    raise ValueError("selected model features contain non-finite values")
                x_aug = np.r_[feat_scaled, 1.0]
                se = model.t_val * np.sqrt(model.MSE + model.MSE * x_aug @ inv_des @ x_aug.T)
                raw_output = float(feat_scaled @ model.B1 + model.y_mu)
            except Exception as exc:
                row_status["status"] = "unavailable"
                row_status["error"] = str(exc)
        intermediate.append({"cell": cell, "row_status": row_status, "raw_output": raw_output, "se": se})

    chosen_transform = infer_target_transform(
        np.asarray([row["raw_output"] for row in intermediate], dtype=float),
        requested_transform=target_transform,
    )
    out_rows: list[dict[str, Any]] = []
    for row in intermediate:
        cell = row["cell"]
        row_status = row["row_status"]
        raw_output = float(row["raw_output"])
        se = float(row["se"])
        pred = ci_lo = ci_hi = ci_width = np.nan
        anomaly = False
        if row_status["status"] != "unavailable" and np.isfinite(raw_output) and np.isfinite(se):
            pred, ci_lo, ci_hi = transform_prediction(raw_output, se, chosen_transform)
            ci_width = ci_hi - ci_lo
            anomaly = bool(np.isfinite(ci_width) and ci_width > 2000)
            if anomaly:
                pred = -1.0
        out_rows.append(
            {
                "cell_id": cell.cell_id,
                "batch_id": cell.batch_id,
                "channel": cell.channel,
                "barcode": cell.barcode,
                "protocol_readable": cell.protocol_readable,
                "C1": cell.C1,
                "C2": cell.C2,
                "C3": cell.C3,
                "C4": cell.C4,
                "raw_output": raw_output,
                "target_transform": chosen_transform,
                "Prediction": pred,
                "CI_Lo": ci_lo,
                "CI_Hi": ci_hi,
                "ci_width": ci_width,
                "Lifetime": cell.lifetime,
                "anomaly_flag": bool(anomaly),
                "model_file": str(Path(model_path)),
                "cutoff_cycle": cutoff_cycle,
                "exact_feature_status": row_status["status"],
                "exact_feature_error": row_status["error"],
                "batch_name": batch_name,
            }
        )
    return pd.DataFrame(out_rows)


def infer_target_transform(raw_output: np.ndarray, requested_transform: str = "auto") -> str:
    if requested_transform not in TARGET_TRANSFORMS:
        raise ValueError(f"requested_transform must be one of {TARGET_TRANSFORMS}, got {requested_transform}")
    if requested_transform != "auto":
        return requested_transform
    raw = np.asarray(raw_output, dtype=float)
    raw = raw[np.isfinite(raw)]
    if raw.size == 0:
        raise ValueError("Cannot infer author model target transform without finite raw outputs")
    direct = 10**raw
    inverse = np.divide(1.0, direct, out=np.full_like(direct, np.nan), where=direct > 0)
    direct_median = float(np.nanmedian(direct))
    inverse_median = float(np.nanmedian(inverse))
    if 100 <= direct_median <= 3000:
        return "log10_cycle_life"
    if 100 <= inverse_median <= 3000 and direct_median < 1:
        return "log10_inverse_cycle_life"
    raise ValueError(
        "Unable to infer author-model target transform from raw outputs: "
        f"median(10**raw)={direct_median:.6g}, median(1/(10**raw))={inverse_median:.6g}"
    )


def transform_prediction(raw_output: float, se: float, target_transform: str) -> tuple[float, float, float]:
    if target_transform == "log10_cycle_life":
        prediction_cycles = 10**raw_output
        ci_lo_cycles = 10 ** (raw_output - se)
        ci_hi_cycles = 10 ** (raw_output + se)
    elif target_transform == "log10_inverse_cycle_life":
        inv_pred = 10**raw_output
        inv_lo = 10 ** (raw_output - se)
        inv_hi = 10 ** (raw_output + se)
        prediction_cycles = 1 / inv_pred
        ci_lo_cycles = 1 / inv_hi
        ci_hi_cycles = 1 / inv_lo
    else:
        raise ValueError(f"Unsupported target transform: {target_transform}")
    return float(prediction_cycles), float(ci_lo_cycles), float(ci_hi_cycles)


def _finite_stats(values: pd.Series | np.ndarray) -> dict[str, float | None]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"min": None, "median": None, "max": None}
    return {"min": float(np.min(arr)), "median": float(np.median(arr)), "max": float(np.max(arr))}


def prediction_scale_diagnostics(df: pd.DataFrame) -> dict[str, Any]:
    raw = np.asarray(df.get("raw_output", pd.Series(dtype=float)), dtype=float)
    direct = 10**raw
    inverse = np.divide(1.0, direct, out=np.full_like(direct, np.nan), where=direct > 0)
    ci_width = np.asarray(df.get("ci_width", pd.Series(dtype=float)), dtype=float)
    pred = np.asarray(df.get("Prediction", pd.Series(dtype=float)), dtype=float)
    transform = None
    if "target_transform" in df and not df["target_transform"].dropna().empty:
        transform = str(df["target_transform"].dropna().iloc[0])
    return {
        "raw_output": _finite_stats(raw),
        "ten_power_raw_output": _finite_stats(direct),
        "inverse_ten_power_raw_output": _finite_stats(inverse),
        "chosen_target_transform": transform,
        "prediction": _finite_stats(pred[pred > 0]),
        "ci_lo": _finite_stats(df.get("CI_Lo", pd.Series(dtype=float))),
        "ci_hi": _finite_stats(df.get("CI_Hi", pd.Series(dtype=float))),
        "ci_width": _finite_stats(ci_width),
        "n_ci_width_gt_2000": int(np.sum(np.isfinite(ci_width) & (ci_width > 2000))),
        "n_prediction_le_0": int(np.sum(np.isfinite(pred) & (pred <= 0))),
    }


def write_bayesgap_input(predictions: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    cols = ["C1", "C2", "C3", "C4", "Prediction"]
    missing = [col for col in cols if col not in predictions.columns]
    if missing:
        raise ValueError(f"prediction dataframe missing BayesGap columns: {missing}")
    mask = np.ones(len(predictions), dtype=bool)
    for col in cols:
        mask &= np.isfinite(pd.to_numeric(predictions[col], errors="coerce").to_numpy(float))
    mask &= pd.to_numeric(predictions["Prediction"], errors="coerce").to_numpy(float) > 0
    if "anomaly_flag" in predictions:
        mask &= ~predictions["anomaly_flag"].fillna(False).astype(bool).to_numpy()
    out = predictions.loc[mask, cols].copy()
    validate_bayesgap_prediction_scale(out)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def validate_bayesgap_prediction_scale(df: pd.DataFrame) -> None:
    if df.empty:
        return
    positive = pd.to_numeric(df["Prediction"], errors="coerce")
    positive = positive[np.isfinite(positive) & (positive > 0)]
    if positive.empty:
        return
    median = float(positive.median())
    if not (100 < median < 5000):
        raise ValueError(f"BayesGap prediction median is not on cycle-life scale: median={median:.6g}")


def bayesgap_exclusion_audit(predictions: pd.DataFrame, included: pd.DataFrame) -> dict[str, Any]:
    reasons: dict[str, int] = {}
    excluded_rows: list[dict[str, Any]] = []
    included_index = set(included.index)
    for idx, row in predictions.iterrows():
        if idx in included_index:
            continue
        row_reasons: list[str] = []
        for col in ["C1", "C2", "C3", "C4"]:
            if col not in row or not np.isfinite(pd.to_numeric(row[col], errors="coerce")):
                row_reasons.append(f"nonfinite_{col}")
        pred = pd.to_numeric(row.get("Prediction", np.nan), errors="coerce")
        if not np.isfinite(pred):
            row_reasons.append("nonfinite_prediction")
        elif pred <= 0:
            row_reasons.append("prediction_le_0")
        if bool(row.get("anomaly_flag", False)):
            row_reasons.append("anomalous_ci_width_gt_2000")
        if not row_reasons:
            row_reasons.append("filtered_unknown")
        for reason in row_reasons:
            reasons[reason] = reasons.get(reason, 0) + 1
        excluded_rows.append(
            {
                "cell_id": row.get("cell_id", ""),
                "channel": row.get("channel", ""),
                "reasons": row_reasons,
            }
        )
    return {
        "bayesgap_rows": int(len(included)),
        "bayesgap_exclusion_reasons": reasons,
        "bayesgap_excluded_rows": excluded_rows,
    }


def prediction_summary(df: pd.DataFrame) -> dict[str, Any]:
    pred = pd.to_numeric(df.get("Prediction", pd.Series(dtype=float)), errors="coerce")
    exact_status = df.get("exact_feature_status", pd.Series(dtype=str))
    return {
        "rows": int(len(df)),
        "cells_with_exact_features_available": int((exact_status != "unavailable").sum()),
        "finite_predictions": int(np.isfinite(pred).sum()),
        "available_predictions": int(np.sum(np.isfinite(pred) & (pred > 0))),
        "anomalous_predictions": int(df.get("anomaly_flag", pd.Series(dtype=bool)).sum()),
        "unavailable_features": int((exact_status == "unavailable").sum()),
    }
