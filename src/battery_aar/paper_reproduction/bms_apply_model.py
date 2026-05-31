from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .bms_features import BatteryCell, build_feature_matrix
from .mat_model_loader import OEDMatModel, load_oed_mat_model


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
) -> pd.DataFrame:
    model = load_oed_mat_model(model_path)
    feat, status = build_feature_matrix(cells, cutoff_cycle)
    selected = model.feat_ind_python
    out_rows: list[dict[str, Any]] = []
    inv_des = np.linalg.inv(model.des_mat)

    for idx, cell in enumerate(cells):
        row_status = status.iloc[idx].to_dict() if idx < len(status) else {"status": "unavailable", "error": "missing status"}
        pred = ci_lo = ci_hi = np.nan
        anomaly = False
        if row_status["status"] != "unavailable":
            try:
                feat_scaled_full = (feat[idx] - model.mu) / model.sigma
                feat_scaled = feat_scaled_full[selected]
                if not np.all(np.isfinite(feat_scaled)):
                    raise ValueError("selected model features contain non-finite values")
                x_aug = np.r_[feat_scaled, 1.0]
                se = model.t_val * np.sqrt(model.MSE + model.MSE * x_aug @ inv_des @ x_aug.T)
                ypred_log = float(feat_scaled @ model.B1 + model.y_mu)
                pred = float(10**ypred_log)
                ci_lo = float(10 ** (ypred_log - se))
                ci_hi = float(10 ** (ypred_log + se))
                if (ci_hi - ci_lo) > 2000:
                    pred = -1.0
                    anomaly = True
            except Exception as exc:
                row_status["status"] = "unavailable"
                row_status["error"] = str(exc)

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
                "Prediction": pred,
                "CI_Lo": ci_lo,
                "CI_Hi": ci_hi,
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


def write_bayesgap_input(predictions: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    cols = ["C1", "C2", "C3", "C4", "Prediction"]
    missing = [col for col in cols if col not in predictions.columns]
    if missing:
        raise ValueError(f"prediction dataframe missing BayesGap columns: {missing}")
    out = predictions.loc[:, cols].dropna(subset=["C1", "C2", "C3", "C4", "Prediction"])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def prediction_summary(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(df)),
        "available_predictions": int(np.isfinite(df["Prediction"]).sum()) if "Prediction" in df else 0,
        "anomalous_predictions": int(df.get("anomaly_flag", pd.Series(dtype=bool)).sum()),
        "unavailable_features": int((df.get("exact_feature_status", pd.Series(dtype=str)) == "unavailable").sum()),
    }
