from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr

from .paths import validation_status, validation_status_label


def _protocol_key(df: pd.DataFrame) -> pd.Series:
    return (
        df["C1"].round(3).astype(str)
        + "|"
        + df["C2"].round(3).astype(str)
        + "|"
        + df["C3"].round(3).astype(str)
        + "|"
        + df["C4"].round(3).astype(str)
    )


def _metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int | None]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        return {"n": 0, "pearson": None, "kendall": None, "rmse": None, "mae": None}
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    pearson = None
    kendall = None
    if y_true.size >= 2 and np.nanstd(y_true) > 0 and np.nanstd(y_pred) > 0:
        pearson = float(pearsonr(y_true, y_pred).statistic)
        kendall = float(kendalltau(y_true, y_pred).statistic)
    return {"n": int(y_true.size), "pearson": pearson, "kendall": kendall, "rmse": rmse, "mae": mae}


def build_validation_protocol_ranking(validation_rich: pd.DataFrame, final_posterior_ranking: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    if validation_rich.empty:
        empty = pd.DataFrame()
        return empty, {
            "validation_cells_parsed": 0,
            "validation_protocols_parsed": 0,
            "protocol_level_rows_used": 0,
            "posterior_vs_observed": _metric_dict(np.array([]), np.array([])),
            "early_prediction_vs_observed": _metric_dict(np.array([]), np.array([])),
        }

    rich = validation_rich.copy()
    for col in ["C1", "C2", "C3", "C4", "Prediction", "Lifetime"]:
        if col in rich:
            rich[col] = pd.to_numeric(rich[col], errors="coerce")
    anomaly = rich["anomaly_flag"].fillna(False).astype(bool) if "anomaly_flag" in rich else pd.Series(False, index=rich.index)
    rich["valid_early_prediction"] = np.isfinite(rich["Prediction"]) & (rich["Prediction"] > 0) & ~anomaly
    rich["protocol_key"] = _protocol_key(rich)

    observed = (
        rich[np.isfinite(rich["Lifetime"])]
        .groupby("protocol_key")
        .agg(
            C1=("C1", "first"),
            C2=("C2", "first"),
            C3=("C3", "first"),
            C4=("C4", "first"),
            validation_cell_count=("cell_id", "count"),
            observed_cycle_life_mean=("Lifetime", "mean"),
            observed_cycle_life_median=("Lifetime", "median"),
        )
        .reset_index()
    )
    early = (
        rich[rich["valid_early_prediction"]]
        .groupby("protocol_key")
        .agg(
            author_early_prediction_cell_count=("Prediction", "count"),
            author_early_prediction_mean=("Prediction", "mean"),
            author_early_prediction_median=("Prediction", "median"),
        )
        .reset_index()
    )
    ranking = final_posterior_ranking.copy()
    ranking["protocol_key"] = _protocol_key(ranking)
    ranking_cols = [
        "protocol_key",
        "final_posterior_rank",
        "final_posterior_mean",
        "posterior_half_width",
        "final_posterior_lower",
        "final_posterior_upper",
    ]
    merged = observed.merge(early, on="protocol_key", how="left").merge(ranking[ranking_cols], on="protocol_key", how="left")
    merged["exists_in_final_posterior"] = np.isfinite(merged["final_posterior_mean"])
    merged = merged.sort_values("observed_cycle_life_mean", ascending=False).reset_index(drop=True)
    merged.insert(0, "observed_protocol_rank", np.arange(1, len(merged) + 1))
    merged["posterior_minus_observed"] = merged["final_posterior_mean"] - merged["observed_cycle_life_mean"]
    merged["early_prediction_minus_observed"] = merged["author_early_prediction_mean"] - merged["observed_cycle_life_mean"]

    usable_posterior = merged[np.isfinite(merged["observed_cycle_life_mean"]) & np.isfinite(merged["final_posterior_mean"])]
    usable_early = merged[np.isfinite(merged["observed_cycle_life_mean"]) & np.isfinite(merged["author_early_prediction_mean"])]
    metrics = {
        "validation_cells_parsed": int(len(rich)),
        "validation_protocols_parsed": int(rich["protocol_key"].nunique()),
        "protocol_level_rows_used": int(len(usable_posterior)),
        "posterior_vs_observed": _metric_dict(
            usable_posterior["observed_cycle_life_mean"].to_numpy(float),
            usable_posterior["final_posterior_mean"].to_numpy(float),
        ),
        "early_prediction_vs_observed": _metric_dict(
            usable_early["observed_cycle_life_mean"].to_numpy(float),
            usable_early["author_early_prediction_mean"].to_numpy(float),
        ),
    }
    return merged.drop(columns=["protocol_key"]), metrics


def write_validation_outputs(
    validation_rich: pd.DataFrame,
    final_posterior_ranking: pd.DataFrame,
    ranking_path: str | Path,
    metrics_path: str | Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    ranking, metrics = build_validation_protocol_ranking(validation_rich, final_posterior_ranking)
    ranking_path = Path(ranking_path)
    ranking_path.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(ranking_path, index=False)
    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return ranking, metrics


__all__ = [
    "build_validation_protocol_ranking",
    "validation_status",
    "validation_status_label",
    "write_validation_outputs",
]
