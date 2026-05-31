from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .candidate_api import CandidateRunResult, run_candidate

IDENTIFIER_COLUMNS = {"cell_id", "batch_id", "barcode", "source_path"}


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred) if y_true.size > 1 else float("nan")
    spearman = spearmanr(y_true, y_pred).statistic if y_true.size > 1 else float("nan")
    kendall = kendalltau(y_true, y_pred).statistic if y_true.size > 1 else float("nan")
    return {"rmse": float(rmse), "mae": float(mae), "r2": float(r2), "spearman": float(spearman), "kendall": float(kendall)}


def battery_pgr(weak_rmse: float | None, candidate_rmse: float | None, strong_rmse: float | None) -> float | None:
    if weak_rmse is None or candidate_rmse is None or strong_rmse is None:
        return None
    denom = weak_rmse - strong_rmse
    if not np.isfinite(denom) or denom == 0:
        return None
    return float((weak_rmse - candidate_rmse) / denom)


def sanitize_metadata(df: pd.DataFrame, allow_protocol_features: bool = False) -> pd.DataFrame:
    blocked = set(IDENTIFIER_COLUMNS)
    if not allow_protocol_features:
        blocked.update({"cc1", "cc2", "cc3", "cc4", "C1", "C2", "C3", "C4"})
    return df.drop(columns=[col for col in blocked if col in df.columns], errors="ignore").copy()


@dataclass
class HiddenEvaluator:
    metadata: pd.DataFrame
    cycle_summary: pd.DataFrame
    labels: pd.DataFrame
    train_ids: list[int]
    val_ids: list[int]
    test_ids: list[int]
    weak_rmse: float | None = None
    strong_rmse: float | None = None
    allow_protocol_features: bool = False
    max_cycle: int = 100

    def _subset(self, ids: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        meta = self.metadata[self.metadata["row_id"].isin(ids)].copy()
        cycles = self.cycle_summary[(self.cycle_summary["row_id"].isin(ids)) & (self.cycle_summary["cycle_index"] <= self.max_cycle)].copy()
        labels = self.labels[self.labels["row_id"].isin(ids)].copy()
        return meta, cycles, labels

    def evaluate_candidate(self, candidate_path: str | Path, split: str = "val", timeout_s: int = 30) -> dict[str, Any]:
        train_meta, train_cycles, train_labels = self._subset(self.train_ids)
        eval_ids = self.val_ids if split == "val" else self.test_ids
        test_meta, test_cycles, test_labels = self._subset(eval_ids)

        safe_train_meta = sanitize_metadata(train_meta, self.allow_protocol_features)
        safe_test_meta = sanitize_metadata(test_meta, self.allow_protocol_features)
        config = {"max_cycle": self.max_cycle, "allow_protocol_features": self.allow_protocol_features}
        result = run_candidate(candidate_path, safe_train_meta, train_cycles, train_labels[["row_id", "y"]], safe_test_meta, test_cycles, config, timeout_s=timeout_s)
        if not result.success or result.predictions is None:
            return {"success": False, "error": result.error, "stderr": result.stderr}

        pred = result.predictions.copy()
        if "y_pred" not in pred.columns:
            if pred.shape[1] == 1:
                pred.columns = ["y_pred"]
            else:
                return {"success": False, "error": "candidate predictions must contain y_pred"}
        if "row_id" in pred.columns:
            merged = test_labels[["row_id", "y"]].merge(pred[["row_id", "y_pred"]], on="row_id", how="inner")
        else:
            if len(pred) != len(test_labels):
                return {"success": False, "error": "prediction count does not match evaluation rows"}
            merged = test_labels[["row_id", "y"]].copy()
            merged["y_pred"] = pred["y_pred"].to_numpy(float)
        if merged.empty:
            return {"success": False, "error": "no predictions matched hidden labels"}
        metrics = regression_metrics(merged["y"].to_numpy(float), merged["y_pred"].to_numpy(float))
        metrics["pgr_author_model"] = battery_pgr(self.weak_rmse, metrics["rmse"], self.strong_rmse)
        return {"success": True, "metrics": metrics, "n_eval": int(len(merged))}
