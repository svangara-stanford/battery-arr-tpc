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

IDENTIFIER_COLUMNS = {
    "cell_id",
    "original_cell_id",
    "batch_id",
    "original_batch_id",
    "barcode",
    "source_path",
    "file_path",
    "filename",
    "path",
    "channel",
    "cycle_life",
    "Lifetime",
    "protocol_readable",
    "policy_readable",
}
PROTOCOL_COLUMNS = {"cc1", "cc2", "cc3", "cc4", "C1", "C2", "C3", "C4"}
REQUIRED_CYCLE_COLUMNS = ("cycle_index", "discharge_capacity", "charge_capacity")


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


def _anonymized_cell_id_map(row_ids: pd.Series | list[Any]) -> dict[Any, str]:
    values = pd.Series(row_ids).dropna().drop_duplicates().tolist()
    values = sorted(values, key=lambda value: str(value))
    return {row_id: f"anon_cell_{idx:05d}" for idx, row_id in enumerate(values)}


def _add_candidate_join_keys(df: pd.DataFrame, cell_id_map: dict[Any, str]) -> pd.DataFrame:
    if "row_id" not in df.columns:
        raise ValueError("candidate-facing dataframe requires row_id")
    out = df.copy()
    out["cell_id"] = out["row_id"].map(cell_id_map)
    if out["cell_id"].isna().any():
        missing = out.loc[out["cell_id"].isna(), "row_id"].drop_duplicates().tolist()
        raise ValueError(f"missing anonymized cell_id mapping for row_id values: {missing[:5]}")
    return out


def sanitize_metadata(df: pd.DataFrame, allow_protocol_features: bool = False, cell_id_map: dict[Any, str] | None = None) -> pd.DataFrame:
    blocked = set(IDENTIFIER_COLUMNS)
    if not allow_protocol_features:
        blocked.update(PROTOCOL_COLUMNS)
    out = df.drop(columns=[col for col in blocked if col in df.columns and col != "row_id"], errors="ignore").copy()
    if cell_id_map is None:
        cell_id_map = _anonymized_cell_id_map(out["row_id"])
    out = _add_candidate_join_keys(out, cell_id_map)
    key_cols = ["row_id", "cell_id"]
    other_cols = [col for col in out.columns if col not in key_cols and pd.api.types.is_numeric_dtype(out[col])]
    return out[key_cols + other_cols]


def sanitize_cycle_summary(df: pd.DataFrame, cell_id_map: dict[Any, str]) -> pd.DataFrame:
    blocked = set(IDENTIFIER_COLUMNS) | {"y", "label", "target"}
    allowed = ["row_id"]
    for col in REQUIRED_CYCLE_COLUMNS:
        if col in df.columns:
            allowed.append(col)
    numeric_cols = [
        col
        for col in df.select_dtypes(include=[np.number]).columns
        if col not in set(allowed) | blocked | PROTOCOL_COLUMNS
    ]
    out = df[allowed + numeric_cols].copy()
    for col in REQUIRED_CYCLE_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    out = _add_candidate_join_keys(out, cell_id_map)
    ordered = ["row_id", "cell_id", *REQUIRED_CYCLE_COLUMNS]
    other_cols = [col for col in out.columns if col not in ordered]
    return out[ordered + other_cols]


def sanitize_labels(df: pd.DataFrame, cell_id_map: dict[Any, str]) -> pd.DataFrame:
    out = df[["row_id", "y"]].copy()
    out = _add_candidate_join_keys(out, cell_id_map)
    return out[["row_id", "cell_id", "y"]]


def _candidate_failure(
    candidate_path: str | Path,
    failure_reason: str,
    error_type: str = "CandidateError",
    result: CandidateRunResult | None = None,
) -> dict[str, Any]:
    return {
        "success": False,
        "error": failure_reason,
        "error_type": error_type,
        "failure_reason": failure_reason,
        "traceback": result.traceback if result else None,
        "stdout": result.stdout if result else "",
        "stderr": result.stderr if result else "",
        "candidate_syntax_status": result.syntax_status if result else "unknown",
        "candidate_path": str(candidate_path),
    }


def evaluate_candidate_train_test(
    candidate_path: str | Path,
    train_meta: pd.DataFrame,
    train_cycles: pd.DataFrame,
    train_labels: pd.DataFrame,
    test_meta: pd.DataFrame,
    test_cycles: pd.DataFrame,
    test_labels: pd.DataFrame,
    weak_rmse: float | None = None,
    strong_rmse: float | None = None,
    allow_protocol_features: bool = False,
    max_cycle: int = 100,
    timeout_s: int = 30,
    return_predictions: bool = False,
) -> dict[str, Any]:
    train_cycles = train_cycles[train_cycles["cycle_index"] <= max_cycle].copy()
    test_cycles = test_cycles[test_cycles["cycle_index"] <= max_cycle].copy()
    row_ids = pd.concat(
        [
            train_meta["row_id"],
            train_cycles["row_id"],
            train_labels["row_id"],
            test_meta["row_id"],
            test_cycles["row_id"],
            test_labels["row_id"],
        ],
        ignore_index=True,
    )
    cell_id_map = _anonymized_cell_id_map(row_ids)
    safe_train_meta = sanitize_metadata(train_meta, allow_protocol_features, cell_id_map)
    safe_test_meta = sanitize_metadata(test_meta, allow_protocol_features, cell_id_map)
    safe_train_cycles = sanitize_cycle_summary(train_cycles, cell_id_map)
    safe_test_cycles = sanitize_cycle_summary(test_cycles, cell_id_map)
    safe_train_labels = sanitize_labels(train_labels, cell_id_map)
    config = {"max_cycle": max_cycle, "allow_protocol_features": allow_protocol_features}
    result = run_candidate(candidate_path, safe_train_meta, safe_train_cycles, safe_train_labels, safe_test_meta, safe_test_cycles, config, timeout_s=timeout_s)
    if not result.success or result.predictions is None:
        return _candidate_failure(
            candidate_path,
            result.failure_reason or result.error or "candidate failed",
            result.error_type or "CandidateRuntimeError",
            result,
        )

    pred = result.predictions.copy()
    if "y_pred" not in pred.columns:
        if pred.shape[1] == 1:
            pred.columns = ["y_pred"]
        else:
            return _candidate_failure(candidate_path, "candidate predictions must contain y_pred", "PredictionValidationError", result)
    expected = test_labels[["row_id", "y"]].copy()
    expected_ids = expected["row_id"].tolist()
    test_order_ids = safe_test_meta["row_id"].tolist()
    if "row_id" in pred.columns:
        row_pred = pred[["row_id", "y_pred"]].copy()
    elif "cell_id" in pred.columns:
        reverse_map = {alias: row_id for row_id, alias in cell_id_map.items()}
        row_pred = pred[["cell_id", "y_pred"]].copy()
        row_pred["row_id"] = row_pred["cell_id"].map(reverse_map)
        if row_pred["row_id"].isna().any():
            return _candidate_failure(candidate_path, "candidate returned unknown cell_id value(s)", "PredictionValidationError", result)
        row_pred = row_pred[["row_id", "y_pred"]]
    else:
        if len(pred) != len(expected):
            return _candidate_failure(
                candidate_path,
                "candidate predictions must include row_id or cell_id when prediction count does not match evaluation rows",
                "PredictionValidationError",
                result,
            )
        row_pred = pd.DataFrame({"row_id": test_order_ids, "y_pred": pred["y_pred"].to_numpy()})

    if row_pred["row_id"].duplicated().any():
        return _candidate_failure(candidate_path, "candidate returned duplicate row_id predictions", "PredictionValidationError", result)
    expected_set = set(expected_ids)
    predicted_set = set(row_pred["row_id"].tolist())
    if predicted_set != expected_set:
        missing = sorted(expected_set - predicted_set, key=lambda value: str(value))
        extra = sorted(predicted_set - expected_set, key=lambda value: str(value))
        return _candidate_failure(
            candidate_path,
            f"candidate predictions did not cover expected rows: missing={missing[:5]}, extra={extra[:5]}",
            "PredictionValidationError",
            result,
        )
    row_pred["y_pred"] = pd.to_numeric(row_pred["y_pred"], errors="coerce")
    if not np.isfinite(row_pred["y_pred"].to_numpy(float)).all():
        return _candidate_failure(candidate_path, "candidate predictions must all be finite numeric values", "PredictionValidationError", result)
    merged = expected.merge(row_pred, on="row_id", how="inner")
    if len(merged) != len(expected):
        return _candidate_failure(candidate_path, "no predictions matched hidden labels", "PredictionValidationError", result)
    metrics = regression_metrics(merged["y"].to_numpy(float), merged["y_pred"].to_numpy(float))
    metrics["pgr_author_model"] = battery_pgr(weak_rmse, metrics["rmse"], strong_rmse)
    output: dict[str, Any] = {
        "success": True,
        "metrics": metrics,
        "n_eval": int(len(merged)),
        "stdout": result.stdout,
        "stderr": result.stderr,
        "candidate_syntax_status": result.syntax_status,
        "candidate_path": str(candidate_path),
    }
    if return_predictions:
        output["predictions"] = merged[["row_id", "y", "y_pred"]].copy()
    return output


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

    def _cell_id_map(self) -> dict[Any, str]:
        row_ids = pd.concat(
            [
                self.metadata["row_id"],
                self.cycle_summary["row_id"],
                self.labels["row_id"],
            ],
            ignore_index=True,
        )
        return _anonymized_cell_id_map(row_ids)

    def evaluate_candidate(self, candidate_path: str | Path, split: str = "val", timeout_s: int = 30) -> dict[str, Any]:
        train_meta, train_cycles, train_labels = self._subset(self.train_ids)
        eval_ids = self.val_ids if split == "val" else self.test_ids
        test_meta, test_cycles, test_labels = self._subset(eval_ids)
        return evaluate_candidate_train_test(
            candidate_path,
            train_meta,
            train_cycles,
            train_labels,
            test_meta,
            test_cycles,
            test_labels,
            weak_rmse=self.weak_rmse,
            strong_rmse=self.strong_rmse,
            allow_protocol_features=self.allow_protocol_features,
            max_cycle=self.max_cycle,
            timeout_s=timeout_s,
        )
