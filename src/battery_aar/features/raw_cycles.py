from __future__ import annotations

import json
import gzip
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

CANONICAL_CYCLE_COLUMNS = [
    "row_id",
    "cell_id",
    "cycle_index",
    "step_type",
    "test_time",
    "voltage",
    "current",
    "charge_capacity",
    "discharge_capacity",
    "charge_energy",
    "discharge_energy",
    "internal_resistance",
    "temperature",
]

NUMERIC_CYCLE_COLUMNS = [
    "cycle_index",
    "test_time",
    "voltage",
    "current",
    "charge_capacity",
    "discharge_capacity",
    "charge_energy",
    "discharge_energy",
    "internal_resistance",
    "temperature",
]


def load_attia_json_payload(path: Union[str, Path]) -> Dict[str, Any]:
    raw_path = Path(path)
    if raw_path.suffix == ".gz":
        with gzip.open(raw_path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(raw_path.read_text())


def _as_1d(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=object)
    arr = np.asarray(value)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr.ravel()


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _canonical_from_mapping(mapping: Dict[str, Any], *, cell_id: Optional[str], row_id: Optional[int], include_step_type: bool) -> pd.DataFrame:
    arrays = {str(key): _as_1d(value) for key, value in mapping.items()}
    lengths = {key: int(len(value)) for key, value in arrays.items() if len(value) > 0}
    if not lengths:
        out = pd.DataFrame(columns=CANONICAL_CYCLE_COLUMNS)
        out.attrs["warnings"] = ["no cycle arrays found"]
        return out
    min_len = min(lengths.values())
    max_len = max(lengths.values())
    warnings: list[str] = []
    if min_len != max_len:
        warnings.append(f"cycles_interpolated arrays have unequal lengths; truncated to {min_len} rows")

    aliases = {
        "cycle_index": ["cycle_index", "cycle", "cycle_number"],
        "step_type": ["step_type", "step", "state"],
        "test_time": ["test_time", "time", "t"],
        "voltage": ["voltage", "V", "v"],
        "current": ["current", "I", "i"],
        "charge_capacity": ["charge_capacity", "QCharge", "q_charge"],
        "discharge_capacity": ["discharge_capacity", "QDischarge", "q_discharge"],
        "charge_energy": ["charge_energy", "ECharge", "e_charge"],
        "discharge_energy": ["discharge_energy", "EDischarge", "e_discharge"],
        "internal_resistance": ["dc_internal_resistance", "internal_resistance", "IR", "resistance"],
        "temperature": ["temperature", "temperature_average", "Tavg", "Tmean", "T", "temp"],
    }
    data: dict[str, Any] = {}
    resolved_signals: dict[str, str] = {}
    for canonical, keys in aliases.items():
        for key in keys:
            if key in arrays and len(arrays[key]) > 0:
                data[canonical] = arrays[key][:min_len]
                resolved_signals[canonical] = key
                break
    out = pd.DataFrame(data)
    if "cycle_index" not in out.columns:
        out["cycle_index"] = np.arange(min_len)
        warnings.append("cycle_index missing; generated zero-based positional index")
    if include_step_type and "step_type" in out.columns:
        out["step_type"] = out["step_type"].astype(str).str.lower().str.strip()
    elif include_step_type:
        out["step_type"] = ""
    for col in NUMERIC_CYCLE_COLUMNS:
        if col in out.columns:
            out[col] = _coerce_numeric(out[col])
    out.insert(0, "cell_id", cell_id)
    out.insert(0, "row_id", row_id)
    for col in CANONICAL_CYCLE_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan if col in NUMERIC_CYCLE_COLUMNS else ""
    out = out[CANONICAL_CYCLE_COLUMNS]
    out.attrs["warnings"] = warnings
    out.attrs["resolved_signals"] = resolved_signals
    return out


def canonicalize_cycles_interpolated(payload: Dict[str, Any], *, cell_id: Optional[str] = None, row_id: Optional[int] = None) -> pd.DataFrame:
    cycles = payload.get("cycles_interpolated") or {}
    if isinstance(cycles, list):
        cycles = {key: [row.get(key) for row in cycles if isinstance(row, dict)] for key in CANONICAL_CYCLE_COLUMNS}
    if not isinstance(cycles, dict):
        out = pd.DataFrame(columns=CANONICAL_CYCLE_COLUMNS)
        out.attrs["warnings"] = ["cycles_interpolated is not a mapping"]
        return out
    return _canonical_from_mapping(cycles, cell_id=cell_id, row_id=row_id, include_step_type=True)


def canonicalize_summary(payload: Dict[str, Any], *, cell_id: Optional[str] = None, row_id: Optional[int] = None) -> pd.DataFrame:
    summary = payload.get("summary") or {}
    if not isinstance(summary, dict):
        out = pd.DataFrame(columns=[col for col in CANONICAL_CYCLE_COLUMNS if col != "step_type"])
        out.attrs["warnings"] = ["summary is not a mapping"]
        return out
    out = _canonical_from_mapping(summary, cell_id=cell_id, row_id=row_id, include_step_type=False)
    out = out.drop(columns=["step_type"], errors="ignore")
    return out


def infer_cycle_index_convention(cycles_df: pd.DataFrame, summary_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    values = pd.to_numeric(cycles_df.get("cycle_index", pd.Series(dtype=float)), errors="coerce").dropna()
    summary_values = pd.Series(dtype=float)
    if summary_df is not None and "cycle_index" in summary_df:
        summary_values = pd.to_numeric(summary_df["cycle_index"], errors="coerce").dropna()
    preview = values.head(10).tolist()
    min_cycle = float(values.min()) if not values.empty else None
    if min_cycle == 0:
        convention = "raw_zero_based"
    elif min_cycle == 1:
        convention = "raw_one_based"
    else:
        convention = "unknown"
    return {
        "cycle_index_convention": convention,
        "cycle_index_min": min_cycle,
        "cycle_index_max": float(values.max()) if not values.empty else None,
        "cycle_index_preview": preview,
        "summary_cycle_index_min": float(summary_values.min()) if not summary_values.empty else None,
        "summary_cycle_index_max": float(summary_values.max()) if not summary_values.empty else None,
    }


def validate_cycles_df(cycles_df: pd.DataFrame) -> List[str]:
    warnings: list[str] = list(cycles_df.attrs.get("warnings", []))
    required = {"row_id", "cell_id", "cycle_index"}
    missing = sorted(required - set(cycles_df.columns))
    if missing:
        warnings.append(f"cycles_df missing required columns: {missing}")
    if "cycle_index" in cycles_df and pd.to_numeric(cycles_df["cycle_index"], errors="coerce").isna().all():
        warnings.append("cycle_index is entirely nonnumeric")
    if "step_type" in cycles_df:
        nonempty = cycles_df["step_type"].astype(str).str.len() > 0
        if nonempty.any() and not cycles_df.loc[nonempty, "step_type"].astype(str).str.islower().all():
            warnings.append("step_type values are not fully normalized to lower-case")
    return warnings


def validate_feature_table(feature_table: pd.DataFrame) -> List[str]:
    warnings: list[str] = []
    if "row_id" not in feature_table.columns and "cell_id" not in feature_table.columns:
        warnings.append("feature_table must contain row_id or cell_id")
    for col in feature_table.columns:
        if col in {"row_id", "cell_id"}:
            continue
        values = pd.to_numeric(feature_table[col], errors="coerce")
        if np.isinf(values.to_numpy(float)).any():
            warnings.append(f"feature column contains infinities: {col}")
    return warnings


def validate_feature_metadata(feature_metadata: pd.DataFrame) -> List[str]:
    warnings: list[str] = []
    required = {"feature_name", "feature_family", "operator_name", "operator_type"}
    missing = sorted(required - set(feature_metadata.columns))
    if missing:
        warnings.append(f"feature_metadata missing required columns: {missing}")
    if "feature_name" in feature_metadata and feature_metadata["feature_name"].duplicated().any():
        warnings.append("feature_metadata contains duplicate feature_name rows")
    return warnings
