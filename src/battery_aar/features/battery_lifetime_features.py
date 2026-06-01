from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import skew

KEY_COLUMNS = ("row_id", "cell_id")
CAPACITY_CYCLES = (1, 2, 10, 50, 95, 98, 100)
PROTOCOL_COLUMNS = ("C1", "C2", "C3", "C4", "cc1", "cc2", "cc3", "cc4")


@dataclass(frozen=True)
class FeatureFamilyMetadata:
    feature_name: str
    family: str
    approximate: bool = False


def _key_column(df: pd.DataFrame) -> str:
    for col in KEY_COLUMNS:
        if col in df.columns:
            return col
    raise ValueError("input dataframe must contain row_id or cell_id")


def _numeric(values: pd.Series | np.ndarray | list[Any]) -> np.ndarray:
    return pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)


def _safe_log10(value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value == 0:
        return float("nan")
    return float(np.log10(abs(value)))


def _linear_fit(cycle: np.ndarray, capacity: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(cycle) & np.isfinite(capacity)
    if mask.sum() < 2:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(cycle[mask], capacity[mask], 1)
    return float(slope), float(intercept)


def _feature_metadata(features: pd.DataFrame, family: str, approximate: bool = False) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_name": list(features.columns),
            "family": family,
            "approximate": approximate,
        }
    )


def _drop_empty_numeric_columns(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    for col in list(out.columns):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna(axis=1, how="all")


def build_capacity_summary_features(cycle_summary: pd.DataFrame, max_cycle: int = 100) -> pd.DataFrame:
    """Build early-cycle capacity and slope features.

    The returned dataframe has one row per candidate-facing cell key. The key is
    stored as the index only; all columns are numeric feature columns.
    """
    key = _key_column(cycle_summary)
    if "cycle_index" not in cycle_summary.columns or "discharge_capacity" not in cycle_summary.columns:
        raise ValueError("cycle_summary must contain cycle_index and discharge_capacity")
    rows: list[dict[str, float | Any]] = []
    for cell_key, grp in cycle_summary.groupby(key, dropna=False):
        g = grp.copy()
        g["cycle_index"] = pd.to_numeric(g["cycle_index"], errors="coerce")
        g["discharge_capacity"] = pd.to_numeric(g["discharge_capacity"], errors="coerce")
        g = g[(g["cycle_index"] <= max_cycle) & np.isfinite(g["cycle_index"])].sort_values("cycle_index")
        row: dict[str, float | Any] = {"_key": cell_key}
        if g.empty:
            rows.append(row)
            continue

        by_cycle = g.dropna(subset=["cycle_index"]).groupby("cycle_index")["discharge_capacity"].mean()
        for cycle in CAPACITY_CYCLES:
            row[f"capacity_cycle_{cycle}"] = float(by_cycle.get(float(cycle), np.nan))

        capacities = g["discharge_capacity"].to_numpy(float)
        cycles = g["cycle_index"].to_numpy(float)
        cycle_2 = float(by_cycle.get(2.0, np.nan))
        cycle_10 = float(by_cycle.get(10.0, np.nan))
        n_cycle = float(max_cycle)
        cycle_n = float(by_cycle.get(n_cycle, np.nan))
        if not np.isfinite(cycle_n) and not by_cycle.empty:
            valid_le_n = by_cycle[by_cycle.index <= max_cycle].dropna()
            cycle_n = float(valid_le_n.iloc[-1]) if not valid_le_n.empty else float("nan")

        max_capacity = float(np.nanmax(capacities)) if np.isfinite(capacities).any() else float("nan")
        first_capacity = float(capacities[np.where(np.isfinite(capacities))[0][0]]) if np.isfinite(capacities).any() else float("nan")
        last_capacity = float(capacities[np.where(np.isfinite(capacities))[0][-1]]) if np.isfinite(capacities).any() else float("nan")
        early_mask = (cycles >= 2) & (cycles <= max_cycle)
        late_start = 91 if max_cycle >= 91 else max(2, max_cycle - 9)
        late_mask = (cycles >= late_start) & (cycles <= max_cycle)
        early_slope, early_intercept = _linear_fit(cycles[early_mask], capacities[early_mask])
        late_slope, late_intercept = _linear_fit(cycles[late_mask], capacities[late_mask])

        row.update(
            {
                "capacity_max_first_n": max_capacity,
                "capacity_max_minus_cycle_2": max_capacity - cycle_2 if np.isfinite(max_capacity) and np.isfinite(cycle_2) else np.nan,
                "capacity_cycle_n_minus_cycle_10": cycle_n - cycle_10 if np.isfinite(cycle_n) and np.isfinite(cycle_10) else np.nan,
                "capacity_slope_cycle_2_to_n": early_slope,
                "capacity_intercept_cycle_2_to_n": early_intercept,
                "capacity_late_slope": late_slope,
                "capacity_late_intercept": late_intercept,
                "capacity_mean_first_n": float(np.nanmean(capacities)) if np.isfinite(capacities).any() else np.nan,
                "capacity_std_first_n": float(np.nanstd(capacities)) if np.isfinite(capacities).any() else np.nan,
                "capacity_min_first_n": float(np.nanmin(capacities)) if np.isfinite(capacities).any() else np.nan,
                "capacity_max_stat_first_n": max_capacity,
                "capacity_first": first_capacity,
                "capacity_last": last_capacity,
                "capacity_first_minus_last": first_capacity - last_capacity if np.isfinite(first_capacity) and np.isfinite(last_capacity) else np.nan,
                "capacity_fractional_fade": (first_capacity - last_capacity) / first_capacity
                if np.isfinite(first_capacity) and first_capacity != 0 and np.isfinite(last_capacity)
                else np.nan,
            }
        )
        rows.append(row)
    features = pd.DataFrame(rows).set_index("_key") if rows else pd.DataFrame()
    features.index.name = key
    return _drop_empty_numeric_columns(features)


def _array_from_value(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(float).ravel()
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=float).ravel()
    if isinstance(value, str):
        stripped = value.strip().strip("[]")
        if not stripped:
            return np.array([], dtype=float)
        sep = "," if "," in stripped else None
        try:
            return np.asarray([float(part) for part in stripped.split(sep)], dtype=float).ravel()
        except Exception:
            return np.array([], dtype=float)
    try:
        return np.asarray([float(value)], dtype=float)
    except Exception:
        return np.array([], dtype=float)


def _curve_difference_for_group(g: pd.DataFrame, max_cycle: int) -> tuple[np.ndarray, bool, float]:
    q_cols = [col for col in ("Qdlin", "qdlin", "q_dlin", "discharge_capacity_curve") if col in g.columns]
    v_cols = [col for col in ("Vdlin", "vdlin", "v_dlin", "voltage_curve") if col in g.columns]
    if q_cols:
        q_col = q_cols[0]
        by_cycle = g.groupby("cycle_index", dropna=False)[q_col].first()
        q10 = _array_from_value(by_cycle.get(10, []))
        qn = _array_from_value(by_cycle.get(max_cycle, []))
        if q10.size and qn.size and q10.size == qn.size:
            energy_diff = np.nan
            if v_cols:
                v_col = v_cols[0]
                v_by_cycle = g.groupby("cycle_index", dropna=False)[v_col].first()
                v10 = _array_from_value(v_by_cycle.get(10, []))
                vn = _array_from_value(v_by_cycle.get(max_cycle, []))
                if v10.size == q10.size and vn.size == qn.size:
                    energy_diff = float(np.trapz(q10, v10) - np.trapz(qn, vn))
            return qn - q10, False, energy_diff

    scalar = g.dropna(subset=["cycle_index"]).copy()
    scalar["cycle_index"] = pd.to_numeric(scalar["cycle_index"], errors="coerce")
    scalar["discharge_capacity"] = pd.to_numeric(scalar["discharge_capacity"], errors="coerce")
    by_cycle = scalar.groupby("cycle_index")["discharge_capacity"].mean()
    if by_cycle.empty:
        return np.array([], dtype=float), True, np.nan
    cycle_10 = float(by_cycle.get(10.0, np.nan))
    cycle_n = float(by_cycle.get(float(max_cycle), np.nan))
    if not np.isfinite(cycle_n):
        valid_le_n = by_cycle[by_cycle.index <= max_cycle].dropna()
        cycle_n = float(valid_le_n.iloc[-1]) if not valid_le_n.empty else float("nan")
    if not np.isfinite(cycle_10) or not np.isfinite(cycle_n):
        return np.array([], dtype=float), True, np.nan
    diff = cycle_n - cycle_10
    return np.asarray([diff], dtype=float), True, diff


def build_curve_difference_features(cycle_summary: pd.DataFrame, max_cycle: int = 100) -> pd.DataFrame:
    """Build cycle-N minus cycle-10 curve-difference statistics.

    If interpolated curves are absent, scalar capacity differences are used as
    approximate proxy features and feature names are prefixed with ``approx_``.
    """
    key = _key_column(cycle_summary)
    if "cycle_index" not in cycle_summary.columns:
        raise ValueError("cycle_summary must contain cycle_index")
    rows: list[dict[str, float | Any]] = []
    for cell_key, grp in cycle_summary.groupby(key, dropna=False):
        diff, approximate, energy_diff = _curve_difference_for_group(grp, max_cycle)
        diff = diff[np.isfinite(diff)]
        prefix = "approx_" if approximate else ""
        row: dict[str, float | Any] = {"_key": cell_key}
        if diff.size == 0:
            rows.append(row)
            continue
        variance = float(np.var(diff))
        skewness = float(skew(diff, bias=False)) if diff.size >= 3 else 0.0
        row.update(
            {
                f"{prefix}qdiff_n_10": float(np.mean(diff)),
                f"{prefix}log_abs_qdiff_min": _safe_log10(float(np.min(diff))),
                f"{prefix}log_abs_qdiff_mean": _safe_log10(float(np.mean(diff))),
                f"{prefix}log_abs_qdiff_var": _safe_log10(variance),
                f"{prefix}log_abs_qdiff_skew": _safe_log10(skewness),
                f"{prefix}log_sum_abs_qdiff": _safe_log10(float(np.sum(np.abs(diff)))),
                f"{prefix}log_sum_sq_qdiff": _safe_log10(float(np.sum(diff**2))),
                f"{prefix}log_abs_energy_diff": _safe_log10(energy_diff),
            }
        )
        rows.append(row)
    features = pd.DataFrame(rows).set_index("_key") if rows else pd.DataFrame()
    features.index.name = key
    return _drop_empty_numeric_columns(features)


def build_protocol_features(metadata: pd.DataFrame) -> pd.DataFrame:
    """Build protocol-current features from candidate-facing metadata."""
    key = _key_column(metadata)
    rows: list[dict[str, float | Any]] = []
    for cell_key, row in metadata.drop_duplicates(subset=[key]).set_index(key).iterrows():
        out: dict[str, float | Any] = {"_key": cell_key}
        currents: list[float] = []
        for canonical, aliases in {
            "C1": ("C1", "cc1"),
            "C2": ("C2", "cc2"),
            "C3": ("C3", "cc3"),
            "C4": ("C4", "cc4"),
        }.items():
            value = np.nan
            for alias in aliases:
                if alias in row.index:
                    value = pd.to_numeric(pd.Series([row[alias]]), errors="coerce").iloc[0]
                    break
            out[f"protocol_{canonical}"] = float(value) if np.isfinite(value) else np.nan
            if np.isfinite(value):
                currents.append(float(value))
        arr = np.asarray(currents, dtype=float)
        if arr.size:
            out.update(
                {
                    "protocol_current_mean": float(np.mean(arr)),
                    "protocol_current_max": float(np.max(arr)),
                    "protocol_current_variance": float(np.var(arr)),
                    "protocol_c1_c2": out.get("protocol_C1", np.nan) * out.get("protocol_C2", np.nan),
                    "protocol_c2_c3": out.get("protocol_C2", np.nan) * out.get("protocol_C3", np.nan),
                    "protocol_c3_c4": out.get("protocol_C3", np.nan) * out.get("protocol_C4", np.nan),
                }
            )
        rows.append(out)
    features = pd.DataFrame(rows).set_index("_key") if rows else pd.DataFrame()
    features.index.name = key
    return _drop_empty_numeric_columns(features)


def build_all_battery_features(
    metadata: pd.DataFrame,
    cycle_summary: pd.DataFrame,
    max_cycle: int = 100,
    include_protocol: bool = True,
    return_feature_metadata: bool = False,
):
    """Build numeric author-inspired early-cycle features.

    The output feature dataframe contains numeric feature columns only. row_id
    or cell_id may appear as the index for alignment, but never as columns.
    """
    capacity = build_capacity_summary_features(cycle_summary, max_cycle=max_cycle)
    curve = build_curve_difference_features(cycle_summary, max_cycle=max_cycle)
    parts = [capacity, curve]
    metadata_rows = [
        _feature_metadata(capacity, "capacity_summary"),
        _feature_metadata(curve, "curve_difference_approximate", approximate=any(col.startswith("approx_") for col in curve.columns)),
    ]
    if include_protocol:
        protocol = build_protocol_features(metadata)
        parts.append(protocol)
        metadata_rows.append(_feature_metadata(protocol, "protocol"))
    features = pd.concat(parts, axis=1, join="outer") if parts else pd.DataFrame()
    features = _drop_empty_numeric_columns(features)
    features = features.loc[:, ~features.columns.duplicated()]
    for forbidden in KEY_COLUMNS:
        if forbidden in features.columns:
            features = features.drop(columns=[forbidden])
    if return_feature_metadata:
        feature_metadata = pd.concat(metadata_rows, ignore_index=True) if metadata_rows else pd.DataFrame(columns=["feature_name", "family", "approximate"])
        feature_metadata = feature_metadata[feature_metadata["feature_name"].isin(features.columns)].drop_duplicates("feature_name")
        return features, feature_metadata
    return features
