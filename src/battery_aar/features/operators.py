from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import skew

from battery_aar.features.operator_registry import FeatureOperatorOutput
from battery_aar.workflows.schemas import FeatureOperatorSpec

SIGNAL_ALIASES = {
    "internal_resistance": ["dc_internal_resistance", "internal_resistance", "IR", "resistance"],
    "temperature": ["temperature", "temperature_average", "temperature_mean", "temp", "T"],
}


def _safe_name(value: Any) -> str:
    return str(value).replace("-", "m").replace(".", "p").replace(" ", "_")


def _signal_family(signal: str, default: str) -> str:
    lowered = signal.lower()
    if "capacity" in lowered:
        return "capacity_summary"
    if "resistance" in lowered:
        return "resistance_summary"
    if "temperature" in lowered or "temp" in lowered:
        return "thermal_summary"
    if "energy" in lowered:
        return "energy_summary"
    return default


def _finite(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _safe_log10(value: float, eps: float) -> float:
    value = float(value)
    if not np.isfinite(value):
        return float("nan")
    return float(np.log10(abs(value) + eps))


def _safe_skew(values: Any, *, atol: float = 1e-12, rtol: float = 1e-9) -> float:
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size < 3:
        return 0.0

    max_abs = float(np.nanmax(np.abs(arr))) if arr.size else 0.0
    scale = max(1.0, max_abs)
    spread = float(np.nanmax(arr) - np.nanmin(arr))
    std = float(np.nanstd(arr))
    tol = float(atol + rtol * scale)
    if spread <= tol or std <= tol:
        return 0.0

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Precision loss occurred in moment calculation.*",
            category=RuntimeWarning,
        )
        value = skew(arr, bias=False)
    return _finite(value)


def _metadata(
    *,
    feature_name: str,
    feature_family: str,
    feature_source: str,
    spec: FeatureOperatorSpec,
    is_true_curve_feature: bool = False,
    is_proxy_feature: bool = False,
    uses_raw_cycles_interpolated: bool = False,
    uses_summary: bool = False,
    uses_protocol: bool = False,
    cycle_indices: list[int] | None = None,
    step_type: str | None = None,
    x_axis: str | None = None,
    y_signal: str | None = None,
    requested_signal: str | None = None,
    resolved_signal: str | None = None,
    aggregation: str | None = None,
    transform: str | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    return {
        "feature_name": feature_name,
        "feature_family": feature_family,
        "family": feature_family,
        "feature_source": feature_source,
        "operator_name": spec.operator_name,
        "operator_type": spec.operator_type,
        "is_true_curve_feature": bool(is_true_curve_feature),
        "is_proxy_feature": bool(is_proxy_feature),
        "uses_raw_cycles_interpolated": bool(uses_raw_cycles_interpolated),
        "uses_summary": bool(uses_summary),
        "uses_protocol": bool(uses_protocol),
        "cycle_indices": ",".join(map(str, cycle_indices or [])),
        "step_type": step_type or "",
        "x_axis": x_axis or "",
        "y_signal": y_signal or "",
        "requested_signal": requested_signal or y_signal or "",
        "resolved_signal": resolved_signal or y_signal or "",
        "aggregation": aggregation or "",
        "transform": transform or "",
        "description": description or spec.description or "",
    }


def _prefixed(spec: FeatureOperatorSpec, name: str) -> str:
    return f"{spec.feature_prefix}_{name}" if spec.feature_prefix else name


def _has_finite_values(series: pd.Series) -> bool:
    return bool(np.isfinite(pd.to_numeric(series, errors="coerce").to_numpy(float)).any())


def _resolve_signal_column(df: pd.DataFrame, requested_signal: str) -> str | None:
    if df.empty:
        return None
    candidates = SIGNAL_ALIASES.get(requested_signal, [requested_signal])
    candidates = list(dict.fromkeys([requested_signal, *candidates]))
    if requested_signal == "internal_resistance":
        candidates = list(dict.fromkeys(["dc_internal_resistance", *candidates]))
    existing = [candidate for candidate in candidates if candidate in df.columns]
    if not existing:
        return None
    for candidate in existing:
        if _has_finite_values(df[candidate]):
            return candidate
    return existing[0]


def _resolved_signal_name(df: pd.DataFrame, requested_signal: str, resolved_column: str | None) -> str:
    if resolved_column is None:
        return requested_signal
    resolved = dict(df.attrs.get("resolved_signals", {}))
    return str(resolved.get(resolved_column, resolved_column))


def _summary_series(summary_df: pd.DataFrame, signal: str) -> tuple[pd.Series, str]:
    resolved_signal = _resolve_signal_column(summary_df, signal)
    if summary_df.empty or "cycle_index" not in summary_df or resolved_signal is None:
        return pd.Series(dtype=float), signal
    df = summary_df[["cycle_index", resolved_signal]].copy()
    df["cycle_index"] = pd.to_numeric(df["cycle_index"], errors="coerce")
    df[resolved_signal] = pd.to_numeric(df[resolved_signal], errors="coerce")
    return (
        df.dropna(subset=["cycle_index"]).groupby("cycle_index")[resolved_signal].mean(),
        _resolved_signal_name(summary_df, signal, resolved_signal),
    )


def _raw_cycle_values(cycles_df: pd.DataFrame, *, cycle: int, signal: str, step_type: str | None) -> tuple[pd.Series, str]:
    resolved_signal = _resolve_signal_column(cycles_df, signal)
    if cycles_df.empty or "cycle_index" not in cycles_df or resolved_signal is None:
        return pd.Series(dtype=float), signal
    df = cycles_df[pd.to_numeric(cycles_df["cycle_index"], errors="coerce") == float(cycle)].copy()
    if step_type and "step_type" in df:
        df = df[df["step_type"].astype(str).str.lower() == step_type.lower()]
    return pd.to_numeric(df[resolved_signal], errors="coerce").dropna(), _resolved_signal_name(cycles_df, signal, resolved_signal)


def _raw_cycle_mean_series(cycles_df: pd.DataFrame, signal: str) -> tuple[pd.Series, str]:
    resolved_signal = _resolve_signal_column(cycles_df, signal)
    if cycles_df.empty or "cycle_index" not in cycles_df or resolved_signal is None:
        return pd.Series(dtype=float), signal
    raw = cycles_df[["cycle_index", resolved_signal]].copy()
    raw["cycle_index"] = pd.to_numeric(raw["cycle_index"], errors="coerce")
    raw[resolved_signal] = pd.to_numeric(raw[resolved_signal], errors="coerce")
    series = raw.dropna(subset=["cycle_index"]).groupby("cycle_index")[resolved_signal].mean()
    return series, _resolved_signal_name(cycles_df, signal, resolved_signal)


def _raw_cycle_aggregate(
    cycles_df: pd.DataFrame,
    *,
    cycle: int,
    signal: str,
    step_type: str | None,
    aggregation: str,
) -> tuple[float, str, bool]:
    raw_values, raw_signal = _raw_cycle_values(cycles_df, cycle=cycle, signal=signal, step_type=step_type)
    if raw_values.empty:
        return np.nan, raw_signal, False
    return _aggregate(raw_values, aggregation), raw_signal, True


def _aggregate(values: pd.Series | np.ndarray, aggregation: str) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    if arr.size == 0:
        return float("nan")
    if aggregation == "last":
        return _finite(arr[-1])
    if aggregation == "max":
        return _finite(np.max(arr))
    if aggregation == "min":
        return _finite(np.min(arr))
    if aggregation == "mean":
        return _finite(np.mean(arr))
    if aggregation == "std":
        return _finite(np.std(arr, ddof=0))
    if aggregation == "var":
        return _finite(np.var(arr))
    if aggregation == "skew":
        return _safe_skew(arr)
    return _finite(arr[-1])


def _linear_stats(cycles: np.ndarray, values: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(cycles) & np.isfinite(values)
    if mask.sum() < 2:
        return {"slope": np.nan, "intercept": np.nan, "r2": np.nan, "curvature": np.nan}
    x = cycles[mask]
    y = values[mask]
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    denom = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - float(np.sum((y - pred) ** 2) / denom) if denom > 0 else np.nan
    curvature = np.nan
    if mask.sum() >= 3:
        coeffs = np.polyfit(x, y, 2)
        curvature = float(coeffs[0])
    return {"slope": float(slope), "intercept": float(intercept), "r2": float(r2), "curvature": float(curvature)}


def cycle_scalar(spec: FeatureOperatorSpec, cycles_df: pd.DataFrame, summary_df: pd.DataFrame, metadata_row: pd.Series | None = None) -> FeatureOperatorOutput:
    params = spec.params
    signals = list(params.get("signals", ["discharge_capacity"]))
    cycle_indices = [int(cycle) for cycle in params.get("cycle_indices", [0, 1, 9, 99])]
    aggregations = list(params.get("aggregations", ["last"]))
    step_type = params.get("step_type")
    features: dict[str, float] = {}
    metadata: list[dict[str, Any]] = []
    warnings: list[str] = []
    for signal in signals:
        by_cycle, summary_signal = _summary_series(summary_df, signal)
        raw_cycle_means = pd.Series(dtype=float)
        raw_mean_signal = signal
        raw_has_finite: bool | None = None

        def _load_raw_cycle_means() -> bool:
            nonlocal raw_cycle_means, raw_mean_signal, raw_has_finite
            if raw_has_finite is None:
                raw_cycle_means, raw_mean_signal = _raw_cycle_mean_series(cycles_df, signal)
                raw_has_finite = _has_finite_values(raw_cycle_means)
            return bool(raw_has_finite)

        def _raw_fallback(cycle: int, aggregation: str) -> tuple[float, str, bool]:
            if not _load_raw_cycle_means():
                return np.nan, raw_mean_signal, False
            if aggregation in {"last", "mean"} and cycle in raw_cycle_means.index:
                value = _finite(raw_cycle_means.get(cycle))
                if np.isfinite(value):
                    return value, raw_mean_signal, True
            return _raw_cycle_aggregate(
                cycles_df,
                cycle=cycle,
                signal=signal,
                step_type=step_type,
                aggregation=aggregation,
            )

        for cycle in cycle_indices:
            for aggregation in aggregations:
                value = np.nan
                source = "summary"
                uses_summary = True
                uses_raw = False
                resolved_signal = summary_signal
                if not by_cycle.empty and cycle in by_cycle.index and aggregation in {"last", "mean"}:
                    value = _finite(by_cycle.get(cycle))
                    if not np.isfinite(value):
                        raw_value, raw_signal, raw_available = _raw_fallback(cycle, aggregation)
                        if raw_available:
                            value = raw_value
                            source = "raw_cycles_interpolated"
                            uses_summary = False
                            uses_raw = True
                            resolved_signal = raw_signal
                else:
                    raw_value, raw_signal, raw_available = _raw_fallback(cycle, aggregation)
                    if raw_available:
                        value = raw_value
                        source = "raw_cycles_interpolated"
                        uses_summary = False
                        uses_raw = True
                        resolved_signal = raw_signal
                if not np.isfinite(value):
                    warnings.append(f"{signal} missing for cycle {cycle}")
                suffix = "" if aggregation == "last" else f"_{aggregation}"
                name = _prefixed(spec, f"{signal}{suffix}_cycle_{cycle}")
                features[name] = value
                metadata.append(
                    _metadata(
                        feature_name=name,
                        feature_family=_signal_family(signal, spec.family),
                        feature_source=source,
                        spec=spec,
                        uses_raw_cycles_interpolated=uses_raw,
                        uses_summary=uses_summary,
                        cycle_indices=[cycle],
                        step_type=step_type,
                        y_signal=signal,
                        requested_signal=signal,
                        resolved_signal=resolved_signal,
                        aggregation=aggregation,
                    )
                )
    return FeatureOperatorOutput(features=features, metadata=metadata, warnings=warnings)


def cycle_window_stats(spec: FeatureOperatorSpec, cycles_df: pd.DataFrame, summary_df: pd.DataFrame, metadata_row: pd.Series | None = None) -> FeatureOperatorOutput:
    params = spec.params
    signals = list(params.get("signals", ["discharge_capacity"]))
    windows = [tuple(map(int, window)) for window in params.get("windows", [[0, 9], [0, 99], [10, 99]])]
    stats = list(params.get("stats", ["mean", "std", "min", "max", "delta", "slope", "intercept"]))
    features: dict[str, float] = {}
    metadata: list[dict[str, Any]] = []
    warnings: list[str] = []
    for signal in signals:
        series, resolved_signal = _summary_series(summary_df, signal)
        feature_source = "summary"
        uses_summary = True
        uses_raw = False
        if not _has_finite_values(series):
            raw_series, raw_resolved_signal = _raw_cycle_mean_series(cycles_df, signal)
            if _has_finite_values(raw_series):
                series = raw_series
                resolved_signal = raw_resolved_signal
                feature_source = "raw_cycles_interpolated_cycle_mean"
                uses_summary = False
                uses_raw = True
        for start, end in windows:
            window = series[(series.index >= start) & (series.index <= end)].dropna()
            if window.empty:
                warnings.append(f"{signal} has no values in window {start}:{end}")
            arr = window.to_numpy(float)
            cycle_arr = window.index.to_numpy(float)
            linear = _linear_stats(cycle_arr, arr)
            stat_values = {
                "mean": _finite(np.mean(arr)) if arr.size else np.nan,
                "std": _finite(np.std(arr, ddof=0)) if arr.size else np.nan,
                "min": _finite(np.min(arr)) if arr.size else np.nan,
                "max": _finite(np.max(arr)) if arr.size else np.nan,
                "delta": _finite(arr[-1] - arr[0]) if arr.size >= 2 else np.nan,
                **linear,
            }
            for stat in stats:
                name = _prefixed(spec, f"{signal}_{stat}_cycle_{start}_to_{end}")
                features[name] = _finite(stat_values.get(stat, np.nan))
                metadata.append(
                    _metadata(
                        feature_name=name,
                        feature_family=_signal_family(signal, spec.family),
                        feature_source=feature_source,
                        spec=spec,
                        uses_summary=uses_summary,
                        uses_raw_cycles_interpolated=uses_raw,
                        cycle_indices=[start, end],
                        y_signal=signal,
                        requested_signal=signal,
                        resolved_signal=resolved_signal,
                        aggregation=stat,
                    )
                )
    return FeatureOperatorOutput(features=features, metadata=metadata, warnings=warnings)


def cross_cycle_scalar_delta(spec: FeatureOperatorSpec, cycles_df: pd.DataFrame, summary_df: pd.DataFrame, metadata_row: pd.Series | None = None) -> FeatureOperatorOutput:
    params = spec.params
    signals = list(params.get("signals", ["discharge_capacity"]))
    cycle_pairs = [tuple(map(int, pair)) for pair in params.get("cycle_pairs", [[0, 9], [9, 99]])]
    operations = list(params.get("operations", ["difference", "ratio", "relative_difference", "log_abs_difference"]))
    eps = float(params.get("eps", 1e-12))
    features: dict[str, float] = {}
    metadata: list[dict[str, Any]] = []
    warnings: list[str] = []
    for signal in signals:
        series, resolved_signal = _summary_series(summary_df, signal)
        feature_source = "summary"
        uses_summary = True
        uses_raw = False
        if not _has_finite_values(series):
            raw_series, raw_resolved_signal = _raw_cycle_mean_series(cycles_df, signal)
            if _has_finite_values(raw_series):
                series = raw_series
                resolved_signal = raw_resolved_signal
                feature_source = "raw_cycles_interpolated_cycle_mean"
                uses_summary = False
                uses_raw = True
        for early, late in cycle_pairs:
            early_value = _finite(series.get(early, np.nan))
            late_value = _finite(series.get(late, np.nan))
            if not np.isfinite(early_value) or not np.isfinite(late_value):
                warnings.append(f"{signal} missing for scalar delta pair {early},{late}")
            diff = late_value - early_value if np.isfinite(early_value) and np.isfinite(late_value) else np.nan
            op_values = {
                "difference": diff,
                "ratio": late_value / (early_value + eps) if np.isfinite(early_value) and np.isfinite(late_value) else np.nan,
                "relative_difference": diff / (abs(early_value) + eps) if np.isfinite(diff) else np.nan,
                "log_abs_difference": _safe_log10(diff, eps),
            }
            for operation in operations:
                name = _prefixed(spec, f"{signal}_{operation}_cycle_{late}_minus_{early}")
                features[name] = _finite(op_values.get(operation, np.nan))
                metadata.append(
                    _metadata(
                        feature_name=name,
                        feature_family=f"{_signal_family(signal, spec.family)}_delta",
                        feature_source=feature_source,
                        spec=spec,
                        uses_summary=uses_summary,
                        uses_raw_cycles_interpolated=uses_raw,
                        cycle_indices=[early, late],
                        y_signal=signal,
                        requested_signal=signal,
                        resolved_signal=resolved_signal,
                        aggregation=operation,
                        transform=operation,
                    )
                )
    return FeatureOperatorOutput(features=features, metadata=metadata, warnings=warnings)


def _curve_for(cycles_df: pd.DataFrame, *, cycle: int, step_type: str, x_axis: str, y_signal: str) -> tuple[np.ndarray, np.ndarray]:
    required = {"cycle_index", "step_type", x_axis, y_signal}
    if cycles_df.empty or not required.issubset(cycles_df.columns):
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    df = cycles_df[
        (pd.to_numeric(cycles_df["cycle_index"], errors="coerce") == float(cycle))
        & (cycles_df["step_type"].astype(str).str.lower() == step_type.lower())
    ][[x_axis, y_signal]].copy()
    df[x_axis] = pd.to_numeric(df[x_axis], errors="coerce")
    df[y_signal] = pd.to_numeric(df[y_signal], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if df.empty:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    grouped = df.groupby(x_axis, as_index=False)[y_signal].mean().sort_values(x_axis)
    return grouped[x_axis].to_numpy(float), grouped[y_signal].to_numpy(float)


def _interpolate_curve(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if x.size < 2 or y.size < 2 or grid.size == 0:
        return np.asarray([], dtype=float)
    return np.interp(grid, x, y)


def curve_shape(spec: FeatureOperatorSpec, cycles_df: pd.DataFrame, summary_df: pd.DataFrame, metadata_row: pd.Series | None = None) -> FeatureOperatorOutput:
    params = spec.params
    step_types = list(params.get("step_types", ["discharge"]))
    cycles = [int(cycle) for cycle in params.get("cycles", [9, 99])]
    x_axis = str(params.get("x_axis", "voltage"))
    y_signals = list(params.get("y_signals", ["discharge_capacity"]))
    grid_size = int(params.get("grid_size", 1000))
    aggregations = list(params.get("aggregations", ["min", "max", "mean", "std", "var", "skew", "area", "slope_mean"]))
    features: dict[str, float] = {}
    metadata: list[dict[str, Any]] = []
    warnings: list[str] = []
    for step_type in step_types:
        for cycle in cycles:
            for y_signal in y_signals:
                x, y = _curve_for(cycles_df, cycle=cycle, step_type=step_type, x_axis=x_axis, y_signal=y_signal)
                if x.size < 2:
                    warnings.append(f"curve missing for {step_type} cycle {cycle} {y_signal}")
                    values = np.asarray([], dtype=float)
                    grid = np.asarray([], dtype=float)
                else:
                    grid = np.linspace(float(np.min(x)), float(np.max(x)), max(2, grid_size))
                    values = _interpolate_curve(x, y, grid)
                gradient = np.gradient(values, grid) if values.size >= 2 and grid.size >= 2 else np.asarray([], dtype=float)
                agg_values = {
                    "min": _aggregate(values, "min"),
                    "max": _aggregate(values, "max"),
                    "mean": _aggregate(values, "mean"),
                    "std": _aggregate(values, "std"),
                    "var": _aggregate(values, "var"),
                    "skew": _aggregate(values, "skew"),
                    "area": _finite(np.trapz(values, grid)) if values.size >= 2 else np.nan,
                    "slope_mean": _aggregate(gradient, "mean"),
                }
                for aggregation in aggregations:
                    name = _prefixed(spec, f"{step_type}_{y_signal}_vs_{x_axis}_{aggregation}_cycle_{cycle}")
                    features[name] = _finite(agg_values.get(aggregation, np.nan))
                    metadata.append(
                        _metadata(
                            feature_name=name,
                            feature_family="true_curve_shape",
                            feature_source="raw_cycles_interpolated",
                            spec=spec,
                            is_true_curve_feature=True,
                            uses_raw_cycles_interpolated=True,
                            cycle_indices=[cycle],
                            step_type=step_type,
                            x_axis=x_axis,
                            y_signal=y_signal,
                            aggregation=aggregation,
                        )
                    )
    return FeatureOperatorOutput(features=features, metadata=metadata, warnings=warnings)


def _delta_values(delta: np.ndarray, grid: np.ndarray, transform: str, eps: float) -> np.ndarray:
    if transform == "identity":
        return delta
    if transform == "abs":
        return np.abs(delta)
    if transform == "log_abs":
        return np.log10(np.abs(delta) + eps)
    return delta


def _curve_delta_aggregate(values: np.ndarray, grid: np.ndarray, aggregation: str) -> float:
    values = np.asarray(values, dtype=float).ravel()
    grid = np.asarray(grid, dtype=float).ravel()
    mask = np.isfinite(values)
    if grid.size == values.size:
        mask = mask & np.isfinite(grid)
    values = values[mask]
    if grid.size == mask.size:
        grid = grid[mask]
    if values.size == 0:
        return float("nan")
    if aggregation == "sum_abs":
        return _finite(np.sum(np.abs(values)))
    if aggregation == "sum_sq":
        return _finite(np.sum(values**2))
    if aggregation == "area_abs":
        x_axis = grid if grid.size == values.size else np.arange(values.size, dtype=float)
        return _finite(np.trapz(np.abs(values), x_axis)) if values.size >= 2 else np.nan
    if aggregation == "area_signed":
        x_axis = grid if grid.size == values.size else np.arange(values.size, dtype=float)
        return _finite(np.trapz(values, x_axis)) if values.size >= 2 else np.nan
    return _aggregate(values, aggregation)


def cross_cycle_curve_delta(spec: FeatureOperatorSpec, cycles_df: pd.DataFrame, summary_df: pd.DataFrame, metadata_row: pd.Series | None = None) -> FeatureOperatorOutput:
    params = spec.params
    step_type = str(params.get("step_type", "discharge"))
    cycle_pairs = [tuple(map(int, pair)) for pair in params.get("cycle_pairs", [[9, 99]])]
    x_axis = str(params.get("x_axis", "voltage"))
    y_signal = str(params.get("y_signal", "discharge_capacity"))
    grid_size = int(params.get("grid_size", 1000))
    direction = str(params.get("delta_direction", "later_minus_earlier"))
    transforms = list(params.get("transforms", ["identity", "log_abs"]))
    aggregations = list(params.get("aggregations", ["min", "max", "mean", "std", "var", "skew", "sum_abs", "sum_sq", "area_abs", "area_signed"]))
    eps = float(params.get("eps", 1e-12))
    features: dict[str, float] = {}
    metadata: list[dict[str, Any]] = []
    warnings: list[str] = []
    for early, late in cycle_pairs:
        x_early, y_early = _curve_for(cycles_df, cycle=early, step_type=step_type, x_axis=x_axis, y_signal=y_signal)
        x_late, y_late = _curve_for(cycles_df, cycle=late, step_type=step_type, x_axis=x_axis, y_signal=y_signal)
        if x_early.size < 2 or x_late.size < 2:
            warnings.append(f"curve pair missing for {step_type} {y_signal} cycles {early},{late}")
            grid = np.asarray([], dtype=float)
            delta = np.asarray([], dtype=float)
        else:
            lo = max(float(np.min(x_early)), float(np.min(x_late)))
            hi = min(float(np.max(x_early)), float(np.max(x_late)))
            if hi <= lo:
                warnings.append(f"curve pair has no overlapping {x_axis} range for cycles {early},{late}")
                grid = np.asarray([], dtype=float)
                delta = np.asarray([], dtype=float)
            else:
                grid = np.linspace(lo, hi, max(2, grid_size))
                early_values = _interpolate_curve(x_early, y_early, grid)
                late_values = _interpolate_curve(x_late, y_late, grid)
                delta = late_values - early_values if direction == "later_minus_earlier" else early_values - late_values
        for transform in transforms:
            transformed = _delta_values(delta, grid, transform, eps)
            for aggregation in aggregations:
                name = _prefixed(spec, f"{step_type}_{y_signal}_curve_delta_{transform}_{aggregation}_cycle_{late}_minus_{early}")
                features[name] = _finite(_curve_delta_aggregate(transformed, grid, aggregation))
                metadata.append(
                    _metadata(
                        feature_name=name,
                        feature_family="true_curve_difference",
                        feature_source="raw_cycles_interpolated",
                        spec=spec,
                        is_true_curve_feature=True,
                        uses_raw_cycles_interpolated=True,
                        cycle_indices=[early, late],
                        step_type=step_type,
                        x_axis=x_axis,
                        y_signal=y_signal,
                        aggregation=aggregation,
                        transform=transform,
                    )
                )
    return FeatureOperatorOutput(features=features, metadata=metadata, warnings=warnings)


def protocol(spec: FeatureOperatorSpec, cycles_df: pd.DataFrame, summary_df: pd.DataFrame, metadata_row: pd.Series | None = None) -> FeatureOperatorOutput:
    if metadata_row is None:
        return FeatureOperatorOutput(status="warning", warnings=["protocol operator requires metadata_row"])
    params = spec.params
    columns = list(params.get("columns", ["C1", "C2", "C3", "C4", "cc1", "cc2", "cc3", "cc4"]))
    values: list[float] = []
    features: dict[str, float] = {}
    metadata: list[dict[str, Any]] = []
    for col in columns:
        if col not in metadata_row.index:
            continue
        value = _finite(metadata_row[col])
        name = _prefixed(spec, f"protocol_{col}")
        features[name] = value
        if np.isfinite(value):
            values.append(value)
        metadata.append(
            _metadata(
                feature_name=name,
                feature_family="protocol",
                feature_source="processed_metadata",
                spec=spec,
                uses_protocol=True,
                y_signal=col,
                aggregation="value",
            )
        )
    arr = np.asarray(values, dtype=float)
    for agg, value in {
        "mean": np.mean(arr) if arr.size else np.nan,
        "max": np.max(arr) if arr.size else np.nan,
        "variance": np.var(arr) if arr.size else np.nan,
    }.items():
        name = _prefixed(spec, f"protocol_current_{agg}")
        features[name] = _finite(value)
        metadata.append(
            _metadata(
                feature_name=name,
                feature_family="protocol",
                feature_source="processed_metadata",
                spec=spec,
                uses_protocol=True,
                aggregation=agg,
            )
        )
    return FeatureOperatorOutput(features=features, metadata=metadata)


def learned_embedding_placeholder(spec: FeatureOperatorSpec, cycles_df: pd.DataFrame, summary_df: pd.DataFrame, metadata_row: pd.Series | None = None) -> FeatureOperatorOutput:
    return FeatureOperatorOutput(
        status="warning",
        warnings=["learned_embedding_placeholder is declared but not implemented in this safe first-generation substrate."],
    )


def generic_timeseries_placeholder(spec: FeatureOperatorSpec, cycles_df: pd.DataFrame, summary_df: pd.DataFrame, metadata_row: pd.Series | None = None) -> FeatureOperatorOutput:
    return FeatureOperatorOutput(
        status="warning",
        warnings=["generic_timeseries_placeholder is reserved for future tsfresh-like backends, but is intentionally disabled in this first safe implementation."],
    )
