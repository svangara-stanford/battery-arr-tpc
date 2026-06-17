from __future__ import annotations

from typing import Callable, Dict

from battery_aar.features.schemas import FeatureOperatorSpec, FeatureProgram


def _program(program_id: str, name: str, description: str, operators: list[FeatureOperatorSpec], first_n_cycles: int, include_protocol: bool = False) -> FeatureProgram:
    return FeatureProgram(
        run_id="feature_program_library",
        human_readable_summary=description,
        program_id=program_id,
        name=name,
        description=description,
        operators=operators,
        include_protocol_features=include_protocol,
        cycle_index_convention="raw_zero_based",
        first_n_cycles=first_n_cycles,
        proposed_by="deterministic_program_library",
        rationale="Trusted deterministic feature-program recipe.",
    )


def make_scalar_baseline_program(first_n_cycles: int = 100) -> FeatureProgram:
    return _program(
        "scalar_baseline",
        "Scalar Baseline",
        "Capacity, resistance, temperature, and energy summary features over early-cycle windows.",
        [
            FeatureOperatorSpec(
                operator_name="cycle_scalar",
                operator_type="cycle_scalar",
                family="capacity_summary",
                params={
                    "signals": ["discharge_capacity", "charge_capacity", "internal_resistance", "temperature", "discharge_energy", "charge_energy"],
                    "cycle_indices": [0, 1, 2, 4, 9, 10, 49, 99],
                    "aggregations": ["last", "mean"],
                },
            ),
            FeatureOperatorSpec(
                operator_name="cycle_window_stats",
                operator_type="cycle_window_stats",
                family="capacity_summary",
                params={
                    "signals": ["discharge_capacity", "charge_capacity", "internal_resistance", "temperature", "discharge_energy", "charge_energy"],
                    "windows": [[0, 4], [0, 9], [0, 49], [0, min(99, first_n_cycles - 1)], [10, min(99, first_n_cycles - 1)]],
                    "stats": ["mean", "std", "min", "max", "delta", "slope", "intercept", "r2", "curvature"],
                },
            ),
            FeatureOperatorSpec(
                operator_name="cross_cycle_scalar_delta",
                operator_type="cross_cycle_scalar_delta",
                family="scalar_delta",
                params={
                    "signals": ["discharge_capacity", "charge_capacity", "internal_resistance", "temperature", "discharge_energy", "charge_energy"],
                    "cycle_pairs": [[0, 9], [1, min(99, first_n_cycles - 1)], [9, min(99, first_n_cycles - 1)]],
                    "operations": ["difference", "ratio", "relative_difference", "log_abs_difference"],
                },
            ),
        ],
        first_n_cycles,
    )


def make_curve_delta_program(
    cycle_early: int = 9,
    cycle_late: int = 99,
    voltage_window: tuple[float, float] | None = None,
) -> FeatureProgram:
    """Curve-delta recipe.

    Parameters
    ----------
    cycle_early, cycle_late
        Cycle indices to compare.
    voltage_window
        Optional ``(V_low, V_high)`` clip applied to both curves before
        interpolation. When ``None`` (default) the per-cell intersection of
        voltage ranges is used (preserves existing behaviour). For Severson-style
        LFP benchmarks the convention is ``voltage_window=(2.0, 3.5)``.
    """
    params: dict = {
        "step_type": "discharge",
        "cycle_pairs": [[cycle_early, cycle_late]],
        "x_axis": "voltage",
        "y_signal": "discharge_capacity",
        "grid_size": 1000,
        "delta_direction": "later_minus_earlier",
        "transforms": ["identity", "abs", "log_abs"],
        "aggregations": ["min", "max", "mean", "std", "var", "log_var", "skew", "sum_abs", "sum_sq", "area_abs", "area_signed"],
    }
    if voltage_window is not None:
        params["voltage_window"] = [float(voltage_window[0]), float(voltage_window[1])]
    return _program(
        f"curve_delta_idx{cycle_early}_{cycle_late}",
        "Curve Delta",
        "True raw discharge curve-difference features between two raw cycle indices.",
        [
            FeatureOperatorSpec(
                operator_name="cross_cycle_curve_delta",
                operator_type="cross_cycle_curve_delta",
                family="true_curve_difference",
                params=params,
            )
        ],
        max(cycle_late + 1, 100),
    )


def make_attia_severson_like_program(cycle_early: int = 9, cycle_late: int = 99) -> FeatureProgram:
    return _program(
        f"attia_severson_like_idx{cycle_early}_{cycle_late}",
        "Attia/Severson-like",
        "Literature-inspired capacity and discharge curve-delta terms using generic trusted operators.",
        [
            FeatureOperatorSpec(
                operator_name="cycle_scalar",
                operator_type="cycle_scalar",
                family="capacity_summary",
                params={"signals": ["discharge_capacity"], "cycle_indices": [1, cycle_early, cycle_late], "aggregations": ["last"]},
            ),
            FeatureOperatorSpec(
                operator_name="cycle_window_stats",
                operator_type="cycle_window_stats",
                family="capacity_summary",
                params={
                    "signals": ["discharge_capacity"],
                    "windows": [[0, cycle_late], [max(0, cycle_late - 8), cycle_late]],
                    "stats": ["max", "delta", "slope", "intercept"],
                },
            ),
            FeatureOperatorSpec(
                operator_name="cross_cycle_curve_delta",
                operator_type="cross_cycle_curve_delta",
                family="true_curve_difference",
                params={
                    "step_type": "discharge",
                    "cycle_pairs": [[cycle_early, cycle_late]],
                    "x_axis": "voltage",
                    "y_signal": "discharge_capacity",
                    "grid_size": 1000,
                    "delta_direction": "later_minus_earlier",
                    "transforms": ["identity", "log_abs"],
                    "aggregations": ["min", "mean", "var", "log_var", "skew", "sum_abs", "sum_sq"],
                },
            ),
        ],
        max(cycle_late + 1, 100),
    )


def make_broad_physics_program(
    first_n_cycles: int = 100,
    voltage_window: tuple[float, float] | None = None,
) -> FeatureProgram:
    """Broad physics recipe.

    Parameters
    ----------
    first_n_cycles
        How many early cycles to use.
    voltage_window
        Optional ``(V_low, V_high)`` clip applied to both curves of every
        cross-cycle-curve-delta operator before interpolation. When ``None``
        (default) the per-cell intersection of voltage ranges is used
        (preserves existing behaviour). For Severson-style LFP benchmarks the
        convention is ``voltage_window=(2.0, 3.5)``.
    """
    late = max(1, first_n_cycles - 1)
    curve_shape_params: dict = {
        "step_types": ["discharge", "charge"],
        "cycles": [0, 1, 9, min(49, late), late],
        "x_axis": "voltage",
        "y_signals": ["discharge_capacity", "charge_capacity", "current", "temperature"],
        "grid_size": 1000,
        "aggregations": ["min", "max", "mean", "std", "var", "area", "slope_mean"],
    }
    curve_delta_params: dict = {
        "step_type": "discharge",
        "cycle_pairs": [[9, late]],
        "x_axis": "voltage",
        "y_signal": "discharge_capacity",
        "grid_size": 1000,
        "transforms": ["identity", "abs", "log_abs"],
        "aggregations": ["min", "max", "mean", "std", "var", "log_var", "skew", "sum_abs", "sum_sq", "area_abs", "area_signed"],
    }
    if voltage_window is not None:
        window_list = [float(voltage_window[0]), float(voltage_window[1])]
        curve_shape_params["voltage_window"] = window_list
        curve_delta_params["voltage_window"] = window_list
    return _program(
        "broad_physics",
        "Broad Physics",
        "Broad capacity, resistance, thermal, energy, charge/discharge curve shape, and curve-delta feature program.",
        [
            *make_scalar_baseline_program(first_n_cycles).operators,
            FeatureOperatorSpec(
                operator_name="curve_shape",
                operator_type="curve_shape",
                family="true_curve_shape",
                params=curve_shape_params,
            ),
            FeatureOperatorSpec(
                operator_name="cross_cycle_curve_delta",
                operator_type="cross_cycle_curve_delta",
                family="true_curve_difference",
                params=curve_delta_params,
            ),
            FeatureOperatorSpec(
                operator_name="learned_embedding_placeholder",
                operator_type="learned_embedding_placeholder",
                family="learned_embedding_placeholder",
                enabled=False,
            ),
            FeatureOperatorSpec(
                operator_name="generic_timeseries_placeholder",
                operator_type="generic_timeseries_placeholder",
                family="generic_timeseries_placeholder",
                enabled=False,
            ),
        ],
        first_n_cycles,
    )


def make_minimal_debug_program() -> FeatureProgram:
    return _program(
        "minimal_debug",
        "Minimal Debug",
        "Small deterministic feature program for tests and smoke runs.",
        [
            FeatureOperatorSpec(
                operator_name="cycle_scalar",
                operator_type="cycle_scalar",
                family="capacity_summary",
                params={"signals": ["discharge_capacity"], "cycle_indices": [0, 1, 9], "aggregations": ["last"]},
            ),
            FeatureOperatorSpec(
                operator_name="cross_cycle_scalar_delta",
                operator_type="cross_cycle_scalar_delta",
                family="scalar_delta",
                params={"signals": ["discharge_capacity"], "cycle_pairs": [[0, 9]], "operations": ["difference", "log_abs_difference"]},
            ),
        ],
        10,
    )


PROGRAM_RECIPES: Dict[str, Callable[..., FeatureProgram]] = {
    "scalar_baseline": make_scalar_baseline_program,
    "curve_delta": make_curve_delta_program,
    "attia_severson_like": make_attia_severson_like_program,
    "broad_physics": make_broad_physics_program,
    "minimal_debug": make_minimal_debug_program,
}


def available_program_recipes() -> list[str]:
    return sorted(PROGRAM_RECIPES)
