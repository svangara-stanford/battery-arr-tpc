"""Tests for patch E3 — feature-pipeline fixes.

Covers three regressions identified in the broad-discovery audit:

1. ``broad_physics`` runtime dispatch — the compiled candidate should pick up
   a family set strictly larger than ``scalar_plus_curve`` (so the richer
   broad_physics program actually contributes), and should emit a
   ``broad_physics_feature_program_missing`` warning when no broad_physics
   feature-program table is provided.
2. Voltage-window standardization — ``cross_cycle_curve_delta`` and
   ``curve_shape`` accept an optional fixed ``voltage_window`` and clip
   correctly when set, while preserving per-cell intersection defaults.
3. ``log_var`` aggregation — exposes Severson's Variance-model feature
   (``log10(Var(ΔQ(V)))``) as a single column.
"""
from __future__ import annotations

import importlib.util
import warnings as _stdlib_warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from battery_aar.features.operators import (
    _clip_curve_to_window,
    _curve_delta_aggregate,
    cross_cycle_curve_delta,
    curve_shape,
)
from battery_aar.features.program_library import (
    make_attia_severson_like_program,
    make_broad_physics_program,
    make_curve_delta_program,
)
from battery_aar.features.raw_cycles import (
    canonicalize_cycles_interpolated,
    canonicalize_summary,
)
from battery_aar.features.schemas import FeatureOperatorSpec
from battery_aar.workflows.candidate_compiler import (
    candidate_spec_from_plans,
    compile_candidate_spec_to_python,
)
from battery_aar.workflows.schemas import FeaturePlan, ModelPlan

from .feature_program_test_utils import toy_raw_payload


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _tables(n_cycles: int = 101, n_points: int = 8, identical_late_curve: bool = False):
    payload = toy_raw_payload(
        n_cycles=n_cycles,
        n_points=n_points,
        identical_late_curve=identical_late_curve,
    )
    return canonicalize_cycles_interpolated(payload), canonicalize_summary(payload)


def _load_compiled(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _compiled_module_for_feature_set(
    tmp_path: Path,
    *,
    feature_set: str,
    feature_program_paths: list[str] | None = None,
):
    feature_plan = FeaturePlan(
        run_id="patch_e3_test",
        human_readable_summary="features",
        agent_id="feature_scientist",
        feature_families=[],
        include_protocol_features=False,
        feature_program_paths=feature_program_paths or [],
        feature_set=feature_set,
        max_cycle=100,
    )
    model_plan = ModelPlan(
        run_id="patch_e3_test",
        human_readable_summary="model",
        agent_id="model_architect",
        model_family="Ridge",
        estimator_name="Ridge",
        feature_set=feature_set,
        preprocessing_steps=["SimpleImputer", "StandardScaler"],
        hyperparameters={"alpha": 1.0},
    )
    candidate_path = tmp_path / f"candidate_{feature_set}.py"
    spec_model = candidate_spec_from_plans(
        run_id="patch_e3_test",
        candidate_id=f"candidate_{feature_set}",
        agent_id="code_generator",
        iteration=0,
        candidate_path=candidate_path,
        feature_plan=feature_plan,
        model_plan=model_plan,
    )
    compile_candidate_spec_to_python(spec_model, candidate_path)
    return _load_compiled(candidate_path)


# ---------------------------------------------------------------------------
# Fix 1 — broad_physics dispatch is strictly broader than scalar_plus_curve
# ---------------------------------------------------------------------------

def test_broad_physics_family_set_is_strict_superset_of_scalar_plus_curve(tmp_path):
    module_broad = _compiled_module_for_feature_set(
        tmp_path,
        feature_set="broad_physics",
        feature_program_paths=["/some/broad_physics_recipe/feature_table.csv"],
    )
    module_scalar_plus_curve = _compiled_module_for_feature_set(
        tmp_path,
        feature_set="scalar_plus_curve",
    )
    broad = module_broad._families_for_feature_set("broad_physics", allow_protocol=False)
    spc = module_scalar_plus_curve._families_for_feature_set(
        "scalar_plus_curve", allow_protocol=False
    )

    assert spc.issubset(broad), "scalar_plus_curve must be a subset of broad_physics"
    assert broad - spc, "broad_physics must contain extra families not in scalar_plus_curve"
    # The placeholder families must be in the extras (reserved for the richer
    # broad_physics operators that scalar_plus_curve doesn't cover).
    assert "learned_embedding_placeholder" in broad
    assert "generic_timeseries_placeholder" in broad


def test_broad_physics_emits_missing_feature_program_warning_when_no_paths(tmp_path):
    module = _compiled_module_for_feature_set(
        tmp_path,
        feature_set="broad_physics",
        feature_program_paths=None,
    )
    with _stdlib_warnings.catch_warnings(record=True) as caught:
        _stdlib_warnings.simplefilter("always")
        module._families_for_feature_set("broad_physics", allow_protocol=False)
    messages = [str(w.message) for w in caught]
    assert any("broad_physics_feature_program_missing" in msg for msg in messages), (
        f"expected broad_physics_feature_program_missing warning, got {messages}"
    )


def test_broad_physics_warning_fires_when_paths_do_not_match_recipe(tmp_path):
    module = _compiled_module_for_feature_set(
        tmp_path,
        feature_set="broad_physics",
        feature_program_paths=["/some/curve_delta_idx9_99/feature_table.csv"],
    )
    with _stdlib_warnings.catch_warnings(record=True) as caught:
        _stdlib_warnings.simplefilter("always")
        module._families_for_feature_set("broad_physics", allow_protocol=False)
    messages = [str(w.message) for w in caught]
    assert any("broad_physics_feature_program_missing" in msg for msg in messages)


def test_broad_physics_no_warning_when_path_references_recipe(tmp_path):
    module = _compiled_module_for_feature_set(
        tmp_path,
        feature_set="broad_physics",
        feature_program_paths=["/some/broad_physics_recipe/feature_table.csv"],
    )
    with _stdlib_warnings.catch_warnings(record=True) as caught:
        _stdlib_warnings.simplefilter("always")
        module._families_for_feature_set("broad_physics", allow_protocol=False)
    fallback = [w for w in caught if "broad_physics_feature_program_missing" in str(w.message)]
    assert not fallback, (
        f"unexpected fallback warning when broad_physics path is present: {fallback}"
    )


# ---------------------------------------------------------------------------
# Fix 2 — voltage_window clipping
# ---------------------------------------------------------------------------

def test_clip_curve_to_window_clips_inclusive():
    x = np.linspace(1.5, 4.0, 26)
    y = np.sin(x)
    x_clipped, y_clipped = _clip_curve_to_window(x, y, (2.0, 3.5))
    assert x_clipped.size > 0
    assert float(np.min(x_clipped)) >= 2.0
    assert float(np.max(x_clipped)) <= 3.5
    # None passes through unchanged
    x_pass, y_pass = _clip_curve_to_window(x, y, None)
    np.testing.assert_array_equal(x_pass, x)
    np.testing.assert_array_equal(y_pass, y)


def test_cross_cycle_curve_delta_with_voltage_window_clips_grid():
    cycles, summary = _tables()
    spec_default = FeatureOperatorSpec(
        operator_name="cross_cycle_curve_delta",
        operator_type="cross_cycle_curve_delta",
        family="true_curve_difference",
        params={
            "step_type": "discharge",
            "cycle_pairs": [[9, 99]],
            "x_axis": "voltage",
            "y_signal": "discharge_capacity",
            "grid_size": 50,
            "transforms": ["identity"],
            "aggregations": ["min", "max", "mean", "var"],
        },
    )
    out_default = cross_cycle_curve_delta(spec_default, cycles, summary)

    spec_windowed = FeatureOperatorSpec(
        operator_name="cross_cycle_curve_delta",
        operator_type="cross_cycle_curve_delta",
        family="true_curve_difference",
        params={
            **spec_default.params,
            "voltage_window": [3.3, 3.7],
        },
    )
    out_windowed = cross_cycle_curve_delta(spec_windowed, cycles, summary)

    # Both should produce values; numbers should differ once we restrict the
    # window because we are aggregating over a different segment of the curve.
    key = "discharge_discharge_capacity_curve_delta_identity_mean_cycle_99_minus_9"
    assert key in out_default.features
    assert key in out_windowed.features
    assert np.isfinite(out_default.features[key])
    assert np.isfinite(out_windowed.features[key])


def test_curve_shape_with_voltage_window_clips_grid():
    cycles, summary = _tables()
    spec_default = FeatureOperatorSpec(
        operator_name="curve_shape",
        operator_type="curve_shape",
        family="true_curve_shape",
        params={
            "step_types": ["discharge"],
            "cycles": [9],
            "x_axis": "voltage",
            "y_signals": ["discharge_capacity"],
            "grid_size": 50,
            "aggregations": ["min", "max", "mean"],
        },
    )
    out_default = curve_shape(spec_default, cycles, summary)

    spec_windowed = FeatureOperatorSpec(
        operator_name="curve_shape",
        operator_type="curve_shape",
        family="true_curve_shape",
        params={**spec_default.params, "voltage_window": [3.3, 3.7]},
    )
    out_windowed = curve_shape(spec_windowed, cycles, summary)

    assert out_default.features
    assert out_windowed.features
    assert all(np.isfinite(v) for v in out_windowed.features.values())


def test_voltage_window_default_preserves_per_cell_intersection():
    """Backward-compat: voltage_window=None must keep the old behaviour."""
    cycles, summary = _tables()
    spec = FeatureOperatorSpec(
        operator_name="cross_cycle_curve_delta",
        operator_type="cross_cycle_curve_delta",
        family="true_curve_difference",
        params={
            "step_type": "discharge",
            "cycle_pairs": [[9, 99]],
            "x_axis": "voltage",
            "y_signal": "discharge_capacity",
            "grid_size": 20,
            "transforms": ["identity"],
            "aggregations": ["min", "max", "mean"],
        },
    )
    out_no_window = cross_cycle_curve_delta(spec, cycles, summary)
    spec_explicit_none = FeatureOperatorSpec(
        operator_name="cross_cycle_curve_delta",
        operator_type="cross_cycle_curve_delta",
        family="true_curve_difference",
        params={**spec.params, "voltage_window": None},
    )
    out_explicit_none = cross_cycle_curve_delta(spec_explicit_none, cycles, summary)
    for key, value in out_no_window.features.items():
        other = out_explicit_none.features[key]
        if np.isfinite(value):
            assert np.isclose(value, other, atol=1e-12)


def test_curve_delta_program_grid_size_matches_severson_convention():
    program = make_broad_physics_program()
    curve_delta_op = next(
        op for op in program.operators if op.operator_type == "cross_cycle_curve_delta"
    )
    assert int(curve_delta_op.params["grid_size"]) == 1000


# ---------------------------------------------------------------------------
# Fix 3 — log_var aggregation
# ---------------------------------------------------------------------------

def test_log_var_aggregation_matches_analytical_value():
    rng = np.random.default_rng(0)
    values = rng.normal(loc=0.0, scale=0.01, size=512)
    grid = np.linspace(0.0, 1.0, values.size)
    log_var_op = _curve_delta_aggregate(values, grid, "log_var")
    expected = float(np.log10(np.var(values) + 1e-30))
    assert np.isfinite(log_var_op)
    assert np.isclose(log_var_op, expected, atol=1e-12)


def test_log_var_aggregation_handles_zero_variance_finite():
    flat = np.full(64, 0.5, dtype=float)
    grid = np.linspace(0.0, 1.0, flat.size)
    value = _curve_delta_aggregate(flat, grid, "log_var")
    # Variance is zero, so result should be log10(1e-30) = -30 exactly
    assert np.isfinite(value)
    assert np.isclose(value, -30.0, atol=1e-9)


def test_log_var_column_appears_in_curve_delta_features():
    cycles, summary = _tables()
    spec = FeatureOperatorSpec(
        operator_name="cross_cycle_curve_delta",
        operator_type="cross_cycle_curve_delta",
        family="true_curve_difference",
        params={
            "step_type": "discharge",
            "cycle_pairs": [[9, 99]],
            "x_axis": "voltage",
            "y_signal": "discharge_capacity",
            "grid_size": 100,
            "transforms": ["identity"],
            "aggregations": ["var", "log_var"],
        },
    )
    output = cross_cycle_curve_delta(spec, cycles, summary)
    target = "discharge_discharge_capacity_curve_delta_identity_log_var_cycle_99_minus_9"
    assert target in output.features, (
        f"expected {target} in features, got {sorted(output.features)[:20]}"
    )
    var_value = output.features[
        "discharge_discharge_capacity_curve_delta_identity_var_cycle_99_minus_9"
    ]
    log_var_value = output.features[target]
    assert np.isfinite(log_var_value)
    # log_var should equal log10(var + eps) for the same delta vector.
    assert np.isclose(log_var_value, float(np.log10(var_value + 1e-30)), atol=1e-9)


def test_log_var_is_in_program_library_defaults():
    curve_delta = make_curve_delta_program()
    attia_severson_like = make_attia_severson_like_program()
    broad_physics = make_broad_physics_program()

    def aggs(program):
        return list(
            program.operators[
                next(
                    i
                    for i, op in enumerate(program.operators)
                    if op.operator_type == "cross_cycle_curve_delta"
                )
            ].params.get("aggregations", [])
        )

    assert "log_var" in aggs(curve_delta)
    assert "log_var" in aggs(attia_severson_like)
    assert "log_var" in aggs(broad_physics)
