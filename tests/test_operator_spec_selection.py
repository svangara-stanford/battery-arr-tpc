"""Offline tests for the operator-spec feature-selection path (Tier B).

These prove the FeatureScientist's structured operator choices deterministically
drive which feature columns the model trains on -- no LLM, no raw data. See
docs/structured_feature_specs_plan.md.
"""

from __future__ import annotations

import pandas as pd
import pytest

from battery_aar.features.operator_spec_selector import select_columns_for_specs
from battery_aar.workflows.candidate_compiler import (
    candidate_spec_from_plans,
    compile_candidate_spec_to_python,
)
from battery_aar.workflows.roles import _parse_feature_operators
from battery_aar.workflows.schemas import FeatureOperatorSpec, FeaturePlan, ModelPlan


def _metadata() -> pd.DataFrame:
    # Minimal stand-in for a materialized feature_metadata table.
    rows = []
    for i in range(10):
        rows.append({"feature_name": f"dc_delta_{i}", "operator_type": "cross_cycle_scalar_delta", "y_signal": "discharge_capacity", "aggregation": "delta"})
    for i in range(10):
        rows.append({"feature_name": f"dc_curve_{i}", "operator_type": "curve_shape", "y_signal": "discharge_capacity", "aggregation": "area"})
    for i in range(10):
        rows.append({"feature_name": f"ce_stat_{i}", "operator_type": "cycle_window_stats", "y_signal": "charge_energy", "aggregation": "mean"})
    return pd.DataFrame(rows)


def test_specs_select_a_subset_not_the_whole_pool():
    md = _metadata()
    specs = [{"operator_type": "cross_cycle_scalar_delta", "params": {"y_signal": "discharge_capacity"}}]
    sel = select_columns_for_specs(md, specs, budget=None)
    assert set(sel.columns) == {f"dc_delta_{i}" for i in range(10)}
    assert len(sel.columns) < len(md)  # strictly a subset


def test_budget_is_enforced_and_shared_across_specs():
    md = _metadata()
    specs = [
        {"operator_type": "cross_cycle_scalar_delta", "params": {"y_signal": "discharge_capacity"}},
        {"operator_type": "curve_shape", "params": {"y_signal": "discharge_capacity"}},
    ]
    sel = select_columns_for_specs(md, specs, budget=6)
    assert len(sel.columns) == 6
    # Round-robin => both operators represented, not 6 from the first.
    assert any(c.startswith("dc_delta_") for c in sel.columns)
    assert any(c.startswith("dc_curve_") for c in sel.columns)


def test_aggregation_param_narrows_selection():
    md = _metadata()
    specs = [{"operator_type": "cycle_window_stats", "params": {"y_signal": "charge_energy", "aggregation": "mean"}}]
    sel = select_columns_for_specs(md, specs, budget=None)
    assert set(sel.columns) == {f"ce_stat_{i}" for i in range(10)}
    # A non-existent aggregation selects nothing.
    empty = select_columns_for_specs(md, [{"operator_type": "cycle_window_stats", "params": {"aggregation": "nope"}}], budget=None)
    assert empty.columns == []


def test_selection_is_deterministic():
    md = _metadata()
    specs = [{"operator_type": "curve_shape", "params": {"y_signal": "discharge_capacity"}}]
    a = select_columns_for_specs(md, specs, budget=5)
    b = select_columns_for_specs(md, specs, budget=5)
    assert a.columns == b.columns


def test_invalid_operator_name_is_dropped_not_executed():
    specs, info = _parse_feature_operators(
        [
            {"operator_type": "curve_shape", "y_signal": "discharge_capacity"},
            {"operator_type": "definitely_not_an_operator", "y_signal": "x"},
        ],
        feature_budget=None,
    )
    assert [s.operator_type for s in specs] == ["curve_shape"]
    assert info["n_proposed"] == 2 and info["n_valid"] == 1
    assert "definitely_not_an_operator" in info["dropped_operator_names"]


def test_parse_enforces_budget_on_valid_specs():
    raw = [
        {"operator_type": "curve_shape", "y_signal": "discharge_capacity"},
        {"operator_type": "cross_cycle_scalar_delta", "y_signal": "discharge_capacity"},
        {"operator_type": "cycle_window_stats", "y_signal": "charge_energy"},
    ]
    specs, info = _parse_feature_operators(raw, feature_budget=2)
    assert len(specs) == 2
    assert info["budget_enforced"] is True
    assert info["n_valid"] == 3 and info["n_kept"] == 2


def test_empty_or_all_invalid_specs_yield_no_allowlist_fallback():
    # No operators on the plan => empty allowlist => compiler falls back to the
    # normal feature_set behavior (candidate still compiles and is not gutted).
    fp = FeaturePlan(
        run_id="t", agent_id="fs", human_readable_summary="x",
        feature_program_paths=[], feature_operators=[],
    )
    mp = ModelPlan(run_id="t", agent_id="ma", human_readable_summary="x",
                   model_family="Ridge", target_transform="log10")
    spec = candidate_spec_from_plans(
        run_id="t", candidate_id="c", agent_id="fs", iteration=0,
        candidate_path="/tmp/c.py", feature_plan=fp, model_plan=mp,
    )
    assert spec.feature_column_allowlist == []
    code = compile_candidate_spec_to_python(spec)
    assert "FEATURE_COLUMN_ALLOWLIST = []" in code


def test_features_used_reports_actual_fit_columns_not_builder_output():
    # Regression for the counter bug: the builder emits 630 columns, but an
    # allowlist narrows the model to a few. n_fit_features must reflect what the
    # model trained on (from model state), with the builder count kept separately.
    from battery_aar.agents.candidate_api import _features_used_summary

    builder_calls = [{"phase": "fit", "n_rows": 100,
                      "n_features": 630,
                      "feature_columns": [f"f{i}" for i in range(630)]}]
    summary = _features_used_summary(builder_calls, fit_columns=["f1", "f2", "f3"])
    assert summary["n_fit_features"] == 3
    assert summary["fit_feature_columns"] == ["f1", "f2", "f3"]
    assert summary["n_builder_features"] == 630
    assert summary["capture_method"] == "model_state_feature_columns"

    # Freeform candidates without a feature_columns state fall back to builder.
    fallback = _features_used_summary(builder_calls, fit_columns=None)
    assert fallback["n_fit_features"] == 630
    assert fallback["capture_method"] == "runtime_wrap_build_all_battery_features"
