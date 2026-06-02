from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from battery_aar.features.operators import (
    _safe_skew,
    cross_cycle_curve_delta,
    cross_cycle_scalar_delta,
    curve_shape,
    cycle_scalar,
    cycle_window_stats,
    generic_timeseries_placeholder,
    learned_embedding_placeholder,
    protocol,
)
from battery_aar.features.raw_cycles import canonicalize_cycles_interpolated, canonicalize_summary
from battery_aar.workflows.schemas import FeatureOperatorSpec

from .feature_program_test_utils import toy_raw_payload


def _tables(identical_late_curve: bool = False):
    payload = toy_raw_payload(n_cycles=101, n_points=8, identical_late_curve=identical_late_curve)
    return canonicalize_cycles_interpolated(payload), canonicalize_summary(payload)


def _dc_resistance_tables():
    payload = toy_raw_payload(n_cycles=12, n_points=4)
    payload["summary"]["dc_internal_resistance"] = [0.02 + 0.001 * cycle for cycle in payload["summary"]["cycle_index"]]
    payload["summary"].pop("internal_resistance", None)
    return canonicalize_cycles_interpolated(payload), canonicalize_summary(payload)


def _precision_loss_warnings(caught):
    return [
        warning
        for warning in caught
        if "Precision loss occurred in moment calculation" in str(warning.message)
    ]


def test_safe_skew_constant_array_no_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = _safe_skew([1.0, 1.0, 1.0, 1.0])

    assert value == 0.0
    assert not _precision_loss_warnings(caught)


def test_safe_skew_near_constant_array_no_warning():
    values = [1.0, 1.0 + 1e-13, 1.0 - 1e-13, 1.0 + 2e-13]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = _safe_skew(values)

    assert value == 0.0
    assert not _precision_loss_warnings(caught)


def test_safe_skew_asymmetric_array_is_finite_and_nonzero():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = _safe_skew([0.0, 0.1, 0.2, 0.3, 4.0])

    assert np.isfinite(value)
    assert abs(value) > 0.1
    assert not _precision_loss_warnings(caught)


def test_cycle_scalar_extracts_selected_cycle_values_with_metadata():
    cycles, summary = _tables()
    spec = FeatureOperatorSpec(
        operator_name="cycle_scalar",
        operator_type="cycle_scalar",
        family="capacity_summary",
        params={"signals": ["discharge_capacity"], "cycle_indices": [9], "aggregations": ["last"]},
    )
    output = cycle_scalar(spec, cycles, summary)

    assert output.status == "ok"
    assert np.isclose(output.features["discharge_capacity_cycle_9"], 1.12 - 0.009)
    assert output.metadata[0]["feature_family"] == "capacity_summary"
    assert output.metadata[0]["uses_summary"] is True


def test_internal_resistance_request_resolves_to_dc_internal_resistance_summary_alias():
    cycles, summary = _dc_resistance_tables()
    spec = FeatureOperatorSpec(
        operator_name="cycle_scalar",
        operator_type="cycle_scalar",
        family="resistance_summary",
        params={"signals": ["internal_resistance"], "cycle_indices": [9], "aggregations": ["last"]},
    )
    output = cycle_scalar(spec, cycles, summary)

    assert np.isfinite(output.features["internal_resistance_cycle_9"])
    assert np.isclose(output.features["internal_resistance_cycle_9"], 0.029)
    assert output.metadata[0]["feature_family"] == "resistance_summary"
    assert output.metadata[0]["requested_signal"] == "internal_resistance"
    assert output.metadata[0]["resolved_signal"] == "dc_internal_resistance"


def test_internal_resistance_window_features_are_not_all_nan_with_dc_alias():
    cycles, summary = _dc_resistance_tables()
    spec = FeatureOperatorSpec(
        operator_name="cycle_window_stats",
        operator_type="cycle_window_stats",
        family="resistance_summary",
        params={"signals": ["internal_resistance"], "windows": [[0, 9]], "stats": ["mean", "slope", "delta"]},
    )
    output = cycle_window_stats(spec, cycles, summary)

    values = list(output.features.values())
    assert values
    assert any(np.isfinite(value) for value in values)
    assert {row["resolved_signal"] for row in output.metadata} == {"dc_internal_resistance"}


def test_cycle_window_stats_computes_slope_and_summary_stats():
    cycles, summary = _tables()
    spec = FeatureOperatorSpec(
        operator_name="cycle_window_stats",
        operator_type="cycle_window_stats",
        family="capacity_summary",
        params={"signals": ["discharge_capacity"], "windows": [[0, 9]], "stats": ["mean", "delta", "slope", "intercept", "curvature"]},
    )
    output = cycle_window_stats(spec, cycles, summary)

    assert np.isfinite(output.features["discharge_capacity_mean_cycle_0_to_9"])
    assert np.isclose(output.features["discharge_capacity_slope_cycle_0_to_9"], -0.001)
    assert "discharge_capacity_curvature_cycle_0_to_9" in output.features


def test_cross_cycle_scalar_delta_computes_finite_difference_ratio_and_log():
    cycles, summary = _tables()
    spec = FeatureOperatorSpec(
        operator_name="cross_cycle_scalar_delta",
        operator_type="cross_cycle_scalar_delta",
        family="scalar_delta",
        params={"signals": ["discharge_capacity"], "cycle_pairs": [[0, 9]], "operations": ["difference", "ratio", "relative_difference", "log_abs_difference"]},
    )
    output = cross_cycle_scalar_delta(spec, cycles, summary)

    assert np.isclose(output.features["discharge_capacity_difference_cycle_9_minus_0"], -0.009)
    assert np.isfinite(output.features["discharge_capacity_ratio_cycle_9_minus_0"])
    assert np.isfinite(output.features["discharge_capacity_log_abs_difference_cycle_9_minus_0"])


def test_curve_shape_filters_to_requested_step_type_and_marks_true_curve_feature():
    cycles, summary = _tables()
    spec = FeatureOperatorSpec(
        operator_name="curve_shape",
        operator_type="curve_shape",
        family="true_curve_shape",
        params={
            "step_types": ["discharge"],
            "cycles": [9],
            "x_axis": "voltage",
            "y_signals": ["discharge_capacity"],
            "grid_size": 20,
            "aggregations": ["mean", "std", "area"],
        },
    )
    output = curve_shape(spec, cycles, summary)

    assert output.features
    assert all(np.isfinite(value) for value in output.features.values())
    assert all(row["step_type"] == "discharge" for row in output.metadata)
    assert all(row["is_true_curve_feature"] is True for row in output.metadata)


def test_cross_cycle_curve_delta_interpolates_and_log_abs_zero_delta_is_finite():
    cycles, summary = _tables(identical_late_curve=True)
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
            "transforms": ["identity", "log_abs"],
            "aggregations": ["mean", "var", "skew", "sum_abs", "sum_sq", "area_abs"],
        },
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        output = cross_cycle_curve_delta(spec, cycles, summary)

    assert output.features
    assert all(np.isfinite(value) for value in output.features.values())
    assert output.features["discharge_discharge_capacity_curve_delta_identity_sum_abs_cycle_99_minus_9"] == 0.0
    assert output.features["discharge_discharge_capacity_curve_delta_identity_skew_cycle_99_minus_9"] == 0.0
    assert all(row["feature_family"] == "true_curve_difference" for row in output.metadata)
    assert all(row["is_true_curve_feature"] is True for row in output.metadata)
    assert all(row["is_proxy_feature"] is False for row in output.metadata)
    assert not _precision_loss_warnings(caught)


def test_protocol_features_are_explicit_and_opt_in_at_program_layer():
    cycles, summary = _tables()
    metadata_row = pd.Series({"C1": 4.0, "C2": 4.4, "C3": 4.8, "C4": 3.5})
    spec = FeatureOperatorSpec(operator_name="protocol", operator_type="protocol", family="protocol")
    output = protocol(spec, cycles, summary, metadata_row)

    assert output.features["protocol_C1"] == 4.0
    assert output.features["protocol_current_mean"] > 0
    assert all(row["uses_protocol"] is True for row in output.metadata)


def test_placeholder_operators_noop_with_warnings():
    cycles, summary = _tables()
    learned = learned_embedding_placeholder(
        FeatureOperatorSpec(operator_name="learned_embedding_placeholder", operator_type="learned_embedding_placeholder", family="learned_embedding_placeholder"),
        cycles,
        summary,
    )
    generic = generic_timeseries_placeholder(
        FeatureOperatorSpec(operator_name="generic_timeseries_placeholder", operator_type="generic_timeseries_placeholder", family="generic_timeseries_placeholder"),
        cycles,
        summary,
    )

    assert learned.features == {}
    assert generic.features == {}
    assert "not implemented" in learned.warnings[0]
    assert "intentionally disabled" in generic.warnings[0]
