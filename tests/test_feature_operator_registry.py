from __future__ import annotations

import pytest

from battery_aar.features.operator_registry import FeatureOperatorRegistry, default_operator_registry


def test_default_operator_registry_lists_expected_trusted_operators():
    registry = default_operator_registry()
    names = {row["name"] for row in registry.available_operators()}

    assert {
        "cycle_scalar",
        "cycle_window_stats",
        "cross_cycle_scalar_delta",
        "curve_shape",
        "cross_cycle_curve_delta",
        "protocol",
        "learned_embedding_placeholder",
        "generic_timeseries_placeholder",
    }.issubset(names)


def test_operator_registry_unknown_operator_fails_clearly():
    registry = FeatureOperatorRegistry()

    with pytest.raises(KeyError, match="Unknown feature operator"):
        registry.get("missing_operator")
