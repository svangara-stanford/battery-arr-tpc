import numpy as np
import pandas as pd

from battery_aar.features.battery_lifetime_features import build_all_battery_features, build_capacity_summary_features, build_protocol_features


def _toy_tables():
    metadata = pd.DataFrame(
        {
            "row_id": [1, 2],
            "cell_id": ["a", "b"],
            "C1": [4.0, 5.0],
            "C2": [4.4, 5.2],
            "C3": [4.8, 5.4],
            "C4": [np.nan, np.nan],
            "batch_id": ["hidden_a", "hidden_b"],
        }
    )
    cycles = []
    for row_id, fade in [(1, 0.001), (2, 0.002)]:
        for cycle in range(1, 101):
            cycles.append(
                {
                    "row_id": row_id,
                    "cell_id": f"cell_{row_id}",
                    "cycle_index": cycle,
                    "discharge_capacity": 1.1 - fade * cycle,
                    "charge_capacity": np.nan,
                }
            )
    return metadata, pd.DataFrame(cycles)


def test_feature_toolbox_returns_numeric_row_per_cell_without_identifier_columns():
    metadata, cycles = _toy_tables()
    features = build_all_battery_features(metadata, cycles, max_cycle=100, include_protocol=True)

    assert len(features) == 2
    assert "row_id" not in features.columns
    assert "cell_id" not in features.columns
    assert "batch_id" not in features.columns
    assert all(pd.api.types.is_numeric_dtype(features[col]) for col in features.columns)
    assert "capacity_cycle_10" in features.columns
    assert "capacity_cycle_100" in features.columns
    assert "capacity_late_slope" in features.columns
    assert "approx_log_sum_abs_qdiff" in features.columns


def test_feature_toolbox_drops_all_nan_columns():
    metadata, cycles = _toy_tables()
    protocol = build_protocol_features(metadata)

    assert "protocol_C4" not in protocol.columns
    assert not any(protocol[col].isna().all() for col in protocol.columns)


def test_protocol_features_can_be_included_or_excluded():
    metadata, cycles = _toy_tables()
    with_protocol = build_all_battery_features(metadata, cycles, max_cycle=100, include_protocol=True)
    without_protocol = build_all_battery_features(metadata, cycles, max_cycle=100, include_protocol=False)

    assert any(col.startswith("protocol_") for col in with_protocol.columns)
    assert not any(col.startswith("protocol_") for col in without_protocol.columns)


def test_feature_toolbox_accepts_include_protocol_features_alias():
    metadata, cycles = _toy_tables()
    features = build_all_battery_features(metadata, cycles, max_cycle=100, include_protocol_features=True)

    assert any(col.startswith("protocol_") for col in features.columns)


def test_feature_toolbox_rejects_disagreeing_protocol_aliases():
    metadata, cycles = _toy_tables()

    try:
        build_all_battery_features(metadata, cycles, max_cycle=100, include_protocol=True, include_protocol_features=False)
    except ValueError as exc:
        assert "disagree" in str(exc)
    else:
        raise AssertionError("Expected ValueError for disagreeing protocol flags")


def test_capacity_summary_handles_missing_cycles():
    _, cycles = _toy_tables()
    sparse = cycles[~cycles["cycle_index"].isin([50, 95, 98])]
    features = build_capacity_summary_features(sparse, max_cycle=100)

    assert len(features) == 2
    assert "capacity_cycle_50" not in features.columns
    assert "capacity_cycle_100" in features.columns
