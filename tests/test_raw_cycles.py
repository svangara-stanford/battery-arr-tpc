from __future__ import annotations

import numpy as np

from battery_aar.features.raw_cycles import (
    canonicalize_cycles_interpolated,
    canonicalize_summary,
    infer_cycle_index_convention,
    validate_cycles_df,
)

from .feature_program_test_utils import toy_raw_payload


def test_canonical_raw_cycle_loader_handles_flat_cycles_interpolated_arrays():
    payload = toy_raw_payload(n_cycles=3, n_points=4)
    cycles = canonicalize_cycles_interpolated(payload, cell_id="cell_a", row_id=7)

    assert {"row_id", "cell_id", "cycle_index", "step_type", "voltage", "discharge_capacity"}.issubset(cycles.columns)
    assert len(cycles) == 3 * 2 * 4
    assert cycles["row_id"].eq(7).all()
    assert cycles["cell_id"].eq("cell_a").all()
    assert set(cycles["step_type"]) == {"charge", "discharge"}
    assert np.isfinite(cycles["voltage"]).all()
    assert validate_cycles_df(cycles) == []


def test_cycle_index_convention_inference_detects_zero_based_data():
    payload = toy_raw_payload(n_cycles=5, n_points=2)
    cycles = canonicalize_cycles_interpolated(payload)
    summary = canonicalize_summary(payload)
    convention = infer_cycle_index_convention(cycles, summary)

    assert convention["cycle_index_convention"] == "raw_zero_based"
    assert convention["cycle_index_min"] == 0.0
    assert convention["summary_cycle_index_min"] == 0.0


def test_unequal_raw_arrays_are_truncated_with_warning():
    payload = toy_raw_payload(n_cycles=2, n_points=3)
    payload["cycles_interpolated"]["voltage"] = payload["cycles_interpolated"]["voltage"][:-2]
    cycles = canonicalize_cycles_interpolated(payload)

    assert "unequal lengths" in cycles.attrs["warnings"][0]
    assert len(cycles) == len(payload["cycles_interpolated"]["voltage"])
