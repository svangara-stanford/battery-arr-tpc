import numpy as np

from battery_aar.paper_reproduction.bms_features import BatteryCell, _extract_summary_q_discharge, build_feature_vector


def synthetic_cell(cutoff=100):
    q = 1.1 - 0.0005 * np.arange(1, cutoff + 2)
    q10 = np.linspace(0.0, 1.0, 1000)
    qn = 0.9 * q10 - 0.001 * q10**2 - 0.001
    return BatteryCell(
        cell_id="cell1",
        batch_id="batch",
        channel=1,
        barcode="bc",
        protocol_readable="4.0-4.4-4.8-3.2",
        C1=4.0,
        C2=4.4,
        C3=4.8,
        C4=3.2,
        q_discharge=q,
        qdlin_by_cycle={10: q10, cutoff: qn},
        vdlin=np.linspace(2.8, 3.5, 1000),
        lifetime=500,
    )


def test_build_feature_vector_uses_matlab_summary_indexing():
    cell = synthetic_cell()
    feat, status = build_feature_vector(cell, 100)
    assert feat.shape == (15,)
    assert np.isclose(feat[0], cell.q_discharge[1])
    assert np.isclose(feat[2], cell.q_discharge[99])
    assert status == "ok"
    assert np.isfinite(feat[:15]).all()


def test_summary_cycle_zero_is_dropped_for_matlab_alignment():
    q = _extract_summary_q_discharge({"cycle_index": [0, 1, 2, 3], "discharge_capacity": [1.5, 1.08, 1.07, 1.06]})
    assert q.tolist() == [1.08, 1.07, 1.06]
