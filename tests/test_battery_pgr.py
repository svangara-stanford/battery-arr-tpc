from battery_aar.agents.evaluator import battery_pgr


def test_battery_pgr_handles_missing_strong_reference():
    assert battery_pgr(200.0, 150.0, None) is None
    assert battery_pgr(200.0, 150.0, 100.0) == 0.5
    assert battery_pgr(100.0, 90.0, 100.0) is None
