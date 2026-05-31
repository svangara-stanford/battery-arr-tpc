from battery_aar.agents.orchestrator import run_rediscovery


def test_offline_agent_run_completes_on_synthetic_data(tmp_path):
    report = run_rediscovery(
        processed_dir=tmp_path / "missing",
        reference_run=None,
        out=tmp_path / "run",
        reports_dir=tmp_path / "reports",
        agents=1,
        iterations=1,
        offline=True,
        seed=2,
    )
    assert report["mode"] == "offline"
    assert (tmp_path / "run" / "leaderboard.csv").exists()
    assert (tmp_path / "reports" / "agent_rediscovery.json").exists()
