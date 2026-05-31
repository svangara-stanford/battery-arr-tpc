from battery_aar.agents.orchestrator import run_rediscovery


def test_agent_rediscovery_does_not_require_batch9(tmp_path):
    report = run_rediscovery(
        processed_dir=tmp_path / "missing",
        reference_run=tmp_path / "missing_reference",
        out=tmp_path / "run",
        reports_dir=tmp_path / "reports",
        agents=1,
        iterations=1,
        offline=True,
    )
    assert report["batch9_status"] == "skipped_not_required"
    assert report["author_model_validation_metrics_unavailable_batch9_skipped"]
