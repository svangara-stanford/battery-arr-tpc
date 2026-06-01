import json

from battery_aar.agents.llm_client import AgentResponse
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


def test_candidate_failure_events_include_traceback_and_reason(tmp_path, monkeypatch):
    class BadAgent:
        agent_id = "bad_agent"

        def propose(self, prompt, iteration):
            return AgentResponse(
                code="""
def fit(train_metadata, train_cycle_summary, train_labels, config):
    raise KeyError('cell_id')

def predict(model, test_metadata, test_cycle_summary, config):
    return []
""",
                prompt=prompt,
                response_text="bad candidate",
            )

    monkeypatch.setattr("battery_aar.agents.orchestrator.make_agent", lambda *args, **kwargs: BadAgent())
    report = run_rediscovery(
        processed_dir=tmp_path / "missing",
        reference_run=None,
        out=tmp_path / "run",
        reports_dir=tmp_path / "reports",
        agents=1,
        iterations=1,
        offline=False,
        seed=3,
    )

    event = json.loads((tmp_path / "run" / "events.jsonl").read_text().splitlines()[0])
    assert event["success"] is False
    assert event["failure_reason"] == "'cell_id'"
    assert event["error_type"] == "KeyError"
    assert "Traceback" in event["traceback"]
    assert event["candidate_syntax_status"] == "passed"
    assert report["best_candidate"] is None
    assert report["candidate_failures"]
    assert "## Candidate Failures" in (tmp_path / "reports" / "agent_rediscovery.md").read_text()
