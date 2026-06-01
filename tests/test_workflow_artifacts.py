import json

import pandas as pd

from battery_aar.agents.orchestrator import run_rediscovery
from battery_aar.workflows.artifacts import ArtifactStore, build_dataset_profile_artifact
from battery_aar.workflows.schemas import AgentRole, CandidateSpec, EvaluationReport, RunManifest
from battery_aar.workflows.trace import TraceLogger


def test_pydantic_artifact_schemas_serialize_deserialize():
    manifest = RunManifest(
        run_id="run_a",
        output_dir="runs/open_battery_agents/run_a",
        human_readable_summary="test manifest",
        config={"offline": True},
    )
    candidate = CandidateSpec(
        run_id="run_a",
        parent_artifact_ids=[manifest.artifact_id],
        human_readable_summary="candidate",
        candidate_id="agent_0_iter_0",
        agent_id="agent_0",
        agent_role=AgentRole.LLM_CANDIDATE,
        iteration=0,
        candidate_path="candidates/agent_0_iter_0.py",
        uses_toolbox=True,
    )
    evaluation = EvaluationReport(
        run_id="run_a",
        parent_artifact_ids=[candidate.artifact_id],
        human_readable_summary="evaluation",
        candidate_id=candidate.candidate_id,
        candidate_path=candidate.candidate_path,
        agent_id="agent_0",
        iteration=0,
        split_mode="random",
        success=True,
        rmse=12.5,
        mae=10.0,
    )

    round_tripped = EvaluationReport.model_validate_json(evaluation.model_dump_json())
    assert round_tripped.schema_version == evaluation.schema_version
    assert round_tripped.rmse == 12.5
    assert candidate.agent_role == AgentRole.LLM_CANDIDATE


def test_artifact_store_writes_json_and_index_entries(tmp_path):
    store = ArtifactStore(tmp_path / "run_a", run_id="run_a")
    manifest = RunManifest(run_id="run_a", output_dir=str(tmp_path / "run_a"), human_readable_summary="manifest")
    path = store.write_artifact(manifest)

    assert path.exists()
    payload = json.loads(path.read_text())
    index = json.loads((tmp_path / "run_a" / "artifacts" / "index.json").read_text())
    assert payload["artifact_type"] == "RunManifest"
    assert index["artifacts"][0]["artifact_id"] == manifest.artifact_id
    assert index["artifacts"][0]["path"] == "artifacts/run_manifest.json"


def test_dataset_profile_artifact_captures_basic_profile():
    metadata = pd.DataFrame(
        {
            "row_id": [1, 2],
            "batch_id": ["b0", "b1"],
            "protocol_readable": ["p0", "p0"],
            "label_source": ["synthetic", "synthetic"],
        }
    )
    cycles = pd.DataFrame(
        {
            "row_id": [1, 1, 2],
            "cycle_index": [1, 2, 1],
            "discharge_capacity": [1.0, None, 1.1],
        }
    )
    labels = pd.DataFrame({"row_id": [1, 2], "y": [100.0, 120.0]})
    profile = build_dataset_profile_artifact("run_a", metadata, cycles, labels, "synthetic_demo", "synthetic")

    assert profile.metadata_row_count == 2
    assert profile.cycle_summary_row_count == 3
    assert profile.labeled_cell_count == 2
    assert profile.nan_counts["cycle_summary.discharge_capacity"] == 1
    assert profile.cycle_index_min == 1.0
    assert profile.cycle_index_max == 2.0
    assert profile.batch_id_counts == {"b0": 1, "b1": 1}
    assert profile.protocol_counts == {"p0": 2}


def test_trace_logger_writes_jsonl_records(tmp_path):
    logger = TraceLogger(tmp_path / "artifacts", run_id="run_a")
    logger.log_event(
        event_type="candidate_evaluated",
        iteration=0,
        agent_role=AgentRole.EVALUATOR,
        input_artifact_ids=["a"],
        output_artifact_ids=["b"],
        success=True,
    )
    logger.log_tool_call(tool_name="evaluate_candidate", iteration=0, agent_role=AgentRole.EVALUATOR, success=True)
    logger.log_agent_message(
        event_type="candidate_response",
        iteration=0,
        agent_role=AgentRole.LLM_CANDIDATE,
        agent_id="agent_0",
        message_summary="candidate code",
    )

    event = json.loads((tmp_path / "artifacts" / "events.jsonl").read_text().splitlines()[0])
    tool = json.loads((tmp_path / "artifacts" / "tool_calls.jsonl").read_text().splitlines()[0])
    message = json.loads((tmp_path / "artifacts" / "agent_messages.jsonl").read_text().splitlines()[0])
    assert event["run_id"] == "run_a"
    assert event["agent_role"] == "evaluator"
    assert tool["tool_name"] == "evaluate_candidate"
    assert message["agent_id"] == "agent_0"


def test_run_agentic_rediscovery_emit_artifact_trace_creates_artifacts(tmp_path):
    report = run_rediscovery(
        processed_dir=tmp_path / "missing",
        reference_run=tmp_path / "missing_reference",
        out=tmp_path / "run_trace",
        reports_dir=tmp_path / "reports",
        agents=1,
        iterations=1,
        offline=True,
        seed=7,
        emit_artifact_trace=True,
    )

    artifact_dir = tmp_path / "run_trace" / "artifacts"
    assert report["emit_artifact_trace"] is True
    assert (artifact_dir / "run_manifest.json").exists()
    assert (artifact_dir / "dataset_profile.json").exists()
    assert (artifact_dir / "split_artifact.json").exists()
    assert (artifact_dir / "experiment_state.json").exists()
    assert (artifact_dir / "iteration_000" / "candidate_spec_agent_0.json").exists()
    assert (artifact_dir / "iteration_000" / "evaluation_report_agent_0.json").exists()
    index = json.loads((artifact_dir / "index.json").read_text())
    artifact_types = {entry["artifact_type"] for entry in index["artifacts"]}
    assert {"RunManifest", "DatasetProfileArtifact", "SplitArtifact", "CandidateSpec", "EvaluationReport", "ExperimentState"}.issubset(artifact_types)
    assert (artifact_dir / "events.jsonl").exists()
    assert (artifact_dir / "tool_calls.jsonl").exists()
    assert (artifact_dir / "agent_messages.jsonl").exists()
    events = [json.loads(line) for line in (artifact_dir / "events.jsonl").read_text().splitlines()]
    event_types = [event["event_type"] for event in events]
    assert "run_started" in event_types
    assert "run_completed" in event_types


def test_run_agentic_rediscovery_without_artifact_trace_does_not_create_artifacts(tmp_path):
    report = run_rediscovery(
        processed_dir=tmp_path / "missing",
        reference_run=tmp_path / "missing_reference",
        out=tmp_path / "run_plain",
        reports_dir=tmp_path / "reports",
        agents=1,
        iterations=1,
        offline=True,
        seed=8,
    )

    assert report["emit_artifact_trace"] is False
    assert not (tmp_path / "run_plain" / "artifacts").exists()
