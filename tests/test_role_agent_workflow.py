import json
from pathlib import Path

import pandas as pd

from battery_aar.agents.orchestrator import run_rediscovery
from battery_aar.workflows.role_graph import run_role_workflow


def _write_toy_processed_dataset(base: Path) -> Path:
    processed = base / "processed"
    processed.mkdir()
    metadata_rows = []
    cycle_rows = []
    for row_id in range(12):
        batch_id = "batch_a" if row_id < 6 else "batch_b"
        protocol = f"p{row_id % 4}"
        c1 = 4.0 + 0.2 * (row_id % 3)
        metadata_rows.append(
            {
                "row_id": row_id,
                "cell_id": f"cell_{row_id}",
                "batch_id": batch_id,
                "protocol_readable": protocol,
                "C1": c1,
                "C2": 4.4,
                "C3": 4.8,
                "C4": 3.5,
                "cycle_life": 900.0 - 8.0 * row_id,
                "label_source": "toy_surrogate",
            }
        )
        for cycle in range(1, 13):
            cycle_rows.append(
                {
                    "row_id": row_id,
                    "cell_id": f"cell_{row_id}",
                    "cycle_index": cycle,
                    "discharge_capacity": 1.1 - 0.001 * cycle - 0.0005 * row_id,
                    "charge_capacity": 1.12 - 0.001 * cycle,
                }
            )
    pd.DataFrame(metadata_rows).to_csv(processed / "cell_metadata.csv", index=False)
    pd.DataFrame(cycle_rows).to_csv(processed / "cycle_summary.csv", index=False)
    return processed


def test_offline_role_graph_runs_end_to_end_and_writes_artifacts(tmp_path):
    processed = _write_toy_processed_dataset(tmp_path)
    out = tmp_path / "role_run"
    report = run_role_workflow(
        processed_dir=processed,
        reference_run=tmp_path / "missing_reference",
        out=out,
        reports_dir=tmp_path / "reports",
        split_mode="random",
        validation_fraction=0.25,
        split_seed=2,
        offline=True,
        iterations=1,
    )

    artifact_dir = out / "artifacts"
    assert report["role_sequence"] == [
        "DatasetProfiler",
        "FeatureScientist",
        "ModelArchitect",
        "CodeGenerator",
        "CodeReviewer",
        "Evaluator",
        "ScientistCritic",
    ]
    assert (artifact_dir / "run_manifest.json").exists()
    assert (artifact_dir / "dataset_profile.json").exists()
    assert (artifact_dir / "split_artifact.json").exists()
    assert (artifact_dir / "iteration_000" / "feature_plan_feature_scientist.json").exists()
    assert (artifact_dir / "iteration_000" / "model_plan_model_architect.json").exists()
    assert (artifact_dir / "iteration_000" / "candidate_spec_code_generator.json").exists()
    assert (artifact_dir / "iteration_000" / "review_report_code_reviewer.json").exists()
    assert (artifact_dir / "iteration_000" / "evaluation_report_evaluator.json").exists()
    assert (artifact_dir / "iteration_000" / "critique_report_scientist_critic.json").exists()
    assert (artifact_dir / "experiment_state.json").exists()
    assert (tmp_path / "reports" / "role_agent_workflow.md").exists()

    index = json.loads((artifact_dir / "index.json").read_text())
    artifact_types = {entry["artifact_type"] for entry in index["artifacts"]}
    assert {
        "DatasetProfileArtifact",
        "FeaturePlan",
        "ModelPlan",
        "CandidateSpec",
        "ReviewReport",
        "EvaluationReport",
        "CritiqueReport",
        "ExperimentState",
    }.issubset(artifact_types)


def test_role_graph_tool_calls_and_candidate_toolbox_usage(tmp_path):
    processed = _write_toy_processed_dataset(tmp_path)
    out = tmp_path / "role_run"
    report = run_role_workflow(
        processed_dir=processed,
        reference_run=None,
        out=out,
        reports_dir=tmp_path / "reports",
        offline=True,
        iterations=1,
    )

    tool_calls = [json.loads(line) for line in (out / "artifacts" / "tool_calls.jsonl").read_text().splitlines() if line.strip()]
    tool_names = [call["tool_name"] for call in tool_calls]
    assert "profile_dataset" in tool_names
    assert "build_battery_features" in tool_names
    assert "review_candidate" in tool_names
    assert "evaluate_candidate" in tool_names

    candidate_path = Path(report["candidate_path"])
    assert "build_all_battery_features" in candidate_path.read_text()


def test_run_agentic_rediscovery_still_works_without_role_graph(tmp_path):
    report = run_rediscovery(
        processed_dir=tmp_path / "missing",
        reference_run=tmp_path / "missing_reference",
        out=tmp_path / "plain_rediscovery",
        reports_dir=tmp_path / "reports",
        agents=1,
        iterations=1,
        offline=True,
        seed=19,
    )

    assert report["mode"] == "offline"
    assert (tmp_path / "plain_rediscovery" / "leaderboard.csv").exists()
