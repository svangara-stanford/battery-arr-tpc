import json
from pathlib import Path

import pandas as pd

from battery_aar.agents.orchestrator import run_rediscovery
import battery_aar.workflows.roles as workflow_roles
from battery_aar.workflows.role_graph import run_role_workflow
from battery_aar.workflows.schemas import EvaluationReport


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


def _write_toy_batch9(base: Path, n_cells: int = 5) -> Path:
    batch9 = base / "2019-01-24_batch9"
    batch9.mkdir(parents=True, exist_ok=True)
    for idx in range(n_cells):
        q0 = 1.12 - 0.002 * idx
        fade = 0.0018 + 0.00015 * idx
        cycles = list(range(1, 141))
        discharge = [q0 - fade * cycle for cycle in cycles]
        charge = [value + 0.015 for value in discharge]
        payload = {
            "channel_id": idx + 1,
            "barcode": f"batch9_barcode_{idx}",
            "protocol": f"protocol-4.{idx % 3}-4.4-4.8-3.5.sdu",
            "summary": {
                "cycle_index": cycles,
                "discharge_capacity": discharge,
                "charge_capacity": charge,
            },
        }
        (batch9 / f"2019-01-24_batch9_CH{idx + 1}_structure.json").write_text(json.dumps(payload))
    return batch9


def _write_author_validation_metrics(reference_run: Path, rmse: float | None = 75.0) -> Path:
    reference_run.mkdir(parents=True, exist_ok=True)
    payload = {"early_prediction_vs_observed": {"rmse": rmse, "mae": 50.0}}
    (reference_run / "validation_metrics.json").write_text(json.dumps(payload))
    return reference_run


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
    assert "Auto-generated by Open Battery Agents trusted candidate compiler" in candidate_path.read_text()
    assert report["candidate_spec"]["compiled_candidate"] is True
    assert report["candidate_spec"]["model_family"] == "Ridge"
    assert report["candidate_spec"]["feature_families"]
    assert report["best_review_status"] in {"pass", "pass_with_warnings"}
    assert report["final_review_status"] in {"pass", "pass_with_warnings"}


def test_role_graph_repair_loop_creates_repaired_candidate_after_review_failure(tmp_path, monkeypatch):
    processed = _write_toy_processed_dataset(tmp_path)

    def bad_candidate_code():
        return """
import pandas as pd
from battery_aar.features.battery_lifetime_features import build_all_battery_features

def fit(train_metadata, train_cycle_summary, train_labels, config):
    build_all_battery_features(train_metadata, train_cycle_summary, max_cycle=100, include_protocol_currents=True)
    return {}

def predict(model, test_metadata, test_cycle_summary, config):
    return pd.DataFrame({"row_id": test_metadata["row_id"], "y_pred": [1.0] * len(test_metadata)})
"""

    monkeypatch.setattr(workflow_roles, "offline_candidate_code", bad_candidate_code)
    out = tmp_path / "role_repair_run"
    report = run_role_workflow(
        processed_dir=processed,
        reference_run=None,
        out=out,
        reports_dir=tmp_path / "reports",
        offline=True,
        iterations=1,
        allow_freeform_code=True,
    )

    artifact_dir = out / "artifacts"
    assert (artifact_dir / "iteration_000" / "candidate_spec_code_generator.json").exists()
    assert (artifact_dir / "iteration_000" / "candidate_spec_code_generator_repair.json").exists()
    assert Path(report["candidate_path"]).name == "role_graph_iter_000_repair_1.py"
    assert "build_all_battery_features" in Path(report["candidate_path"]).read_text()
    events = [json.loads(line) for line in (artifact_dir / "events.jsonl").read_text().splitlines() if line.strip()]
    assert any(event["event_type"] == "candidate_repair_attempted" for event in events)


def test_role_graph_reports_best_successful_candidate_not_final_iteration(tmp_path, monkeypatch):
    processed = _write_toy_processed_dataset(tmp_path)
    rmse_by_iteration = {0: 5.0, 1: 1.0, 2: 3.0}

    def fake_evaluate(self, ctx, candidate, review, iteration):
        rmse = rmse_by_iteration[iteration]
        report = EvaluationReport(
            run_id=ctx.run_id,
            parent_artifact_ids=[candidate.artifact_id, review.artifact_id],
            human_readable_summary=f"fake evaluation {iteration}",
            candidate_id=candidate.candidate_id,
            candidate_path=candidate.candidate_path,
            agent_id="evaluator",
            candidate_name=candidate.candidate_name,
            iteration=iteration,
            split_mode=ctx.split_mode,
            success=True,
            rmse=rmse,
            mae=rmse / 2,
            r2=1.0 - rmse,
        )
        ctx.store.write_artifact(report)
        return report

    monkeypatch.setattr(workflow_roles.Evaluator, "run", fake_evaluate)
    out = tmp_path / "role_best_run"
    report = run_role_workflow(
        processed_dir=processed,
        reference_run=None,
        out=out,
        reports_dir=tmp_path / "reports",
        offline=True,
        iterations=3,
    )

    state = json.loads((out / "artifacts" / "experiment_state.json").read_text())
    assert report["best_iteration"] == 1
    assert report["validation_metrics"]["rmse"] == 1.0
    assert Path(report["candidate_path"]).name == "role_graph_iter_001.py"
    assert Path(report["final_iteration_candidate_path"]).name == "role_graph_iter_002.py"
    assert state["best_iteration"] == 1
    assert state["best_candidate_path"].endswith("role_graph_iter_001.py")
    assert [row["rmse"] for row in report["per_iteration_metrics"]] == [5.0, 1.0, 3.0]


def test_role_graph_candidates_per_iteration_generates_distinct_specs(tmp_path):
    processed = _write_toy_processed_dataset(tmp_path)
    out = tmp_path / "role_diversity_run"
    report = run_role_workflow(
        processed_dir=processed,
        reference_run=None,
        out=out,
        reports_dir=tmp_path / "reports",
        offline=True,
        iterations=1,
        candidates_per_iteration=4,
        allow_protocol_features=True,
    )

    rows = report["all_candidate_metrics"]
    assert len(rows) == 4
    assert len({row["candidate_id"] for row in rows}) == 4
    assert len({row["model_family"] for row in rows}) > 1
    assert {row["target_transform"] for row in rows}.issubset({"raw", "log10"})
    assert any(row["target_transform"] == "log10" for row in rows)
    assert all(row["prediction_path"] for row in rows if row["success"])
    candidate_spec_paths = sorted((out / "artifacts" / "iteration_000").glob("candidate_spec_role_graph_iter_000_variant_*.json"))
    assert len(candidate_spec_paths) == 4
    assert report["best_candidate_id"] in {row["candidate_id"] for row in rows}


def test_role_graph_best_selection_across_multiple_candidates_and_iterations(tmp_path, monkeypatch):
    processed = _write_toy_processed_dataset(tmp_path)

    def fake_evaluate(self, ctx, candidate, review, iteration):
        if candidate.candidate_id.endswith("variant_01") and iteration == 0:
            rmse = 0.5
        elif candidate.candidate_id.endswith("variant_00") and iteration == 1:
            rmse = 1.0
        else:
            rmse = 5.0 + iteration
        report = EvaluationReport(
            run_id=ctx.run_id,
            parent_artifact_ids=[candidate.artifact_id, review.artifact_id],
            human_readable_summary=f"fake evaluation {candidate.candidate_id}",
            candidate_id=candidate.candidate_id,
            candidate_path=candidate.candidate_path,
            agent_id="evaluator",
            candidate_name=candidate.candidate_name,
            iteration=iteration,
            split_mode=ctx.split_mode,
            success=True,
            rmse=rmse,
            mae=rmse / 2,
            r2=1.0 - rmse,
            extra_metrics={"y_pred_mean": 800.0 + rmse, "y_true_mean": 800.0, "n_predictions": 3},
        )
        ctx.store.write_artifact(report)
        return report

    monkeypatch.setattr(workflow_roles.Evaluator, "run", fake_evaluate)
    out = tmp_path / "role_best_variant_run"
    report = run_role_workflow(
        processed_dir=processed,
        reference_run=None,
        out=out,
        reports_dir=tmp_path / "reports",
        offline=True,
        iterations=2,
        candidates_per_iteration=2,
    )

    state = json.loads((out / "artifacts" / "experiment_state.json").read_text())
    assert report["best_iteration"] == 0
    assert report["best_candidate_id"] == "role_graph_iter_000_variant_01"
    assert report["validation_metrics"]["rmse"] == 0.5
    assert Path(report["candidate_path"]).name == "role_graph_iter_000_variant_01.py"
    assert state["best_candidate_id"] == "role_graph_iter_000_variant_01"
    assert len(report["all_candidate_metrics"]) == 4


def test_role_graph_locked_batch9_validation_writes_metrics_and_predictions(tmp_path):
    processed = _write_toy_processed_dataset(tmp_path)
    batch9_path = _write_toy_batch9(tmp_path)
    reference_run = _write_author_validation_metrics(tmp_path / "reference_run", rmse=60.0)
    out = tmp_path / "role_locked_run"
    report = run_role_workflow(
        processed_dir=processed,
        reference_run=reference_run,
        out=out,
        reports_dir=tmp_path / "reports",
        offline=True,
        iterations=1,
        final_batch9_validation=True,
        batch9_path=batch9_path,
    )

    metrics_path = out / "final_batch9_metrics.json"
    predictions_path = out / "final_batch9_predictions.csv"
    assert metrics_path.exists()
    assert predictions_path.exists()
    metrics = json.loads(metrics_path.read_text())
    predictions = pd.read_csv(predictions_path)
    assert report["locked_batch9_validation"]["status"] == "ok"
    assert report["final_batch9_metrics"]["rmse"] == metrics["rmse"]
    assert {"row_id", "cell_id", "y_true", "y_pred", "residual", "protocol_readable"}.issubset(predictions.columns)
    assert metrics["n_predictions"] == len(predictions)
    assert "batch9_weak_baseline_rmse" in metrics
    assert metrics["author_model_batch9_rmse"] == 60.0
    assert metrics["battery_pgr_author_model_batch9"] is not None
    state = json.loads((out / "artifacts" / "experiment_state.json").read_text())
    assert state["final_batch9_metrics_path"] == str(metrics_path)
    assert state["final_batch9_predictions_path"] == str(predictions_path)
    assert state["best_locked_validation_candidate_path"] == report["candidate_path"]


def test_role_graph_batch9_not_used_during_surrogate_search(tmp_path):
    processed = _write_toy_processed_dataset(tmp_path)
    batch9_path = _write_toy_batch9(tmp_path)
    out = tmp_path / "role_locked_isolated_run"
    report = run_role_workflow(
        processed_dir=processed,
        reference_run=None,
        out=out,
        reports_dir=tmp_path / "reports",
        offline=True,
        iterations=1,
        final_batch9_validation=True,
        batch9_path=batch9_path,
    )

    assignments = pd.read_csv(out / "split_assignments.csv")
    assert not assignments.get("batch_id", pd.Series(dtype=str)).astype(str).str.contains("2019-01-24_batch9", regex=False).any()
    search_prediction_paths = [Path(row["prediction_path"]) for row in report["all_candidate_metrics"] if row.get("prediction_path")]
    assert search_prediction_paths
    for prediction_path in search_prediction_paths:
        pred = pd.read_csv(prediction_path)
        if "batch_id" in pred.columns:
            assert not pred["batch_id"].astype(str).str.contains("2019-01-24_batch9", regex=False).any()
    assert pd.read_csv(out / "final_batch9_predictions.csv")["cell_id"].astype(str).str.contains("batch9_cell").all()


def test_role_graph_final_batch9_topk_runs_after_search(tmp_path):
    processed = _write_toy_processed_dataset(tmp_path)
    batch9_path = _write_toy_batch9(tmp_path)
    out = tmp_path / "role_topk_run"
    report = run_role_workflow(
        processed_dir=processed,
        reference_run=None,
        out=out,
        reports_dir=tmp_path / "reports",
        offline=True,
        iterations=1,
        candidates_per_iteration=3,
        final_batch9_top_k=2,
        batch9_path=batch9_path,
    )

    topk_path = out / "final_batch9_topk_metrics.csv"
    topk_dir = out / "final_batch9_topk_predictions"
    assert topk_path.exists()
    topk = pd.read_csv(topk_path)
    assert len(topk) == 2
    assert topk_dir.is_dir()
    assert len(list(topk_dir.glob("*.csv"))) == 2
    assert report["locked_batch9_validation"]["status"] == "ok"
    assert report["locked_batch9_validation"]["top_k_only"] == 2
    assert len(report["final_batch9_topk_rows"]) == 2


def test_role_graph_report_includes_locked_validation_section(tmp_path):
    processed = _write_toy_processed_dataset(tmp_path)
    batch9_path = _write_toy_batch9(tmp_path)
    run_role_workflow(
        processed_dir=processed,
        reference_run=None,
        out=tmp_path / "role_report_locked_run",
        reports_dir=tmp_path / "reports",
        offline=True,
        iterations=1,
        final_batch9_validation=True,
        batch9_path=batch9_path,
    )

    report_text = (tmp_path / "reports" / "role_agent_workflow.md").read_text()
    assert "## Locked Validation Batch" in report_text
    assert "Batch 9 was used only after search for locked final validation." in report_text


def test_role_graph_batch9_pgr_requires_author_rmse(tmp_path):
    processed = _write_toy_processed_dataset(tmp_path)
    batch9_path = _write_toy_batch9(tmp_path)
    out = tmp_path / "role_no_author_pgr_run"
    report = run_role_workflow(
        processed_dir=processed,
        reference_run=None,
        out=out,
        reports_dir=tmp_path / "reports",
        offline=True,
        iterations=1,
        final_batch9_validation=True,
        batch9_path=batch9_path,
    )

    metrics = json.loads((out / "final_batch9_metrics.json").read_text())
    assert metrics["batch9_weak_baseline_rmse"] is not None
    assert metrics["author_model_batch9_rmse"] is None
    assert metrics["battery_pgr_author_model_batch9"] is None
    assert report["batch9_pgr"] is None


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
