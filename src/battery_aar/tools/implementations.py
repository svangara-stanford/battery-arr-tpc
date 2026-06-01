from __future__ import annotations

import hashlib
import json
import time
import traceback
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np
import pandas as pd

from battery_aar.agents.evaluator import evaluate_candidate_train_test, regression_metrics
from battery_aar.agents.sandbox import validate_code_safety
from battery_aar.features.battery_lifetime_features import build_all_battery_features
from battery_aar.workflows.artifacts import ArtifactStore, build_dataset_profile_artifact
from battery_aar.workflows.schemas import AgentRole, CritiqueReport, EvaluationReport, FeaturePlan, ReviewReport
from battery_aar.workflows.trace import TraceLogger

from .schemas import (
    BuildFeaturesRequest,
    BuildFeaturesResponse,
    CandidateEvaluateRequest,
    CandidateEvaluateResponse,
    CandidateReviewRequest,
    CandidateReviewResponse,
    DatasetProfileRequest,
    DatasetProfileResponse,
    RunCompareRequest,
    RunCompareResponse,
    ToolDescriptor,
    ToolListResponse,
)

ResponseT = TypeVar("ResponseT")


TOOL_DESCRIPTORS = [
    ToolDescriptor(name="profile_dataset", endpoint="/dataset/profile", description="Profile metadata, cycle summaries, labels, batches, protocols, and missingness."),
    ToolDescriptor(name="build_battery_features", endpoint="/features/build", description="Build author-inspired battery lifetime feature tables."),
    ToolDescriptor(name="review_candidate", endpoint="/candidate/review", description="Statically review candidate code for safety and leakage risks."),
    ToolDescriptor(name="evaluate_candidate", endpoint="/candidate/evaluate", description="Evaluate a candidate against a train/validation split using the existing evaluator."),
    ToolDescriptor(name="compare_runs", endpoint="/runs/compare", description="Compare Open Battery Agents run summaries."),
]


def _run_dir(request_run_dir: str | None, run_id: str) -> Path:
    return Path(request_run_dir) if request_run_dir else Path("runs") / "open_battery_agents" / run_id


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _request_hash(request: Any) -> str:
    payload = request.model_dump_json(exclude={"tool_call_id"})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _trace_and_store(run_id: str, run_dir: str | None) -> tuple[ArtifactStore, TraceLogger]:
    store = ArtifactStore(_run_dir(run_dir, run_id), run_id=run_id)
    return store, TraceLogger(store.artifact_dir, run_id=run_id)


def _log_tool_call(
    request: Any,
    tool_name: str,
    response: Any,
    duration_ms: float,
    error_type: str | None = None,
    error_message: str | None = None,
) -> None:
    _store, trace = _trace_and_store(request.run_id, request.run_dir)
    trace.log_tool_call(
        tool_name=tool_name,
        tool_call_id=request.tool_call_id,
        iteration=request.iteration,
        agent_role=request.agent_role,
        input_artifact_ids=request.input_artifact_ids,
        output_artifact_ids=getattr(response, "output_artifact_ids", []),
        duration_ms=duration_ms,
        success=bool(getattr(response, "success", False)),
        error_type=error_type,
        error_message=error_message,
        arguments_summary={"request_sha256": _request_hash(request)},
    )


def _handle_tool(request: Any, tool_name: str, response_cls: type[ResponseT], fn: Callable[[ArtifactStore], dict[str, Any]]) -> ResponseT:
    started = time.perf_counter()
    store, _trace = _trace_and_store(request.run_id, request.run_dir)
    try:
        payload = fn(store)
        duration_ms = (time.perf_counter() - started) * 1000
        success = bool(payload.pop("_success", True))
        error_type = payload.pop("_error_type", None)
        error_message = payload.pop("_error_message", None)
        response = response_cls(
            tool_name=tool_name,
            tool_call_id=request.tool_call_id,
            run_id=request.run_id,
            success=success,
            error_type=error_type,
            error_message=error_message,
            duration_ms=duration_ms,
            **payload,
        )
        _log_tool_call(request, tool_name, response, duration_ms, error_type, error_message)
        return response
    except Exception as exc:
        duration_ms = (time.perf_counter() - started) * 1000
        response = response_cls(
            tool_name=tool_name,
            tool_call_id=request.tool_call_id,
            run_id=request.run_id,
            success=False,
            error_type=type(exc).__name__,
            error_message=str(exc),
            duration_ms=duration_ms,
        )
        _log_tool_call(request, tool_name, response, duration_ms, type(exc).__name__, str(exc))
        return response


def _load_processed_or_paths(request: DatasetProfileRequest) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str | None]:
    if request.processed_dir:
        base = Path(request.processed_dir)
        metadata_path = base / "cell_metadata.csv"
        if not metadata_path.exists():
            metadata_path = base / "metadata.csv"
        cycle_path = base / "cycle_summary.csv"
        labels_path = base / "labels.csv"
        metadata = pd.read_csv(metadata_path)
        cycles = pd.read_csv(cycle_path)
        if labels_path.exists():
            labels = pd.read_csv(labels_path)
        elif "cycle_life" in metadata.columns:
            labels = metadata[["row_id", "cycle_life"]].rename(columns={"cycle_life": "y"}).copy()
        else:
            labels = pd.DataFrame(columns=["row_id", "y"])
        label_source = request.label_source or (metadata["label_source"].dropna().iloc[0] if "label_source" in metadata and not metadata["label_source"].dropna().empty else None)
        return metadata, cycles, labels, label_source
    if not request.metadata_path or not request.cycle_summary_path:
        raise ValueError("Provide processed_dir or both metadata_path and cycle_summary_path")
    metadata = pd.read_csv(request.metadata_path)
    cycles = pd.read_csv(request.cycle_summary_path)
    if request.labels_path:
        labels = pd.read_csv(request.labels_path)
    elif "cycle_life" in metadata.columns:
        labels = metadata[["row_id", "cycle_life"]].rename(columns={"cycle_life": "y"}).copy()
    else:
        labels = pd.DataFrame(columns=["row_id", "y"])
    return metadata, cycles, labels, request.label_source


def profile_dataset(request: DatasetProfileRequest) -> DatasetProfileResponse:
    def run(store: ArtifactStore) -> dict[str, Any]:
        metadata, cycles, labels, label_source = _load_processed_or_paths(request)
        artifact = build_dataset_profile_artifact(
            run_id=request.run_id,
            metadata=metadata,
            cycle_summary=cycles,
            labels=labels,
            data_source=request.data_source,
            label_source=label_source,
            parent_artifact_ids=request.input_artifact_ids,
        )
        path = store.write_artifact(artifact)
        return {
            "output_artifact_ids": [artifact.artifact_id],
            "output_paths": {"dataset_profile": str(path)},
            "profile": artifact.model_dump(mode="json"),
        }

    return _handle_tool(request, "profile_dataset", DatasetProfileResponse, run)


def build_battery_features(request: BuildFeaturesRequest) -> BuildFeaturesResponse:
    def run(store: ArtifactStore) -> dict[str, Any]:
        metadata = pd.read_csv(request.metadata_path)
        cycles = pd.read_csv(request.cycle_summary_path)
        if request.return_feature_metadata:
            features, feature_meta = build_all_battery_features(
                metadata,
                cycles,
                max_cycle=request.max_cycle,
                include_protocol=request.include_protocol,
                return_feature_metadata=True,
            )
        else:
            features = build_all_battery_features(metadata, cycles, max_cycle=request.max_cycle, include_protocol=request.include_protocol)
            feature_meta = pd.DataFrame()
        output_path = Path(request.output_path) if request.output_path else store.artifact_dir / "tool_outputs" / "battery_features.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(output_path, index=True)
        feature_meta_path = output_path.with_name(output_path.stem + "_metadata.csv")
        if not feature_meta.empty:
            feature_meta.to_csv(feature_meta_path, index=False)
        artifact = FeaturePlan(
            run_id=request.run_id,
            parent_artifact_ids=request.input_artifact_ids,
            human_readable_summary=f"Built {features.shape[1]} battery feature columns for {features.shape[0]} cells.",
            agent_id=request.agent_role or "tool",
            feature_families=sorted(set(feature_meta["family"].astype(str))) if not feature_meta.empty and "family" in feature_meta else [],
            selected_columns=list(map(str, features.columns)),
            include_protocol_features=request.include_protocol,
            max_cycle=request.max_cycle,
            rationale="Generated by build_battery_features tool.",
        )
        artifact_path = store.write_artifact(artifact, "tool_outputs/feature_plan.json")
        output_paths = {"features": str(output_path), "feature_plan": str(artifact_path)}
        if feature_meta_path.exists():
            output_paths["feature_metadata"] = str(feature_meta_path)
        return {
            "output_artifact_ids": [artifact.artifact_id],
            "output_paths": output_paths,
            "n_rows": int(features.shape[0]),
            "n_features": int(features.shape[1]),
            "feature_columns": list(map(str, features.columns)),
        }

    return _handle_tool(request, "build_battery_features", BuildFeaturesResponse, run)


def _review_candidate_code(candidate_path: str | Path) -> tuple[str, list[str], list[str]]:
    code = Path(candidate_path).read_text()
    issues: list[str] = []
    recommendations: list[str] = []
    try:
        validate_code_safety(code)
    except Exception as exc:
        issues.append(str(exc))
    lowered = code.lower()
    for token in ["row_id", "cell_id"]:
        if token in lowered and "drop" not in lowered and "feature" in lowered:
            issues.append(f"{token} appears near feature construction; verify it is used only as a join key")
    if "batch_id" in lowered:
        issues.append("batch_id appears in candidate code; batch identifiers should not be model features")
    if "build_all_battery_features" not in code:
        recommendations.append("Consider using build_all_battery_features for author-inspired feature coverage")
    if "simpleimputer" not in lowered and "fillna" not in lowered:
        recommendations.append("Add explicit missing-value handling")
    verdict = "pass" if not issues else "needs_attention"
    return verdict, issues, recommendations


def review_candidate(request: CandidateReviewRequest) -> CandidateReviewResponse:
    def run(store: ArtifactStore) -> dict[str, Any]:
        verdict, issues, recommendations = _review_candidate_code(request.candidate_path)
        artifact = ReviewReport(
            run_id=request.run_id,
            parent_artifact_ids=request.input_artifact_ids,
            human_readable_summary=f"Candidate review verdict: {verdict}.",
            reviewer_id=request.agent_role or "tool_reviewer",
            target_artifact_ids=request.input_artifact_ids,
            verdict=verdict,
            issues=issues,
            recommendations=recommendations,
        )
        path = store.write_artifact(artifact, "tool_outputs/review_report.json")
        return {
            "output_artifact_ids": [artifact.artifact_id],
            "output_paths": {"review_report": str(path)},
            "verdict": verdict,
            "issues": issues,
            "recommendations": recommendations,
        }

    return _handle_tool(request, "review_candidate", CandidateReviewResponse, run)


def _load_labels(metadata: pd.DataFrame, labels_path: str | None) -> pd.DataFrame:
    if labels_path:
        labels = pd.read_csv(labels_path)
    elif "cycle_life" in metadata.columns:
        labels = metadata[["row_id", "cycle_life"]].rename(columns={"cycle_life": "y"}).copy()
    else:
        raise ValueError("labels_path is required unless metadata contains cycle_life")
    if "y" not in labels.columns and "cycle_life" in labels.columns:
        labels = labels.rename(columns={"cycle_life": "y"})
    labels["y"] = pd.to_numeric(labels["y"], errors="coerce")
    return labels[np.isfinite(labels["y"])].copy()


def _train_val_ids(metadata: pd.DataFrame, labels: pd.DataFrame, split_assignments_path: str | None) -> tuple[list[int], list[int]]:
    labeled = set(pd.to_numeric(labels["row_id"], errors="raise").astype(int).tolist())
    if split_assignments_path:
        assignments = pd.read_csv(split_assignments_path)
        assignments["row_id"] = pd.to_numeric(assignments["row_id"], errors="raise").astype(int)
        assignments = assignments[assignments["row_id"].isin(labeled)]
        split_values = assignments["split"].astype(str).str.lower()
        train = assignments.loc[split_values == "train", "row_id"].astype(int).tolist()
        val = assignments.loc[split_values.isin(["val", "validation"]), "row_id"].astype(int).tolist()
        if train and val:
            return train, val
    ids = sorted(labeled)
    if len(ids) < 2:
        raise ValueError("At least two labeled rows are required for candidate evaluation")
    n_val = max(1, int(round(len(ids) * 0.25)))
    return ids[:-n_val], ids[-n_val:]


def evaluate_candidate(request: CandidateEvaluateRequest) -> CandidateEvaluateResponse:
    def run(store: ArtifactStore) -> dict[str, Any]:
        metadata = pd.read_csv(request.metadata_path)
        cycles = pd.read_csv(request.cycle_summary_path)
        labels = _load_labels(metadata, request.labels_path)
        train_ids, val_ids = _train_val_ids(metadata, labels, request.split_assignments_path)
        train_labels = labels[labels["row_id"].isin(train_ids)]["y"].to_numpy(float)
        val_labels = labels[labels["row_id"].isin(val_ids)]["y"].to_numpy(float)
        weak_rmse = request.weak_rmse
        if weak_rmse is None:
            weak_pred = np.full_like(val_labels, float(np.mean(train_labels)), dtype=float)
            weak_rmse = regression_metrics(val_labels, weak_pred)["rmse"]
        result = evaluate_candidate_train_test(
            request.candidate_path,
            metadata[metadata["row_id"].isin(train_ids)].copy(),
            cycles[cycles["row_id"].isin(train_ids)].copy(),
            labels[labels["row_id"].isin(train_ids)].copy(),
            metadata[metadata["row_id"].isin(val_ids)].copy(),
            cycles[cycles["row_id"].isin(val_ids)].copy(),
            labels[labels["row_id"].isin(val_ids)].copy(),
            weak_rmse=weak_rmse,
            strong_rmse=request.strong_rmse,
            allow_protocol_features=request.allow_protocol_features,
            max_cycle=request.max_cycle,
            timeout_s=request.timeout_s,
        )
        metrics = result.get("metrics") or {}
        artifact = EvaluationReport(
            run_id=request.run_id,
            parent_artifact_ids=request.input_artifact_ids,
            human_readable_summary=f"Candidate evaluation success={bool(result.get('success'))}.",
            candidate_id=Path(request.candidate_path).stem,
            candidate_path=request.candidate_path,
            agent_id=request.agent_role,
            iteration=request.iteration,
            split_mode="tool_validation",
            success=bool(result.get("success")),
            rmse=metrics.get("rmse"),
            mae=metrics.get("mae"),
            r2=metrics.get("r2"),
            spearman=metrics.get("spearman"),
            kendall=metrics.get("kendall"),
            pgr=metrics.get("pgr_author_model"),
            failure_reason=result.get("failure_reason") or result.get("error"),
            traceback=result.get("traceback"),
            stdout_excerpt=(result.get("stdout") or "")[:2000],
            stderr_excerpt=(result.get("stderr") or "")[:2000],
            extra_metrics={k: v for k, v in metrics.items() if k not in {"rmse", "mae", "r2", "spearman", "kendall", "pgr_author_model"}},
        )
        path = store.write_artifact(artifact, "tool_outputs/evaluation_report.json")
        return {
            "_success": bool(result.get("success")),
            "_error_type": result.get("error_type"),
            "_error_message": result.get("failure_reason") or result.get("error"),
            "output_artifact_ids": [artifact.artifact_id],
            "output_paths": {"evaluation_report": str(path)},
            "metrics": _json_ready(metrics),
            "n_eval": result.get("n_eval"),
            "failure_reason": result.get("failure_reason") or result.get("error"),
        }

    return _handle_tool(request, "evaluate_candidate", CandidateEvaluateResponse, run)


def compare_runs(request: RunCompareRequest) -> RunCompareResponse:
    def run(store: ArtifactStore) -> dict[str, Any]:
        summary_path = Path(request.summary_csv) if request.summary_csv else Path(request.reports_dir) / "agent_rediscovery_runs_summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError(f"run summary CSV not found: {summary_path}")
        summary = pd.read_csv(summary_path)
        if request.run_ids:
            summary = summary[summary["run_id"].astype(str).isin([str(run_id) for run_id in request.run_ids])]
        output_path = Path(request.output_path) if request.output_path else store.artifact_dir / "tool_outputs" / "run_comparison.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_path, index=False)
        rows = summary.to_dict(orient="records")
        artifact = CritiqueReport(
            run_id=request.run_id,
            parent_artifact_ids=request.input_artifact_ids,
            human_readable_summary=f"Compared {len(rows)} Open Battery Agents runs.",
            critic_id=request.agent_role or "tool_run_comparer",
            strengths=[],
            weaknesses=[],
            proposed_next_steps=["Inspect run comparison CSV for candidate and split tradeoffs."],
        )
        artifact_path = store.write_artifact(artifact, "tool_outputs/run_compare_report.json")
        return {
            "output_artifact_ids": [artifact.artifact_id],
            "output_paths": {"comparison_csv": str(output_path), "comparison_report": str(artifact_path)},
            "comparison_rows": _json_ready(rows),
        }

    return _handle_tool(request, "compare_runs", RunCompareResponse, run)


def list_tools(run_id: str = "server", tool_call_id: str = "tool_list") -> ToolListResponse:
    return ToolListResponse(
        tool_name="list_tools",
        tool_call_id=tool_call_id,
        run_id=run_id,
        success=True,
        tools=TOOL_DESCRIPTORS,
    )
