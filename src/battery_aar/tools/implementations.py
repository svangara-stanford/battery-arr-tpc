from __future__ import annotations

import hashlib
import json
import re
import time
import traceback
import types
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np
import pandas as pd

from battery_aar.agents.candidate_api import load_candidate
from battery_aar.agents.evaluator import evaluate_candidate_train_test, regression_metrics
from battery_aar.agents.sandbox import validate_code_safety
from battery_aar.features.battery_lifetime_features import build_all_battery_features
from battery_aar.features.operator_registry import default_operator_registry
from battery_aar.features.program_library import available_program_recipes
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
    FeatureProgramsResponse,
    RunCompareRequest,
    RunCompareResponse,
    ToolDescriptor,
    ToolListResponse,
)

ResponseT = TypeVar("ResponseT")


TOOL_DESCRIPTORS = [
    ToolDescriptor(name="profile_dataset", endpoint="/dataset/profile", description="Profile metadata, cycle summaries, labels, batches, protocols, and missingness."),
    ToolDescriptor(name="build_battery_features", endpoint="/features/build", description="Build author-inspired battery lifetime feature tables."),
    ToolDescriptor(name="list_feature_programs", endpoint="/features/programs", description="List trusted feature-program recipes and operators."),
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
                feature_program_paths=request.feature_program_paths,
                feature_program_mode=request.feature_program_mode,
                include_feature_programs=request.include_feature_programs,
                feature_family_filter=request.feature_family_filter,
            )
        else:
            features = build_all_battery_features(
                metadata,
                cycles,
                max_cycle=request.max_cycle,
                include_protocol=request.include_protocol,
                feature_program_paths=request.feature_program_paths,
                feature_program_mode=request.feature_program_mode,
                include_feature_programs=request.include_feature_programs,
                feature_family_filter=request.feature_family_filter,
            )
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
        family_col = "feature_family" if "feature_family" in feature_meta.columns else "family" if "family" in feature_meta.columns else None
        family_counts = {str(k): int(v) for k, v in feature_meta[family_col].value_counts().to_dict().items()} if family_col else {}
        feature_program_columns = 0
        if not feature_meta.empty and "source_feature_program_table" in feature_meta.columns:
            feature_program_columns = int(feature_meta["source_feature_program_table"].notna().sum())
        matched_rows = int(len(features))
        missing_rows = 0
        if request.feature_program_paths:
            key = "row_id" if "row_id" in metadata.columns else "cell_id" if "cell_id" in metadata.columns else None
            if key:
                expected = set(metadata[key].dropna().astype(str))
                seen = set(map(str, features.index.dropna().tolist()))
                missing_rows = int(len(expected - seen))
        return {
            "output_artifact_ids": [artifact.artifact_id],
            "output_paths": output_paths,
            "n_rows": int(features.shape[0]),
            "n_features": int(features.shape[1]),
            "feature_columns": list(map(str, features.columns)),
            "feature_programs_used": list(request.feature_program_paths),
            "n_feature_program_columns": feature_program_columns,
            "feature_family_counts": family_counts,
            "n_matched_rows": matched_rows,
            "n_missing_rows": missing_rows,
            "true_raw_curve_features_used": bool(feature_meta.get("is_true_curve_feature", pd.Series(dtype=bool)).fillna(False).astype(bool).any()),
            "proxy_features_used": bool(feature_meta.get("is_proxy_feature", pd.Series(dtype=bool)).fillna(False).astype(bool).any()),
            "protocol_features_used": bool(feature_meta.get("uses_protocol", pd.Series(dtype=bool)).fillna(False).astype(bool).any()),
        }

    return _handle_tool(request, "build_battery_features", BuildFeaturesResponse, run)


def list_feature_programs(
    run_id: str = "server",
    tool_call_id: str = "feature_programs",
    run_dir: str | None = None,
    iteration: int | None = None,
    agent_role: str | None = None,
    input_artifact_ids: list[str] | None = None,
) -> FeatureProgramsResponse:
    started = time.perf_counter()
    response = FeatureProgramsResponse(
        tool_name="list_feature_programs",
        tool_call_id=tool_call_id,
        run_id=run_id,
        success=True,
        recipes=available_program_recipes(),
        operators=default_operator_registry().available_operators(),
        duration_ms=0.0,
    )
    response.duration_ms = (time.perf_counter() - started) * 1000
    _store, trace = _trace_and_store(run_id, run_dir)
    trace.log_tool_call(
        tool_name="list_feature_programs",
        tool_call_id=tool_call_id,
        iteration=iteration,
        agent_role=agent_role,
        input_artifact_ids=input_artifact_ids or [],
        output_artifact_ids=[],
        duration_ms=response.duration_ms,
        success=True,
        arguments_summary={"request_sha256": hashlib.sha256(f"{run_id}:{tool_call_id}".encode("utf-8")).hexdigest()},
    )
    return response


def _mock_candidate_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train_metadata = pd.DataFrame(
        {
            "row_id": [0, 1, 2],
            "cell_id": ["anon_cell_00000", "anon_cell_00001", "anon_cell_00002"],
            "C1": [4.0, 4.4, 4.8],
            "C2": [4.4, 4.8, 5.2],
            "C3": [4.8, 5.2, 5.2],
            "C4": [3.5, 3.6, 3.7],
        }
    )
    test_metadata = pd.DataFrame(
        {
            "row_id": [3, 4],
            "cell_id": ["anon_cell_00003", "anon_cell_00004"],
            "C1": [4.2, 4.6],
            "C2": [4.6, 5.0],
            "C3": [5.0, 5.2],
            "C4": [3.55, 3.65],
        }
    )
    rows: list[dict[str, Any]] = []
    for row_id in [0, 1, 2, 3, 4]:
        for cycle in [1, 2, 10, 50, 91, 100]:
            rows.append(
                {
                    "row_id": row_id,
                    "cell_id": f"anon_cell_{row_id:05d}",
                    "cycle_index": cycle,
                    "discharge_capacity": 1.10 - 0.001 * cycle - 0.0005 * row_id,
                    "charge_capacity": 1.12 - 0.001 * cycle,
                }
            )
    cycle_summary = pd.DataFrame(rows)
    train_labels = pd.DataFrame(
        {
            "row_id": [0, 1, 2],
            "cell_id": ["anon_cell_00000", "anon_cell_00001", "anon_cell_00002"],
            "y": [900.0, 850.0, 800.0],
        }
    )
    config = {"max_cycle": 100, "allow_protocol_features": True}
    return (
        train_metadata,
        cycle_summary[cycle_summary["row_id"].isin([0, 1, 2])].copy(),
        train_labels,
        test_metadata,
        cycle_summary[cycle_summary["row_id"].isin([3, 4])].copy(),
        config,
    )


def _preflight_candidate(candidate_path: str | Path) -> tuple[bool, str | None, str | None]:
    try:
        train_metadata, train_cycles, train_labels, test_metadata, test_cycles, config = _mock_candidate_tables()
        candidate = load_candidate(candidate_path)
        if isinstance(candidate, types.ModuleType):
            model = candidate.fit(train_metadata, train_cycles, train_labels, config)
            pred = candidate.predict(model, test_metadata, test_cycles, config)
        else:
            model = candidate.fit(train_metadata, train_cycles, train_labels, config)
            pred = candidate.predict(test_metadata, test_cycles, config)
        if not isinstance(pred, pd.DataFrame):
            pred = pd.DataFrame(pred)
        if "y_pred" not in pred.columns:
            if pred.shape[1] == 1:
                pred.columns = ["y_pred"]
            else:
                raise ValueError("preflight predictions must contain y_pred")
        if "row_id" not in pred.columns and "cell_id" not in pred.columns and len(pred) != len(test_metadata):
            raise ValueError("preflight predictions must include row_id/cell_id or match test row order")
        y_pred = pd.to_numeric(pred["y_pred"], errors="coerce").to_numpy(float)
        if len(y_pred) != len(test_metadata):
            raise ValueError(f"preflight prediction row count {len(y_pred)} did not match expected {len(test_metadata)}")
        if not np.isfinite(y_pred).all():
            raise ValueError("preflight predictions must be finite")
        return True, None, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}", traceback.format_exc()


IDENTIFIER_REVIEW_TOKENS = {
    "batch_id",
    "source_path",
    "protocol_readable",
    "anonymized_cell_id",
    "cell_id",
    "row_id",
    "barcode",
    "channel",
    "file_path",
    "filename",
    "path",
}


def _identifier_token_is_allowed_context(line: str, token: str) -> bool:
    lowered = line.lower()
    if "identifier" in lowered or "leakage" in lowered:
        return True
    if "drop" in lowered or "exclude" in lowered or "forbidden" in lowered or "blocked" in lowered:
        return True
    if "columns" in lowered and ("{" in line or "[" in line or "(" in line):
        return True
    if token in {"row_id", "cell_id"} and (
        "merge" in lowered
        or "join" in lowered
        or "index" in lowered
        or "output" in lowered
        or "insert" in lowered
        or "drop_duplicates" in lowered
        or "columns" in lowered
        or "[[" in lowered
    ):
        return True
    return False


def _identifier_feature_issues(code: str) -> list[str]:
    issues: list[str] = []
    in_identifier_block = False
    def contains_token(line: str, token: str) -> bool:
        return re.search(rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])", line) is not None

    for line_no, line in enumerate(code.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lowered = stripped.lower()
        if in_identifier_block:
            if "}" in stripped or "]" in stripped or ")" in stripped:
                in_identifier_block = False
            continue
        if ("identifier" in lowered or "exclude" in lowered or "drop" in lowered) and ("{" in stripped or "[" in stripped or "(" in stripped):
            if not ("}" in stripped or "]" in stripped or ")" in stripped):
                in_identifier_block = True
            continue
        for token in IDENTIFIER_REVIEW_TOKENS:
            if not contains_token(lowered, token):
                continue
            if _identifier_token_is_allowed_context(stripped, token):
                continue
            if (
                "feature" in lowered
                or re.search(r"\bX\s*=", stripped)
                or re.search(r"\bfeatures\s*=", stripped)
                or ".fit(" in stripped
            ):
                issues.append(f"{token} may be used as a model feature on line {line_no}")
    return issues


def _review_candidate_code(candidate_path: str | Path) -> tuple[str, list[str], list[str], str | None]:
    code = Path(candidate_path).read_text()
    issues: list[str] = []
    recommendations: list[str] = []
    failure_reason: str | None = None
    try:
        validate_code_safety(code)
    except Exception as exc:
        issues.append(str(exc))
    lowered = code.lower()
    issues.extend(_identifier_feature_issues(code))
    if "build_all_battery_features" not in code:
        recommendations.append("Consider using build_all_battery_features for author-inspired feature coverage")
    if "simpleimputer" not in lowered and "fillna" not in lowered:
        recommendations.append("Add explicit missing-value handling")
    preflight_ok, preflight_reason, preflight_traceback = _preflight_candidate(candidate_path)
    if not preflight_ok:
        failure_reason = preflight_reason
        issues.append(f"preflight_failure: {preflight_reason}")
        if preflight_traceback:
            recommendations.append("Preflight traceback:\n" + preflight_traceback)
    if failure_reason:
        verdict = "needs_repair"
    elif issues:
        verdict = "needs_attention"
    else:
        verdict = "pass"
    return verdict, issues, recommendations, failure_reason


def review_candidate(request: CandidateReviewRequest) -> CandidateReviewResponse:
    def run(store: ArtifactStore) -> dict[str, Any]:
        verdict, issues, recommendations, failure_reason = _review_candidate_code(request.candidate_path)
        artifact = ReviewReport(
            run_id=request.run_id,
            parent_artifact_ids=request.input_artifact_ids,
            human_readable_summary=f"Candidate review verdict: {verdict}." + (f" Failure: {failure_reason}" if failure_reason else ""),
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
            "failure_reason": failure_reason,
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


def _prediction_diagnostics(predictions: pd.DataFrame) -> dict[str, Any]:
    if predictions.empty:
        return {
            "n_predictions": 0,
            "n_negative_predictions": 0,
            "n_nonfinite_predictions": 0,
        }
    y_true = pd.to_numeric(predictions["y_true"], errors="coerce").to_numpy(float)
    y_pred = pd.to_numeric(predictions["y_pred"], errors="coerce").to_numpy(float)
    residual = y_pred - y_true
    finite_pred = np.isfinite(y_pred)
    finite_true = np.isfinite(y_true)
    finite_residual = np.isfinite(residual)
    diagnostics = {
        "y_true_mean": float(np.mean(y_true[finite_true])) if finite_true.any() else None,
        "y_pred_mean": float(np.mean(y_pred[finite_pred])) if finite_pred.any() else None,
        "y_true_min": float(np.min(y_true[finite_true])) if finite_true.any() else None,
        "y_true_max": float(np.max(y_true[finite_true])) if finite_true.any() else None,
        "y_pred_min": float(np.min(y_pred[finite_pred])) if finite_pred.any() else None,
        "y_pred_max": float(np.max(y_pred[finite_pred])) if finite_pred.any() else None,
        "residual_mean": float(np.mean(residual[finite_residual])) if finite_residual.any() else None,
        "residual_std": float(np.std(residual[finite_residual], ddof=0)) if finite_residual.any() else None,
        "n_predictions": int(len(predictions)),
        "n_negative_predictions": int(np.sum(finite_pred & (y_pred < 0))),
        "n_nonfinite_predictions": int(np.sum(~finite_pred)),
    }
    return diagnostics


def _write_evaluation_predictions(
    *,
    store: ArtifactStore,
    request: CandidateEvaluateRequest,
    candidate_id: str,
    predictions: pd.DataFrame,
    metadata: pd.DataFrame,
    val_ids: list[int],
) -> tuple[Path, dict[str, Any]]:
    output = predictions.rename(columns={"y": "y_true"}).copy()
    if "y_true" not in output.columns:
        raise ValueError("prediction diagnostics require y_true")
    output["residual"] = pd.to_numeric(output["y_pred"], errors="coerce") - pd.to_numeric(output["y_true"], errors="coerce")
    output["split_mode"] = request.split_mode
    meta_cols = ["row_id"]
    for col in ["cell_id", "batch_id", "protocol_readable", "C1", "C2", "C3", "C4"]:
        if col in metadata.columns:
            meta_cols.append(col)
    meta = metadata.loc[metadata["row_id"].isin(val_ids), meta_cols].drop_duplicates("row_id").copy()
    output = output.merge(meta, on="row_id", how="left")
    ordered_cols = [
        "row_id",
        "cell_id",
        "y_true",
        "y_pred",
        "residual",
        "split_mode",
        "batch_id",
        "protocol_readable",
        "C1",
        "C2",
        "C3",
        "C4",
    ]
    output = output[[col for col in ordered_cols if col in output.columns]]
    prediction_path = store.artifact_dir / f"iteration_{int(request.iteration or 0):03d}" / f"predictions_{candidate_id}.csv"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(prediction_path, index=False)
    return prediction_path, _prediction_diagnostics(output)


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
            return_predictions=True,
            feature_program_paths=request.feature_program_paths,
            feature_program_mode=request.feature_program_mode,
            include_feature_programs=request.include_feature_programs,
            feature_family_filter=request.feature_family_filter,
        )
        metrics = result.get("metrics") or {}
        candidate_id = Path(request.candidate_path).stem
        prediction_path: Path | None = None
        prediction_diagnostics: dict[str, Any] = {}
        if bool(result.get("success")) and isinstance(result.get("predictions"), pd.DataFrame):
            prediction_path, prediction_diagnostics = _write_evaluation_predictions(
                store=store,
                request=request,
                candidate_id=candidate_id,
                predictions=result["predictions"],
                metadata=metadata,
                val_ids=val_ids,
            )
            metrics = {**metrics, **prediction_diagnostics}
        artifact = EvaluationReport(
            run_id=request.run_id,
            parent_artifact_ids=request.input_artifact_ids,
            human_readable_summary=f"Candidate evaluation success={bool(result.get('success'))}.",
            candidate_id=candidate_id,
            candidate_path=request.candidate_path,
            agent_id=request.agent_role,
            iteration=request.iteration,
            split_mode=request.split_mode,
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
            prediction_path=str(prediction_path) if prediction_path else None,
            extra_metrics={k: v for k, v in metrics.items() if k not in {"rmse", "mae", "r2", "spearman", "kendall", "pgr_author_model"}},
        )
        path = store.write_artifact(artifact, "tool_outputs/evaluation_report.json")
        output_paths = {"evaluation_report": str(path)}
        if prediction_path:
            output_paths["predictions"] = str(prediction_path)
        return {
            "_success": bool(result.get("success")),
            "_error_type": result.get("error_type"),
            "_error_message": result.get("failure_reason") or result.get("error"),
            "output_artifact_ids": [artifact.artifact_id],
            "output_paths": output_paths,
            "metrics": _json_ready(metrics),
            "n_eval": result.get("n_eval"),
            "prediction_path": str(prediction_path) if prediction_path else None,
            "failure_reason": result.get("failure_reason") or result.get("error"),
            "traceback": result.get("traceback"),
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
