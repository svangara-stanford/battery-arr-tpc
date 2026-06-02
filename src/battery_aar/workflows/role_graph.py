from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from battery_aar.agents.attia_data_bridge import load_author_validation_metrics, load_batch9_holdout
from battery_aar.agents.evaluator import battery_pgr, evaluate_candidate_train_test, regression_metrics
from battery_aar.agents.orchestrator import make_search_split, weak_baseline_rmse_against
from battery_aar.tools.client import HTTPToolClient, NativeToolClient
from battery_aar.workflows.artifacts import ArtifactStore, build_split_artifact
from battery_aar.workflows.roles import (
    ROLE_SEQUENCE,
    CodeGenerator,
    CodeReviewer,
    DatasetProfiler,
    Evaluator,
    FeatureScientist,
    ModelArchitect,
    RoleContext,
    ScientistCritic,
)
from battery_aar.workflows.schemas import AgentRole, ExperimentState, RunManifest
from battery_aar.workflows.trace import TraceLogger


@dataclass
class RoleGraphConfig:
    processed_dir: Path
    reference_run: Path | None
    out: Path
    reports_dir: Path
    split_mode: str = "random"
    validation_fraction: float = 0.25
    split_seed: int = 0
    validation_batch_id: str | None = None
    offline: bool = True
    iterations: int = 1
    use_http_tools: bool = False
    tool_server_url: str | None = None
    model: str | None = None
    max_cycle: int = 100
    allow_protocol_features: bool = False
    allow_freeform_code: bool = False
    candidates_per_iteration: int = 1
    final_batch9_validation: bool = False
    battery_fast_charging_root: Path | None = None
    batch9_path: Path | None = None
    final_batch9_top_k: int = 0
    feature_program_paths: list[Path] | None = None
    batch9_feature_program_path: Path | None = None
    include_feature_programs: bool = False
    feature_program_mode: str = "none"
    feature_program_recipe: str | None = None
    feature_family_filter: list[str] | None = None
    cycle_early_index: int = 9
    cycle_late_index: int = 99


def _load_processed(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path | None]:
    metadata_path = processed_dir / "cell_metadata.csv"
    if not metadata_path.exists():
        metadata_path = processed_dir / "metadata.csv"
    cycle_path = processed_dir / "cycle_summary.csv"
    labels_path = processed_dir / "labels.csv"
    if not metadata_path.exists() or not cycle_path.exists():
        raise FileNotFoundError(f"processed dataset requires metadata/cell_metadata and cycle_summary CSVs: {processed_dir}")
    metadata = pd.read_csv(metadata_path)
    cycles = pd.read_csv(cycle_path)
    if labels_path.exists():
        labels = pd.read_csv(labels_path)
        if "y" not in labels.columns and "cycle_life" in labels.columns:
            labels = labels.rename(columns={"cycle_life": "y"})
        return metadata, cycles, labels, labels_path
    if "cycle_life" not in metadata.columns:
        raise ValueError("processed metadata must contain cycle_life or processed_dir must contain labels.csv")
    labels = metadata[["row_id", "cycle_life"]].rename(columns={"cycle_life": "y"}).copy()
    return metadata, cycles, labels, None


def _metadata_path(processed_dir: Path) -> Path:
    path = processed_dir / "cell_metadata.csv"
    return path if path.exists() else processed_dir / "metadata.csv"


def _finite_labels(labels: pd.DataFrame) -> pd.DataFrame:
    out = labels.copy()
    if "y" not in out.columns and "cycle_life" in out.columns:
        out = out.rename(columns={"cycle_life": "y"})
    out["row_id"] = pd.to_numeric(out["row_id"], errors="raise").astype(int)
    out["y"] = pd.to_numeric(out["y"], errors="coerce")
    return out[np.isfinite(out["y"])].copy()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def _resolve_feature_table_path(path: Path) -> Path:
    return path / "feature_table.csv" if path.is_dir() else path


def _resolve_feature_metadata_path(table_path: Path) -> Path:
    if table_path.name == "feature_table.csv":
        return table_path.with_name("feature_metadata.csv")
    candidate = table_path.with_name(table_path.stem + "_metadata.csv")
    return candidate if candidate.exists() else table_path.with_name("feature_metadata.csv")


def _load_feature_metadata_for_report(feature_program_paths: list[Path] | None) -> dict[str, Any]:
    paths = [Path(path) for path in (feature_program_paths or [])]
    if not paths:
        return {
            "feature_programs_used": [],
            "feature_family_counts": {},
            "true_raw_curve_features_used": False,
            "proxy_features_used": False,
            "protocol_features_used": False,
            "n_feature_program_columns": 0,
        }
    metadata_parts: list[pd.DataFrame] = []
    feature_columns: set[str] = set()
    for raw_path in paths:
        table_path = _resolve_feature_table_path(raw_path)
        if table_path.exists():
            try:
                table = pd.read_csv(table_path, nrows=1)
                feature_columns.update(col for col in table.columns if col not in {"row_id", "cell_id"})
            except Exception:
                pass
        meta_path = _resolve_feature_metadata_path(table_path)
        if meta_path.exists():
            try:
                metadata_parts.append(pd.read_csv(meta_path))
            except Exception:
                pass
    if not metadata_parts:
        return {
            "feature_programs_used": [str(path) for path in paths],
            "feature_family_counts": {},
            "true_raw_curve_features_used": False,
            "proxy_features_used": False,
            "protocol_features_used": False,
            "n_feature_program_columns": len(feature_columns),
        }
    metadata = pd.concat(metadata_parts, ignore_index=True)
    if "feature_family" not in metadata.columns and "family" in metadata.columns:
        metadata["feature_family"] = metadata["family"]
    family_counts = (
        {str(k): int(v) for k, v in metadata["feature_family"].astype(str).value_counts().to_dict().items()}
        if "feature_family" in metadata.columns
        else {}
    )
    return {
        "feature_programs_used": [str(path) for path in paths],
        "feature_family_counts": family_counts,
        "true_raw_curve_features_used": bool(metadata.get("is_true_curve_feature", pd.Series(dtype=bool)).fillna(False).astype(bool).any()),
        "proxy_features_used": bool(metadata.get("is_proxy_feature", pd.Series(dtype=bool)).fillna(False).astype(bool).any()),
        "protocol_features_used": bool(metadata.get("uses_protocol", pd.Series(dtype=bool)).fillna(False).astype(bool).any()),
        "n_feature_program_columns": int(len(set(metadata.get("feature_name", pd.Series(dtype=str)).dropna().astype(str))) or len(feature_columns)),
    }


def _combine_locked_feature_program_tables(
    *,
    search_paths: list[Path],
    batch9_path: Path | None,
    out_dir: Path,
    row_offset: int,
) -> list[str]:
    if not search_paths:
        return []
    if batch9_path is None:
        return [str(_resolve_feature_table_path(path)) for path in search_paths]
    batch9_table_path = _resolve_feature_table_path(batch9_path)
    if not batch9_table_path.exists():
        raise FileNotFoundError(f"Batch 9 feature-program table not found: {batch9_table_path}")
    batch9_table = pd.read_csv(batch9_table_path)
    if "row_id" in batch9_table.columns:
        batch9_table = batch9_table.copy()
        batch9_table["row_id"] = pd.to_numeric(batch9_table["row_id"], errors="raise").astype(int) + int(row_offset)
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_paths: list[str] = []
    for idx, search_path in enumerate(search_paths):
        search_table_path = _resolve_feature_table_path(search_path)
        if not search_table_path.exists():
            raise FileNotFoundError(f"search feature-program table not found: {search_table_path}")
        search_table = pd.read_csv(search_table_path)
        combined = pd.concat([search_table, batch9_table], ignore_index=True, sort=False)
        combined_path = out_dir / f"combined_feature_table_{idx:02d}.csv"
        combined.to_csv(combined_path, index=False)

        meta_parts: list[pd.DataFrame] = []
        for table_path in [search_table_path, batch9_table_path]:
            meta_path = _resolve_feature_metadata_path(table_path)
            if meta_path.exists():
                meta_parts.append(pd.read_csv(meta_path))
        if meta_parts:
            pd.concat(meta_parts, ignore_index=True).drop_duplicates("feature_name").to_csv(
                combined_path.with_name("feature_metadata.csv") if idx == 0 else combined_path.with_name(f"combined_feature_table_{idx:02d}_metadata.csv"),
                index=False,
            )
        combined_paths.append(str(combined_path))
    return combined_paths


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _review_needs_repair(verdict: str | None) -> bool:
    return str(verdict or "").lower() in {"fail", "needs_repair"}


def _final_review_status(verdict: str | None, issues: list[str], evaluation_success: bool) -> str:
    normalized = str(verdict or "unknown").lower()
    if normalized == "pass":
        return "pass"
    if evaluation_success and normalized == "needs_attention":
        return "pass_with_warnings" if issues else "pass"
    return normalized


def _candidate_metric_row(record: dict[str, Any]) -> dict[str, Any]:
    candidate = record["candidate"]
    review = record["review"]
    evaluation = record["evaluation"]
    extra = evaluation.extra_metrics or {}
    return {
        "iteration": int(evaluation.iteration if evaluation.iteration is not None else record.get("iteration", -1)),
        "candidate_path": candidate.candidate_path,
        "candidate_id": candidate.candidate_id,
        "model_family": candidate.model_family,
        "target_transform": candidate.target_transform,
        "include_protocol_features": candidate.include_protocol_features,
        "compiled_candidate": candidate.compiled_candidate,
        "feature_set": candidate.feature_set,
        "feature_program_count": len(candidate.feature_program_paths or []),
        "rmse": evaluation.rmse,
        "mae": evaluation.mae,
        "r2": evaluation.r2,
        "spearman": evaluation.spearman,
        "kendall": evaluation.kendall,
        "prediction_path": evaluation.prediction_path,
        "y_pred_mean": extra.get("y_pred_mean"),
        "y_true_mean": extra.get("y_true_mean"),
        "n_predictions": extra.get("n_predictions"),
        "n_negative_predictions": extra.get("n_negative_predictions"),
        "n_nonfinite_predictions": extra.get("n_nonfinite_predictions"),
        "review_verdict": review.verdict,
        "review_status": _final_review_status(review.verdict, review.issues, bool(evaluation.success)),
        "success": bool(evaluation.success),
    }


def _best_successful_record(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    successful = [record for record in records if record["evaluation"].success and record["evaluation"].rmse is not None]
    if not successful:
        return None
    return min(successful, key=lambda record: float(record["evaluation"].rmse))


def _best_metric_rows_by(rows: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not row.get("success") or row.get("rmse") is None:
            continue
        key = str(row.get(group_key) or "unknown")
        if key not in best or float(row["rmse"]) < float(best[key]["rmse"]):
            best[key] = row
    return [best[key] for key in sorted(best)]


def _prediction_diagnostics(predictions: pd.DataFrame) -> dict[str, Any]:
    if predictions.empty:
        return {"n_predictions": 0, "n_negative_predictions": 0, "n_nonfinite_predictions": 0}
    y_true = pd.to_numeric(predictions["y_true"], errors="coerce").to_numpy(float)
    y_pred = pd.to_numeric(predictions["y_pred"], errors="coerce").to_numpy(float)
    residual = y_pred - y_true
    finite_true = np.isfinite(y_true)
    finite_pred = np.isfinite(y_pred)
    finite_residual = np.isfinite(residual)
    return {
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


def _locked_prediction_frame(predictions: pd.DataFrame, holdout_meta: pd.DataFrame) -> pd.DataFrame:
    output = predictions.rename(columns={"y": "y_true"}).copy()
    output["residual"] = pd.to_numeric(output["y_pred"], errors="coerce") - pd.to_numeric(output["y_true"], errors="coerce")
    meta_cols = ["row_id"]
    for col in ["cell_id", "protocol_readable", "C1", "C2", "C3", "C4"]:
        if col in holdout_meta.columns:
            meta_cols.append(col)
    output = output.merge(holdout_meta[meta_cols].drop_duplicates("row_id"), on="row_id", how="left")
    ordered = ["row_id", "cell_id", "y_true", "y_pred", "residual", "protocol_readable", "C1", "C2", "C3", "C4"]
    return output[[col for col in ordered if col in output.columns]]


def _write_report(report_path: Path, report: dict[str, Any]) -> None:
    tool_calls = report.get("tool_calls", [])
    metrics = report.get("validation_metrics") or {}
    candidate_spec = report.get("candidate_spec") or {}
    per_iteration_rows = report.get("per_iteration_metrics", [])
    locked = report.get("locked_batch9_validation") or {"status": "not_run"}
    locked_metrics = report.get("final_batch9_metrics") or {}
    topk_rows = report.get("final_batch9_topk_rows") or []
    feature_program_report = report.get("feature_program_report") or {}
    best_by_feature_set = report.get("best_by_feature_set") or []
    best_by_target_transform = report.get("best_by_target_transform") or []
    lines = [
        "# Open Battery Agents Role Workflow",
        "",
        f"run_id: `{report.get('run_id')}`",
        f"split_mode: `{report.get('split_mode')}`",
        f"offline: `{report.get('offline')}`",
        f"iterations: `{report.get('iterations')}`",
        f"candidates_per_iteration: `{report.get('candidates_per_iteration')}`",
        "",
        "## Role Sequence",
        "",
        ", ".join(report.get("role_sequence", [])),
        "",
        "## Split",
        "",
        f"- train cells: {report.get('n_train_cells')}",
        f"- validation cells: {report.get('n_validation_cells')}",
        f"- validation fraction: {report.get('validation_fraction')}",
        f"- split seed: {report.get('split_seed')}",
        "",
        "Batch 9 was not used during surrogate search.",
        "",
        "## Feature Programs",
        "",
        f"- include feature programs: `{report.get('include_feature_programs')}`",
        f"- feature program mode: `{report.get('feature_program_mode')}`",
        f"- recipe hint: `{report.get('feature_program_recipe')}`",
        f"- feature programs used: {len(feature_program_report.get('feature_programs_used', []))}",
        f"- feature-program columns: {feature_program_report.get('n_feature_program_columns')}",
        f"- feature-family counts: `{feature_program_report.get('feature_family_counts', {})}`",
        f"- true raw curve features used: `{feature_program_report.get('true_raw_curve_features_used')}`",
        f"- proxy features used: `{feature_program_report.get('proxy_features_used')}`",
        f"- protocol features used: `{feature_program_report.get('protocol_features_used')}`",
        "",
        "## Tool Calls",
        "",
    ]
    if tool_calls:
        for call in tool_calls:
            lines.append(f"- {call.get('tool_name')} success={call.get('success')} duration_ms={call.get('duration_ms')}")
    else:
        lines.append("- none recorded")
    lines.extend(
        [
            "",
            "## Candidate",
            "",
            f"- best candidate id: `{report.get('best_candidate_id')}`",
            f"- best candidate path: `{report.get('candidate_path')}`",
            f"- best iteration: `{report.get('best_iteration')}`",
        f"- final iteration candidate path: `{report.get('final_iteration_candidate_path')}`",
        f"- compiled candidate: `{candidate_spec.get('compiled_candidate')}`",
        f"- model family: `{candidate_spec.get('model_family')}`",
        f"- target transform: `{candidate_spec.get('target_transform')}`",
        f"- feature families: `{', '.join(candidate_spec.get('feature_families', []))}`",
        f"- include protocol features: `{candidate_spec.get('include_protocol_features')}`",
        f"- preprocessing: `{', '.join(candidate_spec.get('preprocessing', []))}`",
        f"- best review status: `{report.get('best_review_status')}`",
        f"- review verdict: `{report.get('review_verdict')}`",
            "",
            "## Per-Iteration Candidates",
            "",
            "| iteration | candidate_id | model_family | feature_set | target_transform | include_protocol_features | success | RMSE | MAE | R2 | y_pred_mean | y_true_mean | prediction_path |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    if per_iteration_rows:
        for row in per_iteration_rows:
            lines.append(
                "| {iteration} | {candidate_id} | {model_family} | {feature_set} | {target_transform} | {include_protocol_features} | {success} | {rmse} | {mae} | {r2} | {y_pred_mean} | {y_true_mean} | `{prediction_path}` |".format(
                    **row
                )
            )
    else:
        lines.append("| | | | | | | | | | | | |")
    lines.extend(
        [
            "",
            "### Best By Feature Set",
            "",
            "| feature_set | candidate_id | model_family | target_transform | RMSE | MAE | R2 |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    if best_by_feature_set:
        for row in best_by_feature_set:
            lines.append(
                "| {feature_set} | {candidate_id} | {model_family} | {target_transform} | {rmse} | {mae} | {r2} |".format(**row)
            )
    else:
        lines.append("| | | | | | | |")
    lines.extend(
        [
            "",
            "### Best By Target Transform",
            "",
            "| target_transform | candidate_id | feature_set | model_family | RMSE | MAE | R2 |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    if best_by_target_transform:
        for row in best_by_target_transform:
            lines.append(
                "| {target_transform} | {candidate_id} | {feature_set} | {model_family} | {rmse} | {mae} | {r2} |".format(**row)
            )
    else:
        lines.append("| | | | | | | |")
    lines.extend(
        [
            "",
            "## Validation Metrics",
            "",
            f"- rmse: {metrics.get('rmse')}",
            f"- mae: {metrics.get('mae')}",
            f"- r2: {metrics.get('r2')}",
            f"- spearman: {metrics.get('spearman')}",
            f"- kendall: {metrics.get('kendall')}",
            "",
            "## Prediction Diagnostics",
            "",
            f"- y_true_mean: {metrics.get('extra_metrics', {}).get('y_true_mean')}",
            f"- y_pred_mean: {metrics.get('extra_metrics', {}).get('y_pred_mean')}",
            f"- y_true_min/max: {metrics.get('extra_metrics', {}).get('y_true_min')} / {metrics.get('extra_metrics', {}).get('y_true_max')}",
            f"- y_pred_min/max: {metrics.get('extra_metrics', {}).get('y_pred_min')} / {metrics.get('extra_metrics', {}).get('y_pred_max')}",
            f"- residual_mean: {metrics.get('extra_metrics', {}).get('residual_mean')}",
            f"- residual_std: {metrics.get('extra_metrics', {}).get('residual_std')}",
            f"- n_predictions: {metrics.get('extra_metrics', {}).get('n_predictions')}",
            f"- n_negative_predictions: {metrics.get('extra_metrics', {}).get('n_negative_predictions')}",
            f"- n_nonfinite_predictions: {metrics.get('extra_metrics', {}).get('n_nonfinite_predictions')}",
            "",
            "## Locked Validation Batch",
            "",
            f"- status: `{locked.get('status', 'not_run')}`",
            f"- role-agent surrogate validation RMSE: `{metrics.get('rmse')}`",
            f"- role-agent locked Batch 9 RMSE: `{locked_metrics.get('rmse')}`",
            f"- Batch 9 weak baseline RMSE: `{locked_metrics.get('batch9_weak_baseline_rmse')}`",
            f"- author/literature Batch 9 RMSE: `{locked_metrics.get('author_model_batch9_rmse')}`",
            f"- Battery-PGR on Batch 9: `{locked_metrics.get('battery_pgr_author_model_batch9')}`",
            f"- predictions: `{locked.get('predictions_path')}`",
            f"- metrics: `{locked.get('metrics_path')}`",
            "",
            "Batch 9 was used only after search for locked final validation.",
        "",
        ]
    )
    if topk_rows:
        lines.extend(
            [
                "",
                "### Locked Batch 9 Top-K",
                "",
                "| rank | candidate_id | surrogate_rmse | batch9_rmse | batch9_mae | batch9_pgr | predictions_path |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in topk_rows:
            lines.append(
                "| {rank_by_surrogate_rmse} | {candidate_id} | {surrogate_rmse} | {batch9_rmse} | {batch9_mae} | {batch9_pgr} | `{predictions_path}` |".format(
                    **row
                )
            )
    lines.extend(
        [
            "",
            "## Critique",
            "",
            report.get("critique_summary") or "No critique summary available.",
            "",
            "## Artifacts",
            "",
            f"- artifact index: `{report.get('artifact_index_path')}`",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")


class RoleGraphRunner:
    def __init__(self, config: RoleGraphConfig):
        self.config = config
        self.run_id = config.out.name
        self.store = ArtifactStore(config.out, run_id=self.run_id)
        self.trace = TraceLogger(self.store.artifact_dir, run_id=self.run_id)
        if config.use_http_tools:
            self.tool_client = HTTPToolClient(config.tool_server_url or "http://127.0.0.1:8000")
        else:
            self.tool_client = NativeToolClient()

    def run(self) -> dict[str, Any]:
        cfg = self.config
        cfg.out.mkdir(parents=True, exist_ok=True)
        cfg.reports_dir.mkdir(parents=True, exist_ok=True)
        self.trace.log_event(
            event_type="run_started",
            agent_role=AgentRole.ORCHESTRATOR,
            success=True,
            extra={"workflow": "role_graph"},
        )
        manifest = RunManifest(
            run_id=self.run_id,
            output_dir=str(cfg.out),
            human_readable_summary="Open Battery Agents role-specialized workflow run.",
            config={
                "processed_dir": str(cfg.processed_dir),
                "reference_run": str(cfg.reference_run) if cfg.reference_run else None,
                "split_mode": cfg.split_mode,
                "validation_fraction": cfg.validation_fraction,
                "split_seed": cfg.split_seed,
                "validation_batch_id": cfg.validation_batch_id,
                "offline": cfg.offline,
                "iterations": cfg.iterations,
                "use_http_tools": cfg.use_http_tools,
                "tool_server_url": cfg.tool_server_url,
                "max_cycle": cfg.max_cycle,
                "allow_protocol_features": cfg.allow_protocol_features,
                "allow_freeform_code": cfg.allow_freeform_code,
                "candidates_per_iteration": cfg.candidates_per_iteration,
                "final_batch9_validation": cfg.final_batch9_validation,
                "battery_fast_charging_root": str(cfg.battery_fast_charging_root) if cfg.battery_fast_charging_root else None,
                "batch9_path": str(cfg.batch9_path) if cfg.batch9_path else None,
                "final_batch9_top_k": cfg.final_batch9_top_k,
                "feature_program_paths": [str(path) for path in (cfg.feature_program_paths or [])],
                "batch9_feature_program_path": str(cfg.batch9_feature_program_path) if cfg.batch9_feature_program_path else None,
                "include_feature_programs": cfg.include_feature_programs,
                "feature_program_mode": cfg.feature_program_mode,
                "feature_program_recipe": cfg.feature_program_recipe,
                "feature_family_filter": list(cfg.feature_family_filter or []),
                "cycle_early_index": cfg.cycle_early_index,
                "cycle_late_index": cfg.cycle_late_index,
            },
            tags=["role_graph", "open_battery_agents_v2"],
        )
        self.store.write_artifact(manifest)

        metadata, _cycles, labels, labels_path = _load_processed(cfg.processed_dir)
        labels = _finite_labels(labels)
        train_ids, val_ids, _test_ids, split_manifest, assignments = make_search_split(
            metadata,
            labels,
            split_mode=cfg.split_mode,
            validation_fraction=cfg.validation_fraction,
            split_seed=cfg.split_seed,
            validation_batch_id=cfg.validation_batch_id,
        )
        split_assignments_path = cfg.out / "split_assignments.csv"
        split_manifest_path = cfg.out / "split_manifest.json"
        assignments.to_csv(split_assignments_path, index=False)
        _write_json(split_manifest_path, split_manifest)
        split_artifact = build_split_artifact(self.run_id, split_manifest, split_assignments_path, parent_artifact_ids=[manifest.artifact_id])
        self.store.write_artifact(split_artifact)

        ctx = RoleContext(
            run_id=self.run_id,
            run_dir=cfg.out,
            processed_dir=cfg.processed_dir,
            metadata_path=_metadata_path(cfg.processed_dir),
            cycle_summary_path=cfg.processed_dir / "cycle_summary.csv",
            labels_path=labels_path,
            split_assignments_path=split_assignments_path,
            split_mode=cfg.split_mode,
            max_cycle=cfg.max_cycle,
            allow_protocol_features=cfg.allow_protocol_features,
            offline=cfg.offline,
            model=cfg.model,
            store=self.store,
            trace=self.trace,
            tool_client=self.tool_client,
            allow_freeform_code=cfg.allow_freeform_code,
            candidates_per_iteration=cfg.candidates_per_iteration,
            feature_program_paths=[str(path) for path in (cfg.feature_program_paths or [])],
            feature_program_mode=cfg.feature_program_mode,
            include_feature_programs=cfg.include_feature_programs,
            feature_family_filter=list(cfg.feature_family_filter or []),
            feature_program_recipe=cfg.feature_program_recipe,
        )

        profile, profile_artifact_ids = DatasetProfiler().run(ctx, parent_ids=[manifest.artifact_id])
        parent_ids = [manifest.artifact_id, split_artifact.artifact_id, *profile_artifact_ids]
        final_candidate = None
        final_review = None
        final_eval = None
        final_critique = None
        eval_reports = []
        candidate_specs = []
        candidate_records: list[dict[str, Any]] = []
        per_iteration_records: list[dict[str, Any]] = []
        for iteration in range(int(cfg.iterations)):
            feature_plan = FeatureScientist().run(ctx, profile, parent_ids, iteration)
            model_plan = ModelArchitect().run(ctx, feature_plan, profile, iteration)
            code_generator = CodeGenerator()
            generated_candidates = code_generator.run_many(ctx, feature_plan, model_plan, iteration)
            iteration_records: list[dict[str, Any]] = []
            for candidate in generated_candidates:
                candidate_specs.append(candidate)
                review = CodeReviewer().run(ctx, candidate, iteration)
                repaired = False
                if _review_needs_repair(review.verdict):
                    candidate = code_generator.repair(
                        ctx,
                        feature_plan,
                        model_plan,
                        candidate,
                        error_message="; ".join(review.issues) or f"review verdict={review.verdict}",
                        traceback_text="\n".join(item for item in review.recommendations if "traceback" in item.lower()) or None,
                        iteration=iteration,
                    )
                    candidate_specs.append(candidate)
                    review = CodeReviewer().run(ctx, candidate, iteration)
                    repaired = True
                evaluation = Evaluator().run(ctx, candidate, review, iteration)
                eval_reports.append(evaluation)
                current_record = {"iteration": iteration, "candidate": candidate, "review": review, "evaluation": evaluation}
                candidate_records.append(current_record)
                if not evaluation.success and not repaired:
                    candidate = code_generator.repair(
                        ctx,
                        feature_plan,
                        model_plan,
                        candidate,
                        error_message=evaluation.failure_reason or "evaluation failed",
                        traceback_text=evaluation.traceback,
                        iteration=iteration,
                    )
                    candidate_specs.append(candidate)
                    review = CodeReviewer().run(ctx, candidate, iteration)
                    evaluation = Evaluator().run(ctx, candidate, review, iteration)
                    eval_reports.append(evaluation)
                    current_record = {"iteration": iteration, "candidate": candidate, "review": review, "evaluation": evaluation}
                    candidate_records.append(current_record)
                iteration_records.append(current_record)
                final_candidate = candidate
                final_review = review
                final_eval = evaluation
            if iteration_records:
                best_iteration_record = _best_successful_record(iteration_records) or iteration_records[-1]
                critique = ScientistCritic().run(ctx, best_iteration_record["review"], best_iteration_record["evaluation"], iteration)
                per_iteration_records.extend(iteration_records)
            else:
                critique = None
            final_critique = critique

        best_record = _best_successful_record(candidate_records)
        if best_record is None and per_iteration_records:
            best_record = per_iteration_records[-1]
        best_eval = best_record["evaluation"] if best_record else None
        best_candidate = best_record["candidate"] if best_record else final_candidate
        best_review = best_record["review"] if best_record else final_review
        successful = [record for record in candidate_records if record["evaluation"].success and record["evaluation"].rmse is not None]
        all_candidate_metrics = [_candidate_metric_row(record) for record in candidate_records]
        per_iteration_metrics = [_candidate_metric_row(record) for record in per_iteration_records]
        best_by_feature_set = _best_metric_rows_by(all_candidate_metrics, "feature_set")
        best_by_target_transform = _best_metric_rows_by(all_candidate_metrics, "target_transform")
        author_validation = load_author_validation_metrics(cfg.reference_run)
        locked_batch9 = {"status": "not_run"}
        final_batch9_metrics: dict[str, Any] | None = None
        final_batch9_predictions_path: Path | None = None
        final_batch9_metrics_path: Path | None = None
        final_batch9_topk_metrics_path: Path | None = None
        final_batch9_topk_rows: list[dict[str, Any]] = []
        best_locked_validation_candidate_path: str | None = None
        if cfg.final_batch9_validation or int(cfg.final_batch9_top_k) > 0:
            locked_batch9 = {"status": "not_run"}
            try:
                if best_candidate is None or best_eval is None or not best_eval.success:
                    raise RuntimeError("No successful candidate is available for locked Batch 9 validation")
                if cfg.battery_fast_charging_root is None and cfg.batch9_path is None:
                    raise ValueError("locked Batch 9 validation requires --battery-fast-charging-root or --batch9-path")
                holdout_meta, holdout_cycles, holdout_labels = load_batch9_holdout(
                    battery_fast_charging_root=cfg.battery_fast_charging_root,
                    batch9_path=cfg.batch9_path,
                    first_n_cycles=cfg.max_cycle,
                )
                row_offset = int(pd.to_numeric(metadata["row_id"], errors="coerce").max()) + 1
                holdout_meta = holdout_meta.copy()
                holdout_cycles = holdout_cycles.copy()
                holdout_labels = holdout_labels.copy()
                holdout_meta["row_id"] = pd.to_numeric(holdout_meta["row_id"], errors="raise").astype(int) + row_offset
                holdout_cycles["row_id"] = pd.to_numeric(holdout_cycles["row_id"], errors="raise").astype(int) + row_offset
                holdout_labels["row_id"] = pd.to_numeric(holdout_labels["row_id"], errors="raise").astype(int) + row_offset
                batch9_weak_rmse = weak_baseline_rmse_against(labels, holdout_labels)
                author_rmse = author_validation.get("author_model_batch9_rmse")
                locked_feature_program_paths = [str(path) for path in (cfg.feature_program_paths or [])]
                if cfg.include_feature_programs and cfg.feature_program_paths:
                    locked_feature_program_paths = _combine_locked_feature_program_tables(
                        search_paths=list(cfg.feature_program_paths),
                        batch9_path=cfg.batch9_feature_program_path,
                        out_dir=cfg.out / "locked_batch9_feature_programs",
                        row_offset=row_offset,
                    )

                def evaluate_locked_candidate(record: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
                    candidate = record["candidate"]
                    result = evaluate_candidate_train_test(
                        candidate.candidate_path,
                        metadata,
                        _cycles,
                        labels,
                        holdout_meta,
                        holdout_cycles,
                        holdout_labels,
                        weak_rmse=batch9_weak_rmse,
                        strong_rmse=author_rmse,
                        allow_protocol_features=cfg.allow_protocol_features,
                        max_cycle=cfg.max_cycle,
                        return_predictions=True,
                        feature_program_paths=locked_feature_program_paths,
                        feature_program_mode=cfg.feature_program_mode,
                        include_feature_programs=cfg.include_feature_programs,
                        feature_family_filter=cfg.feature_family_filter or [],
                    )
                    if not result.get("success"):
                        raise RuntimeError(result.get("failure_reason") or result.get("error") or "Batch 9 candidate evaluation failed")
                    predictions = _locked_prediction_frame(result["predictions"], holdout_meta)
                    metrics = dict(result.get("metrics") or {})
                    metrics.update(_prediction_diagnostics(predictions))
                    metrics["batch9_weak_baseline_rmse"] = batch9_weak_rmse
                    metrics["author_model_batch9_rmse"] = author_rmse
                    metrics["author_model_batch9_mae"] = author_validation.get("author_model_batch9_mae")
                    metrics["battery_pgr_author_model_batch9"] = battery_pgr(batch9_weak_rmse, metrics.get("rmse"), author_rmse)
                    metrics["candidate_id"] = candidate.candidate_id
                    metrics["candidate_path"] = candidate.candidate_path
                    self.trace.log_event(
                        event_type="locked_batch9_candidate_evaluated",
                        agent_role=AgentRole.EVALUATOR,
                        input_artifact_ids=[candidate.artifact_id],
                        success=True,
                        extra={"candidate_id": candidate.candidate_id, "rmse": metrics.get("rmse")},
                    )
                    return metrics, predictions

                if cfg.final_batch9_validation:
                    final_batch9_metrics, final_predictions = evaluate_locked_candidate(best_record)
                    final_batch9_predictions_path = cfg.out / "final_batch9_predictions.csv"
                    final_batch9_metrics_path = cfg.out / "final_batch9_metrics.json"
                    final_predictions.to_csv(final_batch9_predictions_path, index=False)
                    _write_json(final_batch9_metrics_path, final_batch9_metrics)
                    best_locked_validation_candidate_path = best_candidate.candidate_path
                    locked_batch9 = {
                        "status": "ok",
                        "n_cells": int(len(holdout_labels)),
                        "predictions_path": str(final_batch9_predictions_path),
                        "metrics_path": str(final_batch9_metrics_path),
                    }
                if int(cfg.final_batch9_top_k) > 0:
                    topk_dir = cfg.out / "final_batch9_topk_predictions"
                    topk_dir.mkdir(parents=True, exist_ok=True)
                    top_records = sorted(successful, key=lambda record: float(record["evaluation"].rmse))[: int(cfg.final_batch9_top_k)]
                    for rank, record in enumerate(top_records, start=1):
                        metrics, predictions = evaluate_locked_candidate(record)
                        candidate = record["candidate"]
                        predictions_path = topk_dir / f"rank_{rank:02d}_{candidate.candidate_id}.csv"
                        predictions.to_csv(predictions_path, index=False)
                        final_batch9_topk_rows.append(
                            {
                                "rank_by_surrogate_rmse": rank,
                                "candidate_id": candidate.candidate_id,
                                "candidate_path": candidate.candidate_path,
                                "surrogate_rmse": record["evaluation"].rmse,
                                "batch9_rmse": metrics.get("rmse"),
                                "batch9_mae": metrics.get("mae"),
                                "batch9_pgr": metrics.get("battery_pgr_author_model_batch9"),
                                "predictions_path": str(predictions_path),
                            }
                        )
                    final_batch9_topk_metrics_path = cfg.out / "final_batch9_topk_metrics.csv"
                    pd.DataFrame(final_batch9_topk_rows).to_csv(final_batch9_topk_metrics_path, index=False)
                if not cfg.final_batch9_validation:
                    locked_batch9 = {"status": "ok", "top_k_only": int(cfg.final_batch9_top_k)}
            except Exception as exc:
                locked_batch9 = {"status": "failed", "error": str(exc)}
                self.trace.log_event(
                    event_type="locked_batch9_validation_failed",
                    agent_role=AgentRole.EVALUATOR,
                    success=False,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
        state = ExperimentState(
            run_id=self.run_id,
            parent_artifact_ids=[report.artifact_id for report in eval_reports],
            human_readable_summary="Role graph workflow completed.",
            status="complete" if final_eval is not None else "failed",
            completed_iterations=int(cfg.iterations),
            candidate_count=len(candidate_specs),
            successful_candidate_count=len(successful),
            best_candidate_path=best_candidate.candidate_path if best_candidate else None,
            best_candidate_id=best_candidate.candidate_id if best_candidate else None,
            best_iteration=best_eval.iteration if best_eval else None,
            best_metrics=best_eval.model_dump(mode="json") if best_eval else {},
            all_candidate_metrics=all_candidate_metrics,
            final_batch9_metrics_path=str(final_batch9_metrics_path) if final_batch9_metrics_path else None,
            final_batch9_predictions_path=str(final_batch9_predictions_path) if final_batch9_predictions_path else None,
            final_batch9_topk_metrics_path=str(final_batch9_topk_metrics_path) if final_batch9_topk_metrics_path else None,
            best_locked_validation_candidate_path=best_locked_validation_candidate_path,
            artifact_index_path=str(self.store.index_path),
            output_paths={
                "split_assignments": str(split_assignments_path),
                "split_manifest": str(split_manifest_path),
                "final_batch9_metrics": str(final_batch9_metrics_path) if final_batch9_metrics_path else "",
                "final_batch9_predictions": str(final_batch9_predictions_path) if final_batch9_predictions_path else "",
                "final_batch9_topk_metrics": str(final_batch9_topk_metrics_path) if final_batch9_topk_metrics_path else "",
            },
        )
        self.store.write_artifact(state)
        self.trace.log_event(
            event_type="run_completed",
            agent_role=AgentRole.ORCHESTRATOR,
            output_artifact_ids=[state.artifact_id],
            success=True,
            extra={"workflow": "role_graph"},
        )

        tool_calls = _read_jsonl(self.store.artifact_dir / "tool_calls.jsonl")
        feature_program_report = _load_feature_metadata_for_report(cfg.feature_program_paths)
        report = {
            "run_id": self.run_id,
            "out": str(cfg.out),
            "reports_dir": str(cfg.reports_dir),
            "split_mode": cfg.split_mode,
            "validation_fraction": cfg.validation_fraction,
            "split_seed": cfg.split_seed,
            "offline": cfg.offline,
            "iterations": cfg.iterations,
            "role_sequence": ROLE_SEQUENCE,
            "allow_freeform_code": cfg.allow_freeform_code,
            "candidates_per_iteration": cfg.candidates_per_iteration,
            "include_feature_programs": cfg.include_feature_programs,
            "feature_program_mode": cfg.feature_program_mode,
            "feature_program_recipe": cfg.feature_program_recipe,
            "feature_program_paths": [str(path) for path in (cfg.feature_program_paths or [])],
            "feature_family_filter": list(cfg.feature_family_filter or []),
            "cycle_early_index": cfg.cycle_early_index,
            "cycle_late_index": cfg.cycle_late_index,
            "feature_program_report": feature_program_report,
            "n_train_cells": len(train_ids),
            "n_validation_cells": len(val_ids),
            "tool_calls": tool_calls,
            "candidate_path": best_candidate.candidate_path if best_candidate else None,
            "best_candidate_id": best_candidate.candidate_id if best_candidate else None,
            "candidate_spec": best_candidate.model_dump(mode="json") if best_candidate else {},
            "best_iteration": best_eval.iteration if best_eval else None,
            "final_iteration_candidate_path": final_candidate.candidate_path if final_candidate else None,
            "final_iteration_metrics": final_eval.model_dump(mode="json") if final_eval else {},
            "all_candidate_metrics": all_candidate_metrics,
            "per_iteration_metrics": per_iteration_metrics,
            "best_by_feature_set": best_by_feature_set,
            "best_by_target_transform": best_by_target_transform,
            "review_verdict": best_review.verdict if best_review else None,
            "best_review_status": _final_review_status(best_review.verdict if best_review else None, best_review.issues if best_review else [], best_eval.success if best_eval else False),
            "final_review_status": _final_review_status(final_review.verdict if final_review else None, final_review.issues if final_review else [], final_eval.success if final_eval else False),
            "validation_metrics": best_eval.model_dump(mode="json") if best_eval else {},
            "locked_batch9_validation": locked_batch9,
            "final_batch9_metrics": final_batch9_metrics,
            "final_batch9_top_k": cfg.final_batch9_top_k,
            "final_batch9_topk_metrics_path": str(final_batch9_topk_metrics_path) if final_batch9_topk_metrics_path else None,
            "final_batch9_topk_predictions_dir": str(cfg.out / "final_batch9_topk_predictions") if final_batch9_topk_metrics_path else None,
            "final_batch9_topk_rows": final_batch9_topk_rows,
            "locked_batch9_weak_baseline_rmse": final_batch9_metrics.get("batch9_weak_baseline_rmse") if final_batch9_metrics else None,
            "author_literature_batch9_rmse": author_validation.get("author_model_batch9_rmse"),
            "author_literature_batch9_metrics": author_validation,
            "batch9_pgr": final_batch9_metrics.get("battery_pgr_author_model_batch9") if final_batch9_metrics else None,
            "critique_summary": final_critique.human_readable_summary if final_critique else None,
            "artifact_index_path": str(self.store.index_path),
            "split_assignments_path": str(split_assignments_path),
            "split_manifest_path": str(split_manifest_path),
        }
        _write_report(cfg.reports_dir / "role_agent_workflow.md", report)
        _write_json(cfg.reports_dir / "role_agent_workflow.json", report)
        return report


def run_role_workflow(
    *,
    processed_dir: str | Path,
    reference_run: str | Path | None,
    out: str | Path,
    reports_dir: str | Path,
    split_mode: str = "random",
    validation_fraction: float = 0.25,
    split_seed: int = 0,
    validation_batch_id: str | None = None,
    offline: bool = True,
    iterations: int = 1,
    use_http_tools: bool = False,
    tool_server_url: str | None = None,
    model: str | None = None,
    max_cycle: int = 100,
    allow_protocol_features: bool = False,
    allow_freeform_code: bool = False,
    candidates_per_iteration: int = 1,
    final_batch9_validation: bool = False,
    battery_fast_charging_root: str | Path | None = None,
    batch9_path: str | Path | None = None,
    final_batch9_top_k: int = 0,
    feature_program_paths: list[str | Path] | None = None,
    batch9_feature_program_path: str | Path | None = None,
    include_feature_programs: bool = False,
    feature_program_mode: str = "none",
    feature_program_recipe: str | None = None,
    feature_family_filter: list[str] | None = None,
    cycle_early_index: int = 9,
    cycle_late_index: int = 99,
) -> dict[str, Any]:
    config = RoleGraphConfig(
        processed_dir=Path(processed_dir),
        reference_run=Path(reference_run) if reference_run else None,
        out=Path(out),
        reports_dir=Path(reports_dir),
        split_mode=split_mode,
        validation_fraction=validation_fraction,
        split_seed=split_seed,
        validation_batch_id=validation_batch_id,
        offline=offline,
        iterations=iterations,
        use_http_tools=use_http_tools,
        tool_server_url=tool_server_url,
        model=model,
        max_cycle=max_cycle,
        allow_protocol_features=allow_protocol_features,
        allow_freeform_code=allow_freeform_code,
        candidates_per_iteration=candidates_per_iteration,
        final_batch9_validation=final_batch9_validation,
        battery_fast_charging_root=Path(battery_fast_charging_root) if battery_fast_charging_root else None,
        batch9_path=Path(batch9_path) if batch9_path else None,
        final_batch9_top_k=final_batch9_top_k,
        feature_program_paths=[Path(path) for path in (feature_program_paths or [])],
        batch9_feature_program_path=Path(batch9_feature_program_path) if batch9_feature_program_path else None,
        include_feature_programs=include_feature_programs,
        feature_program_mode=feature_program_mode,
        feature_program_recipe=feature_program_recipe,
        feature_family_filter=list(feature_family_filter or []),
        cycle_early_index=cycle_early_index,
        cycle_late_index=cycle_late_index,
    )
    return RoleGraphRunner(config).run()
