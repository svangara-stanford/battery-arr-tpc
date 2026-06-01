from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from battery_aar.agents.orchestrator import make_search_split
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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_report(report_path: Path, report: dict[str, Any]) -> None:
    tool_calls = report.get("tool_calls", [])
    metrics = report.get("validation_metrics") or {}
    lines = [
        "# Open Battery Agents Role Workflow",
        "",
        f"run_id: `{report.get('run_id')}`",
        f"split_mode: `{report.get('split_mode')}`",
        f"offline: `{report.get('offline')}`",
        f"iterations: `{report.get('iterations')}`",
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
            f"- candidate path: `{report.get('candidate_path')}`",
            f"- review verdict: `{report.get('review_verdict')}`",
            "",
            "## Validation Metrics",
            "",
            f"- rmse: {metrics.get('rmse')}",
            f"- mae: {metrics.get('mae')}",
            f"- r2: {metrics.get('r2')}",
            f"- spearman: {metrics.get('spearman')}",
            f"- kendall: {metrics.get('kendall')}",
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
        )

        profile, profile_artifact_ids = DatasetProfiler().run(ctx, parent_ids=[manifest.artifact_id])
        parent_ids = [manifest.artifact_id, split_artifact.artifact_id, *profile_artifact_ids]
        final_candidate = None
        final_review = None
        final_eval = None
        final_critique = None
        eval_reports = []
        for iteration in range(int(cfg.iterations)):
            feature_plan = FeatureScientist().run(ctx, profile, parent_ids, iteration)
            model_plan = ModelArchitect().run(ctx, feature_plan, profile, iteration)
            candidate = CodeGenerator().run(ctx, feature_plan, model_plan, iteration)
            review = CodeReviewer().run(ctx, candidate, iteration)
            evaluation = Evaluator().run(ctx, candidate, review, iteration)
            critique = ScientistCritic().run(ctx, review, evaluation, iteration)
            eval_reports.append(evaluation)
            final_candidate = candidate
            final_review = review
            final_eval = evaluation
            final_critique = critique

        successful = [report for report in eval_reports if report.success and report.rmse is not None]
        best = min(successful, key=lambda report: float(report.rmse)) if successful else final_eval
        state = ExperimentState(
            run_id=self.run_id,
            parent_artifact_ids=[report.artifact_id for report in eval_reports],
            human_readable_summary="Role graph workflow completed.",
            status="complete" if final_eval is not None else "failed",
            completed_iterations=int(cfg.iterations),
            candidate_count=int(cfg.iterations),
            successful_candidate_count=len(successful),
            best_candidate_path=best.candidate_path if best else None,
            best_metrics=best.model_dump(mode="json") if best else {},
            artifact_index_path=str(self.store.index_path),
            output_paths={
                "split_assignments": str(split_assignments_path),
                "split_manifest": str(split_manifest_path),
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
            "n_train_cells": len(train_ids),
            "n_validation_cells": len(val_ids),
            "tool_calls": tool_calls,
            "candidate_path": final_candidate.candidate_path if final_candidate else None,
            "review_verdict": final_review.verdict if final_review else None,
            "validation_metrics": final_eval.model_dump(mode="json") if final_eval else {},
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
    )
    return RoleGraphRunner(config).run()
