from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

FEATURE_FAMILIES = (
    "early discharge capacity",
    "max early capacity change",
    "late-cycle capacity slope",
    "difference between cycle 10 and cycle N curves",
    "log-transformed curve statistics",
    "energy-like integral changes",
)


def posthoc_feature_overlap(candidate_code: str) -> list[str]:
    code = candidate_code.lower()
    overlap: list[str] = []
    if "capacity" in code or "q" in code:
        overlap.append("early discharge capacity")
    if "max_delta" in code or "nanmax" in code:
        overlap.append("max early capacity change")
    if "slope" in code or "polyfit" in code:
        overlap.append("late-cycle capacity slope")
    if "diff" in code:
        overlap.append("difference between cycle 10 and cycle N curves")
    if "log" in code:
        overlap.append("log-transformed curve statistics")
    if "trapz" in code or "integr" in code or "energy" in code:
        overlap.append("energy-like integral changes")
    return sorted(set(overlap))


SUMMARY_COLUMNS = [
    "run_id",
    "split_mode",
    "agents",
    "iterations",
    "best_candidate",
    "surrogate_rmse",
    "batch9_rmse",
    "batch9_weak_rmse",
    "author_rmse",
    "pgr",
]


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _summary_row(report: dict[str, Any]) -> dict[str, Any]:
    final_metrics = report.get("final_batch9_metrics") or {}
    author_metrics = report.get("author_literature_batch9_metrics") or {}
    return {
        "run_id": report.get("run_id"),
        "split_mode": report.get("split_mode"),
        "agents": report.get("agents"),
        "iterations": report.get("iterations"),
        "best_candidate": report.get("best_candidate"),
        "surrogate_rmse": _first_present(report.get("surrogate_search_validation_rmse"), report.get("best_metrics", {}).get("rmse")),
        "batch9_rmse": _first_present(report.get("locked_batch9_rmse"), final_metrics.get("rmse")),
        "batch9_weak_rmse": _first_present(report.get("locked_batch9_weak_baseline_rmse"), final_metrics.get("batch9_weak_baseline_rmse")),
        "author_rmse": _first_present(report.get("author_literature_batch9_rmse"), author_metrics.get("author_model_batch9_rmse")),
        "pgr": _first_present(report.get("batch9_pgr"), final_metrics.get("battery_pgr_author_model_batch9")),
    }


def update_run_summary(report: dict[str, Any], reports_dir: str | Path) -> pd.DataFrame:
    out = Path(reports_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "agent_rediscovery_runs_summary.csv"
    row = _summary_row(report)
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
    else:
        summary = pd.DataFrame(columns=SUMMARY_COLUMNS)
    if "run_id" in summary.columns and row["run_id"] is not None:
        summary = summary[summary["run_id"].astype(str) != str(row["run_id"])]
    if summary.empty:
        summary = pd.DataFrame([row])
    else:
        summary = pd.concat([summary, pd.DataFrame([row])], ignore_index=True)
    for col in SUMMARY_COLUMNS:
        if col not in summary.columns:
            summary[col] = None
    summary = summary[SUMMARY_COLUMNS].sort_values("run_id", na_position="last").reset_index(drop=True)
    summary.to_csv(summary_path, index=False)
    return summary


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> list[str]:
    if df.empty:
        return ["No runs summarized yet."]
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df[columns].iterrows():
        values = []
        for col in columns:
            value = row.get(col)
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                text = str(value)
                values.append(text if len(text) <= 80 else text[:77] + "...")
        rows.append("| " + " | ".join(values) + " |")
    return rows


def write_agent_reports(report: dict[str, Any], reports_dir: str | Path) -> None:
    out = Path(reports_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = update_run_summary(report, out)
    (out / "agent_rediscovery.json").write_text(json.dumps(report, indent=2, default=str, sort_keys=True) + "\n")
    run_id = report.get("run_id")
    if run_id:
        safe_run_id = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(run_id))
        (out / f"agent_rediscovery_{safe_run_id}.json").write_text(json.dumps(report, indent=2, default=str, sort_keys=True) + "\n")
    lines = [
        "# Open Battery Agents Rediscovery",
        "",
        f"run_id: `{report.get('run_id')}`",
        f"mode: `{report.get('mode')}`",
        f"real_data_used: `{report.get('real_data_used')}`",
        f"synthetic_fallback_used: `{report.get('synthetic_fallback_used')}`",
        f"label_source: `{report.get('label_source')}`",
        f"split_mode: `{report.get('split_mode')}`",
        f"validation_fraction: `{report.get('validation_fraction')}`",
        f"split_seed: `{report.get('split_seed')}`",
        f"batch9_status: `{report.get('batch9_status')}`",
        f"author_model_predictions_available: `{report.get('author_model_predictions_available')}`",
        f"author_model_validation_metrics_unavailable_batch9_skipped: `{report.get('author_model_validation_metrics_unavailable_batch9_skipped')}`",
        "",
        "## Search Split",
        f"- split_mode: `{report.get('split_mode')}`",
        f"- validation_fraction: `{report.get('validation_fraction')}`",
        f"- split_seed: `{report.get('split_seed')}`",
        f"- group type: `{(report.get('split_manifest') or {}).get('group_type')}`",
        f"- train cells: `{(report.get('split_manifest') or {}).get('n_train_cells')}`",
        f"- validation cells: `{(report.get('split_manifest') or {}).get('n_validation_cells')}`",
        f"- train groups: `{(report.get('split_manifest') or {}).get('n_train_groups')}`",
        f"- validation groups: `{(report.get('split_manifest') or {}).get('n_validation_groups')}`",
        "Batch 9 was not used during surrogate search; it was only used for locked final validation when requested.",
        "",
        "## Baselines",
        f"- surrogate-search weak baseline RMSE: `{report.get('surrogate_search_weak_baseline_rmse')}`",
        f"- Batch 9 weak baseline RMSE: `{report.get('locked_batch9_weak_baseline_rmse')}`",
        f"- author/literature Batch 9 RMSE: `{report.get('author_literature_batch9_rmse')}`",
        "",
        "## Best Candidate",
        f"- candidate: `{report.get('best_candidate')}`",
        f"- surrogate-search validation RMSE: `{report.get('surrogate_search_validation_rmse')}`",
        f"- surrogate search Battery-PGR against author model: `{report.get('best_metrics', {}).get('pgr_author_model')}`",
        f"- post-hoc feature-family overlap: `{', '.join(report.get('posthoc_feature_overlap', []))}`",
        "",
    ]
    final_metrics = report.get("final_batch9_metrics") or {}
    author_metrics = report.get("author_literature_batch9_metrics") or {}
    split_manifest = report.get("split_manifest") or {}
    if report.get("split_mode") == "protocol":
        heldout = split_manifest.get("heldout_protocols") or []
        lines.extend(
            [
                "## Held-Out Protocols",
                f"- count: `{len(heldout)}`",
                f"- protocols: `{', '.join(map(str, heldout[:20]))}`" + (" ..." if len(heldout) > 20 else ""),
                "",
            ]
        )
    if report.get("split_mode") in {"batch", "leave_one_batch_out"}:
        heldout = split_manifest.get("heldout_batch_ids") or []
        lines.extend(
            [
                "## Held-Out Batches",
                f"- batch IDs: `{', '.join(map(str, heldout))}`",
                "",
            ]
        )
    if report.get("final_batch9_validation"):
        lines.extend(
            [
                "## Locked Batch 9 Validation",
                f"- status: `{report.get('final_batch9_validation', {}).get('status')}`",
                f"- best agent locked Batch 9 RMSE: `{report.get('locked_batch9_rmse')}`",
                f"- best agent Batch 9 MAE: `{final_metrics.get('mae')}`",
                f"- Batch 9 weak baseline RMSE: `{report.get('locked_batch9_weak_baseline_rmse')}`",
                f"- author/literature model Batch 9 RMSE: `{report.get('author_literature_batch9_rmse')}`",
                f"- Battery-PGR on Batch 9: `{report.get('batch9_pgr')}`",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Locked Batch 9 Validation",
                "- not run",
                "",
            ]
        )
    failures = report.get("candidate_failures", [])
    if report.get("best_candidate") is None and failures:
        lines.extend(["## Candidate Failures", ""])
        for failure in failures:
            lines.append(f"- `{failure.get('error_type')}` x{failure.get('count')}: {failure.get('failure_reason')}")
        lines.append("")
    lines.extend(["## Run Comparison", ""])
    lines.extend(_markdown_table(summary, SUMMARY_COLUMNS))
    lines.append("")
    lines.append("## Caveats")
    for caveat in report.get("caveats", []):
        lines.append(f"- {caveat}")
    (out / "agent_rediscovery.md").write_text("\n".join(lines) + "\n")
