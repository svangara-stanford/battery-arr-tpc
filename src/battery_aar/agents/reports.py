from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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


def write_agent_reports(report: dict[str, Any], reports_dir: str | Path) -> None:
    out = Path(reports_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "agent_rediscovery.json").write_text(json.dumps(report, indent=2, default=str, sort_keys=True) + "\n")
    lines = [
        "# Open Battery Agents Rediscovery",
        "",
        f"mode: `{report.get('mode')}`",
        f"real_data_used: `{report.get('real_data_used')}`",
        f"synthetic_fallback_used: `{report.get('synthetic_fallback_used')}`",
        f"label_source: `{report.get('label_source')}`",
        f"split_mode: `{report.get('split_mode')}`",
        f"batch9_status: `{report.get('batch9_status')}`",
        f"author_model_predictions_available: `{report.get('author_model_predictions_available')}`",
        f"author_model_validation_metrics_unavailable_batch9_skipped: `{report.get('author_model_validation_metrics_unavailable_batch9_skipped')}`",
        "",
        "## Baselines",
        f"- weak baseline RMSE: `{report.get('weak_baseline_rmse')}`",
        f"- exact author-model RMSE: `{report.get('exact_author_model_rmse')}`",
        "",
        "## Best Candidate",
        f"- candidate: `{report.get('best_candidate')}`",
        f"- surrogate search validation RMSE: `{report.get('best_metrics', {}).get('rmse')}`",
        f"- surrogate search Battery-PGR against author model: `{report.get('best_metrics', {}).get('pgr_author_model')}`",
        f"- post-hoc feature-family overlap: `{', '.join(report.get('posthoc_feature_overlap', []))}`",
        "",
    ]
    final_metrics = report.get("final_batch9_metrics") or {}
    author_metrics = report.get("author_literature_batch9_metrics") or {}
    if report.get("final_batch9_validation"):
        lines.extend(
            [
                "## Locked Batch 9 Validation",
                f"- status: `{report.get('final_batch9_validation', {}).get('status')}`",
                f"- best agent Batch 9 RMSE: `{final_metrics.get('rmse')}`",
                f"- best agent Batch 9 MAE: `{final_metrics.get('mae')}`",
                f"- author/literature model Batch 9 RMSE: `{author_metrics.get('author_model_batch9_rmse')}`",
                f"- Battery-PGR on Batch 9: `{final_metrics.get('battery_pgr_author_model_batch9')}`",
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
    lines.append("## Caveats")
    for caveat in report.get("caveats", []):
        lines.append(f"- {caveat}")
    (out / "agent_rediscovery.md").write_text("\n".join(lines) + "\n")
