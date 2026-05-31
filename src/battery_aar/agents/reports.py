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
        f"- validation RMSE: `{report.get('best_metrics', {}).get('rmse')}`",
        f"- Battery-PGR against author model: `{report.get('best_metrics', {}).get('pgr_author_model')}`",
        f"- post-hoc feature-family overlap: `{', '.join(report.get('posthoc_feature_overlap', []))}`",
        "",
        "## Caveats",
    ]
    for caveat in report.get("caveats", []):
        lines.append(f"- {caveat}")
    (out / "agent_rediscovery.md").write_text("\n".join(lines) + "\n")
