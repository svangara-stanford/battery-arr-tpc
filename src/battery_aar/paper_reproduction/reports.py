from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json_report(report: dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")


def write_markdown_report(report: dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Attia Reference Reproduction",
        "",
        f"status: `{report.get('status')}`",
        f"validation_status: `{report.get('validation_status')}`",
        f"validation_status_label: `{report.get('validation_status_label')}`",
        "",
        "## Model Files",
        f"- oed_model.mat found: `{report.get('oed_model_found')}`",
        f"- oed_model_batch1.mat found: `{report.get('oed_model_batch1_found')}`",
        f"- all required model variables found: `{report.get('all_required_model_variables_found')}`",
        "",
        "## Batch Mapping",
    ]
    for folder, batch_name in (report.get("batch_name_mapping") or {}).items():
        lines.append(f"- `{folder}` -> `{batch_name}`")
    lines.extend(["", "## Batch Results"])
    for item in report.get("batches", []):
        lines.append(
            f"- `{item.get('folder')}`: model `{item.get('model_file')}`, cutoff `{item.get('cutoff_cycle')}`, "
            f"cells `{item.get('cells_parsed')}`, predictions `{item.get('available_predictions')}`, "
            f"unavailable `{item.get('unavailable_features')}`"
        )
    if report.get("bayesgap_top_protocols"):
        lines.extend(["", "## Top BayesGap Recommended Protocols"])
        for item in report["bayesgap_top_protocols"][:10]:
            lines.append(f"- C1={item.get('C1')}, C2={item.get('C2')}, C3={item.get('C3')}, C4={item.get('C4')}")
    lines.extend(["", "## Caveats"])
    for caveat in report.get("caveats", []):
        lines.append(f"- {caveat}")
    p.write_text("\n".join(lines) + "\n")
