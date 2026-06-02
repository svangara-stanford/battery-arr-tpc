#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from battery_aar.features.severson_matr import audit_severson_mat_dir


def _write_markdown(path: Path, audit: Dict[str, Any]) -> None:
    lines = [
        "# Severson MatR Data Audit",
        "",
        f"mat_dir: `{audit.get('mat_dir')}`",
        f"n_files: `{audit.get('n_files')}`",
        "",
        "## Files",
        "",
    ]
    for file_report in audit.get("files", []):
        lines.extend(
            [
                f"### {file_report.get('file_name')}",
                "",
                f"- path: `{file_report.get('path')}`",
                f"- load_method: `{file_report.get('load_method')}`",
                f"- top_level_keys: `{file_report.get('top_level_keys')}`",
                f"- n_cells: `{file_report.get('n_cells')}`",
                f"- true_cycle_life_present: `{file_report.get('true_cycle_life_present')}`",
                f"- cycle_life min/mean/max: `{file_report.get('cycle_life_min')}` / `{file_report.get('cycle_life_mean')}` / `{file_report.get('cycle_life_max')}`",
                f"- cycle_index_conventions: `{file_report.get('cycle_index_conventions')}`",
                f"- cells_with_first_100_cycles: `{file_report.get('cells_with_first_100_cycles')}`",
                f"- summary_fields: `{file_report.get('summary_fields')}`",
                f"- cycle_level_fields: `{file_report.get('cycle_level_fields')}`",
                f"- field_availability: `{file_report.get('field_availability')}`",
                "",
            ]
        )
        warnings = file_report.get("warnings") or []
        if warnings:
            lines.append("Warnings:")
            for warning in warnings:
                lines.append(f"- {warning}")
            lines.append("")
    lines.extend(["## Warnings", ""])
    if audit.get("warnings"):
        for warning in audit["warnings"]:
            lines.append(f"- {warning}")
    else:
        lines.append("- none")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Severson/MatR true cycle-life MATLAB files.")
    parser.add_argument("--mat-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    audit = audit_severson_mat_dir(args.mat_dir)
    md_path = args.out.with_suffix(".md") if args.out.suffix else args.out.parent / f"{args.out.name}.md"
    json_path = args.out.with_suffix(".json") if args.out.suffix else args.out.parent / f"{args.out.name}.json"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    _write_markdown(md_path, audit)
    json_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(f"audit markdown: {md_path}")
    print(f"audit json: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
