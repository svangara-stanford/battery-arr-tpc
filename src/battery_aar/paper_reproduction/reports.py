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
        f"author_model_target_transform_requested: `{report.get('author_model_target_transform_requested')}`",
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
            f"raw files `{item.get('total_raw_cell_files')}`, parsed `{item.get('cells_parsed')}`, "
            f"exact features `{item.get('cells_with_exact_features_available')}`, finite predictions `{item.get('finite_predictions')}`, "
            f"anomalous `{item.get('anomalous_predictions')}`, BayesGap rows `{item.get('bayesgap_rows')}`, "
            f"unavailable `{item.get('unavailable_features')}`"
        )
        if item.get("excluded_cells"):
            lines.append(f"  excluded load-time cells: `{len(item.get('excluded_cells') or [])}`")
            for excluded in item.get("excluded_cells") or []:
                lines.append(
                    f"  - load exclusion cell `{excluded.get('cell_id')}`, channel `{excluded.get('channel')}`: {excluded.get('reason')}"
                )
        if item.get("bayesgap_exclusion_reasons"):
            lines.append(f"  BayesGap exclusion reasons: `{item.get('bayesgap_exclusion_reasons')}`")
        if item.get("bayesgap_excluded_rows"):
            for excluded in item.get("bayesgap_excluded_rows") or []:
                lines.append(
                    f"  - BayesGap exclusion cell `{excluded.get('cell_id')}`, channel `{excluded.get('channel')}`: "
                    f"{', '.join(excluded.get('reasons') or [])}"
                )
    if report.get("prediction_scale_diagnostics"):
        lines.extend(["", "## Prediction Scale Diagnostics"])
        for batch, diag in report["prediction_scale_diagnostics"].items():
            lines.append(
                f"- `{batch}`: transform `{diag.get('chosen_target_transform')}`, "
                f"raw median `{(diag.get('raw_output') or {}).get('median')}`, "
                f"10^raw median `{(diag.get('ten_power_raw_output') or {}).get('median')}`, "
                f"1/(10^raw) median `{(diag.get('inverse_ten_power_raw_output') or {}).get('median')}`, "
                f"Prediction median `{(diag.get('prediction') or {}).get('median')}`, "
                f"CI width median `{(diag.get('ci_width') or {}).get('median')}`, "
                f"CI width >2000 `{diag.get('n_ci_width_gt_2000')}`, "
                f"Prediction <=0 `{diag.get('n_prediction_le_0')}`"
            )
    if report.get("bayesgap_rounds"):
        lines.extend(["", "## BayesGap Round Indexing"])
        lines.append("| round_idx | consumed_prediction_file | input_rows | output_file |")
        lines.append("| --- | --- | ---: | --- |")
        for item in report["bayesgap_rounds"]:
            lines.append(
                f"| {item.get('round_idx')} | `{item.get('consumed_prediction_file')}` | "
                f"{item.get('input_rows')} | `{item.get('output_file')}` |"
            )
    if report.get("next_batch_acquisition_recommendations"):
        lines.extend(["", "## Next-Batch Acquisition Recommendations"])
        for item in report["next_batch_acquisition_recommendations"][:20]:
            lines.append(
                f"- round {item.get('round_idx')}: C1={item.get('C1')}, C2={item.get('C2')}, "
                f"C3={item.get('C3')}, C4={item.get('C4')}"
            )
    if report.get("final_posterior_ranking_top"):
        lines.extend(["", "## Final Posterior Mean Ranking"])
        lines.append(f"ranking_file: `{report.get('final_posterior_ranking_path')}`")
        for item in report["final_posterior_ranking_top"][:10]:
            lines.append(
                f"- rank {item.get('final_posterior_rank')}: C1={item.get('C1')}, C2={item.get('C2')}, "
                f"C3={item.get('C3')}, C4={item.get('C4')}, mean={item.get('final_posterior_mean')}, "
                f"uncertainty={item.get('posterior_half_width')}"
            )
    if report.get("paper_top_protocol_check"):
        lines.extend(["", "## Paper Top Protocol Sanity Check"])
        lines.append(f"check_file: `{report.get('paper_top_protocol_check_path')}`")
        lines.append(f"final top three exactly match paper: `{report.get('final_top_three_match_paper')}`")
        for item in report["paper_top_protocol_check"]:
            lines.append(
                f"- `{item.get('protocol')}`: exists `{item.get('exists_in_policy_space')}`, "
                f"rank `{item.get('final_posterior_rank')}`, mean `{item.get('final_posterior_mean')}`, "
                f"uncertainty `{item.get('final_posterior_uncertainty')}`"
            )
        if not report.get("final_top_three_match_paper"):
            lines.append(
                "- Audit note: mismatch may reflect implementation differences, exclusion handling, raw-data parsing, "
                "or skipped validation; this report does not claim exact paper Figure 3 reproduction."
            )
    lines.extend(["", "## Caveats"])
    for caveat in report.get("caveats", []):
        lines.append(f"- {caveat}")
    p.write_text("\n".join(lines) + "\n")
