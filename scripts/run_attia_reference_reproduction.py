#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from battery_aar.paper_reproduction.bayesgap import BayesGapConfig, run_closed_loop
from battery_aar.paper_reproduction.bms_apply_model import (
    apply_oed_model,
    bayesgap_exclusion_audit,
    cutoff_for_batch_name,
    model_path_for_batch_name,
    prediction_scale_diagnostics,
    prediction_summary,
    write_bayesgap_input,
)
from battery_aar.paper_reproduction.bms_features import load_cells_from_batch_with_status
from battery_aar.paper_reproduction.mat_model_loader import inspect_mat_model_file
from battery_aar.paper_reproduction.paths import (
    find_oed_batch_paths,
    infer_batch_name_map,
    parse_batch_name_map,
    resolve_paper_paths,
    validation_status,
    validation_status_label,
)
from battery_aar.paper_reproduction.policy_space import assert_author_policy_count, save_policy_space
from battery_aar.paper_reproduction.reports import write_json_report, write_markdown_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay Attia/Chueh author-model predictions and BayesGap selection.")
    parser.add_argument("--battery-fast-charging-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("runs/attia_reference_reproduction"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--overwrite", action="store_true", help="Clean known script-owned outputs under --out before regenerating.")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--require-exact-author-model", action="store_true")
    parser.add_argument("--use-provided-predictions", action="store_true")
    parser.add_argument("--max-cells-per-batch", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--batch-name-map", nargs="*", default=None)
    parser.add_argument("--skip-validation-batch", action="store_true", default=True)
    parser.add_argument("--include-validation-batch", action="store_true")
    parser.add_argument("--validation-batch-path", type=Path, default=None)
    parser.add_argument(
        "--author-model-target-transform",
        choices=["auto", "log10_cycle_life", "log10_inverse_cycle_life"],
        default="auto",
    )
    return parser.parse_args()


SCRIPT_OWNED_OUTPUTS = (
    "policies_all.csv",
    "model_diagnostics",
    "early_predictions",
    "early_predictions_rich",
    "bayesgap",
    "bayesgap_provided_predictions",
)


def _prepare_output_dir(out: Path, overwrite: bool) -> None:
    if not out.exists():
        out.mkdir(parents=True, exist_ok=True)
        return
    # Stale reciprocal-scale BayesGap files are more dangerous than rerunning
    # known outputs. Only remove artifacts this script owns.
    for name in SCRIPT_OWNED_OUTPUTS:
        path = out / name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    out.mkdir(parents=True, exist_ok=True)


def _write_model_diag(path: Path, out_dir: Path) -> tuple[bool, bool]:
    found = path.exists()
    ok = False
    diag = {"path": str(path), "found": found, "load_status": "missing"}
    if found:
        diag = inspect_mat_model_file(path)
        ok = diag.get("load_status") == "ok" and not diag.get("missing_required_variables")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{path.stem}.json").write_text(json.dumps(diag, indent=2, default=str) + "\n")
    return found, ok


def _provided_prediction_csvs(paths) -> list[Path]:
    candidates = []
    for rel in ("figures/fig3/pred", "figures/SI/misc_results/pred", "figures/SI/predictions_evolution/pred"):
        pred_dir = paths.optimization_dir / rel
        if pred_dir.is_dir():
            candidates.extend(sorted(pred_dir.glob("*.csv")))
    return candidates


def main() -> int:
    args = parse_args()
    if args.smoke:
        args.allow_partial = True
        args.max_cells_per_batch = args.max_cells_per_batch or 1

    out = args.out
    reports_dir = args.reports_dir
    _prepare_output_dir(out, args.overwrite)
    (out / "model_diagnostics").mkdir(parents=True, exist_ok=True)
    (out / "early_predictions").mkdir(parents=True, exist_ok=True)
    (out / "early_predictions_rich").mkdir(parents=True, exist_ok=True)
    (out / "bayesgap").mkdir(parents=True, exist_ok=True)

    caveats: list[str] = []
    status = "failed"
    exact = False
    batch_summaries: list[dict[str, object]] = []
    prediction_csvs: list[Path] = []
    scale_diagnostics: dict[str, object] = {}

    try:
        paths = resolve_paper_paths(args.battery_fast_charging_root)
    except Exception as exc:
        report = {
            "status": "author_model_missing",
            "error": str(exc),
            "validation_status": "skipped_batch9_missing",
            "validation_status_label": "validation_skipped_batch9_missing",
            "caveats": ["Paper materials were not found; smoke/partial mode only."],
        }
        write_json_report(report, reports_dir / "attia_reference_reproduction.json")
        write_markdown_report(report, reports_dir / "attia_reference_reproduction.md")
        if args.require_exact_author_model and not args.allow_partial:
            return 1
        return 0

    val_status = validation_status(paths.data_dir, args.include_validation_batch, args.validation_batch_path)
    val_label = validation_status_label(val_status)

    oed_model = paths.bms_autoanalysis_dir / "oed_model.mat"
    oed_model_batch1 = paths.bms_autoanalysis_dir / "oed_model_batch1.mat"
    oed_found, oed_ok = _write_model_diag(oed_model, out / "model_diagnostics")
    batch1_found, batch1_ok = _write_model_diag(oed_model_batch1, out / "model_diagnostics")

    policies = save_policy_space(out / "policies_all.csv")
    assert_author_policy_count(policies)

    batch_paths = find_oed_batch_paths(paths.data_dir, include_validation_batch=args.include_validation_batch, validation_batch_path=args.validation_batch_path)
    inferred_map = infer_batch_name_map(batch_paths)
    inferred_map.update(parse_batch_name_map(args.batch_name_map))

    if not (oed_ok and batch1_ok):
        caveats.append("One or more author MATLAB model files were missing or invalid.")

    for batch_path in batch_paths:
        if batch_path.name.startswith("2019-01-24_batch9") and not args.include_validation_batch:
            continue
        batch_name = inferred_map.get(batch_path.name)
        cutoff = cutoff_for_batch_name(batch_name)
        model_path = model_path_for_batch_name(paths.bms_autoanalysis_dir, batch_name)
        summary = {
            "folder": batch_path.name,
            "batch_name": batch_name,
            "cutoff_cycle": cutoff,
            "model_file": str(model_path),
            "total_raw_cell_files": len(list(batch_path.glob("*_structure.json"))),
            "cells_parsed": 0,
            "cells_with_exact_features_available": 0,
            "finite_predictions": 0,
            "available_predictions": 0,
            "unavailable_features": 0,
            "anomalous_predictions": 0,
            "bayesgap_rows": 0,
            "bayesgap_exclusion_reasons": {},
            "bayesgap_excluded_rows": [],
            "excluded_cells": [],
            "status": "not_run",
        }
        try:
            cells, excluded = load_cells_from_batch_with_status(
                batch_path,
                max_cells=args.max_cells_per_batch,
                required_cycles=(10, cutoff),
            )
            summary["excluded_cells"] = excluded
            summary["cells_parsed"] = len(cells)
            if not cells:
                raise RuntimeError(f"No cell JSON files found in {batch_path}")
            rich = apply_oed_model(
                cells,
                model_path,
                cutoff_cycle=cutoff,
                batch_name=batch_name,
                target_transform=args.author_model_target_transform,
            )
            pred_summary = prediction_summary(rich)
            summary.update(pred_summary)
            summary["status"] = "ok" if pred_summary["available_predictions"] else "no_predictions"
            rich_path = out / "early_predictions_rich" / f"{batch_path.name}.csv"
            preferred_cols = [
                "cell_id",
                "batch_id",
                "channel",
                "barcode",
                "protocol_readable",
                "C1",
                "C2",
                "C3",
                "C4",
                "raw_output",
                "target_transform",
                "Prediction",
                "CI_Lo",
                "CI_Hi",
                "ci_width",
                "anomaly_flag",
                "model_file",
                "cutoff_cycle",
                "exact_feature_status",
            ]
            extra_cols = [col for col in rich.columns if col not in preferred_cols]
            rich = rich.loc[:, preferred_cols + extra_cols]
            rich.to_csv(rich_path, index=False)
            scale_diagnostics[batch_path.name] = prediction_scale_diagnostics(rich)
            bg_path = out / "early_predictions" / f"{batch_path.name}.csv"
            bg = write_bayesgap_input(rich, bg_path)
            summary.update(bayesgap_exclusion_audit(rich, bg))
            if not bg.empty:
                prediction_csvs.append(bg_path)
        except Exception as exc:
            summary["status"] = "failed"
            summary["error"] = str(exc)
            caveats.append(f"{batch_path.name}: exact author-model replay failed: {exc}")
            if args.require_exact_author_model and not args.allow_partial:
                batch_summaries.append(summary)
                break
        batch_summaries.append(summary)

    available_batches = [b for b in batch_summaries if b.get("available_predictions", 0)]
    exact = bool(available_batches) and all(b.get("status") == "ok" for b in batch_summaries if not str(b.get("folder", "")).startswith("2019-01-24"))
    if exact:
        status = "exact_author_model_replay"
    elif oed_found or batch1_found:
        status = "author_model_found_but_curve_arrays_missing"
    else:
        status = "author_model_missing"

    bayesgap_top: list[dict[str, object]] = []
    final_posterior_ranking_top: list[dict[str, object]] = []
    next_batch_recommendations: list[dict[str, object]] = []
    bayesgap_rounds: list[dict[str, object]] = []
    paper_top_protocol_check: list[dict[str, object]] = []
    final_top_three_exact_match = False
    final_posterior_ranking_path = None
    paper_top_protocol_check_path = None
    if prediction_csvs:
        results = run_closed_loop(out / "policies_all.csv", out / "bayesgap", prediction_csvs=prediction_csvs, config=BayesGapConfig(seed=0))
        for result in results:
            bayesgap_rounds.append(
                {
                    "round_idx": int(result["round_idx"]),
                    "consumed_prediction_file": result.get("consumed_prediction_file"),
                    "input_rows": int(result.get("input_rows", 0)),
                    "output_file": str(result.get("next_batch_path")),
                    "round_description": result.get("round_description"),
                }
            )
            next_batch = result.get("next_batch")
            if hasattr(next_batch, "head"):
                for item in next_batch.head(10).to_dict("records"):
                    next_batch_recommendations.append({"round_idx": int(result["round_idx"]), **item})
        if results and "final_posterior_ranking" in results[-1]:
            final_ranking = results[-1]["final_posterior_ranking"]
            final_posterior_ranking_path = str(results[-1]["final_posterior_ranking_path"])
            paper_top_protocol_check_path = str(results[-1]["paper_top_protocol_check_path"])
            if hasattr(final_ranking, "head"):
                final_posterior_ranking_top = final_ranking.head(10).to_dict("records")
                bayesgap_top = final_posterior_ranking_top
            paper_check = results[-1].get("paper_top_protocol_check")
            if hasattr(paper_check, "to_dict"):
                paper_top_protocol_check = paper_check.to_dict("records")
                if paper_top_protocol_check:
                    final_top_three_exact_match = bool(paper_top_protocol_check[0].get("final_top_three_exact_match"))
                    if not final_top_three_exact_match:
                        caveats.append(
                            "Final top three posterior-mean protocols do not exactly match the paper-reported CLO top protocols; "
                            "this may reflect implementation differences, exclusion handling, raw-data parsing, or skipped validation."
                        )
    else:
        caveats.append("No exact early-prediction CSVs were generated; BayesGap was not advanced beyond initialization.")
        if args.use_provided_predictions:
            provided = _provided_prediction_csvs(paths)
            if provided:
                run_closed_loop(out / "policies_all.csv", out / "bayesgap_provided_predictions", prediction_csvs=provided[:4], config=BayesGapConfig(seed=0))
                status = "used_author_prediction_csv_only"
                caveats.append("BayesGap used author-provided prediction CSVs as a separately labeled fallback.")

    if args.include_validation_batch:
        caveats.append("Validation batch inclusion is experimental in this prototype.")
    else:
        caveats.append("Batch 9 validation was skipped by default; no final validation ranking was written.")

    all_model_vars = bool(oed_ok and batch1_ok)
    scale_diag_path = out / "model_diagnostics" / "prediction_scale_diagnostics.json"
    scale_diag_path.write_text(json.dumps(scale_diagnostics, indent=2, default=str, sort_keys=True) + "\n")
    report = {
        "status": status,
        "exact_author_model_replay": exact,
        "author_model_target_transform_requested": args.author_model_target_transform,
        "oed_model_found": oed_found,
        "oed_model_batch1_found": batch1_found,
        "all_required_model_variables_found": all_model_vars,
        "batch_name_mapping": inferred_map,
        "batches": batch_summaries,
        "prediction_scale_diagnostics": scale_diagnostics,
        "prediction_scale_diagnostics_path": str(scale_diag_path),
        "bayesgap_rounds": bayesgap_rounds,
        "next_batch_acquisition_recommendations": next_batch_recommendations,
        "final_posterior_ranking_top": final_posterior_ranking_top,
        "final_posterior_ranking_path": final_posterior_ranking_path,
        "paper_top_protocol_check": paper_top_protocol_check,
        "paper_top_protocol_check_path": paper_top_protocol_check_path,
        "final_top_three_match_paper": final_top_three_exact_match,
        "validation_status": val_status,
        "validation_status_label": val_label,
        "bayesgap_top_protocols": bayesgap_top,
        "author_model_predictions_available": bool(prediction_csvs),
        "author_model_validation_metrics_unavailable_batch9_skipped": not args.include_validation_batch,
        "caveats": caveats,
    }
    write_json_report(report, reports_dir / "attia_reference_reproduction.json")
    write_markdown_report(report, reports_dir / "attia_reference_reproduction.md")

    if args.require_exact_author_model and not exact:
        return 1
    if status == "failed" and not args.allow_partial:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
