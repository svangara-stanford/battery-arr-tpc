#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        return {"_read_error": f"{type(exc).__name__}: {exc}"}


def _run_recipe(report: dict[str, Any], report_path: Path) -> str:
    recipe = report.get("feature_program_recipe")
    if recipe:
        return str(recipe)
    text = str(report.get("feature_program_paths") or "") + " " + str(report_path)
    for candidate in ["minimal_debug", "scalar_baseline", "curve_delta", "broad_physics", "attia_severson_like"]:
        if candidate in text:
            return candidate
    m = re.search(r"trackA_([^/]+?)_(random|batch|protocol|leave_one_batch_out)_", str(report_path))
    return m.group(1) if m else "unknown"


def _normalized_config_key(row: dict[str, Any], run_recipe: str) -> str:
    payload = {
        "run_recipe": run_recipe,
        "model_family": row.get("model_family"),
        "feature_set": row.get("feature_set"),
        "target_transform": row.get("target_transform"),
        "include_protocol_features": bool(row.get("include_protocol_features", False)),
        "compiled_candidate": bool(row.get("compiled_candidate", False)),
        "feature_family_filter": row.get("feature_family_filter") or [],
        "hyperparameters": row.get("hyperparameters") or {},
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Track A Battery-ARR role-agent discovery outputs.")
    parser.add_argument("--reports-root", type=Path, default=Path("reports"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/trackA_summary"))
    parser.add_argument("--pattern", default="trackA_*/role_agent_workflow.json")
    parser.add_argument("--min-evals", type=int, default=2)
    args = parser.parse_args()

    paths = sorted(args.reports_root.glob(args.pattern))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []

    for path in paths:
        report = _read_json(path)
        recipe = _run_recipe(report, path)
        metrics = report.get("validation_metrics") or {}
        extra = metrics.get("extra_metrics") or {}
        spec = report.get("candidate_spec") or {}
        locked = report.get("locked_batch9_validation") or {}
        b9 = report.get("final_batch9_metrics") or {}
        run_row = {
            "report_path": str(path),
            "run_id": report.get("run_id"),
            "recipe": recipe,
            "split_mode": report.get("split_mode"),
            "split_seed": report.get("split_seed"),
            "offline": report.get("offline"),
            "iterations": report.get("iterations"),
            "candidates_per_iteration": report.get("candidates_per_iteration"),
            "n_train_cells": report.get("n_train_cells"),
            "n_validation_cells": report.get("n_validation_cells"),
            "best_candidate_id": report.get("best_candidate_id"),
            "best_model_family": spec.get("model_family"),
            "best_feature_set": spec.get("feature_set"),
            "best_target_transform": spec.get("target_transform"),
            "best_rmse": metrics.get("rmse"),
            "best_mae": metrics.get("mae"),
            "best_r2": metrics.get("r2"),
            "best_spearman": metrics.get("spearman"),
            "best_kendall": metrics.get("kendall"),
            "best_y_true_mean": extra.get("y_true_mean"),
            "best_y_pred_mean": extra.get("y_pred_mean"),
            "locked_batch9_status": locked.get("status", "not_run"),
            "batch9_rmse": b9.get("rmse"),
            "batch9_mae": b9.get("mae"),
            "candidate_path": report.get("candidate_path"),
            "read_error": report.get("_read_error"),
        }
        run_rows.append(run_row)

        for row in report.get("all_candidate_metrics", []) or []:
            merged = {
                "report_path": str(path),
                "run_id": report.get("run_id"),
                "recipe": recipe,
                "split_mode": report.get("split_mode"),
                "split_seed": report.get("split_seed"),
                "offline": report.get("offline"),
                **row,
            }
            merged["normalized_config_key"] = _normalized_config_key(merged, recipe)
            try:
                merged["abs_bias"] = abs(float(merged.get("y_pred_mean")) - float(merged.get("y_true_mean")))
            except Exception:
                merged["abs_bias"] = None
            candidate_rows.append(merged)

    runs = pd.DataFrame(run_rows)
    candidates = pd.DataFrame(candidate_rows)

    runs.to_csv(args.out_dir / "trackA_run_summary.csv", index=False)
    candidates.to_csv(args.out_dir / "trackA_all_candidates.csv", index=False)

    if candidates.empty:
        print(f"Found {len(paths)} report files, but no candidate rows.")
        return 0

    c = candidates.copy()
    if "success" in c.columns:
        c = c[c["success"].astype(bool)]
    c["rmse"] = pd.to_numeric(c.get("rmse"), errors="coerce")
    c["mae"] = pd.to_numeric(c.get("mae"), errors="coerce")
    c["r2"] = pd.to_numeric(c.get("r2"), errors="coerce")
    c["spearman"] = pd.to_numeric(c.get("spearman"), errors="coerce")
    c["kendall"] = pd.to_numeric(c.get("kendall"), errors="coerce")
    c["abs_bias"] = pd.to_numeric(c.get("abs_bias"), errors="coerce")
    c = c[c["rmse"].notna()]

    agg = (
        c.groupby("normalized_config_key", dropna=False)
        .agg(
            n_evals=("rmse", "count"),
            n_runs=("run_id", "nunique"),
            n_split_modes=("split_mode", "nunique"),
            split_modes=("split_mode", lambda x: ",".join(sorted(set(map(str, x))))),
            recipes=("recipe", lambda x: ",".join(sorted(set(map(str, x))))),
            mean_rmse=("rmse", "mean"),
            std_rmse=("rmse", "std"),
            min_rmse=("rmse", "min"),
            max_rmse=("rmse", "max"),
            mean_mae=("mae", "mean"),
            mean_r2=("r2", "mean"),
            mean_spearman=("spearman", "mean"),
            mean_kendall=("kendall", "mean"),
            mean_abs_bias=("abs_bias", "mean"),
            model_family=("model_family", "first"),
            feature_set=("feature_set", "first"),
            target_transform=("target_transform", "first"),
            include_protocol_features=("include_protocol_features", "first"),
            example_candidate_id=("candidate_id", "first"),
            example_candidate_path=("candidate_path", "first"),
            example_report_path=("report_path", "first"),
            hyperparameters=("hyperparameters", "first"),
        )
        .reset_index()
    )
    agg["std_rmse"] = agg["std_rmse"].fillna(0.0)
    agg["mean_abs_bias"] = agg["mean_abs_bias"].fillna(0.0)

    # Validation-only no-leakage selector. Batch 9 metrics, if present, are ignored here.
    # Penalize instability and systematic bias; prefer configs seen more than once.
    agg["selection_score"] = agg["mean_rmse"] + 0.25 * agg["std_rmse"] + 0.10 * agg["mean_abs_bias"]
    eligible = agg[agg["n_evals"] >= int(args.min_evals)].copy()
    eligible = eligible.sort_values(["selection_score", "mean_rmse", "std_rmse", "mean_mae"])
    agg = agg.sort_values(["selection_score", "mean_rmse", "std_rmse", "mean_mae"])

    agg.to_csv(args.out_dir / "trackA_config_aggregate_all.csv", index=False)
    eligible.to_csv(args.out_dir / "trackA_config_aggregate_eligible.csv", index=False)

    print(f"Found {len(paths)} report files")
    print(f"Wrote {args.out_dir / 'trackA_run_summary.csv'}")
    print(f"Wrote {args.out_dir / 'trackA_all_candidates.csv'}")
    print(f"Wrote {args.out_dir / 'trackA_config_aggregate_all.csv'}")
    print(f"Wrote {args.out_dir / 'trackA_config_aggregate_eligible.csv'}")
    print("\nTop eligible validation-only configs:")
    show_cols = [
        "selection_score", "n_evals", "n_runs", "split_modes", "recipes", "mean_rmse", "std_rmse", "mean_mae", "mean_abs_bias",
        "model_family", "feature_set", "target_transform", "example_candidate_id", "hyperparameters", "example_report_path"
    ]
    if not eligible.empty:
        print(eligible[show_cols].head(30).to_string(index=False))
    else:
        print("No eligible configs. Try --min-evals 1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
