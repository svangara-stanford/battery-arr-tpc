#!/usr/bin/env python
"""DEPRECATED scorer: trains candidates on Severson and scores on Attia Batch 9.

.. warning::
    Patch E4 switched the locked secondary test from Attia Batch 9 (transfer)
    to Severson b3 (in-distribution). Prefer the new
    ``scripts/score_candidates_on_severson_b3.py`` script. This script is
    retained only for the explicit ``--secondary-test-source=attia_batch9``
    transfer experiments.

Given a JSON list of candidate specs from completed Track A runs, this script
trains each candidate on its full Severson dataset and predicts on Batch 9 once,
writing a single leaderboard CSV plus per-candidate prediction CSVs.

The script does NOT run discovery, does NOT mutate any of the input run dirs,
and is safe to run many times on the same candidate list (output dir is
overwritten only for entries in the current list).

Example invocation
------------------

    module purge >/dev/null 2>&1 || true
    module load python/3.12.1 >/dev/null 2>&1
    source /home/groups/darve/svangara/battery-arr-venvs/oba/bin/activate
    cd /home/groups/darve/svangara/battery-arr-tpc

    python3 scripts/score_candidates_on_batch9.py \
      --candidate-list /tmp/champions.json \
      --battery-fast-charging-root \
          /home/groups/darve/svangara/battery-arr-tpc/literature_models_and_data/battery-fast-charging \
      --out /scratch/users/svangara/battery-arr/champion_selection/20260608/batch9_scoring/ \
      --max-cycle 100

Each entry in ``candidates.json`` must include at least ``candidate_id``,
``source_run_id``, ``candidate_path``, ``feature_program_path``, and
``processed_dir``. The remaining keys (``recipe``, ``split_mode``,
``split_seed``, ``model_family``, ``feature_set``, ``target_transform``,
``surrogate_rmse``, ``surrogate_mae``, ``surrogate_spearman``) are echoed into
the leaderboard for traceability.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from battery_aar.agents.attia_data_bridge import load_author_validation_metrics, load_batch9_holdout
from battery_aar.agents.evaluator import battery_pgr, evaluate_candidate_train_test
from battery_aar.agents.orchestrator import weak_baseline_rmse_against
from battery_aar.workflows.role_graph import (  # noqa: WPS437 - intentional reuse
    _combine_locked_feature_program_tables,
    _finite_labels,
    _load_processed,
    _locked_prediction_frame,
    _prediction_diagnostics,
)


REPO_ROOT = Path("/home/groups/darve/svangara/battery-arr-tpc")
BUILD_BATCH9_FEATURE_SCRIPT = REPO_ROOT / "scripts" / "build_battery_feature_program.py"
# Slurm sherlock_trackA_discovery.slurm passes this placeholder for batch9 builds
# because --processed-dir is required by the script but unused for batch9 source.
BATCH9_PROCESSED_DIR_PLACEHOLDER = "data/processed/chueh_toyota_fast_charge_agent_surrogate"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate-list", type=Path, required=True, help="Path to JSON list of candidate specs.")
    parser.add_argument(
        "--battery-fast-charging-root",
        type=Path,
        required=True,
        help="Path to the local clone of battery-fast-charging (used for Batch 9 raw cells).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Directory to write batch9_leaderboard.csv, batch9_scoring_metadata.json, and per-candidate prediction CSVs.",
    )
    parser.add_argument("--max-cycle", type=int, default=100, help="First N cycles to use when loading Batch 9 (matches training).")
    parser.add_argument(
        "--candidate-timeout-s",
        type=int,
        default=120,
        help="Timeout in seconds for each candidate's fit+predict subprocess (default 120s; default in candidate runner is 30s, too short for RF).",
    )
    parser.add_argument(
        "--allow-protocol-features",
        action="store_true",
        help="Allow C1..C4 protocol currents to leak into candidate features (off by default).",
    )
    parser.add_argument(
        "--reference-run",
        type=Path,
        default=None,
        help="Optional path to an attia reference run with validation_metrics.json (for author Batch 9 RMSE baseline).",
    )
    parser.add_argument(
        "--batch9-feature-program-root",
        type=Path,
        default=None,
        help="Where to cache (recipe, cycle_early, cycle_late) -> Batch 9 feature_table.csv builds. Defaults to <out>/batch9_feature_programs.",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python interpreter used to invoke build_battery_feature_program.py (default: current).",
    )
    return parser.parse_args()


def _load_candidate_list(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list) or not all(isinstance(entry, dict) for entry in payload):
        raise ValueError(f"Candidate list must be a JSON list of objects: {path}")
    required = {"candidate_id", "candidate_path", "feature_program_path", "processed_dir", "source_run_id"}
    for idx, entry in enumerate(payload):
        missing = sorted(required - entry.keys())
        if missing:
            raise ValueError(f"Candidate entry #{idx} is missing required keys: {missing}")
    return payload


def _resolve_recipe_settings(entry: dict[str, Any]) -> tuple[str, int, int, int]:
    """Pull (recipe, cycle_early, cycle_late, first_n_cycles) from the entry, falling back to the
    feature_program_path basename (e.g. ``severson_broad_physics_idx9_99``)."""
    recipe = str(entry.get("recipe") or "").strip()
    cycle_early = entry.get("cycle_early_index")
    cycle_late = entry.get("cycle_late_index")
    first_n = entry.get("first_n_cycles")
    fp_path = Path(entry["feature_program_path"])
    program_dir = fp_path.parent if fp_path.name == "feature_table.csv" else fp_path
    stem = program_dir.name
    # Expected pattern: <prefix>_<recipe>_idx<early>_<late>, e.g. severson_broad_physics_idx9_99.
    if "_idx" in stem:
        head, _, tail = stem.partition("_idx")
        if not recipe:
            for prefix in ("severson_", "oed_", "batch9_", "all_"):
                if head.startswith(prefix):
                    recipe = head[len(prefix):]
                    break
            else:
                recipe = head
        if "_" in tail and (cycle_early is None or cycle_late is None):
            early_str, _, late_str = tail.partition("_")
            try:
                if cycle_early is None:
                    cycle_early = int(early_str)
                if cycle_late is None:
                    cycle_late = int(late_str)
            except ValueError:
                pass
    if not recipe:
        raise ValueError(
            f"Could not infer feature program recipe for candidate {entry['candidate_id']}; "
            "set 'recipe' explicitly in the candidate spec."
        )
    if cycle_early is None:
        cycle_early = 9
    if cycle_late is None:
        cycle_late = 99
    if first_n is None:
        first_n = max(int(cycle_late) + 1, 100)
    return recipe, int(cycle_early), int(cycle_late), int(first_n)


def _build_batch9_feature_table(
    *,
    recipe: str,
    cycle_early: int,
    cycle_late: int,
    first_n_cycles: int,
    include_protocol_features: bool,
    battery_fast_charging_root: Path,
    cache_root: Path,
    python_bin: Path,
) -> Path:
    """Build (or return cached) Batch 9 feature_table.csv for the given recipe configuration."""
    key = f"batch9_{recipe}_idx{cycle_early}_{cycle_late}_firstn{first_n_cycles}_prot{int(include_protocol_features)}"
    out_dir = cache_root / key
    table_path = out_dir / "feature_table.csv"
    if table_path.exists():
        return table_path
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(python_bin),
        str(BUILD_BATCH9_FEATURE_SCRIPT),
        "--battery-fast-charging-root", str(battery_fast_charging_root),
        "--processed-dir", BATCH9_PROCESSED_DIR_PLACEHOLDER,
        "--recipe", recipe,
        "--batch", "batch9",
        "--out", str(out_dir),
        "--first-n-cycles", str(first_n_cycles),
        "--cycle-early-index", str(cycle_early),
        "--cycle-late-index", str(cycle_late),
        "--include-protocol-features", "true" if include_protocol_features else "false",
    ]
    print(f"[batch9_features] building {key} ...", flush=True)
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "build_battery_feature_program.py failed for recipe="
            f"{recipe} idx{cycle_early}_{cycle_late}: "
            f"\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    if not table_path.exists():
        raise RuntimeError(f"Expected batch9 feature_table at {table_path} after build; not found.")
    return table_path


def _row_offset(metadata: pd.DataFrame) -> int:
    return int(pd.to_numeric(metadata["row_id"], errors="coerce").max()) + 1


def _offset_holdout(
    holdout_meta: pd.DataFrame,
    holdout_cycles: pd.DataFrame,
    holdout_labels: pd.DataFrame,
    row_offset: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out_meta = holdout_meta.copy()
    out_cycles = holdout_cycles.copy()
    out_labels = holdout_labels.copy()
    out_meta["row_id"] = pd.to_numeric(out_meta["row_id"], errors="raise").astype(int) + row_offset
    out_cycles["row_id"] = pd.to_numeric(out_cycles["row_id"], errors="raise").astype(int) + row_offset
    out_labels["row_id"] = pd.to_numeric(out_labels["row_id"], errors="raise").astype(int) + row_offset
    return out_meta, out_cycles, out_labels


def _score_one_candidate(
    *,
    entry: dict[str, Any],
    holdout_meta_raw: pd.DataFrame,
    holdout_cycles_raw: pd.DataFrame,
    holdout_labels_raw: pd.DataFrame,
    author_validation: dict[str, Any],
    args: argparse.Namespace,
    predictions_dir: Path,
    feature_cache_root: Path,
) -> dict[str, Any]:
    candidate_id = str(entry["candidate_id"])
    started_at = _utc_now()
    t0 = time.time()
    record: dict[str, Any] = {
        "candidate_id": candidate_id,
        "source_run_id": entry.get("source_run_id"),
        "candidate_path": entry.get("candidate_path"),
        "started_at": started_at,
    }
    try:
        candidate_path = Path(entry["candidate_path"])
        if not candidate_path.exists():
            raise FileNotFoundError(f"candidate_path missing: {candidate_path}")
        processed_dir = Path(entry["processed_dir"])
        if not processed_dir.is_dir():
            raise FileNotFoundError(f"processed_dir missing: {processed_dir}")
        feature_program_path = Path(entry["feature_program_path"])
        if not feature_program_path.exists():
            raise FileNotFoundError(f"feature_program_path missing: {feature_program_path}")

        recipe, cycle_early, cycle_late, first_n = _resolve_recipe_settings(entry)
        include_protocol = bool(entry.get("include_protocol_features", args.allow_protocol_features))

        metadata, cycles, labels, _labels_path, _splits_table = _load_processed(processed_dir)
        labels = _finite_labels(labels)

        holdout_first_n = max(args.max_cycle, first_n)
        # The Batch 9 holdout has already been loaded with first_n_cycles=args.max_cycle once
        # at the caller level; we trust args.max_cycle for the slice used during training/predict.
        row_offset = _row_offset(metadata)
        holdout_meta, holdout_cycles, holdout_labels = _offset_holdout(
            holdout_meta_raw, holdout_cycles_raw, holdout_labels_raw, row_offset
        )

        weak_rmse = weak_baseline_rmse_against(labels, holdout_labels)
        author_rmse = author_validation.get("author_model_batch9_rmse")

        batch9_fp_table = _build_batch9_feature_table(
            recipe=recipe,
            cycle_early=cycle_early,
            cycle_late=cycle_late,
            first_n_cycles=holdout_first_n,
            include_protocol_features=include_protocol,
            battery_fast_charging_root=args.battery_fast_charging_root,
            cache_root=feature_cache_root,
            python_bin=args.python,
        )

        combined_dir = feature_cache_root / f"combined_{candidate_id}"
        combined_paths = _combine_locked_feature_program_tables(
            search_paths=[feature_program_path],
            batch9_path=batch9_fp_table,
            out_dir=combined_dir,
            row_offset=row_offset,
        )

        result = evaluate_candidate_train_test(
            candidate_path,
            metadata,
            cycles,
            labels,
            holdout_meta,
            holdout_cycles,
            holdout_labels,
            weak_rmse=weak_rmse,
            strong_rmse=author_rmse,
            allow_protocol_features=include_protocol,
            max_cycle=args.max_cycle,
            timeout_s=args.candidate_timeout_s,
            return_predictions=True,
            feature_program_paths=combined_paths,
            feature_program_mode=str(entry.get("feature_program_mode", "table")),
            include_feature_programs=True,
            feature_family_filter=list(entry.get("feature_family_filter") or []),
        )
        if not result.get("success"):
            raise RuntimeError(result.get("failure_reason") or result.get("error") or "Batch 9 evaluation failed")

        predictions = _locked_prediction_frame(result["predictions"], holdout_meta)
        predictions_path = predictions_dir / f"{candidate_id}.csv"
        predictions.to_csv(predictions_path, index=False)

        metrics = dict(result.get("metrics") or {})
        metrics.update(_prediction_diagnostics(predictions))
        metrics["batch9_weak_baseline_rmse"] = weak_rmse
        metrics["author_model_batch9_rmse"] = author_rmse
        metrics["author_model_batch9_mae"] = author_validation.get("author_model_batch9_mae")
        metrics["battery_pgr_author_model_batch9"] = battery_pgr(weak_rmse, metrics.get("rmse"), author_rmse)
        record.update(
            {
                "status": "ok",
                "recipe": recipe,
                "cycle_early_index": cycle_early,
                "cycle_late_index": cycle_late,
                "first_n_cycles": first_n,
                "batch9_feature_program_path": str(batch9_fp_table),
                "combined_feature_program_paths": combined_paths,
                "predictions_path": str(predictions_path),
                "metrics": metrics,
                "duration_s": time.time() - t0,
                "finished_at": _utc_now(),
            }
        )
    except Exception as exc:
        record.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
                "duration_s": time.time() - t0,
                "finished_at": _utc_now(),
            }
        )
    return record


def _leaderboard_row(entry: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    metrics = record.get("metrics") or {}
    return {
        "candidate_id": entry.get("candidate_id"),
        "source_run_id": entry.get("source_run_id"),
        "candidate_path": entry.get("candidate_path"),
        "recipe": record.get("recipe") or entry.get("recipe"),
        "split_mode": entry.get("split_mode"),
        "split_seed": entry.get("split_seed"),
        "model_family": entry.get("model_family"),
        "feature_set": entry.get("feature_set"),
        "target_transform": entry.get("target_transform"),
        "surrogate_rmse": entry.get("surrogate_rmse"),
        "surrogate_mae": entry.get("surrogate_mae"),
        "surrogate_spearman": entry.get("surrogate_spearman"),
        "batch9_rmse": metrics.get("rmse"),
        "batch9_mae": metrics.get("mae"),
        "batch9_r2": metrics.get("r2"),
        "batch9_spearman": metrics.get("spearman"),
        "batch9_kendall": metrics.get("kendall"),
        "batch9_pgr": metrics.get("battery_pgr_author_model_batch9"),
        "batch9_weak_baseline_rmse": metrics.get("batch9_weak_baseline_rmse"),
        "batch9_author_model_rmse": metrics.get("author_model_batch9_rmse"),
        "batch9_n_predictions": metrics.get("n_predictions"),
        "batch9_n_negative_predictions": metrics.get("n_negative_predictions"),
        "batch9_n_nonfinite_predictions": metrics.get("n_nonfinite_predictions"),
        "predictions_path": record.get("predictions_path"),
        "status": record.get("status"),
        "error_type": record.get("error_type"),
        "error_message": record.get("error_message"),
    }


def main() -> int:
    import warnings as _warnings

    _warnings.warn(
        "scripts/score_candidates_on_batch9.py is deprecated. The locked "
        "secondary test now defaults to Severson b3; use "
        "scripts/score_candidates_on_severson_b3.py unless you explicitly "
        "want the Attia Batch 9 transfer experiment.",
        DeprecationWarning,
        stacklevel=2,
    )
    args = _parse_args()
    candidates = _load_candidate_list(args.candidate_list)
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = out_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    feature_cache_root = args.batch9_feature_program_root or (out_dir / "batch9_feature_programs")
    feature_cache_root.mkdir(parents=True, exist_ok=True)

    author_validation = load_author_validation_metrics(args.reference_run)

    print(f"[scoring] loading Batch 9 holdout (first_n_cycles={args.max_cycle}) ...", flush=True)
    holdout_meta_raw, holdout_cycles_raw, holdout_labels_raw = load_batch9_holdout(
        battery_fast_charging_root=args.battery_fast_charging_root,
        first_n_cycles=args.max_cycle,
    )
    print(
        f"[scoring] Batch 9: n_cells={len(holdout_labels_raw)}, "
        f"author_rmse={author_validation.get('author_model_batch9_rmse')}",
        flush=True,
    )

    records: list[dict[str, Any]] = []
    leaderboard_rows: list[dict[str, Any]] = []
    for idx, entry in enumerate(candidates, start=1):
        candidate_id = entry.get("candidate_id")
        print(f"[scoring] ({idx}/{len(candidates)}) {candidate_id} ...", flush=True)
        record = _score_one_candidate(
            entry=entry,
            holdout_meta_raw=holdout_meta_raw,
            holdout_cycles_raw=holdout_cycles_raw,
            holdout_labels_raw=holdout_labels_raw,
            author_validation=author_validation,
            args=args,
            predictions_dir=predictions_dir,
            feature_cache_root=feature_cache_root,
        )
        records.append({"candidate_spec": entry, "scoring_record": record})
        leaderboard_rows.append(_leaderboard_row(entry, record))
        status = record.get("status")
        rmse = (record.get("metrics") or {}).get("rmse")
        print(f"[scoring]   -> status={status} batch9_rmse={rmse}", flush=True)

    leaderboard_path = out_dir / "batch9_leaderboard.csv"
    pd.DataFrame(leaderboard_rows).to_csv(leaderboard_path, index=False)
    metadata_path = out_dir / "batch9_scoring_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "generated_at": _utc_now(),
                "candidate_list_path": str(args.candidate_list),
                "battery_fast_charging_root": str(args.battery_fast_charging_root),
                "max_cycle": args.max_cycle,
                "candidate_timeout_s": args.candidate_timeout_s,
                "allow_protocol_features": bool(args.allow_protocol_features),
                "reference_run": str(args.reference_run) if args.reference_run else None,
                "author_validation": author_validation,
                "n_candidates": len(candidates),
                "n_successful": sum(1 for r in records if r["scoring_record"].get("status") == "ok"),
                "n_failed": sum(1 for r in records if r["scoring_record"].get("status") != "ok"),
                "records": records,
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n"
    )
    print(f"[scoring] wrote leaderboard -> {leaderboard_path}", flush=True)
    print(f"[scoring] wrote metadata    -> {metadata_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
