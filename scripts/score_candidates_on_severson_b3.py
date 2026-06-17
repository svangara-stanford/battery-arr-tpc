#!/usr/bin/env python
"""Score already-trained Open Battery Agents candidates on the Severson b3 locked secondary test.

This is the patch-E4 default scorer; it mirrors the structure of
``scripts/score_candidates_on_batch9.py`` but evaluates against the Severson
2018-04-12 (b3) batch instead of Attia Batch 9. Leaderboard columns are
prefixed with ``secondary_test_*`` so downstream tooling can be source-agnostic
while the legacy ``batch9_*`` columns are emitted in parallel for backwards
compatibility.

Each entry in ``candidates.json`` must include at least ``candidate_id``,
``source_run_id``, ``candidate_path``, ``feature_program_path``, and
``processed_dir``. The remaining keys (``recipe``, ``split_mode``,
``split_seed``, ``model_family``, ``feature_set``, ``target_transform``,
``surrogate_rmse``, ``surrogate_mae``, ``surrogate_spearman``) are echoed into
the leaderboard for traceability.

Example invocation
------------------

    module purge >/dev/null 2>&1 || true
    module load python/3.12.1 >/dev/null 2>&1
    source /home/groups/darve/svangara/battery-arr-venvs/oba/bin/activate
    cd /home/groups/darve/svangara/battery-arr-tpc

    python3 scripts/score_candidates_on_severson_b3.py \\
      --candidate-list /tmp/champions.json \\
      --battery-fast-charging-root \\
          /home/groups/darve/svangara/battery-arr-tpc/literature_models_and_data/battery-fast-charging \\
      --out /scratch/users/svangara/battery-arr/champion_selection/<ts>/secondary_test_scoring/ \\
      --max-cycle 100
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from battery_aar.agents.attia_data_bridge import (
    load_author_validation_metrics,
    load_severson_b3_holdout,
)
from battery_aar.agents.evaluator import battery_pgr, evaluate_candidate_train_test
from battery_aar.agents.orchestrator import weak_baseline_rmse_against
from battery_aar.workflows.role_graph import (  # noqa: WPS437 - intentional reuse
    _finite_labels,
    _load_processed,
    _locked_prediction_frame,
    _prediction_diagnostics,
)


REPO_ROOT = Path("/home/groups/darve/svangara/battery-arr-tpc")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate-list", type=Path, required=True, help="Path to JSON list of candidate specs.")
    parser.add_argument(
        "--battery-fast-charging-root",
        type=Path,
        required=False,
        default=None,
        help="Path to the local clone of battery-fast-charging (used to resolve the Severson b3 .mat).",
    )
    parser.add_argument(
        "--severson-b3-mat-path",
        type=Path,
        default=None,
        help="Optional explicit path to the b3 .mat file (overrides --battery-fast-charging-root).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Directory to write secondary_test_leaderboard.csv, secondary_test_scoring_metadata.json, "
        "and per-candidate prediction CSVs.",
    )
    parser.add_argument("--max-cycle", type=int, default=100, help="First N cycles to use when loading b3 (matches training).")
    parser.add_argument(
        "--candidate-timeout-s",
        type=int,
        default=120,
        help="Timeout in seconds for each candidate's fit+predict subprocess.",
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
        help="Optional attia reference run (kept for symmetry; author RMSE is Attia-specific so we record it as null for b3).",
    )
    parser.add_argument(
        "--exclude-cell-ids",
        nargs="*",
        default=[],
        help="Optional list of Severson b3 cell_ids to exclude (e.g. cells that overlapped training).",
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

        include_protocol = bool(entry.get("include_protocol_features", args.allow_protocol_features))

        metadata, cycles, labels, _labels_path, _splits_table = _load_processed(processed_dir)
        labels = _finite_labels(labels)

        row_offset = _row_offset(metadata)
        holdout_meta, holdout_cycles, holdout_labels = _offset_holdout(
            holdout_meta_raw, holdout_cycles_raw, holdout_labels_raw, row_offset
        )

        weak_rmse = weak_baseline_rmse_against(labels, holdout_labels)
        author_rmse = author_validation.get("author_model_batch9_rmse")

        # Severson b3 reuses the search-time Severson feature programs because
        # both training and the secondary test are built from the same Severson
        # MatR dataset. No "combined" feature-program table is required.
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
            feature_program_paths=[str(feature_program_path)],
            feature_program_mode=str(entry.get("feature_program_mode", "table")),
            include_feature_programs=True,
            feature_family_filter=list(entry.get("feature_family_filter") or []),
        )
        if not result.get("success"):
            raise RuntimeError(result.get("failure_reason") or result.get("error") or "Severson b3 evaluation failed")

        predictions = _locked_prediction_frame(result["predictions"], holdout_meta)
        predictions_path = predictions_dir / f"{candidate_id}.csv"
        predictions.to_csv(predictions_path, index=False)

        metrics = dict(result.get("metrics") or {})
        metrics.update(_prediction_diagnostics(predictions))
        # Preserve the same metric semantics as score_candidates_on_batch9 so
        # downstream tools (ChampionAdjudicator) can read either source.
        metrics["secondary_test_weak_baseline_rmse"] = weak_rmse
        metrics["secondary_test_source"] = "severson_b3"
        metrics["batch9_weak_baseline_rmse"] = weak_rmse  # legacy alias
        metrics["author_model_batch9_rmse"] = author_rmse
        metrics["author_model_batch9_mae"] = author_validation.get("author_model_batch9_mae")
        metrics["battery_pgr_author_model_batch9"] = battery_pgr(weak_rmse, metrics.get("rmse"), author_rmse)
        record.update(
            {
                "status": "ok",
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
        # Patch E4: source-agnostic secondary_test_* columns (preferred).
        "secondary_test_source": metrics.get("secondary_test_source", "severson_b3"),
        "secondary_test_rmse": metrics.get("rmse"),
        "secondary_test_mae": metrics.get("mae"),
        "secondary_test_r2": metrics.get("r2"),
        "secondary_test_spearman": metrics.get("spearman"),
        "secondary_test_kendall": metrics.get("kendall"),
        "secondary_test_pgr": metrics.get("battery_pgr_author_model_batch9"),
        "secondary_test_weak_baseline_rmse": metrics.get("secondary_test_weak_baseline_rmse"),
        "secondary_test_n_predictions": metrics.get("n_predictions"),
        "secondary_test_n_negative_predictions": metrics.get("n_negative_predictions"),
        "secondary_test_n_nonfinite_predictions": metrics.get("n_nonfinite_predictions"),
        # Legacy batch9_* columns (deprecated; same values as secondary_test_*).
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
    args = _parse_args()
    candidates = _load_candidate_list(args.candidate_list)
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = out_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    author_validation = load_author_validation_metrics(args.reference_run)

    if args.severson_b3_mat_path is None and args.battery_fast_charging_root is None:
        sys.exit("ERROR: either --severson-b3-mat-path or --battery-fast-charging-root must be set.")

    print(f"[scoring] loading Severson b3 holdout (first_n_cycles={args.max_cycle}) ...", flush=True)
    holdout_meta_raw, holdout_cycles_raw, holdout_labels_raw = load_severson_b3_holdout(
        battery_fast_charging_root=args.battery_fast_charging_root,
        first_n_cycles=args.max_cycle,
        exclude_cell_ids=list(args.exclude_cell_ids or []),
        mat_path=args.severson_b3_mat_path,
    )
    print(
        f"[scoring] Severson b3: n_cells={len(holdout_labels_raw)}",
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
        )
        records.append({"candidate_spec": entry, "scoring_record": record})
        leaderboard_rows.append(_leaderboard_row(entry, record))
        status = record.get("status")
        rmse = (record.get("metrics") or {}).get("rmse")
        print(f"[scoring]   -> status={status} secondary_test_rmse={rmse}", flush=True)

    leaderboard_path = out_dir / "secondary_test_leaderboard.csv"
    # Also write a backwards-compatible alias filename so any consumer still
    # looking for batch9_leaderboard.csv keeps working during the transition.
    alias_path = out_dir / "batch9_leaderboard.csv"
    leaderboard_frame = pd.DataFrame(leaderboard_rows)
    leaderboard_frame.to_csv(leaderboard_path, index=False)
    leaderboard_frame.to_csv(alias_path, index=False)
    metadata_path = out_dir / "secondary_test_scoring_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "generated_at": _utc_now(),
                "secondary_test_source": "severson_b3",
                "candidate_list_path": str(args.candidate_list),
                "battery_fast_charging_root": str(args.battery_fast_charging_root)
                if args.battery_fast_charging_root is not None
                else None,
                "severson_b3_mat_path": str(args.severson_b3_mat_path) if args.severson_b3_mat_path else None,
                "max_cycle": args.max_cycle,
                "candidate_timeout_s": args.candidate_timeout_s,
                "allow_protocol_features": bool(args.allow_protocol_features),
                "reference_run": str(args.reference_run) if args.reference_run else None,
                "author_validation": author_validation,
                "exclude_cell_ids": list(args.exclude_cell_ids or []),
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
    print(f"[scoring] wrote alias       -> {alias_path}", flush=True)
    print(f"[scoring] wrote metadata    -> {metadata_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
