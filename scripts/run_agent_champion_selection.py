#!/usr/bin/env python3
"""End-to-end agentic champion selection for Open Battery Agents Track A.

After the 18-job sealed Track A campaign finishes (and the summarizer has
written `trackA_full_run_summary.csv`, `trackA_full_all_candidates.csv`, and
`trackA_config_aggregate_eligible.csv`), this script:

  1. Loads the cross-run Severson-only evidence and resolves each candidate's
     compiled `.py` path, feature-program path, and processed Severson dir
     from the on-disk run layout.
  2. Invokes `ChampionAggregator` (LLM, or deterministic if --offline) to pick
     a top-K shortlist with explicit selection criteria. Batch 9 is sealed.
  3. Subprocess-invokes `scripts/score_candidates_on_batch9.py` to evaluate
     just those K candidates on Batch 9 once.
  4. Joins the shortlist with the Batch 9 leaderboard and invokes
     `ChampionAdjudicator` to pick the final champion + runner-up.
  5. Writes `champion_decision.json` and a human-readable `champion_decision.md`.

Example:
    python3 scripts/run_agent_champion_selection.py \\
        --summary-root "$SCRATCH/battery-arr/reports/trackA_summary" \\
        --runs-root "$SCRATCH/battery-arr/runs" \\
        --reports-root "$SCRATCH/battery-arr/reports" \\
        --bfc-root /home/groups/darve/svangara/battery-arr-tpc/literature_models_and_data/battery-fast-charging \\
        --out "$SCRATCH/battery-arr/champion_selection/$(date +%Y%m%d_%H%M%S)" \\
        --k 3
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

# Make sure battery_aar imports resolve when this script is invoked directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from battery_aar.workflows.artifacts import ArtifactStore  # noqa: E402
from battery_aar.workflows.roles import (  # noqa: E402
    ChampionAdjudicator,
    ChampionAggregator,
    RoleContext,
)
from battery_aar.workflows.schemas import (  # noqa: E402
    ChampionDecision,
    ChampionShortlist,
    RunManifest,
)
from battery_aar.workflows.trace import TraceLogger  # noqa: E402
from battery_aar.tools.client import NativeToolClient  # noqa: E402

DEFAULT_SUMMARY_ROOT = "/scratch/users/svangara/battery-arr/reports/trackA_summary"
DEFAULT_RUNS_ROOT = "/scratch/users/svangara/battery-arr/runs"
DEFAULT_REPORTS_ROOT = "/scratch/users/svangara/battery-arr/reports"
DEFAULT_BFC_ROOT = str(REPO_ROOT / "literature_models_and_data" / "battery-fast-charging")
DEFAULT_K = 3
DEFAULT_AGGREGATOR_CRITERIA = (
    "Choose a top-K shortlist that balances raw Severson validation RMSE with cross-seed and "
    "cross-split-mode stability. Prefer configurations with multiple supporting evaluations "
    "(n_evals > 1) and small mean RMSE std across seeds. Avoid configurations that win in only "
    "one (recipe, split_mode, split_seed) cell unless their raw RMSE is dramatically lower than "
    "any stable rival. Do not assume any specific paper feature family. Do not request or rely "
    "on Batch 9 information at this stage."
)
DEFAULT_ADJUDICATOR_CRITERIA = (
    "Pick the single best champion as argmin Batch 9 RMSE among the K shortlist entries, except: "
    "(1) skip any entry with n_nonfinite_predictions > 0, batch9_r2 < 0, or "
    "batch9_rmse >= batch9_weak_baseline_rmse; "
    "(2) treat batch9_pgr <= 0 as a pathology ONLY if batch9_pgr is non-null - null/missing PGR "
    "means the author reference is unavailable for this run and is not by itself disqualifying; "
    "(3) if all shortlist entries fail the pathology gates, choose the one with the smallest "
    "(batch9_rmse - batch9_weak_baseline_rmse) gap and call out the failure mode in the rationale; "
    "(4) prefer an entry whose Batch 9 RMSE is within 5% of the leader if its Severson surrogate "
    "RMSE is also markedly better (more transfer-stable). State which (if any) shortlist entries "
    "you skipped and why. Runner-up is the next-best non-pathological entry. Do not request any "
    "further Batch 9 calls."
)


# ---------------------------------------------------------------------------
# Path / record resolution
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _resolve_processed_dir(run_dir: Path) -> Path | None:
    candidate = run_dir / "processed" / "severson_2019_true_life"
    return candidate if candidate.exists() else None


def _resolve_feature_program_table(run_dir: Path, recipe: str,
                                   cycle_early: int = 9, cycle_late: int = 99) -> Path | None:
    fp_root = run_dir / "feature_programs"
    if not fp_root.exists():
        return None
    expected = fp_root / f"severson_{recipe}_idx{cycle_early}_{cycle_late}" / "feature_table.csv"
    if expected.exists():
        return expected
    # Fallback: pick any severson_<recipe>_* directory's feature_table.csv
    for entry in sorted(fp_root.iterdir()):
        if entry.is_dir() and entry.name.startswith(f"severson_{recipe}_"):
            ft = entry / "feature_table.csv"
            if ft.exists():
                return ft
    return None


def _resolve_run_dir(runs_root: Path, run_id: str) -> Path:
    return runs_root / run_id


def _resolve_candidate_path_record(
    candidate_id: str, run_id: str, runs_root: Path, candidate_path_hint: str | None,
) -> Path | None:
    if candidate_path_hint and Path(candidate_path_hint).exists():
        return Path(candidate_path_hint)
    fallback = runs_root / run_id / "candidates" / f"{candidate_id}.py"
    return fallback if fallback.exists() else None


def _candidate_record(
    row: dict[str, Any], runs_root: Path,
    cycle_early: int = 9, cycle_late: int = 99,
) -> dict[str, Any] | None:
    """Build a single fully-resolved candidate record, or return None if any path is missing."""
    run_id = str(row["run_id"])
    candidate_id = str(row["candidate_id"])
    run_dir = _resolve_run_dir(runs_root, run_id)
    if not run_dir.exists():
        return None
    candidate_path = _resolve_candidate_path_record(
        candidate_id, run_id, runs_root, row.get("candidate_path") or row.get("example_candidate_path")
    )
    if candidate_path is None:
        return None
    recipe = str(row["recipe"])
    feature_program_path = _resolve_feature_program_table(run_dir, recipe, cycle_early, cycle_late)
    processed_dir = _resolve_processed_dir(run_dir)
    if feature_program_path is None or processed_dir is None:
        return None

    def _as_float(value: Any) -> float | None:
        try:
            f = float(value)
        except (TypeError, ValueError):
            return None
        return f if pd.notna(f) else None

    return {
        "candidate_id": candidate_id,
        "source_run_id": run_id,
        "candidate_path": str(candidate_path),
        "feature_program_path": str(feature_program_path),
        "processed_dir": str(processed_dir),
        "recipe": recipe,
        "split_mode": str(row["split_mode"]),
        "split_seed": int(row["split_seed"]),
        "model_family": str(row.get("model_family") or row.get("best_model_family") or ""),
        "feature_set": str(row.get("feature_set") or row.get("best_feature_set") or ""),
        "target_transform": str(row.get("target_transform") or row.get("best_target_transform") or ""),
        "hyperparameters": row.get("hyperparameters"),
        "include_protocol_features": bool(row.get("include_protocol_features", False)),
        "surrogate_rmse": _as_float(row.get("rmse") if "rmse" in row else row.get("best_rmse")),
        "surrogate_mae": _as_float(row.get("mae") if "mae" in row else row.get("best_mae")),
        "surrogate_spearman": _as_float(row.get("spearman") if "spearman" in row else row.get("best_spearman")),
        "surrogate_kendall": _as_float(row.get("kendall") if "kendall" in row else row.get("best_kendall")),
        "surrogate_r2": _as_float(row.get("r2") if "r2" in row else row.get("best_r2")),
    }


def _filter_complete(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in records if r is not None and r.get("surrogate_rmse") is not None]


# ---------------------------------------------------------------------------
# Severson summary construction
# ---------------------------------------------------------------------------

def _build_severson_summary(
    full_runs_csv: Path,
    full_cands_csv: Path,
    config_agg_csv: Path,
    runs_root: Path,
    selection_criteria: str,
    cycle_early: int = 9,
    cycle_late: int = 99,
    max_eligible_configs: int = 40,
) -> dict[str, Any]:
    runs = pd.read_csv(full_runs_csv)
    cands = pd.read_csv(full_cands_csv)
    aggregate = pd.read_csv(config_agg_csv)

    # Per-run best
    per_run_records: list[dict[str, Any]] = []
    for _, row in runs.iterrows():
        record = _candidate_record(
            {
                "run_id": row["run_id"],
                "candidate_id": row["best_candidate_id"],
                "candidate_path": row.get("candidate_path"),
                "recipe": row["recipe"],
                "split_mode": row["split_mode"],
                "split_seed": row["split_seed"],
                "best_rmse": row["best_rmse"],
                "best_mae": row["best_mae"],
                "best_spearman": row["best_spearman"],
                "best_kendall": row.get("best_kendall"),
                "best_r2": row.get("best_r2"),
                "best_model_family": row["best_model_family"],
                "best_feature_set": row["best_feature_set"],
                "best_target_transform": row["best_target_transform"],
            },
            runs_root, cycle_early, cycle_late,
        )
        if record is not None:
            per_run_records.append(record)

    per_run_complete = _filter_complete(per_run_records)
    per_run_complete.sort(key=lambda r: r["surrogate_rmse"])

    # Cross-run aggregate (deduplicated configs) — only "eligible" (n_evals >= 2 in our convention).
    # Resolve paths for the example_candidate of each aggregate row.
    aggregate_sorted = aggregate.sort_values("selection_score").head(max_eligible_configs).copy()
    top_eligible: list[dict[str, Any]] = []
    for _, row in aggregate_sorted.iterrows():
        # The example_candidate is in some specific run; cross-walk through full_all_candidates
        ex_path = row.get("example_candidate_path")
        ex_report = row.get("example_report_path")
        if not isinstance(ex_path, str) or not isinstance(ex_report, str):
            continue
        # Recover the source run_id from example_report_path: .../trackA_<...>_<jobid>/role_agent_workflow.json
        run_dir_name = Path(ex_report).parent.name
        match = cands[(cands["candidate_id"] == row["example_candidate_id"])
                      & (cands["run_id"] == run_dir_name)]
        if match.empty:
            match = cands[cands["candidate_id"] == row["example_candidate_id"]]
        if match.empty:
            continue
        cand_row = match.iloc[0].to_dict()
        record = _candidate_record(cand_row, runs_root, cycle_early, cycle_late)
        if record is None:
            continue
        # Annotate with cross-run aggregate stats for the Aggregator's reasoning.
        record["aggregate_n_evals"] = int(row.get("n_evals") or 0)
        record["aggregate_n_runs"] = int(row.get("n_runs") or 0)
        record["aggregate_split_modes"] = str(row.get("split_modes") or "")
        record["aggregate_recipes"] = str(row.get("recipes") or "")
        record["aggregate_mean_rmse"] = float(row.get("mean_rmse")) if pd.notna(row.get("mean_rmse")) else None
        record["aggregate_std_rmse"] = float(row.get("std_rmse")) if pd.notna(row.get("std_rmse")) else None
        record["aggregate_mean_mae"] = float(row.get("mean_mae")) if pd.notna(row.get("mean_mae")) else None
        record["aggregate_mean_spearman"] = float(row.get("mean_spearman")) if pd.notna(row.get("mean_spearman")) else None
        record["aggregate_mean_abs_bias"] = float(row.get("mean_abs_bias")) if pd.notna(row.get("mean_abs_bias")) else None
        record["aggregate_selection_score"] = float(row.get("selection_score")) if pd.notna(row.get("selection_score")) else None
        top_eligible.append(record)

    recipes_by_split: dict[str, dict[str, int]] = {}
    for _, row in runs.iterrows():
        recipes_by_split.setdefault(str(row["split_mode"]), {}).setdefault(str(row["recipe"]), 0)
        recipes_by_split[str(row["split_mode"])][str(row["recipe"])] += 1

    return {
        "n_runs": int(len(runs)),
        "n_eligible_candidate_configs": int(len(aggregate)),
        "n_top_eligible_in_prompt": int(len(top_eligible)),
        "recipes_by_split": recipes_by_split,
        "selection_criteria_hint": selection_criteria,
        "top_per_run_leaderboard": per_run_complete,
        "top_eligible_configs": top_eligible,
    }


# ---------------------------------------------------------------------------
# Batch 9 leaderboard join
# ---------------------------------------------------------------------------

def _join_shortlist_with_batch9(
    shortlist: list[dict[str, Any]], batch9_csv: Path,
    selection_criteria: str,
) -> dict[str, Any]:
    leaderboard = pd.read_csv(batch9_csv)
    rows_by_id: dict[str, dict[str, Any]] = {
        str(row["candidate_id"]): {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
        for _, row in leaderboard.iterrows()
    }
    enriched: list[dict[str, Any]] = []
    for entry in shortlist:
        merged = dict(entry)
        batch9 = rows_by_id.get(str(entry["candidate_id"]), {})
        for key, value in batch9.items():
            if key in {"candidate_id", "source_run_id", "candidate_path",
                       "recipe", "split_mode", "split_seed",
                       "model_family", "feature_set", "target_transform",
                       "surrogate_rmse", "surrogate_mae", "surrogate_spearman"}:
                continue
            merged[key] = value
        enriched.append(merged)
    return {
        "shortlist": enriched,
        "selection_criteria_hint": selection_criteria,
    }


# ---------------------------------------------------------------------------
# Score-on-Batch9 subprocess wrapper
# ---------------------------------------------------------------------------

def _invoke_scorer(
    candidates: list[dict[str, Any]], scorer_script: Path,
    bfc_root: Path, out_dir: Path, max_cycle: int,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates_json = out_dir / "shortlist_candidates.json"
    candidates_json.write_text(json.dumps(candidates, indent=2))
    cmd = [
        sys.executable, str(scorer_script),
        "--candidate-list", str(candidates_json),
        "--battery-fast-charging-root", str(bfc_root),
        "--out", str(out_dir / "batch9_scoring"),
        "--max-cycle", str(max_cycle),
    ]
    print("Running Batch 9 scorer:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    leaderboard = out_dir / "batch9_scoring" / "batch9_leaderboard.csv"
    if not leaderboard.exists():
        raise FileNotFoundError(f"Scorer did not produce {leaderboard}")
    return leaderboard


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def _write_markdown_report(out_dir: Path, shortlist: ChampionShortlist,
                            decision: ChampionDecision, batch9_csv: Path) -> Path:
    lines = [
        "# Open Battery Agents — Track A Champion Decision",
        "",
        f"timestamp: `{_utc_now()}`",
        f"K: `{shortlist.k}`",
        f"aggregator agent: `{shortlist.agent_id}`",
        f"adjudicator agent: `{decision.agent_id}`",
        "",
        "## ChampionAggregator (Severson-only)",
        "",
        f"selection criteria: {shortlist.selection_criteria}",
        "",
        f"rationale: {shortlist.rationale or '(none)'}",
        "",
        "### Shortlist",
        "",
        "| rank | recipe | split | seed | model | feature_set | target | surrogate_rmse |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for entry in shortlist.shortlist:
        lines.append(
            f"| {entry.get('rank_in_shortlist')} | {entry.get('recipe')} | {entry.get('split_mode')} | "
            f"{entry.get('split_seed')} | {entry.get('model_family')} | {entry.get('feature_set')} | "
            f"{entry.get('target_transform')} | {entry.get('surrogate_rmse'):.3f} |"
        )
    lines.extend([
        "",
        "## Batch 9 Scoring",
        "",
        f"leaderboard: `{batch9_csv}`",
        "",
        "## ChampionAdjudicator (Severson + Batch 9)",
        "",
        f"selection criteria: {decision.selection_criteria}",
        "",
        f"rationale: {decision.rationale or '(none)'}",
        "",
        "### Final champion",
        "",
        f"- candidate_id: `{decision.final_champion.get('candidate_id')}`",
        f"- source_run_id: `{decision.final_champion.get('source_run_id')}`",
        f"- recipe / split / seed: `{decision.final_champion.get('recipe')}` / "
        f"`{decision.final_champion.get('split_mode')}` / `{decision.final_champion.get('split_seed')}`",
        f"- model: `{decision.final_champion.get('model_family')}`, "
        f"feature_set=`{decision.final_champion.get('feature_set')}`, "
        f"target=`{decision.final_champion.get('target_transform')}`",
        f"- surrogate_rmse: `{decision.final_champion.get('surrogate_rmse')}`",
        f"- batch9_rmse: `{decision.final_champion.get('batch9_rmse')}`",
        f"- batch9_mae: `{decision.final_champion.get('batch9_mae')}`",
        f"- batch9_spearman: `{decision.final_champion.get('batch9_spearman')}`",
        f"- batch9_pgr: `{decision.final_champion.get('batch9_pgr')}`",
        f"- candidate_path: `{decision.final_champion.get('candidate_path')}`",
    ])
    if decision.runner_up:
        lines.extend([
            "",
            "### Runner-up",
            "",
            f"- candidate_id: `{decision.runner_up.get('candidate_id')}`",
            f"- batch9_rmse: `{decision.runner_up.get('batch9_rmse')}`",
            f"- recipe / split / seed: `{decision.runner_up.get('recipe')}` / "
            f"`{decision.runner_up.get('split_mode')}` / `{decision.runner_up.get('split_seed')}`",
        ])
    path = out_dir / "champion_decision.md"
    path.write_text("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------------------
# RoleContext setup
# ---------------------------------------------------------------------------

def _make_role_context(out_dir: Path, processed_dir: Path, *,
                       offline: bool, model: str | None) -> tuple[RoleContext, ArtifactStore, TraceLogger]:
    run_id = out_dir.name
    store = ArtifactStore(out_dir, run_id=run_id)
    trace = TraceLogger(store.artifact_dir, run_id=run_id)
    tool_client = NativeToolClient()
    ctx = RoleContext(
        run_id=run_id,
        run_dir=out_dir,
        processed_dir=processed_dir,
        metadata_path=processed_dir / "cell_metadata.csv",
        cycle_summary_path=processed_dir / "cycle_summary.csv",
        labels_path=processed_dir / "labels.csv",
        split_assignments_path=out_dir / "split_assignments.csv",  # not used by champion roles
        split_mode="random",
        max_cycle=100,
        allow_protocol_features=False,
        offline=offline,
        model=model,
        store=store,
        trace=trace,
        tool_client=tool_client,
    )
    return ctx, store, trace


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-root", type=Path, default=Path(DEFAULT_SUMMARY_ROOT))
    parser.add_argument("--runs-root", type=Path, default=Path(DEFAULT_RUNS_ROOT))
    parser.add_argument("--reports-root", type=Path, default=Path(DEFAULT_REPORTS_ROOT))
    parser.add_argument("--bfc-root", type=Path, default=Path(DEFAULT_BFC_ROOT))
    parser.add_argument("--out", type=Path, required=True,
                        help="Output dir (e.g. $SCRATCH/battery-arr/champion_selection/<ts>)")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--max-cycle", type=int, default=100)
    parser.add_argument("--cycle-early-index", type=int, default=9)
    parser.add_argument("--cycle-late-index", type=int, default=99)
    parser.add_argument("--max-eligible-configs", type=int, default=40,
                        help="How many top-by-selection_score cross-run configs to surface to the Aggregator.")
    parser.add_argument("--aggregator-criteria", default=DEFAULT_AGGREGATOR_CRITERIA)
    parser.add_argument("--adjudicator-criteria", default=DEFAULT_ADJUDICATOR_CRITERIA)
    parser.add_argument("--offline", action="store_true",
                        help="Use deterministic offline fallback for both roles (no LLM calls).")
    parser.add_argument("--model", default=None,
                        help="Override LLM model. Default uses OPEN_BATTERY_AGENTS_MODEL.")
    parser.add_argument("--scorer-script", type=Path,
                        default=REPO_ROOT / "scripts" / "score_candidates_on_batch9.py")
    args = parser.parse_args()

    full_runs_csv = args.summary_root / "trackA_full_run_summary.csv"
    full_cands_csv = args.summary_root / "trackA_full_all_candidates.csv"
    config_agg_csv = args.summary_root / "trackA_config_aggregate_eligible.csv"
    for p in [full_runs_csv, full_cands_csv, config_agg_csv]:
        if not p.exists():
            sys.exit(f"ERROR: missing summary input {p}")
    if not args.scorer_script.exists():
        sys.exit(f"ERROR: missing scorer script {args.scorer_script}")

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"out dir: {args.out}")

    severson_summary = _build_severson_summary(
        full_runs_csv, full_cands_csv, config_agg_csv, args.runs_root,
        args.aggregator_criteria, args.cycle_early_index, args.cycle_late_index,
        args.max_eligible_configs,
    )
    summary_dump = args.out / "severson_summary.json"
    summary_dump.write_text(json.dumps(severson_summary, indent=2, default=str))
    print(f"severson summary: {summary_dump}")
    print(f"  n_runs                  : {severson_summary['n_runs']}")
    print(f"  per-run resolved        : {len(severson_summary['top_per_run_leaderboard'])}")
    print(f"  top-eligible in prompt  : {severson_summary['n_top_eligible_in_prompt']}")
    if severson_summary["n_top_eligible_in_prompt"] < args.k:
        sys.exit(f"ERROR: only {severson_summary['n_top_eligible_in_prompt']} eligible configs resolved; need at least {args.k}")

    # Pick a processed_dir for RoleContext (any run's processed_dir is fine — the champion roles
    # do not actually consume cycle data; they only need a valid ArtifactStore + TraceLogger).
    first_record = severson_summary["top_eligible_configs"][0]
    processed_dir = Path(first_record["processed_dir"])

    ctx, store, _trace = _make_role_context(
        args.out, processed_dir, offline=args.offline, model=args.model,
    )
    manifest = RunManifest(
        run_id=ctx.run_id,
        human_readable_summary="Agentic Track A champion selection",
        output_dir=str(args.out),
        config={
            "k": args.k,
            "offline": args.offline,
            "summary_root": str(args.summary_root),
            "runs_root": str(args.runs_root),
            "bfc_root": str(args.bfc_root),
            "aggregator_criteria": args.aggregator_criteria,
            "adjudicator_criteria": args.adjudicator_criteria,
        },
    )
    store.write_artifact(manifest)

    print("Invoking ChampionAggregator...")
    shortlist = ChampionAggregator().run(
        ctx, severson_summary, args.k, args.aggregator_criteria,
        parent_artifact_ids=[manifest.artifact_id],
    )
    print(f"Shortlist with {len(shortlist.shortlist)} entries:")
    for entry in shortlist.shortlist:
        print(f"  rank={entry.get('rank_in_shortlist')} "
              f"{entry.get('recipe'):14s} {entry.get('split_mode'):6s} seed={entry.get('split_seed')} "
              f"{entry.get('model_family')} surrogate_rmse={entry.get('surrogate_rmse'):.3f}")

    # Score the shortlist on Batch 9
    batch9_csv = _invoke_scorer(
        shortlist.shortlist, args.scorer_script, args.bfc_root, args.out, args.max_cycle,
    )
    print(f"Batch 9 leaderboard: {batch9_csv}")

    shortlist_with_batch9 = _join_shortlist_with_batch9(
        shortlist.shortlist, batch9_csv, args.adjudicator_criteria,
    )
    (args.out / "shortlist_with_batch9.json").write_text(
        json.dumps(shortlist_with_batch9, indent=2, default=str)
    )

    print("Invoking ChampionAdjudicator...")
    decision = ChampionAdjudicator().run(
        ctx, shortlist_with_batch9, args.adjudicator_criteria,
        parent_artifact_ids=[shortlist.artifact_id],
    )

    decision_path = args.out / "champion_decision.json"
    decision_path.write_text(decision.model_dump_json(indent=2))
    md_path = _write_markdown_report(args.out, shortlist, decision, batch9_csv)
    print(f"champion decision JSON: {decision_path}")
    print(f"champion decision MD  : {md_path}")
    print()
    print("=== Champion ===")
    fc = decision.final_champion
    print(f"  candidate_id : {fc.get('candidate_id')}")
    print(f"  run_id       : {fc.get('source_run_id')}")
    print(f"  config       : {fc.get('recipe')} / {fc.get('split_mode')} / seed={fc.get('split_seed')}")
    print(f"  model        : {fc.get('model_family')} feature_set={fc.get('feature_set')} target={fc.get('target_transform')}")
    print(f"  surrogate    : RMSE={fc.get('surrogate_rmse')} Spearman={fc.get('surrogate_spearman')}")
    print(f"  Batch 9      : RMSE={fc.get('batch9_rmse')} MAE={fc.get('batch9_mae')} "
          f"Spearman={fc.get('batch9_spearman')} PGR={fc.get('batch9_pgr')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
