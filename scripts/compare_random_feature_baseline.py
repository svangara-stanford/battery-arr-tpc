#!/usr/bin/env python
"""Compare a random-feature baseline against the RAG+LLM FeatureScientist.

Ablation design (fix the model, vary only feature selection):

  * random baseline   -- random subset of the build_all_battery_features pool,
                          Ridge(log10). Swept over several seeds and averaged so
                          it is not judged on one lucky draw.
  * agent-features     -- the agent's selected feature set (all_available => the
    control              full pool) with the SAME Ridge(log10) model. Isolates
                          feature choice from model choice.
  * agent end-to-end   -- the best candidate the workflow actually produced
                          (its own model), read from a completed run's
                          evaluation reports. Shown for context.

Every row is scored through the identical evaluate_candidate_train_test harness
on the identical make_search_split, so RMSE/MAPE/R2 are directly comparable.

When comparing against feature-programs-enabled runs, pass the SAME program
table here so the random draw and the agent-features control use the same
expanded feature pool (otherwise the baseline draws from the 28-feature default
while the agent saw hundreds of features -- not apples-to-apples).

Usage:
  # Default 28-feature pool, RAG vs LLM-only agent runs:
  python scripts/compare_random_feature_baseline.py \
      --processed-dir data/processed/chueh_toyota_fast_charge_agent_surrogate \
      --split-mode random --validation-fraction 0.25 --split-seed 0 \
      --n-features 8 --baseline-seeds 0 1 2 3 4 5 6 7 \
      --agent-run runs/open_battery_agents/rag_live_3iter_random \
      --agent-run runs/open_battery_agents/llm_only_3iter_random

  # Feature-programs-enabled pool (~630 features) -- match the agent FP runs:
  python scripts/compare_random_feature_baseline.py \
      --processed-dir data/processed/chueh_toyota_fast_charge_agent_surrogate \
      --split-mode random --validation-fraction 0.25 --split-seed 0 \
      --n-features 8 --baseline-seeds 0 1 2 3 4 5 6 7 \
      --include-feature-programs --feature-program-mode table \
      --feature-program-paths \
        data/processed/chueh_toyota_fast_charge_feature_programs/broad_physics_idx9_99 \
      --agent-run runs/open_battery_agents/rag_fp_3iter_random \
      --agent-run runs/open_battery_agents/llm_only_fp_3iter_random
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
import tempfile
from pathlib import Path

import pandas as pd

from battery_aar.agents.evaluator import evaluate_candidate_train_test
from battery_aar.agents.orchestrator import make_search_split
from battery_aar.agents.random_feature_baseline import (
    random_feature_baseline_candidate,
    random_feature_baseline_candidates,
)
from battery_aar.workflows.role_graph import _finite_labels, _load_processed


def _subset(df: pd.DataFrame, ids: set[int]) -> pd.DataFrame:
    return df[df["row_id"].isin(ids)].copy()


def _evaluate_code(code: str, split_frames, max_cycle: int,
                   feature_program_paths=None, feature_program_mode="none",
                   include_feature_programs=False) -> dict:
    (train_meta, train_cycles, train_labels,
     val_meta, val_cycles, val_labels) = split_frames
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(code)
        candidate_path = fh.name
    result = evaluate_candidate_train_test(
        candidate_path,
        train_meta, train_cycles, train_labels,
        val_meta, val_cycles, val_labels,
        max_cycle=max_cycle,
        feature_program_paths=feature_program_paths,
        feature_program_mode=feature_program_mode,
        include_feature_programs=include_feature_programs,
    )
    Path(candidate_path).unlink(missing_ok=True)
    return result


def _agent_best_metrics(agent_run: Path) -> dict | None:
    """Best (lowest-RMSE) variant metrics from a completed workflow run."""
    reports = sorted(glob.glob(
        str(agent_run / "artifacts" / "iteration_*"
            / "evaluation_report_role_graph*variant*.json")
    ))
    best = None
    for path in reports:
        data = json.load(open(path))
        rmse = data.get("rmse")
        if rmse is None:
            continue
        if best is None or rmse < best["rmse"]:
            best = data
    return best


def _fmt(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "   n/a"
    return f"{value:8.3f}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--processed-dir", type=Path, required=True)
    p.add_argument("--split-mode",
                   choices=["random", "protocol", "batch", "leave_one_batch_out"],
                   default="random")
    p.add_argument("--validation-fraction", type=float, default=0.25)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--validation-batch-id", default=None)
    p.add_argument("--max-cycle", type=int, default=100)
    p.add_argument("--n-features", type=int, default=8,
                   help="How many features the random baseline draws.")
    p.add_argument("--baseline-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--agent-run", type=Path, action="append", default=None,
                   help="Completed workflow run dir to read the agent's "
                        "end-to-end best metrics from. Repeatable: pass once "
                        "per run (e.g. a RAG run and a --no-rag run) to see "
                        "them as separate rows.")
    p.add_argument("--include-feature-programs", action="store_true",
                   help="Expand the feature pool with a feature-program table "
                        "so the baseline matches feature-programs-enabled agent runs.")
    p.add_argument("--feature-program-mode", choices=["none", "table", "auto"],
                   default="none")
    p.add_argument("--feature-program-paths", nargs="*", type=Path, default=[],
                   help="Feature-program table dir(s), e.g. the broad_physics program.")
    args = p.parse_args()

    fp_paths = [str(p_) for p_ in (args.feature_program_paths or [])]
    fp_kwargs = dict(
        feature_program_paths=fp_paths,
        feature_program_mode=args.feature_program_mode,
        include_feature_programs=args.include_feature_programs,
    )

    metadata, cycles, labels, _labels_path, splits_table = _load_processed(args.processed_dir)
    labels = _finite_labels(labels)
    train_ids, val_ids, _test_ids, _manifest, _assignments = make_search_split(
        metadata,
        labels,
        split_mode=args.split_mode,
        validation_fraction=args.validation_fraction,
        split_seed=args.split_seed,
        validation_batch_id=args.validation_batch_id,
        splits_table=splits_table,
    )
    train_ids, val_ids = set(map(int, train_ids)), set(map(int, val_ids))
    split_frames = (
        _subset(metadata, train_ids), _subset(cycles, train_ids), _subset(labels, train_ids),
        _subset(metadata, val_ids), _subset(cycles, val_ids), _subset(labels, val_ids),
    )
    pool_note = f" (+feature programs: {', '.join(fp_paths)})" if fp_paths else ""
    print(f"Split '{args.split_mode}': {len(train_ids)} train / "
          f"{len(val_ids)} val cells{pool_note}\n")

    rows: list[tuple[str, dict]] = []

    # Random-feature baseline, averaged over seeds.
    per_seed: list[dict] = []
    for cand in random_feature_baseline_candidates(
        seeds=args.baseline_seeds, n_features=args.n_features,
        max_cycle=args.max_cycle, ridge_alpha=args.ridge_alpha, **fp_kwargs,
    ):
        res = _evaluate_code(cand.code, split_frames, args.max_cycle, **fp_kwargs)
        if not res["success"]:
            print(f"  [warn] random baseline seed {cand.seed} failed: "
                  f"{res.get('failure_reason')}")
            continue
        per_seed.append(res["metrics"])
    if per_seed:
        def _avg(key):
            vals = [m[key] for m in per_seed if m.get(key) is not None
                    and not math.isnan(m[key])]
            return statistics.mean(vals) if vals else float("nan")
        agg = {k: _avg(k) for k in ("rmse", "mae", "mape", "r2", "spearman", "kendall")}
        label = f"random features (n={args.n_features}, {len(per_seed)} seeds avg)"
        rows.append((label, agg))
        # Also show the spread on the headline metric.
        rmses = [m["rmse"] for m in per_seed]
        print(f"random-baseline RMSE per seed: "
              f"{', '.join(f'{r:.1f}' for r in rmses)} "
              f"(min {min(rmses):.1f}, max {max(rmses):.1f})\n")

    # Agent-features control: same fixed model, full feature pool (all_available).
    control = random_feature_baseline_candidate(
        seed=0, n_features=1_000_000,  # >= pool size => selects all columns
        max_cycle=args.max_cycle, ridge_alpha=args.ridge_alpha, **fp_kwargs,
    )
    res = _evaluate_code(control.code, split_frames, args.max_cycle, **fp_kwargs)
    if res["success"]:
        rows.append(("agent features (all) + same Ridge(log10)", res["metrics"]))

    # Agent end-to-end best (its own model), for context. One row per run so a
    # RAG run and an LLM-only (--no-rag) run appear side by side.
    for agent_run in (args.agent_run or []):
        best = _agent_best_metrics(agent_run)
        if best is not None:
            rows.append((f"agent end-to-end best ({agent_run.name})", {
                "rmse": best.get("rmse"), "mae": best.get("mae"),
                "mape": best.get("mape"), "r2": best.get("r2"),
                "spearman": best.get("spearman"), "kendall": best.get("kendall"),
            }))

    # Print comparison table.
    header = f"{'candidate':<44}{'RMSE':>9}{'MAE':>9}{'MAPE%':>9}{'R2':>9}{'Spear':>9}"
    print(header)
    print("-" * len(header))
    for label, m in rows:
        mape = m.get("mape")
        mape_pct = mape * 100 if mape is not None and not math.isnan(mape) else None
        print(f"{label:<44}{_fmt(m.get('rmse'))}{_fmt(m.get('mae'))}"
              f"{_fmt(mape_pct)}{_fmt(m.get('r2'))}{_fmt(m.get('spearman'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
