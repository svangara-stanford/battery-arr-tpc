#!/usr/bin/env python
"""Compare RAG vs no-RAG FeatureScientist runs under a feature budget.

Runs the role-agent workflow with the operator-spec feature path at a fixed
--feature-budget, once with RAG and once without (optionally also on a stronger
model), plus a random-feature baseline drawn from the same pool. Then prints,
for each run:
  * validation metrics (RMSE / MAE / MAPE / R2 / Spearman) of the best candidate
  * the exact feature columns the best candidate trained on
  * the agent's rationale and per-operator hypotheses (why it chose them)

This reproduces the budget-N equalizer experiment. Because the runs call a live
LLM, exact numbers vary between invocations; use --summarize-only to re-print
from already-completed run directories without re-running.

Examples:
  # Full run at budget 5, weak model only (+ random baseline):
  source .venv/bin/activate && set -a && . ./.env && set +a
  python scripts/compare_rag_feature_budget.py --feature-budget 5

  # Also run the stronger model:
  python scripts/compare_rag_feature_budget.py --feature-budget 8 \
      --strong-model claude-sonnet-4-6

  # Just re-print results from existing runs (no LLM calls):
  python scripts/compare_rag_feature_budget.py --feature-budget 5 --summarize-only
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED = "data/processed/chueh_toyota_fast_charge_agent_surrogate"
DEFAULT_PROGRAM = "data/processed/chueh_toyota_fast_charge_feature_programs/broad_physics_idx9_99"
RUN_ROOT = "runs/open_battery_agents"


def _run_workflow(out_dir: str, *, model: str, rag: bool, budget: int,
                  processed: str, program: str, iterations: int,
                  candidates: int, split_seed: int) -> None:
    """Invoke the role-agent workflow for one cell of the comparison."""
    cmd = [
        sys.executable, "scripts/run_role_agent_workflow.py",
        "--processed-dir", processed,
        "--out", out_dir,
        "--reports-dir", "reports",
        "--split-mode", "random", "--validation-fraction", "0.25",
        "--split-seed", str(split_seed),
        "--iterations", str(iterations),
        "--candidates-per-iteration", str(candidates),
        "--max-cycle", "100",
        "--feature-budget", str(budget),
        "--model", model,
        "--include-feature-programs", "--feature-program-mode", "table",
        "--feature-program-paths", program,
    ]
    if not rag:
        cmd.append("--no-rag")
    print(f"  running {out_dir} (model={model}, rag={rag}, budget={budget}) ...", flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _best_report(run_dir: Path) -> tuple[dict, str] | None:
    """Lowest-RMSE successful evaluation report (champion OR variant).

    Reading both kinds matters: the champion report can be a failed repair
    candidate while variant candidates succeeded.
    """
    reports = sorted(glob.glob(str(run_dir / "artifacts" / "iteration_*" / "evaluation_report_*.json")))
    best = None
    best_path = ""
    for path in reports:
        r = json.load(open(path))
        if not r.get("success") or r.get("rmse") is None:
            continue
        if best is None or r["rmse"] < best["rmse"]:
            best, best_path = r, path
    return (best, best_path) if best is not None else None


def _fit_columns(report: dict) -> list[str]:
    fu = report.get("extra_metrics", {}).get("features_used_path")
    if fu and os.path.exists(fu):
        return json.load(open(fu)).get("fit_feature_columns", [])
    return []


def _feature_plan_for_iteration(run_dir: Path, iteration: int) -> dict | None:
    hits = glob.glob(str(run_dir / "artifacts" / f"iteration_{iteration:03d}" / "feature_plan_*.json"))
    return json.load(open(hits[0])) if hits else None


def _mape_pct(report: dict) -> float:
    mape = report.get("mape")
    if mape is not None:
        return mape * 100.0
    return report.get("test_error_pct", float("nan"))


def _summarize(cells: list[tuple[str, str]], random_row: dict | None) -> None:
    # Metrics table.
    print("\n" + "=" * 78)
    print("VALIDATION METRICS (best candidate per run)")
    print("=" * 78)
    header = f"{'run':<26}{'RMSE':>9}{'MAE':>9}{'MAPE%':>8}{'R2':>8}{'Spear':>8}"
    print(header)
    print("-" * len(header))
    resolved = []
    for label, run in cells:
        found = _best_report(REPO_ROOT / RUN_ROOT / run)
        if not found:
            print(f"{label:<26}{'NO SUCCESSFUL CANDIDATES':>42}")
            resolved.append((label, run, None, ""))
            continue
        r, path = found
        print(f"{label:<26}{r['rmse']:>9.2f}{r['mae']:>9.2f}{_mape_pct(r):>8.2f}{r['r2']:>8.3f}{r['spearman']:>8.3f}")
        resolved.append((label, run, r, path))
    if random_row is not None:
        m = random_row
        mape = (m.get("mape") or 0) * 100
        print(f"{'random (same pool)':<26}{m['rmse']:>9.2f}{m['mae']:>9.2f}{mape:>8.2f}{m['r2']:>8.3f}{m['spearman']:>8.3f}")

    # Features + reasoning per run.
    for label, run, r, _path in resolved:
        if r is None:
            continue
        print("\n" + "=" * 78)
        print(f"{label}  (candidate {r.get('candidate_id')}, iteration {r.get('iteration')})")
        print("=" * 78)
        cols = _fit_columns(r)
        print(f"Features used ({len(cols)}):")
        for c in cols:
            print(f"  - {c}")
        plan = _feature_plan_for_iteration(REPO_ROOT / RUN_ROOT / run, int(r.get("iteration") or 0))
        if plan:
            print("\nAgent rationale:")
            print(f"  {plan.get('rationale')}")
            ops = plan.get("feature_operators", [])
            if ops:
                print("\nPer-operator reasoning:")
                for o in ops:
                    print(f"  - {o.get('operator_type')} {o.get('params')}")
                    if o.get("description"):
                        print(f"      why: {o['description']}")


def _random_baseline(budget: int, processed: str, program: str, split_seed: int) -> dict | None:
    """Run the random-feature baseline at n=budget and parse its avg metrics."""
    cmd = [
        sys.executable, "scripts/compare_random_feature_baseline.py",
        "--processed-dir", processed,
        "--split-mode", "random", "--validation-fraction", "0.25",
        "--split-seed", str(split_seed),
        "--n-features", str(budget),
        "--baseline-seeds", "0", "1", "2", "3", "4", "5", "6", "7",
        "--include-feature-programs", "--feature-program-mode", "table",
        "--feature-program-paths", program,
    ]
    out = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.strip().startswith("random features"):
            # The row label is "random features (n=5, N seeds avg)" followed by
            # the numeric columns. Split off the label at its closing paren so
            # the n=5 / seed count inside it aren't parsed as metric values.
            after_label = line.split(")", 1)[1] if ")" in line else line
            nums = [float(x) for x in after_label.split() if _is_float(x)]
            # columns: RMSE MAE MAPE% R2 Spearman (kendall)
            if len(nums) >= 5:
                return {"rmse": nums[0], "mae": nums[1], "mape": nums[2] / 100,
                        "r2": nums[3], "spearman": nums[4]}
    return None


def _is_float(x: str) -> bool:
    try:
        float(x)
        return True
    except ValueError:
        return False


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--feature-budget", type=int, default=5)
    p.add_argument("--weak-model", default="gpt-4.omini")
    p.add_argument("--strong-model", default=None,
                   help="Optionally also run this model (RAG + no-RAG).")
    p.add_argument("--processed-dir", default=DEFAULT_PROCESSED)
    p.add_argument("--feature-program-path", default=DEFAULT_PROGRAM)
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--candidates-per-iteration", type=int, default=2)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--tag", default=None,
                   help="Suffix for run output dirs (default: b<budget>).")
    p.add_argument("--summarize-only", action="store_true",
                   help="Skip the runs; just re-print from existing run dirs.")
    p.add_argument("--no-random-baseline", action="store_true")
    args = p.parse_args()

    tag = args.tag or f"b{args.feature_budget}"
    b = args.feature_budget

    # (label, run-dir-name, model, rag) for each cell.
    cells_spec = [
        (f"weak + no-RAG (b{b})", f"cmp_{tag}_weak_norag", args.weak_model, False),
        (f"weak + RAG (b{b})", f"cmp_{tag}_weak_rag", args.weak_model, True),
    ]
    if args.strong_model:
        cells_spec += [
            (f"strong + no-RAG (b{b})", f"cmp_{tag}_strong_norag", args.strong_model, False),
            (f"strong + RAG (b{b})", f"cmp_{tag}_strong_rag", args.strong_model, True),
        ]

    if not args.summarize_only:
        print(f"Running {len(cells_spec)} workflow cell(s) at feature-budget={b} ...")
        for _label, run, model, rag in cells_spec:
            _run_workflow(
                f"{RUN_ROOT}/{run}", model=model, rag=rag, budget=b,
                processed=args.processed_dir, program=args.feature_program_path,
                iterations=args.iterations, candidates=args.candidates_per_iteration,
                split_seed=args.split_seed,
            )

    random_row = None
    if not args.no_random_baseline:
        print("Computing random-feature baseline ...", flush=True)
        random_row = _random_baseline(b, args.processed_dir, args.feature_program_path, args.split_seed)

    _summarize([(label, run) for label, run, _m, _r in cells_spec], random_row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
