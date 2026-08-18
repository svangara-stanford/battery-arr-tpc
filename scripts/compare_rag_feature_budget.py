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

Confidence intervals: pass multiple --split-seeds to run every cell once per
seed. Each seed re-shuffles the train/val partition (see orchestrator.make_splits),
so the spread of best-candidate metrics across seeds is a 95% t-interval on model
performance under partition noise. Reported as `mean ± half-width`. A single seed
prints the point estimate with no interval.

There is a second, larger noise source: the FeatureScientist queries the LLM at
temperature > 0 with no fixed sampling seed, so the SAME command at the SAME seed
can pick different features (observed: ~28 RMSE points at fixed seed 0). Use
--replicates N to repeat each (cell, seed) N times; the CI then folds in both
partition and LLM-sampling noise. With a no-RAG/RAG pair the summary also prints a
PAIRED DIFFERENCE table (RAG - no-RAG matched by seed+rep), which cancels the
noise both conditions share on each draw and is the most powerful test of RAG's
effect at a fixed budget.

Examples:
  # Full run at budget 5, weak model only (+ random baseline):
  source .venv/bin/activate && set -a && . ./.env && set +a
  python scripts/compare_rag_feature_budget.py --feature-budget 5

  # Confidence intervals over 5 partition seeds:
  python scripts/compare_rag_feature_budget.py --feature-budget 5 \
      --split-seeds 0 1 2 3 4

  # Fold in LLM-sampling noise: 3 seeds x 3 replicates + paired-difference table:
  python scripts/compare_rag_feature_budget.py --feature-budget 5 \
      --split-seeds 0 1 2 --replicates 3

  # Also run the stronger model:
  python scripts/compare_rag_feature_budget.py --feature-budget 8 \
      --strong-model claude-sonnet-4-6

  # Just re-print results from existing runs (no LLM calls):
  python scripts/compare_rag_feature_budget.py --feature-budget 5 \
      --split-seeds 0 1 2 3 4 --summarize-only
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


def _mean_ci(values: list[float], confidence: float = 0.95) -> tuple[float, float, int]:
    """Return (mean, 95% t-interval half-width, n) for a list of per-seed values.

    Half-width is 0.0 for a single value (no interval). Uses a Student-t
    interval because the seed count is small (typically 3-8); the normal
    approximation would understate the width.
    """
    vals = [v for v in values if v is not None and not math.isnan(v)]
    n = len(vals)
    if n == 0:
        return float("nan"), 0.0, 0
    mean = sum(vals) / n
    if n == 1:
        return mean, 0.0, 1
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    sem = math.sqrt(var / n)
    try:
        from scipy import stats

        tcrit = float(stats.t.ppf(0.5 + confidence / 2.0, df=n - 1))
    except Exception:
        tcrit = 1.96  # normal fallback if scipy is unavailable
    return mean, tcrit * sem, n


def _fmt_ci(mean: float, half: float, decimals: int = 2) -> str:
    """Compact `mean ± half` cell, or just `mean` when there's no interval."""
    if math.isnan(mean):
        return "n/a"
    if half <= 0.0:
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f}±{half:.{decimals}f}"


def _cell_draw_reports(draws: list[tuple[int, int, str]]) -> list[tuple[int, int, dict, str]]:
    """Best (seed, rep, report, path) for each draw that produced a candidate.

    A "draw" is one workflow run at a given (partition seed, replicate). With
    replicates > 1 the same seed appears multiple times, capturing LLM-sampling
    noise on top of the partition noise that varying the seed captures.
    """
    out = []
    for seed, rep, run in draws:
        found = _best_report(REPO_ROOT / RUN_ROOT / run)
        if found is not None:
            r, path = found
            out.append((seed, rep, r, path))
    return out


METRIC_COLS = (("rmse", 2), ("mae", 2), ("mape", 2), ("r2", 3), ("spearman", 3))


def _metric_value(report: dict, key: str) -> float:
    return _mape_pct(report) if key == "mape" else report[key]


def _summarize(cells: list[tuple[str, list[tuple[int, int, str]]]], random_row: dict | None) -> None:
    n_draws = max((len(draws) for _label, draws in cells), default=0)
    n_reps = max((max((rep for _s, rep, _r in draws), default=0) + 1 for _label, draws in cells), default=1)
    multi_draw = n_draws > 1
    # What the CI spans: seeds only, reps only, or both.
    seed_ct = max((len({s for s, _rep, _r in draws}) for _label, draws in cells), default=1)
    if seed_ct > 1 and n_reps > 1:
        ci_over = f"{seed_ct} seeds x {n_reps} reps"
    elif n_reps > 1:
        ci_over = f"{n_reps} reps"
    else:
        ci_over = "seeds"

    # Metrics table. Each numeric column is `mean±halfwidth` of the per-draw best
    # candidate; the interval is a 95% Student-t CI over all draws, folding in both
    # partition noise (varying seed) and LLM-sampling noise (varying replicate).
    print("\n" + "=" * 88)
    title = "VALIDATION METRICS (best candidate per run"
    title += f", mean ± 95% CI over {ci_over})" if multi_draw else ")"
    print(title)
    print("=" * 88)
    w = 15 if multi_draw else 9
    header = f"{'run':<26}{'n':>3}{'RMSE':>{w}}{'MAE':>{w}}{'MAPE%':>{w}}{'R2':>{w}}{'Spear':>{w}}"
    print(header)
    print("-" * len(header))
    resolved = []
    for label, draws in cells:
        reports = _cell_draw_reports(draws)
        if not reports:
            print(f"{label:<26}{'NO SUCCESSFUL CANDIDATES':>{3 + 5 * w}}")
            resolved.append((label, draws, []))
            continue
        cells_fmt = []
        for key, dec in METRIC_COLS:
            mean, half, _n = _mean_ci([_metric_value(r, key) for _s, _rep, r, _p in reports])
            cells_fmt.append(f"{_fmt_ci(mean, half, dec):>{w}}")
        print(f"{label:<26}{len(reports):>3}" + "".join(cells_fmt))
        resolved.append((label, draws, reports))
    if random_row is not None:
        m = random_row
        mape = (m.get("mape") or 0) * 100
        vals = [_fmt_ci(m[k], 0.0, d) for k, d in METRIC_COLS]
        vals[2] = _fmt_ci(mape, 0.0, 2)
        print(f"{'random (same pool)':<26}{'-':>3}" + "".join(f"{v:>{w}}" for v in vals))

    _paired_difference_table(resolved)

    # Features + reasoning. Per draw, so RAG's per-run influence is visible even
    # when the aggregate metrics overlap. When every draw picked the same fit
    # columns, that's called out explicitly (the convergence we keep observing).
    for label, draws, reports in resolved:
        if not reports:
            continue
        print("\n" + "=" * 88)
        print(label)
        print("=" * 88)
        col_sets = {tuple(_fit_columns(r)) for _s, _rep, r, _p in reports}
        if len(col_sets) == 1 and len(reports) > 1:
            print(f"All {len(reports)} draws selected the SAME {len(next(iter(col_sets)))} fit columns:")
            for c in sorted(next(iter(col_sets))):
                print(f"  - {c}")
        else:
            for seed, rep, r, _path in reports:
                cols = _fit_columns(r)
                print(f"\n  seed {seed} rep {rep}: candidate {r.get('candidate_id')}, "
                      f"iteration {r.get('iteration')}, rmse {r['rmse']:.2f} "
                      f"({len(cols)} cols)")
                for c in cols:
                    print(f"    - {c}")
        # Rationale from the first draw's champion (representative when converged).
        seed0, rep0, r0, _p0 = reports[0]
        run0 = {(s, rep): run for s, rep, run in draws}[(seed0, rep0)]
        plan = _feature_plan_for_iteration(REPO_ROOT / RUN_ROOT / run0, int(r0.get("iteration") or 0))
        if plan:
            print(f"\nAgent rationale (seed {seed0} rep {rep0}):")
            print(f"  {plan.get('rationale')}")
            ops = plan.get("feature_operators", [])
            if ops:
                print("\nPer-operator reasoning:")
                for o in ops:
                    print(f"  - {o.get('operator_type')} {o.get('params')}")
                    if o.get("description"):
                        print(f"      why: {o['description']}")


def _paired_difference_table(resolved: list) -> None:
    """RAG minus no-RAG, paired by (seed, rep).

    Pairing cancels the partition/LLM noise the two conditions share on each
    draw, so this isolates RAG's effect far more powerfully than differencing the
    two marginal CIs. Only emitted when a no-RAG / RAG pair exists at the same
    model tier with >= 2 shared draws.
    """
    by_label = {label: reports for label, _draws, reports in resolved}
    pairs = [("weak + no-RAG", "weak + RAG"), ("strong + no-RAG", "strong + RAG")]
    printed_header = False
    for norag_key, rag_key in pairs:
        norag = next((v for k, v in by_label.items() if k.startswith(norag_key)), None)
        rag = next((v for k, v in by_label.items() if k.startswith(rag_key)), None)
        if not norag or not rag:
            continue
        norag_by = {(s, rep): r for s, rep, r, _p in norag}
        rag_by = {(s, rep): r for s, rep, r, _p in rag}
        shared = sorted(set(norag_by) & set(rag_by))
        if len(shared) < 2:
            continue
        if not printed_header:
            print("\n" + "=" * 88)
            print("PAIRED DIFFERENCE  (RAG - no-RAG, matched by seed+rep; negative = RAG better on error)")
            print("=" * 88)
            w = 18
            hdr = f"{'pair':<26}{'n':>3}" + "".join(f"{c.upper():>{w}}" for c, _d in METRIC_COLS)
            print(hdr)
            print("-" * len(hdr))
            printed_header = True
        cells_fmt = []
        for key, dec in METRIC_COLS:
            diffs = [_metric_value(rag_by[k], key) - _metric_value(norag_by[k], key) for k in shared]
            mean, half, _n = _mean_ci(diffs)
            # A CI that excludes 0 is a significant paired effect; star it.
            star = "*" if (not math.isnan(mean) and half > 0 and abs(mean) > half) else " "
            cells_fmt.append(f"{_fmt_ci(mean, half, dec) + star:>{w}}")
        print(f"{(rag_key + ' - ' + norag_key):<26}{len(shared):>3}" + "".join(cells_fmt))
    if printed_header:
        print("\n  * = 95% CI excludes 0 (paired effect distinguishable from noise at this n).")


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
    p.add_argument("--split-seed", type=int, default=0,
                   help="Single partition seed (ignored if --split-seeds is set).")
    p.add_argument("--split-seeds", type=int, nargs="+", default=None,
                   help="Run every cell once per seed and report a 95%% CI over "
                        "the resulting best-candidate metrics. Each seed re-shuffles "
                        "the train/val partition. Overrides --split-seed.")
    p.add_argument("--replicates", type=int, default=1,
                   help="Repeat each (cell, seed) this many times to capture "
                        "LLM-sampling noise (the FeatureScientist runs at temp>0, "
                        "so the same seed gives different feature plans). Folded "
                        "into the CI alongside partition noise.")
    p.add_argument("--tag", default=None,
                   help="Suffix for run output dirs (default: b<budget>).")
    p.add_argument("--summarize-only", action="store_true",
                   help="Skip the runs; just re-print from existing run dirs.")
    p.add_argument("--no-random-baseline", action="store_true")
    args = p.parse_args()

    tag = args.tag or f"b{args.feature_budget}"
    b = args.feature_budget
    seeds = args.split_seeds if args.split_seeds is not None else [args.split_seed]
    reps = max(1, args.replicates)

    # (label, run-suffix, model, rag) for each cell; the seed is appended to the
    # run dir below so each partition gets its own artifacts.
    cells_spec = [
        (f"weak + no-RAG (b{b})", "weak_norag", args.weak_model, False),
        (f"weak + RAG (b{b})", "weak_rag", args.weak_model, True),
    ]
    if args.strong_model:
        cells_spec += [
            (f"strong + no-RAG (b{b})", "strong_norag", args.strong_model, False),
            (f"strong + RAG (b{b})", "strong_rag", args.strong_model, True),
        ]

    def run_dir_name(suffix: str, seed: int, rep: int) -> str:
        # Keep the legacy single-seed name so old runs still re-summarize.
        base = f"cmp_{tag}_{suffix}"
        if len(seeds) == 1 and args.split_seeds is None and reps == 1:
            return base
        name = f"{base}_s{seed}"
        return name if reps == 1 else f"{name}_r{rep}"

    if not args.summarize_only:
        total = len(cells_spec) * len(seeds) * reps
        print(f"Running {total} workflow cell(s) at feature-budget={b} "
              f"over {len(seeds)} seed(s) x {reps} replicate(s): {seeds} ...")
        for seed in seeds:
            for rep in range(reps):
                for _label, suffix, model, rag in cells_spec:
                    _run_workflow(
                        f"{RUN_ROOT}/{run_dir_name(suffix, seed, rep)}", model=model, rag=rag,
                        budget=b, processed=args.processed_dir, program=args.feature_program_path,
                        iterations=args.iterations, candidates=args.candidates_per_iteration,
                        split_seed=seed,
                    )

    random_row = None
    if not args.no_random_baseline:
        print(f"Computing random-feature baseline over {len(seeds)} seed(s) ...", flush=True)
        per_seed = [_random_baseline(b, args.processed_dir, args.feature_program_path, s)
                    for s in seeds]
        per_seed = [m for m in per_seed if m is not None]
        if per_seed:
            random_row = {k: sum(m[k] for m in per_seed) / len(per_seed)
                          for k in ("rmse", "mae", "mape", "r2", "spearman")}

    cells = [(label, [(seed, rep, run_dir_name(suffix, seed, rep))
                      for seed in seeds for rep in range(reps)])
             for label, suffix, _m, _r in cells_spec]
    _summarize(cells, random_row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
