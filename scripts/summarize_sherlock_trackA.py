#!/usr/bin/env python3
"""Summarize a sealed Track A campaign on Sherlock.

Wraps ``scripts/summarize_trackA_results.py`` and then filters to only the
full-campaign runs (8 iterations x 20 candidates) where Batch 9 was not used.

Usage:
    python3 scripts/summarize_sherlock_trackA.py
    python3 scripts/summarize_sherlock_trackA.py --reports-root /custom/path
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _default_reports_root() -> Path:
    scratch = os.environ.get("SCRATCH")
    if not scratch:
        sys.exit("ERROR: $SCRATCH unset; pass --reports-root explicitly.")
    return Path(scratch) / "battery-arr" / "reports"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=None,
                        help="Default: $SCRATCH/battery-arr/reports")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Default: <reports-root>/trackA_summary")
    parser.add_argument("--min-evals", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=8,
                        help="Filter full campaign runs by iterations.")
    parser.add_argument("--candidates-per-iteration", type=int, default=20,
                        help="Filter full campaign runs by candidates/iter.")
    args = parser.parse_args()

    reports_root = args.reports_root or _default_reports_root()
    out_dir = args.out_dir or reports_root / "trackA_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    summarize = REPO_ROOT / "scripts" / "summarize_trackA_results.py"
    if not summarize.exists():
        sys.exit(f"ERROR: missing {summarize}")

    cmd = [
        sys.executable, str(summarize),
        "--reports-root", str(reports_root),
        "--out-dir", str(out_dir),
        "--min-evals", str(args.min_evals),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    import pandas as pd

    runs_csv = out_dir / "trackA_run_summary.csv"
    cands_csv = out_dir / "trackA_all_candidates.csv"
    if not runs_csv.exists():
        sys.exit(f"ERROR: missing {runs_csv}")
    if not cands_csv.exists():
        sys.exit(f"ERROR: missing {cands_csv}")

    runs = pd.read_csv(runs_csv)
    cands = pd.read_csv(cands_csv)

    full_runs = runs[
        (runs["iterations"] == args.iterations)
        & (runs["candidates_per_iteration"] == args.candidates_per_iteration)
        & (runs["locked_batch9_status"].fillna("not_run") == "not_run")
    ].copy()

    full_ids = set(full_runs["run_id"].dropna())
    full_cands = cands[cands["run_id"].isin(full_ids)].copy()

    full_runs_path = out_dir / "trackA_full_run_summary.csv"
    full_cands_path = out_dir / "trackA_full_all_candidates.csv"
    full_runs.to_csv(full_runs_path, index=False)
    full_cands.to_csv(full_cands_path, index=False)

    print()
    print(f"full runs           : {len(full_runs)}  -> {full_runs_path}")
    print(f"full candidate rows : {len(full_cands)} -> {full_cands_path}")
    print()
    if len(full_runs) > 0:
        cols = [c for c in [
            "run_id", "recipe", "split_mode", "split_seed", "best_rmse",
            "best_mae", "best_spearman", "best_model_family",
            "best_feature_set", "best_target_transform",
        ] if c in full_runs.columns]
        print(full_runs[cols].sort_values("best_rmse").head(30).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
