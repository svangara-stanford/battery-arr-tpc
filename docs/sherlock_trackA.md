# Open Battery Agents — Track A on Stanford Sherlock

This document captures the Sherlock-specific recipe for running the sealed
Track A broad-discovery campaign. Track A starts from generic battery-aging
prompts; Batch 9 stays locked until a champion is selected from
Severson-only validation.

## 1. One-time environment setup

```bash
cd /home/groups/darve/svangara/battery-arr-tpc
bash scripts/setup_sherlock_env.sh
```

That script loads `python/3.12.1`, creates the persistent venv at
`/home/groups/darve/svangara/battery-arr-venvs/oba`, installs the pinned
scientific stack plus the agent stack, and runs an import / alias / helper
sanity check. To recreate from scratch use `FORCE_RECREATE=true`.

Activate it manually with:

```bash
module purge
module load python/3.12.1
source /home/groups/darve/svangara/battery-arr-venvs/oba/bin/activate
```

## 2. Smoke tests

Before launching the full campaign, run two cheap jobs. The minimal-debug
job uses ~32 GB during the Severson MatR audit; the scalar-baseline job
exercises the full LLM path that previously caught alias / payload bugs.

```bash
cd /home/groups/darve/svangara/battery-arr-tpc

# minimal_debug (1 iter, 4 candidates)
sbatch \
  --cpus-per-task=4 --mem=32G --time=2:00:00 \
  --export=ALL,OFFLINE=false,FEATURE_RECIPE=minimal_debug,SPLIT_MODE=random,SPLIT_SEED=0,ITERATIONS=1,CANDIDATES_PER_ITERATION=4,DO_FINAL_BATCH9=false,FINAL_BATCH9_TOP_K=0 \
  nersc/sherlock_trackA_discovery.slurm

# scalar_baseline (2 iter, 10 candidates)
sbatch \
  --cpus-per-task=8 --mem=32G --time=2:00:00 \
  --export=ALL,OFFLINE=false,FEATURE_RECIPE=scalar_baseline,SPLIT_MODE=random,SPLIT_SEED=0,ITERATIONS=2,CANDIDATES_PER_ITERATION=10,DO_FINAL_BATCH9=false,FINAL_BATCH9_TOP_K=0 \
  nersc/sherlock_trackA_discovery.slurm
```

Inspect both with:

```bash
tail -180 runs/slurm/oba-trackA-${JOB_ID}.out

python3 - <<PY
import json, os
from pathlib import Path
root = Path(os.environ["SCRATCH"]) / "battery-arr" / "reports"
for p in root.glob("trackA_*/role_agent_workflow.json"):
    r = json.loads(p.read_text())
    print(p.parent.name, "best=", r.get("best_candidate_id"),
          "rmse=", (r.get("validation_metrics") or {}).get("rmse"),
          "batch9=", r.get("final_batch9_metrics"))
PY
```

The smoke tests must show `best_candidate_id` set, validation metrics
populated, and `final_batch9_metrics: None`.

## 3. Full sealed Track A campaign

The campaign is 3 recipes × 2 split modes × 3 seeds = 18 jobs, each
running 8 iterations × 20 candidates with Batch 9 disabled.

```bash
bash scripts/launch_sherlock_trackA_campaign.sh
```

That script writes a CSV launch log under
`$SCRATCH/battery-arr/launch_logs/`. Override defaults via
`ITERATIONS`, `CANDIDATES_PER_ITERATION`, `SLURM_TIME`, `SLURM_MEM`,
`SLURM_CPUS`.

Equivalent manual command:

```bash
mkdir -p "$SCRATCH/battery-arr/launch_logs"
LAUNCH_LOG="$SCRATCH/battery-arr/launch_logs/trackA_sherlock_$(date +%Y%m%d_%H%M%S).csv"
echo "job_id,recipe,split_mode,split_seed,iterations,candidates_per_iteration" > "$LAUNCH_LOG"
for recipe in scalar_baseline curve_delta broad_physics; do
  for mode in random batch; do
    for seed in 0 1 2; do
      J=$(sbatch \
        --export=ALL,OFFLINE=false,FEATURE_RECIPE=$recipe,SPLIT_MODE=$mode,SPLIT_SEED=$seed,ITERATIONS=8,CANDIDATES_PER_ITERATION=20,DO_FINAL_BATCH9=false,FINAL_BATCH9_TOP_K=0 \
        nersc/sherlock_trackA_discovery.slurm | awk '{print $4}')
      echo "$J,$recipe,$mode,$seed,8,20" | tee -a "$LAUNCH_LOG"
    done
  done
done
```

## 4. Summarization

After all 18 jobs complete:

```bash
module load python/3.12.1
source /home/groups/darve/svangara/battery-arr-venvs/oba/bin/activate
cd /home/groups/darve/svangara/battery-arr-tpc

python3 scripts/summarize_sherlock_trackA.py
```

That writes:

- `$SCRATCH/battery-arr/reports/trackA_summary/trackA_run_summary.csv`
- `$SCRATCH/battery-arr/reports/trackA_summary/trackA_all_candidates.csv`
- `$SCRATCH/battery-arr/reports/trackA_summary/trackA_full_run_summary.csv`
- `$SCRATCH/battery-arr/reports/trackA_summary/trackA_full_all_candidates.csv`

The `_full_*` files restrict to runs that are full-campaign
(8 iterations × 20 candidates) and have not touched Batch 9.

Equivalent two-step manual flow:

```bash
python3 scripts/summarize_trackA_results.py \
  --reports-root "$SCRATCH/battery-arr/reports" \
  --out-dir "$SCRATCH/battery-arr/reports/trackA_summary" \
  --min-evals 2

python3 - <<'PY'
import os, pandas as pd
from pathlib import Path
root = Path(os.environ["SCRATCH"]) / "battery-arr" / "reports" / "trackA_summary"
runs = pd.read_csv(root / "trackA_run_summary.csv")
cands = pd.read_csv(root / "trackA_all_candidates.csv")
full = runs[(runs["iterations"]==8) & (runs["candidates_per_iteration"]==20)
            & (runs["locked_batch9_status"].fillna("not_run")=="not_run")]
full.to_csv(root / "trackA_full_run_summary.csv", index=False)
cands[cands["run_id"].isin(full["run_id"])].to_csv(root / "trackA_full_all_candidates.csv", index=False)
print(full[["run_id","recipe","split_mode","split_seed","best_rmse",
            "best_mae","best_spearman","best_model_family",
            "best_feature_set","best_target_transform"]]
      .sort_values("best_rmse").head(30).to_string(index=False))
PY
```

## 5. Batch 9 final validation (post-discovery)

After a champion is chosen from Severson-only validation, run the final
locked Batch 9 evaluation:

```bash
sbatch \
  --export=ALL,OFFLINE=false,FEATURE_RECIPE=<champion-recipe>,SPLIT_MODE=<champion-split>,SPLIT_SEED=<champion-seed>,ITERATIONS=8,CANDIDATES_PER_ITERATION=20,DO_FINAL_BATCH9=true,FINAL_BATCH9_TOP_K=5 \
  nersc/sherlock_trackA_discovery.slurm
```

Do this exactly once per champion and after Track A search is complete.

## 6. Output layout

| Item                       | Path                                                                  |
|----------------------------|-----------------------------------------------------------------------|
| Slurm console logs         | `runs/slurm/oba-trackA-<job>.out` (in repo)                           |
| Per-run artifacts          | `$SCRATCH/battery-arr/runs/<run_id>/`                                 |
| Per-run reports / JSON     | `$SCRATCH/battery-arr/reports/<run_id>/role_agent_workflow.{md,json}` |
| Per-run archive tgz        | `$SCRATCH/battery-arr/archives/oba-trackA-<job>-artifacts.tgz`        |
| Cross-run summary CSVs     | `$SCRATCH/battery-arr/reports/trackA_summary/`                        |
| Launch CSV                 | `$SCRATCH/battery-arr/launch_logs/trackA_sherlock_<ts>.csv`           |
