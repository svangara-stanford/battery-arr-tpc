# Session Context: Iterating Track A to Match Severson 2019

**Date:** 2026-06-09 → 2026-06-12 (single multi-day session)
**Goal:** Iterate on the Open Battery Agents Track A autonomous-discovery pipeline until it produces results comparable to Severson 2019's published cycle-life prediction accuracy on her secondary test set (b3 / MATR2).
**Status at end of session:** Pipeline is verified correct. Autonomous discovery currently lands at ~357 RMSE / 21% MAPE on canonical 40-cell b3. Severson Discharge baseline (149-173 RMSE) is the target. SOTA floor is ~150 RMSE. Three concrete interventions identified to close the gap. **Decision: keep working until we get to Severson-comparable RMSE.**

---

## 1. Where the Session Started

The 2026-06-08 Track A campaign (18 jobs: 3 recipes × 2 split modes × 3 seeds, 8 iters × 20 candidates each) had produced an "honest negative":
- Champion: `role_graph_iter_003_variant_19` from `trackA_curve_delta_batch_seed0_28455878`
- GradientBoostingRegressor / curve_only / raw target
- Surrogate RMSE 264, batch9 RMSE 294, Spearman 0.74
- All 3 shortlisted candidates failed pathology gates on the "Batch 9" test
- Artifacts at `/scratch/users/svangara/battery-arr/champion_selection/20260609T011522Z_28502641/`

**User's directive at the start of session:** "I don't want to publish this negative result. I want to iterate upon our autonomous discovery system until we produce adequate results."

---

## 2. Major Discoveries (in order)

### Discovery 1: We were testing on the WRONG dataset
- "Batch 9" in our codebase = `data/2019-01-24_batch9/` = **Attia 2020 closed-loop optimized cells**, NOT Severson b3
- Attia 2020 is NOT an EOL-prediction paper — it's a closed-loop optimization paper that *uses* Severson's predictor
- Severson's actual secondary test = b3 (`2018-04-12`) at `data/severson_2019_true_life_matr/`
- Our entire pipeline was held-out-testing against a harder dataset than Severson designed for
- **Fix:** retarget Batch 9 → Severson b3 as the locked secondary test

### Discovery 2: The splitter was wrong (non-canonical)
- `severson_matr.py::build_severson_true_life_dataset` used a "source_file_identity_default" splitter: b1=train, b2=validation, b3=test
- Severson's canonical split: train and primary_test are **mixed halves of b1+b2**, secondary_test is **all of b3**
- We weren't even on the same evaluation contract as the paper

### Discovery 3: Multiple infrastructure gaps
- No MAPE / % test error metric anywhere (only RMSE/MAE/R²/Spearman)
- `target_transform` defaulted to `raw`, not `log10` (Severson's convention)
- `broad_physics` recipe was a runtime *duplicate* of `scalar_plus_curve` (the rich recipe in `program_library.py` was never wired through `candidate_compiler.py:312-313`)
- Voltage interpolation grid sizes were inconsistent (1000 for curve_delta, 500 for broad_physics, 300 for curve_shape)

### Discovery 4: A SLEEPER LEAK (after the first re-launch looked stellar)
- 2026-06-09 campaign re-launched with corrections E1-E4 + slurmd-bug heartbeat patch
- Got 6.41% surrogate MAPE on `curve_delta/random/seed=1` — looked like we beat Severson's 9.1%
- **Probe job 28873874 proved it was a leak.** Scored top-5 by-surrogate-MAPE candidates on clean b3 → all collapsed to RMSE 364-486, Spearman 0.06-0.32
- Root cause: `orchestrator.py::make_search_split` was re-splitting the full 138-cell pool (b1+b2+b3) without honoring the canonical `splits.csv` tags written by E1. In `random` mode, 37/44 b3 cells leaked into surrogate training and 7/44 into validation
- E1's canonical splitter wrote the right tags. E4 wired b3 as locked test. **But the workflow's own splitter ignored the canonical tags.**

### Discovery 5: The verified post-leak result is honestly negative — but in line with prior art
- After F1-F4 fixed the splitter to honor canonical tags, surrogate MAPE on b3-clean train was still 6.41% in random mode (looks great), 25-33% in batch mode (terrible)
- Probe job 29135838 scored top-5 on clean 44-cell b3: best candidate 370 RMSE / 21% MAPE
- Champion-selection 29135839 chose scalar_baseline/batch/seed=0 / GBR / scalar_only / raw with secondary_test_rmse=312, all 3 shortlisted candidates failed pathology gates
- **G4 manual reproduction of Severson Variance on our data: 148.4 RMSE / 11.9% MAPE on canonical 40-cell b3 (drop 4 noisy cells)** — within 1 cycle of her published 149.
- **G3 found we keep 4 b3 cells (b3c2, b3c37, b3c42, b3c43) that Severson dropped as noisy channels.** 3 of 4 are upper-tail (1267-1642 cycles) and disproportionately inflate our RMSE
- H6 confirmed: **no published autonomous/agentic system has matched Severson on her own MATR2 split.** Our 357 RMSE is in range with Severson's commonly-cited baseline (214-357) and beats every deep model in BatteryML on MATR2 (LSTM 21k, CNN 228k, Transformer 36k). We are in **unclaimed territory**.

---

## 3. Patches Landed in This Session (Code)

| Patch | Scope | Files Touched |
|---|---|---|
| **E1** | Canonical Severson splitter (build-time) | `src/battery_aar/features/severson_matr.py`, `scripts/build_severson_true_life_dataset.py`, `data/severson_canonical_exclusions.txt`, `tests/test_severson_canonical_split.py` |
| **E2** | MAPE/test_error_pct metric threaded through 18 sites; default `target_transform=log10` | `src/battery_aar/agents/evaluator.py`, `orchestrator.py:288-303`, `reports.py`, `paper_reproduction/validation.py`, `tools/implementations.py:674-685`, `workflows/schemas.py`, `roles.py`, `role_prompts.py`, `role_graph.py:258-289,476-530`, `candidate_compiler.py:80-91,199,395-423`, `scripts/run_agent_champion_selection.py`, `scripts/summarize_sherlock_trackA.py`, `tests/test_mape_metric.py` |
| **E3** | Feature pipeline fixes | `src/battery_aar/features/program_library.py` (broad_physics is now a strict superset, grid_size=1000 standardized), `operators.py` (voltage_window param, `log_var` aggregation so Severson Variance is reachable in a single column), `candidate_compiler.py:312-313` (broad_physics dispatch fix), `tests/test_feature_pipeline_fixes.py` |
| **E4** | Retarget locked test from Attia → Severson b3 | `src/battery_aar/paper_reproduction/paths.py` (kept `VALIDATION_BATCH_NAME = "2019-01-24_batch9"` for Attia, added `SEVERSON_B3_*`), `agents/attia_data_bridge.py` (new `load_severson_b3_holdout`), `agents/orchestrator.py` (new kwargs), `workflows/role_graph.py` (new `secondary_test_*` config), `scripts/run_role_agent_workflow.py` (`--secondary-test-source` CLI), `scripts/run_agent_champion_selection.py`, `scripts/score_candidates_on_severson_b3.py` (NEW), `nersc/sherlock_trackA_discovery.slurm` + `nersc/sherlock_champion_selection.slurm` (env vars), `tests/test_secondary_test_loader.py` |
| **F1** | `make_search_split` honors canonical splits | `src/battery_aar/agents/orchestrator.py:189-470` (new `_classify_split_table`, `_filter_canonical_search_pool`, restricts search pool to `train` rows only, new `split_mode="primary_test"`, hard `RuntimeError` leak guard), `tests/test_canonical_split_honoring.py` (12 tests) |
| **F2 (no-code, audit)** | Verified F1 doesn't reach the workflow via `role_graph` |  — |
| **role_graph follow-up** | Threading the canonical splits table through the production workflow + honest-contract assertion | `src/battery_aar/workflows/role_graph.py:72-94,660-695` (now reads splits.csv, passes `splits_table=` to make_search_split, asserts no `secondary_test` row in train+val) |
| **Slurm defenses** (after slurmd bug #23843 on sh04-13n32) | `nersc/sherlock_trackA_discovery.slurm` — absolute `--output` on Lustre, early heartbeat to `/tmp` + `/scratch/heartbeats/`, `timeout 60 cd`, `--no-requeue`, `--open-mode=append`, `export PYTHONUNBUFFERED=1`, heartbeat cleanup in archive_on_exit trap; `scripts/launch_sherlock_trackA_campaign.sh` — `SLURM_EXCLUDE=sh04-13n32` default |
| **Scorer signature fix** | After my `_load_processed` return-tuple grew from 4→5 | `scripts/score_candidates_on_severson_b3.py:180`, `scripts/score_candidates_on_batch9.py:278` |
| **Summarizer MAPE extraction** | E2 added MAPE to per-run JSON but not the cross-run CSV | `scripts/summarize_trackA_results.py:83-87` (added `best_mape`, `best_test_error_pct`) |
| **Canonical exclusion list populated** | Severson's 6 b3 noisy channels + 5 b1 unfinished cells | `data/severson_canonical_exclusions.txt` |

**Test status at end of session: 204/204 tests pass.**

---

## 4. Live Campaigns Run This Session

| Campaign | Job IDs | Outcome | Notes |
|---|---|---|---|
| Campaign 1 (2026-06-09 evening) | 28650385–28650403 (skip 28650398), summarize 28650541, champion 28650542 | Discovery looked great (broad_physics/random/seed=1 = 7.05% surrogate MAPE) — **revealed as a leak** | Slurmd bug #23843 hit sh04-13n32 → replaced 28650389 with 28681400 |
| Champion-selection 28855944 | Failed: mixed old+new campaigns in summary + ChampionAggregator schema; ran after re-archiving stragglers + adding MAPE extraction | Got fixed second attempt; result: Adjudicator picked `curve_delta/batch/seed=0/GBR/raw` as champion with b3 RMSE 387 (worse than weak baseline) — all entries failed pathology gates |
| Probe 28873874 | Scored top-5 by-surrogate-MAPE candidates on clean b3 (44 cells) → all collapsed to RMSE 364-486 → **proved the leak** | This is the smoking gun for the split-leak investigation |
| Campaign 2 (2026-06-10 evening, "split-fix") | 28893196–28893224 (skip gaps), summarize 28893359, champion 28893362 | 1 LLM-transient failure (JSONDecodeError) → resubmitted as 28903484. Champion-selection 28893362 failed (Aggregator returned only 2 valid entries); rerun as 29135839 succeeded. | Surrogate metrics: curve_delta/random/seed=1 at 6.41% MAPE (real this time, leak closed via F1+role_graph patch) |
| Probe 29132010 | Failed (`ValueError: too many values to unpack` — my `_load_processed` change broke scorer); fixed scorer; resubmitted as 29135838 → COMPLETED |  |
| Probe 29135838 + champion 29135839 | Final clean results | Champion: `scalar_baseline/batch/seed=0 / GBR / scalar_only / raw`, surrogate_rmse 284, secondary_test_rmse 312 (still worse than weak baseline 410). All shortlist entries failed pathology gates. Adjudicator picked smallest-gap. |

**Archived locations:**
- 2026-06-08 reports: `/scratch/users/svangara/battery-arr/reports/_archive_pre_retarget_20260608/`
- 2026-06-09 reports + stale broad_physics_28650400-403: `/scratch/users/svangara/battery-arr/reports/_archive_pre_split_fix_20260610/`
- Live reports: `/scratch/users/svangara/battery-arr/reports/trackA_*` (18 dirs)

---

## 5. Verified Final State (end of session)

**Pipeline correctness verified:**
- G4 manually reproduced Severson Variance model on our exact data and got **RMSE 148.4 / MAPE 11.9% on canonical 40-cell b3** (within 1 cycle of her published 149)
- Sanity variants (voltage window, grid direction, dQ sign) all gave RMSE within 0.3 cycles
- All 138 cells produce finite features; zero NaN exclusions
- Fitted regression: `log10(cycle_life) = 1.6415 + (-0.3120) * log10(Var dQ)` (same sign as Severson)

**Autonomous discovery numbers (best candidates, scored on canonical 40-cell b3):**
| Candidate | Surrogate MAPE | b3 RMSE (40) | b3 MAPE (40) |
|---|---|---|---|
| best (broad_physics/random/2) | 8.83% | ~342 (estimate) | ~20% |
| 2nd (broad_physics/random/1) | 7.19% | ~370 | ~21% |
| Champion (scalar_baseline/batch/0) | — | 312 | — |
| Severson Variance baseline (our data, G4) | — | **148** | **12%** |
| Severson Discharge (paper) | — | **173** | **8.6%** |

**The 4 b3 cells Severson dropped** (G3 finding) cause us to keep an inflated 44-cell test:
- b3c2 (1267), b3c37 (1390), b3c42 (1642), b3c43 (1046)
- Dropping them barely helps us (370 → 357) because our candidates are predicting near the mean — they don't pick up the variance signal Severson exploits
- But dropping them dramatically helps Severson Variance (274 → 148) — calibrated models hit outliers harder

**Diagnostic insight:** our agents aren't choosing `log_var(ΔQ)` even though E3 made it reachable as a single column (`discharge_discharge_capacity_curve_delta_identity_log_var_cycle_99_minus_9`). The Aggregator's stability bias systematically eliminates single-run wonders, even when they're real.

---

## 6. Memory Files Updated

| File | What |
|---|---|
| `memory/project_trackA.md` | New contract: canonical Severson b1+b2 train, b3 secondary test; F1+role_graph split fix; live campaign IDs; previous campaign archives |
| `memory/feedback_sherlock_resources.md` | Slurmd bug defenses (heartbeat, absolute output, `--no-requeue`, `--exclude=sh04-13n32`); resource floors unchanged |
| `memory/sherlock_env.md` | Unchanged |
| `memory/feedback_agentic_orchestration.md` | Unchanged |

---

## 7. Critical Filesystem Paths

```
Repo root:         /home/groups/darve/svangara/battery-arr-tpc
Venv:              /home/groups/darve/svangara/battery-arr-venvs/oba
Data root:         /home/groups/darve/svangara/battery-arr-tpc/literature_models_and_data/battery-fast-charging
Severson .mat:     <data>/data/severson_2019_true_life_matr/{2017-05-12,2017-06-30,2018-04-12}_batchdata_updated_struct_errorcorrect.mat
Attia (demoted):   <data>/data/2019-01-24_batch9/

Scratch root:      /scratch/users/svangara/battery-arr/
Run outputs:       <scratch>/runs/trackA_<recipe>_<split>_seed<N>_<JOBID>/
Reports:           <scratch>/reports/trackA_<recipe>_<split>_seed<N>_<JOBID>/role_agent_workflow.{json,md}
Summary:           <scratch>/reports/trackA_summary/trackA_full_run_summary.csv
Champion sel:      <scratch>/champion_selection/<TS>_<JOBID>/
Slurm logs:        <scratch>/runs/slurm/oba-trackA-<JOBID>.out  (NEW: was runs/slurm/, now Lustre after the slurmd-bug fix)
Heartbeats:        <scratch>/heartbeats/oba-trackA-<JOBID>.heartbeat
Launch logs:       <scratch>/launch_logs/trackA_state.json

One-off scripts:
  Severson Variance reproduction: /scratch/users/svangara/battery-arr/oneoff/severson_variance_reproduction.py
  Predictions output:             /scratch/users/svangara/battery-arr/oneoff/severson_variance_out/
```

---

## 8. Key Commands (Cheatsheet for Next Session)

```bash
# Fresh shell setup
module load python/3.12.1 && source /home/groups/darve/svangara/battery-arr-venvs/oba/bin/activate
cd /home/groups/darve/svangara/battery-arr-tpc

# Run full test suite
python -m pytest tests/ -x --tb=line --no-header -q

# Build the canonical Severson dataset (uses data/severson_canonical_exclusions.txt by default now)
python3 scripts/build_severson_true_life_dataset.py \
  --mat-dir literature_models_and_data/battery-fast-charging/data/severson_2019_true_life_matr \
  --out /tmp/severson_canonical --seed 0
# Expected: 113 cells (138 - 25 exclusions: 6 b3 noisy + 5 b1 unfinished + ... wait, 138 - 11 explicit + 2 auto-dropped = 125. With the new full list 138 - 11 = 127. Verify counts.)
# Actually the exclusion list has: 6 b3 cells + 5 b1 cells = 11 explicit. Some are already auto-dropped (b3c23, b3c32).

# Launch a Track A campaign (18 jobs)
bash scripts/launch_sherlock_trackA_campaign.sh

# Monitor
python3 scripts/monitor_trackA_campaign.py

# Summarize after campaign
SCRATCH=/scratch/users/svangara python3 scripts/summarize_sherlock_trackA.py

# Champion selection (manually, after summarize)
sbatch --export=ALL,SECONDARY_TEST_SOURCE=severson_b3,K=3 nersc/sherlock_champion_selection.slurm

# Probe top-K candidates on Severson b3 directly (bypass champion-selection LLM)
# See /scratch/users/svangara/battery-arr/champion_selection/_top_mape_probe_post_splitfix.json for format
sbatch --partition=normal --time=1:00:00 --mem=96G --cpus-per-task=4 \
  --exclude=sh04-13n32 \
  --job-name=oba-probe \
  --output=/scratch/users/svangara/battery-arr/runs/slurm/oba-probe-%j.out \
  --wrap='module load python/3.12.1 && source /home/groups/darve/svangara/battery-arr-venvs/oba/bin/activate && cd /home/groups/darve/svangara/battery-arr-tpc && python3 scripts/score_candidates_on_severson_b3.py --candidate-list <PATH> --battery-fast-charging-root literature_models_and_data/battery-fast-charging --out /scratch/users/svangara/battery-arr/champion_selection/_probe_<TAG>_$(date -u +%Y%m%dT%H%M%SZ)_${SLURM_JOB_ID} --max-cycle 100'

# Manual Severson Variance reproduction (one-off)
python3 /scratch/users/svangara/battery-arr/oneoff/severson_variance_reproduction.py
```

---

## 9. User's Direction at Session End

> "We are going to keep working until we get a lower RMSE comparable to severson."

User has NOT chosen to accept the honest negative (Framing A). User wants to drive the autonomous-discovery system to Severson Discharge-tier performance (~150-180 RMSE on canonical 40-cell b3).

**The three highest-leverage interventions identified** (from H1-H6 literature survey — see companion doc `LITERATURE_FINDINGS_VS_OUR_PIPELINE.md`):

1. **Centered isotonic regression + quantile-target transform** as a post-hoc calibrator on top of any model. <1 day. Expected drop: 30-60 RMSE. Source: H4 (MDPI Batteries 11(4):145).

2. **Add capacity-fade slope primitive** to the agent substrate: `linear_fit_slope(series, start_cycle, end_cycle)`. The single highest-leverage missing feature; Severson's Discharge stack relies on it as the independent axis to ΔQ variance. 1 day. Expected drop: 20-40 RMSE.

3. **Arrhenius/SEI physics prior** on the predicted capacity-loss curve: `Q_loss(n) = exp(A)·n^B + Q₀`. Two-stage training: stage 1 fits the prior, stage 2 reconstructs residuals. 1-3 days. Gets to ~180 RMSE. Source: H4 (Nicolae et al., arXiv 2404.17174).

**Stretch (1-2 weeks):** BatLiNet-style inter-cell paired regression. Verified ~163 RMSE. Public Code Ocean capsule. Source: H3 (Zhang et al., Nat. Mach. Intell. 2024 / arXiv 2310.05052).

Sequenced, these interventions plausibly drop our autonomous discovery from 357 → 200 → 165 RMSE.

---

## 10. Things to NOT Lose in the New Session

- **The honest-contract assertion** in `role_graph.py:660+` is load-bearing. It would have caught the split-leak immediately. Keep it.
- **The 4 noisy b3 cells** (b3c2, b3c37, b3c42, b3c43) are in `data/severson_canonical_exclusions.txt` now. Any dataset rebuild will exclude them automatically.
- **Slurmd bug defenses**: heartbeat patches, absolute `--output` on Lustre, `--exclude=sh04-13n32` default. These are in the slurm script + launcher.
- **`broad_physics` is a real recipe** post-E3 — not a duplicate of `scalar_plus_curve`. It exposes ΔQ-curve, curve_shape, and other rich families.
- **`log_var` aggregation is reachable** in `curve_delta`, `attia_severson_like`, and `broad_physics` recipes. Severson's Variance feature is a single column: `discharge_discharge_capacity_curve_delta_identity_log_var_cycle_99_minus_9`.
- **The Aggregator's stability bias** (preferring cross-seed mean RMSE over single-run wonders) is correctly rejecting overfit candidates — but it's also rejecting real wins if they exist as single-run. Worth revisiting if interventions 1-3 produce genuine breakthroughs.
- **`scripts/score_candidates_on_severson_b3.py`** loads b3 fresh from `.mat` via `load_severson_b3_holdout` — it does NOT honor the canonical exclusion list. To score on 40 cells instead of 44, either patch the loader or post-filter the predictions CSV.

---

## 11. Open Questions for Next Session

1. Should we rebuild the canonical Severson dataset with the new exclusion list and re-launch all 18 jobs? Or proceed with intervention #1 (post-hoc calibrator) on existing predictions?
2. The Aggregator's "Aggregator returned only 2 valid entries; expected 3" failure has happened 2× now. Is there a structural fix to the LLM schema validation?
3. Do we want to invest in BatLiNet reimplementation (stretch goal, ~163 RMSE) or stop at Severson Discharge-tier (~150-180 RMSE)?
4. If we hit 150-180 RMSE, do we publish as "first autonomous system to match Severson on her own MATR2"? Or push further?
5. Track B (the manually-targeted GBDT/log10 sweep) was never run as a positive control. Should we?

---

## 12. Companion Doc

See `LITERATURE_FINDINGS_VS_OUR_PIPELINE.md` for the detailed H1-H6 literature survey findings: what Severson, BatteryML, BatLiNet, physics-informed, foundation models, and prior agentic systems do that beats us right now, with concrete arxiv IDs / DOIs and reimplementation effort estimates.
