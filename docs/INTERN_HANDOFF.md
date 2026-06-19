# Intern Handoff — Open Battery Agents / Track A

**Written:** 2026-06-19
**For:** incoming interns taking over the autonomous battery-lifetime-discovery project
**Read these next, in order:** this doc → `docs/SESSION_CONTEXT_2026-06-12.md` (the
detailed engineering log) → `docs/LITERATURE_FINDINGS_VS_OUR_PIPELINE.md` (the
leaderboard and what beats us) → `SETUP.md` (getting it running).

---

## 1. What this project is

We are building an **autonomous, LLM-agent-driven scientific discovery
pipeline** and pointing it at a famous battery problem:

> **Predict a lithium-ion cell's total cycle life from only its first 100
> cycles of data** — the task from Severson et al., *Nature Energy* 2019.

Instead of a human hand-crafting the predictive features, role-separated LLM
agents (a Proposer, an Engineer, an Evaluator, an Aggregator/Adjudicator)
autonomously:
1. propose candidate feature functions over the raw charge/discharge curves,
2. compile and run them into a regression model,
3. score them, and
4. select a champion.

The scientific question is not "can we predict battery life" (Severson solved
that in 2019). It is **"can a team of autonomous agents *rediscover*
Severson-quality predictors on their own?"** If they can, that's a publishable
first — see §4.

**"Track A"** is the name for the *sealed broad-discovery campaign*: a fixed
18-job experiment matrix that we launch on the cluster, with a held-out test
set that stays locked until a champion is chosen, so the result is honest.

---

## 2. The data and the evaluation contract (memorize this)

The contract is the most important thing to get right — we burned a week on
two separate data-leak / wrong-dataset bugs that produced beautiful-but-fake
numbers. Details in `SESSION_CONTEXT_2026-06-12.md §2`.

- **Dataset:** Severson 2019 "MATR" cycling data — three `.mat` batches:
  - `b1` = `2017-05-12`, `b2` = `2017-06-30`, `b3` = `2018-04-12`.
- **Canonical split** (this is Severson's, and we now enforce it):
  - **Train + primary_test:** mixed halves of **b1 + b2** (~46 train cells).
  - **Locked secondary test:** **all of b3** (40 cells after dropping 6 noisy
    channels). This is the number we report. It stays sealed during search.
- **Excluded cells** live in `data/severson_canonical_exclusions.txt` (11
  cells: 6 noisy b3 + 5 unfinished b1). Any dataset rebuild drops them
  automatically.
- **Headline metric:** RMSE in cycles on the 40-cell b3 test, with **MAPE**
  reported alongside. Target = Severson "Discharge" model ≈ **173 RMSE / 8.6%
  MAPE** (BatteryML's tighter reproduction hits **149**).
- **Default knobs:** `target_transform=log10`, `SECONDARY_TEST_SOURCE=severson_b3`.

**Two traps we already fell into — do not repeat:**
1. **"Batch 9" is NOT Severson b3.** `data/2019-01-24_batch9/` is the *Attia
   2020 closed-loop* batch — a different paper, a harder/different task. It is
   demoted to an opt-in transfer test. Always test against **b3**.
2. **The splitter used to leak b3 into training** when run in `random` mode,
   giving a fake 6–7% MAPE. Fixed by making `orchestrator.make_search_split`
   honor the canonical `splits.csv` tags, plus a load-bearing
   **honest-contract assertion** in `role_graph.py` that raises if any
   `secondary_test` row reaches train/val. **Do not remove that assertion.**

---

## 3. What has been achieved so far

**Infrastructure (done and verified — 204/204 tests pass as of June 12):**
- Full role-separated agent workflow (`src/battery_aar/workflows/role_graph.py`)
  that runs end-to-end on the cluster.
- Canonical Severson split enforced build-time and run-time, with a leak guard.
- MAPE / % error metric threaded through the whole pipeline; `log10` target
  default; a genuinely physics-aware `broad_physics` feature recipe (it used to
  be a silent duplicate of another recipe).
- Slurm campaign tooling: launcher, monitor, summarizer, agentic
  champion-selection, and a direct b3 scorer that bypasses the LLM.
- Hardened Slurm scripts against a real `slurmd` node bug (heartbeats, absolute
  Lustre output paths, `--no-requeue`, `--exclude=sh04-13n32`).

**Scientific result so far — an honest, in-range number (not yet the goal):**
- Autonomous discovery currently lands at **~357 RMSE / ~21% MAPE** on the
  canonical 40-cell b3 test.
- For context, **this already beats every neural net** in the BatteryML
  benchmark on this split (their LSTM/CNN/Transformer blow up to 20k–228k RMSE).
  But it does **not** yet match Severson's own linear Discharge model
  (149–173). See the full leaderboard in
  `docs/LITERATURE_FINDINGS_VS_OUR_PIPELINE.md §1`.
- **We proved the pipeline is correct, not broken:** a manual reproduction of
  Severson's Variance model *on our exact data* hits **148.4 RMSE / 11.9%
  MAPE** — within one cycle of her published number. So the substrate can
  express a winning feature; the *agents* just aren't selecting it yet.

**Why the agents underperform (the key diagnostic):** the winning Severson
feature — `log Var(ΔQ_{100−10}(V))` — is reachable as a single column, but the
Aggregator's *stability bias* (preferring candidates with low cross-seed
variance) systematically rejects it as a "single-run wonder." The agents
predict near the mean and never lock onto the variance signal.

---

## 4. The goal, and what to do next

**The user's standing directive:** *do not publish the honest negative — keep
iterating the autonomous-discovery system until it produces Severson-comparable
RMSE (~150–180 on the 40-cell b3).* If we get there, the framing is "first
autonomous/agentic system to match Severson on her own MATR2 split."

**Three concrete, pre-scoped interventions (none started yet — these are your
first tasks, in priority order).** Full rationale in
`SESSION_CONTEXT_2026-06-12.md §9` and the literature doc.

1. **Post-hoc calibrator: centered isotonic regression + quantile-target
   transform** on top of any model. ~1 day. Expected drop: **30–60 RMSE**.
   Lowest-risk, highest-ratio win. Start here.
2. **Add a capacity-fade-slope primitive** to the agent feature substrate:
   `linear_fit_slope(series, start_cycle, end_cycle)`. This is the single
   highest-leverage *missing* feature — it's the independent axis Severson's
   Discharge model pairs with ΔQ variance. ~1 day. Expected drop: **20–40
   RMSE**.
3. **Arrhenius / SEI physics prior** on the predicted capacity-loss curve
   (`Q_loss(n) = exp(A)·n^B + Q₀`), two-stage fit. 1–3 days. Gets to **~180
   RMSE**.

Sequenced, these plausibly move us **357 → ~200 → ~165 RMSE**. Stretch goal
(1–2 weeks): a BatLiNet-style inter-cell paired-difference model (verified
~163 RMSE, public Code Ocean capsule).

**Before writing intervention code**, also consider the open questions in
`SESSION_CONTEXT_2026-06-12.md §11` — notably whether to fix the Aggregator's
stability bias so it stops discarding real single-run wins, and whether to run
"Track B" (a manually-targeted GBDT sweep) as a positive control.

---

## 5. How to actually run it (cheat sheet)

This is a **Stanford Sherlock** HPC project. **Never run Python on the login
node** — submit through Slurm or grab `sh_dev`. Each shell is fresh, so load
the module and activate the venv in the *same* command:

```bash
module load python/3.12.1
source /home/groups/darve/svangara/battery-arr-venvs/oba/bin/activate
cd /home/groups/darve/svangara/battery-arr-tpc

# sanity: run the test suite (do this on an sh_dev shell, not the login node)
python -m pytest tests/ -x --tb=line -q

# launch the 18-job Track A campaign
bash scripts/launch_sherlock_trackA_campaign.sh
python3 scripts/monitor_trackA_campaign.py            # check progress
SCRATCH=/scratch/users/svangara python3 scripts/summarize_sherlock_trackA.py

# agentic champion selection (after summarize)
sbatch --export=ALL,SECONDARY_TEST_SOURCE=severson_b3,K=3 \
  nersc/sherlock_champion_selection.slurm

# score candidates directly on clean b3, bypassing the LLM (the honest probe)
python3 scripts/score_candidates_on_severson_b3.py --candidate-list <PATH> \
  --battery-fast-charging-root literature_models_and_data/battery-fast-charging \
  --out <SCRATCH out dir> --max-cycle 100
```

**Where things live:**
- Repo: `/home/groups/darve/svangara/battery-arr-tpc`
- Venv: `/home/groups/darve/svangara/battery-arr-venvs/oba`
- Raw data: `<repo>/literature_models_and_data/battery-fast-charging/data/...`
  (git-ignored, ~19 GB — see `SETUP.md §3` for how to obtain/place it)
- All run output, reports, champion artifacts: **`$SCRATCH/battery-arr/...`**
  (never `$HOME`). Latest clean champion run:
  `/scratch/users/svangara/battery-arr/champion_selection/20260612T001300Z_29135839/`

**Resource floors** (don't undershoot or jobs OOM/timeout): discovery jobs
default 64G / 16 cpu / 8h; champion-selection scoring needs ≥96G / 3h;
discovery audits need ≥32G. Never use the `dev` partition for batch jobs.

---

## 6. Landmines — things not to lose

- The **honest-contract assertion** in `role_graph.py` is load-bearing. Keep it.
- The **4 noisy b3 cells** (b3c2, b3c37, b3c42, b3c43) are in the exclusion
  list. Rebuilds drop them automatically — don't "add them back" thinking
  you're getting more data; they make the number worse and dishonest.
- **`broad_physics` is a real recipe** now. **`log_var(ΔQ)` is reachable** as
  `discharge_discharge_capacity_curve_delta_identity_log_var_cycle_99_minus_9`.
- `score_candidates_on_severson_b3.py` loads b3 fresh from `.mat` and does
  **not** apply the exclusion list — it scores 44 cells, not 40. Post-filter
  the predictions CSV to get the canonical-40 number.
- **Test on b3, report on b3.** If you ever see sub-100 RMSE on MATR2, assume a
  leak or a wrong split before you celebrate — that's exactly how we got
  burned twice.
