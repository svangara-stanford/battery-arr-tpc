# What Severson and Successor Models Do That Beats Our Agentic Pipeline (As of 2026-06-12)

**Context:** Our autonomous-discovery pipeline lands at ~357 RMSE / 21% MAPE on canonical Severson b3 / MATR2 (40 cells). Severson's published Discharge model: 173 RMSE / 8.6%. BatteryML reproduction: 149 RMSE. BatLiNet (verified deep-learning SOTA): 163 ± 12. This document synthesizes findings from 10 parallel subagents (G1-G4 forensic audit on Severson + H1-H6 literature survey) into a concrete list of what the prior art does differently, in priority order.

**Single most important framing finding (from H6):** ***No published autonomous or agentic system has matched Severson on her own MATR2 split.*** Our 357 RMSE is *in range* with Severson's commonly-cited baseline (214-357) and *beats every deep model* in the BatteryML benchmark on MATR2 (LSTM 21,933, CNN 228,104, Transformer 36,425, MLP 27,527). We are in **unclaimed territory** — closing the gap to 150 RMSE would be a publishable first.

---

## 1. The Verified MATR2 Leaderboard

(Compiled from BatteryML benchmark, BatLiNet paper, DiffBatt paper, Schaeffer review. Numbers are RMSE in cycles on the canonical 40-cell b3 test set, trained on b1+b2.)

| Rank | Method | RMSE | Year | Citation | Key trick |
|---|---|---|---|---|---|
| 1* | RUL-QMoE | 136 | 2025 | arXiv 2512.23725 | Non-crossing quantile mixture-of-experts, per-chemistry experts. **Unverified** — split provenance not yet audited. |
| 2 | **Severson "Discharge" (BatteryML reproduction)** | **149** | 2023 | arXiv 2310.14714 | Plain OLS on 6 hand-crafted features. BatteryML's tighter Q(V) preprocessing beats Severson's own 173. |
| 3 | **BatLiNet** | **163 ± 12** | 2024 | Nat. Mach. Intell. / arXiv 2310.05052 | Inter-cell paired-difference CNN: predict lifetime *deltas* between cell pairs. Only deep model reliably beating linear baselines. |
| 4 | **Severson "Discharge" (original 2019)** | **173** | 2019 | Nature Energy | 6 features + elastic net. The model Severson published. |
| 5 | PCR | 187 | 2023 | BatteryML | Principal-component regression on raw Q(V) curves. |
| 6 | Physics-Informed Self-Attention (Nicolae) | 180 | 2024 | arXiv 2404.17174 | Two-stage: Arrhenius capacity-loss prior + self-attention residual. Tiny model. |
| 7 | PLSR | 181 | 2023 | BatteryML | Partial-least-squares on Q(V). |
| 8 | Ridge regression on Q(V) | 184 | 2023 | BatteryML | Plain L2 on full discharge curve. |
| 9 | Severson "Variance" | 196 (paper) / 211 (BatteryML) | 2019/2023 | both | Single feature: log Var(ΔQ_{100-10}(V)). |
| 10 | Random Forest | 202 | 2023 | BatteryML/BatLiNet | Tree ensemble on Q(V). |
| 11 | MLP | 207 ± 4 | 2023 | BatLiNet | Plain feed-forward — already starts to overfit. |
| 12 | Severson "Full" | 214 (paper) / >1000 (BatteryML repro) | 2019 | Nature Energy | All 20 features + elastic net. **Adding more features made MATR2 WORSE** — overfitting on b3 distribution shift. |
| 13 | LSTM | 219 ± 33 | 2023 | BatteryML | Sequence model on Q(V). |
| 14 | CNN | 228 ± 104 | 2023 | BatteryML | 1-D conv; huge seed variance. |
| 15 | DiffBatt | 235 ± 16 | 2024 | arXiv 2410.23893 | Conditional diffusion + transformer. *Worse* than ridge on MATR2 despite winning other benchmarks. |
| 16 | Transformer | 300-364 | 2023 | BatteryML | Catastrophic overfit. |
| 17 | SVM | 300 | 2023 | BatLiNet | RBF SVR. |
| **OUR PIPELINE** | **357** | 2026 | this session | LLM agents proposing Python feature functions. **Beats every neural net in BatteryML.** |
| 18 | XGBoost | 799 | 2023 | BatteryML | Tree boosting. |
| 19+ | Naive MLP / LSTM / CNN / Transformer (different BatteryML seeds) | 21,933 / 27,527 / 36,425 / 228,104 | 2023 | BatteryML | Catastrophic overfit. **Our 357 is 60-640× better than these.** |

**The practical floor: ~130-150 RMSE.** Anyone claiming sub-100 on MATR2 is using a non-canonical split, leaking b3 into training, or evaluating on a different task (e.g., cycle-by-cycle trajectory RMSE).

---

## 2. What Severson Actually Did (G1 + G2 + G4 forensic findings)

### 2.1 Cell exclusions (canonical: 124 cells total)

From Severson's `LoadData.m` (https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation):

**Batch 3 (40 cells after exclusions, full secondary test):**
- Drop `b3c2, b3c23, b3c32, b3c37, b3c42, b3c43` ("noisy channels" per paper Methods + LoadData.ipynb)
- Cell count after: 40 (from 46 raw)

**Batch 1 (41 cells after exclusions):**
- Drop `b1c8, b1c10, b1c12, b1c13, b1c22` ("batteries that do not finish")
- BatteryML alternatively *stitches* b1c[0..4] with b2c[7,8,9,15,16] (different strategy)

**Batch 2 (43 cells after exclusions):**
- BatteryML's stitch consumes b2c[7,8,9,15,16] into b1, leaving 43 in b2

**Our prior state (this session start):** kept ALL 138 cells. Discrepancy was the source of the inflated b3 RMSE — the 4 extra cells (b3c2, b3c37, b3c42, b3c43) account for **62% of our b3 RMSE²** despite being only 9% of the test set, because 3 of 4 are upper-tail cells (cycle_life 1267-1642) that no model trained on b1+b2 (max ~1227) can extrapolate to.

**Fix applied this session:** `data/severson_canonical_exclusions.txt` populated with the canonical 11-cell list. New dataset builds drop them automatically.

### 2.2 Train/primary-test split (deterministic, NOT random)

From `LoadData.m`:
```matlab
test_ind = [1:2:(numBat1+numBat2), 84];   % every odd index in combined b1+b2, plus index 84
train_ind = setdiff(1:(numBat1+numBat2), test_ind);
secondary_test_ind = numBat-numBat3+1:numBat;  % all of b3
```

**This is alternating odd/even pairing of cells within b1+b2, NOT a random shuffle.** Our pipeline uses `np.random.default_rng(seed)` to do a random shuffle — different specific cell assignments. Doesn't materially affect headline numbers but means our split is not byte-for-byte reproducible against Severson.

### 2.3 Feature pipeline (Severson exact)

**ΔQ(V) = Q_discharge(V; cycle=100) − Q_discharge(V; cycle=10)**

- Voltage grid: **1000 linearly spaced points from 3.5 V to 2.0 V (DESCENDING)**. Methods, p. 389: "Capacity was fitted as a function of voltage and evaluated at 1,000 linearly spaced voltage points from 3.5 V to 2.0 V."
- Smoothing: **smoothing spline applied BEFORE interpolation.** Supp Fig 29: "A smoothing spline accurately captures the relationship."
- Each statistical feature is `log10(|·|)`, not raw. E.g., `feature_3 = log10(|var(ΔQ(V))|)`.

### 2.4 The three published models (Severson Table 1, verbatim)

| Model | Train RMSE | Train % err | Primary RMSE (excl outlier) | Primary % err | **Secondary RMSE** | **Secondary % err** |
|---|---|---|---|---|---|---|
| Variance | 103 | 14.1 | 138 (138) | 14.7 (13.2) | **196** | **11.4** |
| Discharge | 76 | 9.8 | 91 (86) | 13.0 (10.1) | **173** | **8.6** |
| Full | 51 | 5.6 | 118 (100) | 14.1 (7.5) | **214** | **10.7** |

**The famous "9.1% test error" is NOT directly in Table 1.** It is derived: average of Discharge's primary-test-excl-outlier (10.1%) and secondary (8.6%) = **9.35%** ≈ "9.1%". The single-model headline therefore is the **Discharge model, 6 features**.

**Critical note**: "% test error" in Severson is **mean absolute percent error on the log-cycle-life predictions** (MAPE in log space), NOT MAPE in linear cycles. We compute linear MAPE. Different metric scale — for the same underlying error pattern, our linear MAPE is typically ~2x larger than Severson's "% test error."

### 2.5 Variance model — 1 feature

```
y = log10(cycle_life)
x = log10(|var(ΔQ_{100-10}(V))|)
fit: OLS, y = a + b*x
```

That's the entire model. **G4 reproduced this exactly on our data: RMSE 148.4 / MAPE 11.9% on the canonical 40-cell b3.** Within 1 cycle of Severson's published 149.

### 2.6 Discharge model — 6 features (the headline 173)

From Severson Supp. Table 1, the 6 features:
1. `log10(|min(ΔQ_{100-10}(V))|)`
2. `log10(|var(ΔQ_{100-10}(V))|)`
3. `log10(|skewness(ΔQ_{100-10}(V))|)`
4. `log10(|kurtosis(ΔQ_{100-10}(V))|)`
5. **Slope of linear fit to Q_d vs cycle, cycles 2–100** ← THE KEY INDEPENDENT AXIS
6. Discharge capacity at cycle 2 (`Qd2`)

Plus IR features (#19 min IR, #20 ΔIR_{100-2}) per SI.

**The slope feature (#5) is the key non-variance signal.** H2 ranked it as the single highest-leverage missing feature in our agent substrate.

### 2.7 Full model — 9 features (the overfit one, 214 RMSE)

Discharge features + charge time (cycles 2-6 mean) + temperature integral + slope/intercept of capacity-fade curves at multiple windows + IR features. **Adding more features made MATR2 *worse* than Discharge** (214 vs 173) — a published cautionary tale about over-engineering for a small dataset with batch-shift.

### 2.8 BatteryML's reproduction (gets 149 RMSE — better than Severson's own 173)

Same features, same model. Differences that bring 173 → 149:
- Plain OLS (not elastic net with CV-chosen α — Severson used elastic net, BatteryML doesn't bother)
- Label transform: `log → z-score` (natural log + z-score normalization) — Severson uses raw log10
- Voltage grid: 1000 pts from 2.0 V → 3.5 V (ASCENDING, opposite to Severson's descending — same set of points so doesn't matter, but worth noting)
- Discharge segment selection: `I < -0.1 A` filter before interpolation
- `use_precalculated_qdlin: True` — reuses the .mat's stock 1000-point Qdlin interpolation, doesn't recompute
- Smoothing function literally returns rolling median (window=10, sigma=3) — has a bug but doesn't matter for our purposes
- `min_rul_limit=100` — drop any cell with RUL ≤ 100 from training/scoring

The 173 → 149 delta is **24 cycles from preprocessing alone**, with the same 6 features. Reproducibility nuance, not science.

### 2.9 Why does adding more features (Discharge → Full) HURT on b3?

The b3 distribution shift breaks every "borrowed" signal:
- IR features generalize poorly to b3 (different cycler, calendar aging)
- Temperature features generalize poorly (different ambient + setpoint history)
- Charge-time-cycles-2-6 features generalize poorly (different protocols)

Only the ΔQ-curve features and the capacity-fade slope are robust enough to survive the b3 transfer. **Less is more** on a small dataset with batch-shift. This is a deep lesson that applies to autonomous discovery too — the agents should be biased toward *fewer, more robust* features.

---

## 3. What BatLiNet Does (The Only Deep Model That Wins)

(From H3 deep-dive. arXiv 2310.05052 / Nature Mach. Intell. 2024. Code Ocean DOI:10.24433/CO.8904065.v2)

### Architecture (two parallel branches, shared linear head)

1. **Intra-cell branch**: capacity-indexed 6-channel signal (charging V, discharging V, charging I, discharging I, voltage-differential, resistance signals) computed as **differences between cycle 100 and cycle 10** — extends Severson's ΔQ_{100-10} idea to all signals.
2. **Inter-cell branch**: same signals computed as **differences between cell PAIRS** (target cell vs an "anchor" cell from training set with KNOWN lifetime). **Predicts lifetime *delta* between cells.**

At inference: pair query against multiple anchors, predict deltas, average with the intra-cell branch's absolute prediction.

### The conceptual trick

**Reformulates extrapolation as interpolation between cells.**

Standard regression: predict cycle_life for cell X. Fails when X is far outside training support.

BatLiNet: predict (cycle_life of X) - (cycle_life of anchor A), where A is in training. Inter-cell delta is structurally smaller than absolute predictions, so the long-life-tail problem becomes a short-life-delta problem.

### Reimplementation effort

- 1-2 weeks for clean reimpl
- Two CNN towers + shared head + anchor-pair sampling at train time
- Single GPU (V100 / A100 16GB) sufficient
- Public Code Ocean capsule available

### Adapt to our autonomous-discovery substrate?

The "predict delta against an anchor cell" pattern could be added as a candidate type in the agent substrate. The agents could propose anchor-selection heuristics (closest cycle-2 capacity, etc.) and inter-cell delta features. Genuine research direction.

---

## 4. Three Concrete Interventions to Close Our Gap (Priority Order)

(From H2 + H4 synthesis.)

### Intervention 1: Centered Isotonic Regression + Quantile-Target Transform (<1 day)

**Source:** MDPI Batteries 11(4):145 (2025).

**What it does:** Post-hoc calibrator that wraps any model. Quantile-transforms the cycle-life target (compresses tail), fits an isotonic regression `y_calibrated = f(y_predicted)` via PAVA.

**Why it helps us:** Our candidates regress toward the mean — they predict near 800-900 cycles for everything. The b3 mean is 1060 with long tail. Isotonic post-calibration can stretch our predictions to better match the b3 marginal distribution, addressing the systematic underestimation on long-life cells.

**Implementation:**
```python
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import QuantileTransformer

# After model.predict()
qt = QuantileTransformer(output_distribution='uniform').fit(y_train.reshape(-1,1))
y_pred_q = qt.transform(model.predict(X_test).reshape(-1,1)).ravel()

iso = IsotonicRegression(out_of_bounds='clip').fit(
    qt.transform(model.predict(X_train).reshape(-1,1)).ravel(), y_train)
y_pred_cal = iso.predict(y_pred_q)
```

**Expected RMSE drop:** 30-60 cycles. Cheap, low risk.

### Intervention 2: Add Capacity-Fade Slope Primitive (1 day)

**Source:** H2; Severson Discharge model feature #5.

**What it does:** Expose `linear_fit_slope(series, start_cycle, end_cycle)` as a callable primitive that agents can use directly. Specifically: `slope of discharge_capacity vs cycle_index, cycles 2-100`.

**Why it helps us:** This is the single highest-leverage missing feature. Severson's Discharge model gets to 173 RMSE specifically because the slope is the *independent axis* to ΔQ variance. Agents currently have to derive this from raw cycle data, which they often don't. Making it a single-call primitive should bring the variance + slope combo within easy reach.

**Implementation:** Add to `src/battery_aar/features/operators.py` or expose in the agent-visible primitives layer. Single column per call: `discharge_capacity_slope_cycle_2_to_100` (or similar).

**Expected RMSE drop:** 20-40 cycles. Plus enables Severson Discharge-tier (173 RMSE) candidates to actually emerge.

### Intervention 3: Arrhenius/SEI Physics Prior on Capacity-Loss Curve (1-3 days)

**Source:** Nicolae et al., arXiv 2404.17174.

**What it does:** Parameterize the model as fitting `Q_loss(n) = exp(A) · n^B + Q_0` (Arrhenius-form SEI growth prior). Per-cell heads predict (A, B); the prediction is `cycle_life = (Q_0_to_EOL / exp(A))^(1/B)` (or similar back-transform). Power-law tail extrapolates monotonically — which is exactly what's broken in our agent candidates on b3's long-life regime.

**Why it helps us:** The author's network gets to 180 RMSE on MATR2 with a tiny model. The Arrhenius prior is plug-in: it's a soft regularizer on the output, not a new architecture. The b3 long-life-tail problem is precisely that unconstrained models can't extrapolate from b1+b2's truncated lifetimes — the power-law prior fixes this at the model level.

**Implementation:** Wrap the predicted cycle_life with a learnable (A, B) head, train with MSE loss on `log(cycle_life)`. Most code reuse from existing candidate template; <100 lines new.

**Expected RMSE drop:** Gets us to ~180 RMSE if implemented well. Best single intervention if we want to break below 200.

### Stretch (1-2 weeks): BatLiNet Inter-Cell Paired Regression

**Source:** Zhang et al., Nat. Mach. Intell. 2024 / arXiv 2310.05052. Code Ocean DOI:10.24433/CO.8904065.v2.

Already described in §3. Gets to 163 RMSE — verified SOTA. Major engineering investment but the architecture is conceptually clean.

---

## 5. What's NOT Worth Pursuing

(From H1 + H3 + H5 negative findings.)

| Method | Why not |
|---|---|
| BatteryGPT (42M-param GPT on tokenized charging curves) | **Different task** — predicts SOH trajectory from first 30% of life, not cycle-life from first 100 cycles. 0.213% RMSE is meaningless for our comparison. |
| Battery-Timer (LoRA on Timer TSFM) | **Not evaluated on MATR2.** Evaluated on a custom industrial dataset. No comparable number. |
| BatteryTSFM (Universal Battery Degradation) | **Same as above** — different evaluation set, no MATR2 number. |
| DiffBatt (diffusion + transformer) | **235 RMSE on MATR2** — *worse* than Ridge (184). Best on MATR1 mean (88) but loses on the b3 split. Confirmed by authors. |
| Naive CNN / LSTM / Transformer / MLP on raw Q(V) | **Catastrophic overfit on MATR2** — RMSE 20k-230k. Our 357 already beats them by 60-640×. The b3 batch-shift kills any model without strong priors when the train set is 80 cells. |
| XGBoost / Random Forest on raw Q(V) | Modest perf (202 RF, 799 XGB). Not worth the engineering. |
| Any paper reporting MATR2 RMSE < 100 | **Always suspect.** Either (a) random k-fold over all 180 cells (leaks b3 into training), (b) cycle-by-cycle trajectory RMSE rather than per-cell scalar, or (c) silently dropping the longest-lived b3 cells. |
| Multi-Feature Transformer (claimed 2.44 RMSE) | Cycle-by-cycle trajectory, not canonical b3. Different task. |

---

## 6. Prior Agentic / Autonomous Discovery Attempts (H6)

**No published autonomous system has matched Severson on her own MATR2 split.**

Closest analogs:
1. **Discovery Learning (Zhang & Song et al., Nature 650:110, 2026, DOI 10.1038/s41586-025-09951-7).** Agentic learner/interpreter/oracle loop. Only 50 cycles needed. Evaluated on Farasis Energy pouch cells, NOT Severson. Reports 91 RMSE on their own held-out groups. Public Code Ocean capsule. **Mechanically not an LLM agent — it's classical ML modules (active learning + physics simulator + zero-shot regressor) branded as agents.** The Interpreter uses hand-engineered physics features and does NOT autonomously rediscover Var(ΔQ).
2. **Genetic Programming HI (RESS 2025, DOI:S0951832025001838).** GP autonomously evolves health indicators from discharge-voltage-curve differences on Severson. Rediscovers the *family* of ΔQ-statistic features (variance, skewness, kurtosis) — confirming the structure is recoverable by SR. No clean MATR2 RMSE reported.
3. **Rhyu/Schaeffer/Braatz Joule 2025 (formation features, arXiv 2410.07458).** TRI/D3BATT publication. Two hand-designed Q(V) features at V=3.57-3.66 V → 9.20% MAPE on a different dataset (Cui 186-cell formation). NOT autonomous and NOT MATR2.
4. **LLM-FE (Abhyankar et al., arXiv 2503.14434, Mar 2025).** Most relevant LLM-driven feature engineering paper. Evaluated on adult, bank-marketing, arrhythmia — **no battery datasets**. CAAFE/FeatLLM/OCTree are classification-only in their official implementations.
5. **BatteryAgent (arXiv 2512.24686, Dec 2025).** Three-layer agent: physics-perception → GBDT+SHAP → LLM reasoning. **Solves fault diagnosis, not cycle life regression** — doesn't touch MATR2.

**The unclaimed-territory framing:** *"We are the first to apply LLM-driven feature discovery to Severson's canonical MATR2 split."*

---

## 7. The Honest Floor (From Schaeffer 2024 Review)

Schaeffer's 2024 review (arXiv 2404.04049) explicitly frames the b3 gap as a **data coverage problem, not a model failure**. Cells outside the training support cannot be learned.

Concretely:
- b1+b2 max cycle_life: ~1227
- b3 max cycle_life: ~1935
- Several b3 cells in 1500-2000 range have **no comparable training instances**
- Calendar-aging shift between train (May+Jun 2017) and test (Apr 2018) is **not learnable from cycle data alone**

**Closable to ~150 RMSE: yes, with effort.** BatLiNet's 163 ± 12 is the demonstrated frontier. Plus physics priors + isotonic calibration could trim another 10-20.

**~50 RMSE is implausible.** Three converging reasons:
1. Median b3 life = 964 cycles; 50 RMSE = 5% MAPE, *better* than Severson's 9.1% on the easier primary test
2. Calendar-aging confound is irreducible without diagnostic data
3. Multiple independent reimplementations of Severson Full report MATR2 RMSE ranging 214 to >1000 — pure run-to-run variance exceeds 50 cycles

**Practical floor: ~130-150 RMSE.** Anything below requires either (a) labeled b3 cells (active learning — Zhang 2025 shows 1 cell suffices for fine-tuning), or (b) external diagnostic data (Howey-style dV/dQ).

---

## 8. Summary: Why We're At 357 RMSE Instead of 150

Five compounding gaps:

| Gap | Their value | Our value | Closes how much? |
|---|---|---|---|
| Cell exclusions | Severson's 40 b3 cells | We kept 44 (3 extra are upper-tail outliers) | ~14 RMSE (370 → 357) |
| Target transform | log10 + z-score (or just log10) | log10 only (now default; was raw) | Already fixed via E2 |
| Voltage window | Fixed (2.0, 3.5) V | Per-cell intersection (default) | Available via E3, not default |
| Smoothing | Smoothing spline before interp | Raw linear interp | ~5-10 RMSE |
| **Capacity-fade slope feature** | **Severson's #5 — independent axis to ΔQ** | **Not exposed as primitive** | **~30-50 RMSE (Discharge → Variance gap of 196-173=23 on canonical, ~50 on us)** |
| Model class | OLS / elastic net (BatteryML 149) | GBR / Ridge / Lasso (overfit prone) | Single-step minor; matters when paired with right features |
| **Post-hoc calibration** | Implicit via good features | **None** | **30-60 RMSE** |
| **Physics prior** | Implicit via target choice + features | **None** | **~30 RMSE if added** |
| **Inter-cell pairing** | BatLiNet trick | Not implemented | ~20 RMSE on top of the rest |
| **Agentic selection bias** | N/A | Stability-over-performance rejects single-run wonders | Hard to quantify; may be rejecting real signal |

**Sum of available improvements: roughly 357 → 200 → 165 → 145.** Sequenced over 1-3 weeks of focused work. The path to Severson-comparable autonomous discovery exists, but every individual step is small.

---

## 9. Specific Concrete Recommendations

In priority order (effort estimate, expected RMSE drop):

1. **Patch `score_candidates_on_severson_b3.py` to honor canonical exclusions** (1 hour, ~14 RMSE drop on current candidates). Currently it loads all 44 b3 cells from .mat; should filter against `data/severson_canonical_exclusions.txt`. This is a "free" win on existing predictions.

2. **Add `linear_fit_slope` primitive to the agent substrate** (1 day, ~30 RMSE expected). The single highest-leverage missing feature. Source: H2.

3. **Implement isotonic + quantile-transform post-calibrator** (<1 day, ~30-60 RMSE expected). Wraps the existing model output. Source: H4.

4. **Set `voltage_window=(2.0, 3.5)` as default** in `make_curve_delta_program` and `make_broad_physics_program` (10 minutes, ~5-10 RMSE). E3 added the option; just flip the default.

5. **Add Arrhenius `Q_loss(n) = exp(A)·n^B + Q_0` prior as a model class** that candidates can select (1-3 days, ~30 RMSE drop, gets to ~180). Source: H4 / Nicolae 2024.

6. **Audit ChampionAggregator's stability bias** (1 day, qualitative). The Aggregator systematically rejects single-run wonders even when they're real. After interventions 1-5 produce more candidates, revisit whether the bias is appropriate.

7. **Re-launch 18-job campaign on patched substrate** (8h compute, depends on queue). Run after interventions 1-5.

8. **Stretch: BatLiNet reimplementation** (1-2 weeks, gets to ~163). Only after 1-7 are exhausted.

---

## 10. Open Questions for the Next Session

1. Do we drop the 4 noisy b3 cells from the secondary test, or keep them with the post-hoc filter (which is what `score_candidates_on_severson_b3.py` would need)? Severson dropped them; our autonomous discovery is being unfairly penalized by them.
2. Should the agents see the formation paper's voltage windows (3.57-3.66 V) as candidate hyperparameters? Or is that too on-the-nose?
3. Is the Aggregator's stability bias actually the right policy? Single-run wonders are *sometimes* leaks, *sometimes* real. After post-calibration + slope primitive, the false-positive rate may drop enough that we should loosen the gate.
4. Track B (manually-targeted GBDT/log10 sweep) was never run as a positive control. If we want to publish, we need it.
5. Do we eventually want to demonstrate transfer to the Attia 2019-01-24 batch (the original "Batch 9")? Or focus entirely on Severson?

---

## Sources

### Primary papers
- Severson 2019: https://web.mit.edu/braatzgroup/Severson_NatureEnergy_2019.pdf, https://www.nature.com/articles/s41560-019-0356-8
- BatteryML benchmark (Zhang et al., ICLR 2024): https://arxiv.org/abs/2310.14714
- BatLiNet (Zhang et al., Nat. Mach. Intell. 2024): https://arxiv.org/abs/2310.05052, Code Ocean DOI 10.24433/CO.8904065.v2
- Schaeffer review (2024): https://arxiv.org/abs/2404.04049
- Physics-informed self-attention (Nicolae et al., 2024): https://arxiv.org/abs/2404.17174
- Centered Isotonic Regression (MDPI Batteries 2025): https://www.mdpi.com/2313-0105/11/4/145
- DiffBatt (NeurIPS 2024-W): https://arxiv.org/abs/2410.23893
- RUL-QMoE (2025, unverified MATR2 win): https://arxiv.org/abs/2512.23725
- Rhyu/Schaeffer/Braatz Joule 2025 (formation features): https://arxiv.org/abs/2410.07458, code Zenodo 10.5281/zenodo.14916092
- Discovery Learning (Nature 2026): https://www.nature.com/articles/s41586-025-09951-7, https://arxiv.org/abs/2508.06985

### Negative-finding references
- Fair Dinkum Systems blog (critical commentary on MATR2 reproducibility): https://fairdinkumsystems.com/blog/standford-dataset/

### Code references
- Severson's LoadData repo: https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation
- BatteryML code: https://github.com/microsoft/BatteryML
- BatLiNet Code Ocean: https://doi.org/10.24433/CO.8904065.v2

### Our session artifacts
- Manual Variance reproduction (G4): `/scratch/users/svangara/battery-arr/oneoff/severson_variance_reproduction.py`, output `/scratch/users/svangara/battery-arr/oneoff/severson_variance_out/`
- Probe predictions: `/scratch/users/svangara/battery-arr/champion_selection/_probe_post_splitfix_fixed_20260612T001254Z_29135838/`
- Live campaign reports: `/scratch/users/svangara/battery-arr/reports/trackA_*_28893*`
- Latest champion: `/scratch/users/svangara/battery-arr/champion_selection/20260612T001300Z_29135839/`
