from __future__ import annotations

import json
from typing import Any


ROLE_SYSTEM_PROMPTS = {
    "FeatureScientist": (
        "You are the FeatureScientist in Open Battery Agents. Return only JSON matching the requested schema. "
        "Do not mention author model coefficients, hidden labels, batch 9 labels, or reference prediction files."
    ),
    "ModelArchitect": (
        "You are the ModelArchitect in Open Battery Agents. Return only JSON matching the requested schema. "
        "Prefer small-data regressors and explicit missing-value handling."
    ),
    "CodeGenerator": (
        "You are the CodeGenerator in Open Battery Agents. Return only Python code for one candidate file. "
        "Do not access the internet, reference runs, paper reproduction code, or hidden validation labels."
    ),
    "ScientistCritic": (
        "You are the ScientistCritic in Open Battery Agents. Return only JSON matching the requested schema. "
        "Critique the completed experiment without using hidden data that was not shown."
    ),
    "ChampionAggregator": (
        "You are the ChampionAggregator in Open Battery Agents. Return only JSON matching the requested schema. "
        "You see ONLY Severson-only validation evidence across all completed Track A runs. You must NOT request, infer, or claim access to Batch 9 results -- Batch 9 is sealed at this stage. "
        "Your task is to nominate a top-K shortlist of candidate configurations that best satisfy the stated selection criteria, with explicit reasoning. Prefer cross-seed and cross-split-mode stability over any single low-RMSE outlier."
    ),
    "ChampionAdjudicator": (
        "You are the ChampionAdjudicator in Open Battery Agents. Return only JSON matching the requested schema. "
        "You see Severson-only AND Batch 9 metrics for a small fixed shortlist (K candidates). Your task is to pick the single final champion and a runner-up using the stated tie-breaking rule, with a one-paragraph rationale. "
        "Default rule: choose the shortlist entry with the lowest Batch 9 RMSE, UNLESS one of them shows clear pathology (gross prediction collapse, negative R2, many non-finite predictions, batch9_rmse >> weak baseline RMSE), in which case skip the pathological entry. State which candidate you skipped and why."
    ),
}


def _json_block(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=str)


def feature_scientist_prompt(dataset_profile: dict[str, Any], feature_probe: dict[str, Any]) -> str:
    return f"""Propose a FeaturePlan JSON object for an early-cycle battery lifetime predictor.

Candidate-facing data use row_id and cell_id only as join keys. They must not be model features.
Allowed feature sources are candidate-facing metadata and first-100-cycle summaries.
When feature-program tables are available, propose declarative feature-program use rather than raw Pandas code.
FeatureProgram objects are compiled by trusted repo code and can expose scalar-only, curve-only,
scalar-plus-curve, and broad-physics feature sets.

Strong plans should reason from general battery degradation principles and choose among:
- early capacity level and retention
- capacity-fade slopes, curvature, variance, and early-window statistics
- resistance, thermal, energy, and efficiency proxies when available
- charge/discharge curve-shape statistics and cross-cycle curve deltas
- conservative protocol-current features only when explicitly allowed
- raw or transformed target modeling when statistically and physically justified

Do not assume a particular paper feature set, coefficient vector, target transform, or model family. The goal is to discover transferable early-life predictors from the available early-cycle data and validation feedback.

Dataset profile:
{_json_block(dataset_profile)}

Feature probe:
{_json_block(feature_probe)}

Return JSON with keys:
agent_id, feature_families, selected_columns, include_protocol_features, feature_program_ids,
feature_program_recipe, feature_set, max_cycle, rationale, constraints
"""


def model_architect_prompt(feature_plan: dict[str, Any], dataset_profile: dict[str, Any]) -> str:
    return f"""Propose a ModelPlan JSON object for a small scientific battery dataset.

Use robust preprocessing. Prefer small-data regressors such as Ridge, ElasticNetCV, LassoCV, RandomForest, or GradientBoosting before neural networks.
Candidates must drop all-NaN features or impute safely and must not use row_id/cell_id as predictors.
Choose target_transform as either "raw" or "log10" and justify it from the target distribution, error structure, and transfer-stability considerations.
Do not assume a paper-specific target transform or model family.

Feature plan:
{_json_block(feature_plan)}

Dataset profile:
{_json_block(dataset_profile)}

Return JSON with keys:
agent_id, model_family, estimator_name, target_transform, feature_set, hyperparameters, preprocessing_steps, rationale
"""


def code_generator_prompt(feature_plan: dict[str, Any], model_plan: dict[str, Any]) -> str:
    return f'''Write one Python candidate file implementing:

def fit(train_metadata, train_cycle_summary, train_labels, config): ...
def predict(model, test_metadata, test_cycle_summary, config): ...

You may import:
from battery_aar.features.battery_lifetime_features import build_all_battery_features

Recommended toolbox call pattern:

X = build_all_battery_features(
    metadata,
    cycle_summary,
    max_cycle=100,
    include_protocol=True,
)

Declarative/compiled candidates are preferred. When FeaturePlan/ModelPlan include feature_program_paths,
feature_set, or feature_program_recipe, do not write raw feature-plumbing code; rely on the trusted compiler
unless explicitly asked for free-form code.

Candidate-facing schema:
- train_metadata/test_metadata include row_id, cell_id, and allowed numeric physical/protocol columns.
- train_cycle_summary/test_cycle_summary include row_id, cell_id, cycle_index, discharge_capacity, charge_capacity, and other allowed numeric early-cycle columns.
- train_labels includes row_id, cell_id, y.

Rules:
- Use row_id/cell_id only for joins and output alignment.
- Do not use source paths, batch identifiers, hidden labels, reference predictions, or author model coefficients.
- Do not invent additional keyword arguments for build_all_battery_features.
- Use include_protocol=True or include_protocol=False, not include_protocol_features.
- Handle NaNs and all-NaN columns safely.
- Return predictions as DataFrame with row_id,y_pred or cell_id,y_pred.

Feature plan:
{_json_block(feature_plan)}

Model plan:
{_json_block(model_plan)}

Return only Python code.
'''


def code_repair_prompt(feature_plan: dict[str, Any], model_plan: dict[str, Any], previous_code: str, error_message: str, traceback_text: str | None) -> str:
    return f'''Repair this Python candidate while preserving the FeaturePlan and ModelPlan.

The previous candidate failed review or evaluation.

Error message:
{error_message}

Traceback:
{traceback_text or ""}

Use the exact toolbox call pattern when using the battery feature helper:

X = build_all_battery_features(
    metadata,
    cycle_summary,
    max_cycle=100,
    include_protocol=True,
)

Do not invent additional keyword arguments. Use include_protocol, not include_protocol_features.

Feature plan:
{_json_block(feature_plan)}

Model plan:
{_json_block(model_plan)}

Previous code:
```python
{previous_code}
```

Return only the complete repaired Python code.
'''


def scientist_critic_prompt(review: dict[str, Any], evaluation: dict[str, Any]) -> str:
    return f"""Write a CritiqueReport JSON object for the candidate experiment.

Review:
{_json_block(review)}

Evaluation:
{_json_block(evaluation)}

Return JSON with keys:
critic_id, strengths, weaknesses, proposed_next_steps
"""


def champion_aggregator_prompt(
    severson_summary: dict[str, Any],
    k: int,
    selection_criteria: str,
) -> str:
    return f"""Nominate a Severson-only top-{k} shortlist of candidate configurations from the completed Track A campaign.

Batch 9 is SEALED at this stage. You must not request, infer, or claim access to Batch 9 metrics, predictions, or any held-out labels. Use only the Severson-only validation evidence supplied below.

Selection criteria:
{selection_criteria}

Reasoning guidance:
- Prefer candidates whose Severson-only metrics are stable across seeds and split modes; a single low-RMSE outlier is not evidence of generalization.
- Prefer small-data sensible model choices (regularized linear, tree ensembles with sane defaults) when ranks are close; avoid over-parameterized configurations on tiny training partitions.
- Do not favor any paper-specific feature family or coefficient set. Reason from the supplied Severson metrics and configuration metadata only.
- If many candidates are statistically tied on RMSE, break ties using MAE and Spearman, then by configuration simplicity.

Severson-only campaign summary:
{_json_block(severson_summary)}

Return JSON with these top-level keys:
- agent_id (string)
- k (int, must equal {k})
- selection_criteria (string, restate the criteria you applied)
- rationale (string, one short paragraph explaining the shortlist as a whole)
- shortlist (list of exactly {k} entries, ordered best-first)

Each entry in `shortlist` MUST contain these keys:
- candidate_id (string)
- source_run_id (string)
- candidate_path (string)
- feature_program_path (string or null)
- processed_dir (string)
- recipe (string)
- split_mode (string)
- split_seed (int)
- model_family (string)
- feature_set (string)
- target_transform (string)
- surrogate_rmse (float)
- surrogate_mae (float)
- surrogate_spearman (float)
- rank_in_shortlist (int, 1-based, matching the entry's position)
"""


def champion_adjudicator_prompt(
    shortlist_with_batch9: dict[str, Any],
    selection_criteria: str,
) -> str:
    return f"""Pick the single final champion and a runner-up from the fixed shortlist below. No further iteration is allowed. Do not request additional Batch 9 evaluations -- the Batch 9 metrics already attached to each entry are final.

Selection criteria:
{selection_criteria}

Default tie-breaking rule: choose the shortlist entry with the lowest `batch9_rmse`, UNLESS that entry shows clear pathology. Pathology checks (skip any entry exhibiting one or more):
- gross prediction collapse (predictions near-constant or with vanishing variance)
- negative `batch9_r2`
- many non-finite predictions (`batch9_n_nonfinite_predictions` materially > 0)
- many negative predictions for a non-negative lifetime target (`batch9_n_negative_predictions` materially > 0)
- `batch9_rmse` much worse than the weak baseline (`batch9_rmse` >> `batch9_weak_baseline_rmse`)

If you skip an entry for pathology, name the entry and the failing check in your rationale.

Shortlist with Batch 9 metrics:
{_json_block(shortlist_with_batch9)}

Return JSON with these top-level keys:
- agent_id (string)
- selection_criteria (string, restate the rule you applied)
- rationale (string, one short paragraph explaining the choice and any pathology skips)
- final_champion (object, the chosen shortlist entry with ALL of its original fields PLUS batch9_rmse, batch9_mae, batch9_spearman, batch9_pgr; must include candidate_id and source_run_id)
- runner_up (object or null, same shape as final_champion; null only if no non-pathological alternative remains)
- shortlist_evaluated (list, the full shortlist as received, preserved verbatim for audit)
"""


def schema_repair_prompt(text: str, schema_hint: str) -> str:
    return f"""The previous response did not parse as valid JSON for this schema:
{schema_hint}

Repair it and return only valid JSON. Previous response:
{text}
"""
