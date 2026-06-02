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
}


def _json_block(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=str)


def feature_scientist_prompt(dataset_profile: dict[str, Any], feature_probe: dict[str, Any]) -> str:
    return f"""Propose a FeaturePlan JSON object for an early-cycle battery lifetime predictor.

Candidate-facing data use row_id and cell_id only as join keys. They must not be model features.
Allowed feature sources are candidate-facing metadata and first-100-cycle summaries.
When feature-program tables are available, propose declarative feature-program use rather than raw Pandas code.
FeatureProgram objects are compiled by trusted repo code and can expose scalar-only, curve-only,
scalar-plus-curve, broad-physics, and Attia/Severson-like feature sets.

Strong plans should consider author-inspired but coefficient-free feature families:
- discharge capacity at cycles 2, 10, and 100 when available
- maximum early capacity minus cycle-2 capacity
- cycle-N minus cycle-10 capacity differences
- early and late capacity slopes
- log-transformed difference-statistic proxies
- protocol-current features only when allowed

The literature early predictor modeled log10(cycle life). Feature plans can support either raw cycle-life
models or log10 target modeling; the ModelArchitect chooses the target transform.

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

Use robust preprocessing. Prefer Ridge, ElasticNet, ElasticNetCV, RandomForest, or GradientBoosting over neural networks.
Candidates must drop all-NaN features or impute safely and must not use row_id/cell_id as predictors.
The literature early predictor modeled log10(cycle life). For positive lifetime targets, log10 target modeling
may improve stability under batch shifts. Choose target_transform as either "raw" or "log10" and justify the choice.

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


def schema_repair_prompt(text: str, schema_hint: str) -> str:
    return f"""The previous response did not parse as valid JSON for this schema:
{schema_hint}

Repair it and return only valid JSON. Previous response:
{text}
"""
