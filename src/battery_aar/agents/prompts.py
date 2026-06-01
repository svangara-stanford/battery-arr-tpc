from __future__ import annotations


def rediscovery_prompt(max_cycle: int, leaderboard: str = "") -> str:
    return f"""You are proposing an early-cycle battery lifetime predictor.

Use only the first {max_cycle} cycles. Candidate code must implement either:

def fit(train_metadata, train_cycle_summary, train_labels, config): ...
def predict(model, test_metadata, test_cycle_summary, config): ...

or a CandidateModel class with fit and predict methods.

Use only numpy, pandas, scipy, and sklearn. Do not use internet access. Do not
use identifiers, batch artifacts, file ordering, or hidden labels as predictors.
You may import and use the provided battery feature toolbox:

from battery_aar.features.battery_lifetime_features import build_all_battery_features

Example:
X = build_all_battery_features(metadata, cycle_summary, max_cycle={max_cycle}, include_protocol=True)

Candidate-facing data schema:
- train_metadata columns: row_id, cell_id, then allowed physical/protocol metadata columns only.
- train_cycle_summary columns: row_id, cell_id, cycle_index, discharge_capacity, charge_capacity, then other allowed early-cycle numeric columns when available.
- train_labels columns: row_id, cell_id, y.
- test_metadata columns: row_id, cell_id, then allowed physical/protocol metadata columns only.
- test_cycle_summary columns: row_id, cell_id, cycle_index, discharge_capacity, charge_capacity, then other allowed early-cycle numeric columns when available.

row_id is the canonical join key. cell_id is an anonymized compatibility alias
for row_id. You may group, merge, or join on row_id or cell_id, but do not use
row_id or cell_id as model features. Validation/test labels are hidden; only
train_labels contains y.

Prediction outputs may be any one of:
- a DataFrame with columns row_id and y_pred,
- a DataFrame with columns cell_id and y_pred,
- or one y_pred value per test row in the same order as test_metadata.

cycle_summary can contain NaNs. charge_capacity may be missing or entirely
all-NaN in some datasets; prefer discharge_capacity features when
charge_capacity is unavailable. Drop all-NaN feature columns or impute numeric
features safely inside a sklearn pipeline.

Strong candidates should try author-inspired, physically interpretable
early-cycle feature families: capacity at cycles 10 and {max_cycle}, capacity
slope, late-cycle slope, cycle-N minus cycle-10 differences, and log-transformed
difference statistics, including energy-like integral changes when available.
Do not use row_id, cell_id, source paths, batch_id, or
other identifiers as model features. Protocol currents are available only when
the run explicitly allows protocol features; otherwise they will be absent from
candidate-facing metadata. For small datasets, prefer Ridge, ElasticNet,
RandomForest, or GradientBoosting over neural networks.

Previous validation feedback:
{leaderboard}
"""
