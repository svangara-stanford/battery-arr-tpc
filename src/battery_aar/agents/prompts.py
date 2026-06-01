from __future__ import annotations


def rediscovery_prompt(max_cycle: int, leaderboard: str = "") -> str:
    return f"""You are proposing an early-cycle battery lifetime predictor.

Use only the first {max_cycle} cycles. Candidate code must implement either:

def fit(train_metadata, train_cycle_summary, train_labels, config): ...
def predict(model, test_metadata, test_cycle_summary, config): ...

or a CandidateModel class with fit and predict methods.

Use only numpy, pandas, scipy, and sklearn. Do not use internet access. Do not
use identifiers, batch artifacts, file ordering, or hidden labels as predictors.

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

Prefer physically interpretable early-cycle capacity and trend features.

Previous validation feedback:
{leaderboard}
"""
