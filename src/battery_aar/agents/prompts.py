from __future__ import annotations


def rediscovery_prompt(max_cycle: int, leaderboard: str = "") -> str:
    return f"""You are proposing an early-cycle battery lifetime predictor.

Use only the first {max_cycle} cycles. Candidate code must implement either:

def fit(train_metadata, train_cycle_summary, train_labels, config): ...
def predict(model, test_metadata, test_cycle_summary, config): ...

or a CandidateModel class with fit and predict methods.

Use only numpy, pandas, scipy, and sklearn. Do not use internet access. Do not
use identifiers, batch artifacts, file ordering, or hidden labels as predictors.
Return a DataFrame with columns row_id and y_pred, or just y_pred in test order.

Allowed data columns are described by the provided metadata and cycle-summary
tables. Prefer physically interpretable early-cycle capacity and trend features.

Previous validation feedback:
{leaderboard}
"""
