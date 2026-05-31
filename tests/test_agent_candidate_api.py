from pathlib import Path

import pandas as pd

from battery_aar.agents.candidate_api import run_candidate


def test_candidate_api_runs_simple_candidate(tmp_path):
    candidate = tmp_path / "candidate.py"
    candidate.write_text(
        """
import pandas as pd

def fit(train_metadata, train_cycle_summary, train_labels, config):
    return float(train_labels['y'].mean())

def predict(model, test_metadata, test_cycle_summary, config):
    return pd.DataFrame({'row_id': test_metadata['row_id'], 'y_pred': model})
"""
    )
    meta = pd.DataFrame({"row_id": [1, 2]})
    cycles = pd.DataFrame({"row_id": [1, 2], "cycle_index": [1, 1], "discharge_capacity": [1.0, 1.0]})
    labels = pd.DataFrame({"row_id": [1, 2], "y": [100.0, 120.0]})
    result = run_candidate(candidate, meta, cycles, labels, meta, cycles, {"max_cycle": 1})
    assert result.success
    assert result.predictions is not None
    assert result.predictions["y_pred"].tolist() == [110.0, 110.0]
