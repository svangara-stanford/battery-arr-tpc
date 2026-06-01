from battery_aar.agents.evaluator import HiddenEvaluator
from battery_aar.agents.llm_client import OfflineHeuristicAgent
from battery_aar.agents.orchestrator import make_splits, synthetic_dataset, weak_baseline_rmse


def _evaluator():
    meta, cycles, labels = synthetic_dataset(seed=4, n_cells=24, max_cycle=20)
    train, val, test = make_splits(labels, seed=4)
    return HiddenEvaluator(meta, cycles, labels, train, val, test, weak_rmse=weak_baseline_rmse(labels, train, val), max_cycle=15)


def _write_candidate(tmp_path, code: str):
    candidate = tmp_path / "candidate.py"
    candidate.write_text(code)
    return candidate


def test_hidden_evaluator_scores_offline_candidate(tmp_path):
    meta, cycles, labels = synthetic_dataset(seed=1, n_cells=30)
    train, val, test = make_splits(labels, seed=1)
    candidate = tmp_path / "candidate.py"
    candidate.write_text(OfflineHeuristicAgent("a").propose("prompt", 0).code)
    evaluator = HiddenEvaluator(meta, cycles, labels, train, val, test, weak_rmse=weak_baseline_rmse(labels, train, val), max_cycle=100)
    result = evaluator.evaluate_candidate(candidate)
    assert result["success"]
    assert result["metrics"]["rmse"] >= 0


def test_candidate_using_row_id_succeeds(tmp_path):
    candidate = _write_candidate(
        tmp_path,
        """
import pandas as pd

def fit(train_metadata, train_cycle_summary, train_labels, config):
    return train_labels.set_index('row_id')['y'].mean()

def predict(model, test_metadata, test_cycle_summary, config):
    return pd.DataFrame({'row_id': test_metadata['row_id'], 'y_pred': model})
""",
    )
    result = _evaluator().evaluate_candidate(candidate)
    assert result["success"]


def test_candidate_using_cell_id_join_key_succeeds(tmp_path):
    candidate = _write_candidate(
        tmp_path,
        """
import pandas as pd

def fit(train_metadata, train_cycle_summary, train_labels, config):
    by_cell = train_cycle_summary.groupby('cell_id')['discharge_capacity'].last().to_frame('q_last')
    labels = train_labels.set_index('cell_id')[['y']]
    return float(by_cell.join(labels, how='inner')['y'].mean())

def predict(model, test_metadata, test_cycle_summary, config):
    cells = test_cycle_summary.groupby('cell_id').size().index
    return pd.DataFrame({'cell_id': cells, 'y_pred': model})
""",
    )
    result = _evaluator().evaluate_candidate(candidate)
    assert result["success"]


def test_candidate_returning_row_id_predictions_succeeds(tmp_path):
    candidate = _write_candidate(
        tmp_path,
        """
import pandas as pd

def fit(train_metadata, train_cycle_summary, train_labels, config):
    return float(train_labels['y'].median())

def predict(model, test_metadata, test_cycle_summary, config):
    return pd.DataFrame({'row_id': test_metadata['row_id'].to_numpy(), 'y_pred': model})
""",
    )
    result = _evaluator().evaluate_candidate(candidate)
    assert result["success"]


def test_candidate_returning_cell_id_predictions_succeeds(tmp_path):
    candidate = _write_candidate(
        tmp_path,
        """
import pandas as pd

def fit(train_metadata, train_cycle_summary, train_labels, config):
    return float(train_labels['y'].mean())

def predict(model, test_metadata, test_cycle_summary, config):
    return pd.DataFrame({'cell_id': test_metadata['cell_id'].to_numpy(), 'y_pred': model})
""",
    )
    result = _evaluator().evaluate_candidate(candidate)
    assert result["success"]


def test_candidate_returning_y_pred_in_test_order_succeeds(tmp_path):
    candidate = _write_candidate(
        tmp_path,
        """
import pandas as pd

def fit(train_metadata, train_cycle_summary, train_labels, config):
    return float(train_labels['y'].mean())

def predict(model, test_metadata, test_cycle_summary, config):
    return pd.DataFrame({'y_pred': [model] * len(test_metadata)})
""",
    )
    result = _evaluator().evaluate_candidate(candidate)
    assert result["success"]


def test_missing_join_key_with_wrong_count_has_clear_failure_reason(tmp_path):
    candidate = _write_candidate(
        tmp_path,
        """
import pandas as pd

def fit(train_metadata, train_cycle_summary, train_labels, config):
    return 1.0

def predict(model, test_metadata, test_cycle_summary, config):
    return pd.DataFrame({'y_pred': [model]})
""",
    )
    result = _evaluator().evaluate_candidate(candidate)
    assert not result["success"]
    assert result["error_type"] == "PredictionValidationError"
    assert "row_id or cell_id" in result["failure_reason"]
