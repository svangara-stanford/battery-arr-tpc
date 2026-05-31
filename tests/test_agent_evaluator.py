from battery_aar.agents.evaluator import HiddenEvaluator
from battery_aar.agents.llm_client import OfflineHeuristicAgent
from battery_aar.agents.orchestrator import make_splits, synthetic_dataset, weak_baseline_rmse


def test_hidden_evaluator_scores_offline_candidate(tmp_path):
    meta, cycles, labels = synthetic_dataset(seed=1, n_cells=30)
    train, val, test = make_splits(labels, seed=1)
    candidate = tmp_path / "candidate.py"
    candidate.write_text(OfflineHeuristicAgent("a").propose("prompt", 0).code)
    evaluator = HiddenEvaluator(meta, cycles, labels, train, val, test, weak_rmse=weak_baseline_rmse(labels, train, val), max_cycle=100)
    result = evaluator.evaluate_candidate(candidate)
    assert result["success"]
    assert result["metrics"]["rmse"] >= 0
