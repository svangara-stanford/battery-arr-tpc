"""Tests for the random-feature baseline control.

The baseline is the no-RAG / no-LLM comparison point for the FeatureScientist.
It must (a) run cleanly through the same candidate evaluator the agent
candidates use, (b) produce finite regression metrics, and (c) be deterministic
per seed while genuinely varying across seeds -- otherwise "averaged over seeds"
is meaningless.
"""

from __future__ import annotations

import numpy as np

from battery_aar.agents.evaluator import HiddenEvaluator
from battery_aar.agents.orchestrator import (
    synthetic_dataset,
    weak_baseline_rmse,
)
from battery_aar.agents.random_feature_baseline import (
    random_feature_baseline_candidate,
    random_feature_baseline_candidates,
    random_feature_baseline_code,
)


def _evaluator():
    meta, cycles, labels = synthetic_dataset(seed=8, n_cells=48, max_cycle=110)
    # Deterministic split built directly (avoids the read-only-array quirk in
    # orchestrator.make_splits under some numpy versions).
    row_ids = sorted(int(r) for r in labels["row_id"].tolist())
    val = row_ids[::4]
    train = [r for r in row_ids if r not in set(val)]
    test: list[int] = []
    return HiddenEvaluator(
        meta,
        cycles,
        labels,
        train,
        val,
        test,
        weak_rmse=weak_baseline_rmse(labels, train, val),
        max_cycle=100,
    )


def test_random_baseline_runs_through_candidate_evaluator(tmp_path):
    evaluator = _evaluator()
    cand = random_feature_baseline_candidate(seed=0, n_features=8)
    path = tmp_path / f"{cand.name}.py"
    path.write_text(cand.code)

    result = evaluator.evaluate_candidate(path)

    assert result["success"], result.get("failure_reason")
    metrics = result["metrics"]
    assert metrics["rmse"] >= 0
    for key in ("rmse", "mae", "mape", "r2"):
        assert np.isfinite(metrics[key]), f"{key} not finite: {metrics[key]}"


def test_random_baseline_is_deterministic_per_seed(tmp_path):
    evaluator = _evaluator()

    def _rmse(seed: int) -> float:
        cand = random_feature_baseline_candidate(seed=seed, n_features=8)
        path = tmp_path / f"{cand.name}.py"
        path.write_text(cand.code)
        result = evaluator.evaluate_candidate(path)
        assert result["success"], result.get("failure_reason")
        return result["metrics"]["rmse"]

    # Same seed reproduces exactly.
    assert _rmse(0) == _rmse(0)


def test_random_baseline_varies_across_seeds():
    # Different seeds should select different feature subsets (with a pool of
    # 20+ features and n=8, identical draws across four seeds is vanishingly
    # unlikely), so the emitted source differs.
    codes = {c.seed: c.code for c in random_feature_baseline_candidates(
        seeds=(0, 1, 2, 3), n_features=8)}
    assert len(set(codes.values())) > 1


def test_large_n_features_selects_full_pool_without_error(tmp_path):
    # The agent-features control asks for more features than exist; the
    # candidate must clamp to the available pool rather than crash.
    evaluator = _evaluator()
    code = random_feature_baseline_code(seed=0, n_features=10_000)
    path = tmp_path / "control.py"
    path.write_text(code)

    result = evaluator.evaluate_candidate(path)

    assert result["success"], result.get("failure_reason")
    assert np.isfinite(result["metrics"]["rmse"])
