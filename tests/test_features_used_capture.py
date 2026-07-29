"""The evaluator must record which features a candidate actually built.

Motivated by the role2 audit (2026-07): feature plans named families that the
generated candidates never used, and nothing recorded the executed features.
"""
from __future__ import annotations

from pathlib import Path

from battery_aar.agents.evaluator import evaluate_candidate_train_test
from battery_aar.agents.orchestrator import synthetic_dataset

CANDIDATE_CODE = '''
import numpy as np
from battery_aar.features.battery_lifetime_features import build_all_battery_features


def fit(train_metadata, train_cycle_summary, train_labels, config):
    X = build_all_battery_features(train_metadata, train_cycle_summary, max_cycle=100).reset_index()
    merged = X.merge(train_labels[["row_id", "y"]], on="row_id")
    return float(np.mean(merged["y"]))


def predict(model, test_metadata, test_cycle_summary, config):
    X = build_all_battery_features(test_metadata, test_cycle_summary, max_cycle=100).reset_index()
    return X[["row_id"]].assign(y_pred=model)
'''


def test_features_used_recorded_for_successful_candidate(tmp_path: Path) -> None:
    metadata, cycles, labels = synthetic_dataset(seed=0, n_cells=12, max_cycle=100)
    train_ids = metadata["row_id"].tolist()[:8]
    test_ids = metadata["row_id"].tolist()[8:]
    candidate_path = tmp_path / "candidate.py"
    candidate_path.write_text(CANDIDATE_CODE)

    result = evaluate_candidate_train_test(
        candidate_path,
        metadata[metadata["row_id"].isin(train_ids)],
        cycles[cycles["row_id"].isin(train_ids)],
        labels[labels["row_id"].isin(train_ids)],
        metadata[metadata["row_id"].isin(test_ids)],
        cycles[cycles["row_id"].isin(test_ids)],
        labels[labels["row_id"].isin(test_ids)],
        timeout_s=120,
    )

    assert result["success"], result.get("failure_reason")
    used = result["features_used"]
    assert used["library_builder_called"] is True
    assert used["n_fit_features"] == len(used["fit_feature_columns"]) > 0
    assert any(col.startswith("capacity_") for col in used["fit_feature_columns"])
    phases = [call["phase"] for call in used["calls"]]
    assert "fit" in phases and "predict" in phases
    fit_call = next(call for call in used["calls"] if call["phase"] == "fit")
    assert fit_call["call_kwargs"].get("max_cycle") == 100


def test_features_used_recorded_even_when_candidate_fails(tmp_path: Path) -> None:
    failing = CANDIDATE_CODE.replace(
        'return X[["row_id"]].assign(y_pred=model)',
        'raise RuntimeError("boom after building features")',
    )
    candidate_path = tmp_path / "failing_candidate.py"
    candidate_path.write_text(failing)
    metadata, cycles, labels = synthetic_dataset(seed=1, n_cells=10, max_cycle=100)
    train_ids = metadata["row_id"].tolist()[:7]
    test_ids = metadata["row_id"].tolist()[7:]

    result = evaluate_candidate_train_test(
        candidate_path,
        metadata[metadata["row_id"].isin(train_ids)],
        cycles[cycles["row_id"].isin(train_ids)],
        labels[labels["row_id"].isin(train_ids)],
        metadata[metadata["row_id"].isin(test_ids)],
        cycles[cycles["row_id"].isin(test_ids)],
        labels[labels["row_id"].isin(test_ids)],
        timeout_s=120,
    )

    assert not result["success"]
    used = result["features_used"]
    assert used is not None and used["library_builder_called"] is True
    assert used["n_fit_features"] > 0
