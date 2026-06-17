"""Patch E2: tests for the MAPE / test_error_pct metrics threaded through the
Open Battery Agents pipeline. Covers the audited sites in the patch ticket:

  * ``regression_metrics`` emits ``mape`` and ``test_error_pct``.
  * The MAPE computation is NaN-safe when ``y_true`` contains zeros / negatives.
  * ``weak_baseline_metrics`` returns both ``rmse`` and ``mape``.
  * ``EvaluationReport`` carries ``mape`` and ``test_error_pct``, and they
    propagate into the role-graph markdown report.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from battery_aar.agents.evaluator import regression_metrics
from battery_aar.agents.orchestrator import (
    weak_baseline_metrics,
    weak_baseline_metrics_against,
    weak_baseline_rmse,
)
from battery_aar.workflows.role_graph import _candidate_metric_row, _write_report
from battery_aar.workflows.schemas import (
    CandidateSpec,
    EvaluationReport,
    ReviewReport,
)


# ---------------------------------------------------------------------------
# regression_metrics
# ---------------------------------------------------------------------------

def test_regression_metrics_emits_mape_and_test_error_pct():
    rng = np.random.default_rng(0)
    y_true = rng.uniform(200.0, 2000.0, size=64)
    # Perturb y_true by a known 10% relative error.
    y_pred = y_true * 1.10
    metrics = regression_metrics(y_true, y_pred)

    assert "mape" in metrics
    assert "test_error_pct" in metrics
    assert np.isfinite(metrics["mape"])
    assert np.isfinite(metrics["test_error_pct"])

    # MAPE of constant +10% perturbation is exactly 0.10.
    assert metrics["mape"] == pytest.approx(0.10, rel=1e-6)
    assert metrics["test_error_pct"] == pytest.approx(10.0, rel=1e-6)

    # Existing RMSE/MAE/R2/Spearman/Kendall still present and finite.
    for key in ("rmse", "mae", "r2", "spearman", "kendall"):
        assert key in metrics
        assert np.isfinite(metrics[key])


def test_regression_metrics_mape_masks_nonpositive_y_true():
    # Mix in a zero and a negative y_true; expect masking to skip them but
    # still produce a finite MAPE from the remaining positive rows.
    y_true = np.array([100.0, 200.0, 0.0, -50.0, 500.0])
    y_pred = np.array([110.0, 180.0, 5.0, 5.0, 450.0])
    metrics = regression_metrics(y_true, y_pred)
    # |10/100| + |20/200| + |50/500| = 0.10 + 0.10 + 0.10 -> mean = 0.10
    assert metrics["mape"] == pytest.approx(0.10, rel=1e-6)
    assert metrics["test_error_pct"] == pytest.approx(10.0, rel=1e-6)


def test_regression_metrics_mape_nan_safe_when_all_y_true_nonpositive():
    y_true = np.array([0.0, -1.0, -10.0])
    y_pred = np.array([1.0, 1.0, 1.0])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        metrics = regression_metrics(y_true, y_pred)
        warned = any(
            issubclass(w.category, RuntimeWarning) and "MAPE" in str(w.message)
            for w in caught
        )
        assert warned, "expected a RuntimeWarning about MAPE when no positive y_true"
    assert np.isnan(metrics["mape"])
    assert np.isnan(metrics["test_error_pct"])
    # Other metrics still compute (RMSE/MAE remain finite, R2 may not).
    assert np.isfinite(metrics["rmse"])
    assert np.isfinite(metrics["mae"])


# ---------------------------------------------------------------------------
# weak_baseline_metrics
# ---------------------------------------------------------------------------

def test_weak_baseline_metrics_returns_rmse_and_mape():
    rng = np.random.default_rng(7)
    n = 40
    labels = pd.DataFrame({
        "row_id": np.arange(n),
        "y": rng.uniform(300.0, 1800.0, size=n),
    })
    train_ids = list(range(0, 30))
    val_ids = list(range(30, n))

    metrics = weak_baseline_metrics(labels, train_ids, val_ids)
    assert set(metrics.keys()) >= {"rmse", "mape", "test_error_pct"}
    assert np.isfinite(metrics["rmse"])
    assert np.isfinite(metrics["mape"])
    assert np.isfinite(metrics["test_error_pct"])
    assert metrics["test_error_pct"] == pytest.approx(metrics["mape"] * 100.0, rel=1e-9)

    # Backward-compat RMSE wrapper agrees with the dict.
    assert weak_baseline_rmse(labels, train_ids, val_ids) == pytest.approx(metrics["rmse"])


def test_weak_baseline_metrics_against_returns_rmse_and_mape():
    rng = np.random.default_rng(11)
    train_labels = pd.DataFrame({"row_id": np.arange(40), "y": rng.uniform(400.0, 1600.0, size=40)})
    eval_labels = pd.DataFrame({"row_id": np.arange(40, 60), "y": rng.uniform(400.0, 1600.0, size=20)})
    metrics = weak_baseline_metrics_against(train_labels, eval_labels)
    assert np.isfinite(metrics["rmse"])
    assert np.isfinite(metrics["mape"])
    assert np.isfinite(metrics["test_error_pct"])


# ---------------------------------------------------------------------------
# End-to-end: EvaluationReport carries mape and report markdown surfaces it
# ---------------------------------------------------------------------------

def _build_record(rmse: float, mape: float) -> dict:
    candidate = CandidateSpec(
        run_id="run_test",
        human_readable_summary="test candidate",
        candidate_id="cand_test_00",
        agent_id="model_builder",
        agent_role="llm_candidate",
        iteration=0,
        candidate_path="/tmp/dummy.py",
        candidate_name="cand_test_00",
        model_family="Ridge",
        feature_set="all_available",
        target_transform="log10",
    )
    review = ReviewReport(
        run_id="run_test",
        human_readable_summary="ok",
        reviewer_id="code_reviewer",
        iteration=0,
        verdict="pass",
        issues=[],
        recommendations=[],
    )
    evaluation = EvaluationReport(
        run_id="run_test",
        human_readable_summary="ok",
        candidate_id=candidate.candidate_id,
        candidate_path=candidate.candidate_path,
        agent_id="evaluator",
        iteration=0,
        split_mode="random",
        success=True,
        rmse=rmse,
        mae=rmse * 0.7,
        r2=0.8,
        spearman=0.7,
        kendall=0.6,
        mape=mape,
        test_error_pct=mape * 100.0,
        pgr=0.5,
        extra_metrics={"y_pred_mean": 800.0, "y_true_mean": 820.0, "n_predictions": 10},
    )
    return {"candidate": candidate, "review": review, "evaluation": evaluation, "iteration": 0}


def test_evaluation_report_carries_mape_and_test_error_pct():
    record = _build_record(rmse=120.0, mape=0.12)
    row = _candidate_metric_row(record)
    assert row["mape"] == pytest.approx(0.12)
    assert row["test_error_pct"] == pytest.approx(12.0)


def test_role_graph_markdown_report_includes_mape_columns(tmp_path: Path):
    record = _build_record(rmse=120.0, mape=0.12)
    per_iteration_rows = [_candidate_metric_row(record)]
    report = {
        "run_id": "run_test",
        "split_mode": "random",
        "offline": True,
        "iterations": 1,
        "candidates_per_iteration": 1,
        "role_sequence": [],
        "n_train_cells": 8,
        "n_validation_cells": 2,
        "validation_fraction": 0.2,
        "split_seed": 0,
        "include_feature_programs": False,
        "feature_program_mode": "none",
        "feature_program_recipe": None,
        "feature_program_report": {"feature_programs_used": [], "n_feature_program_columns": 0,
                                    "feature_family_counts": {}, "true_raw_curve_features_used": False,
                                    "proxy_features_used": False, "protocol_features_used": False},
        "tool_calls": [],
        "candidate_path": record["candidate"].candidate_path,
        "best_candidate_id": record["candidate"].candidate_id,
        "candidate_spec": record["candidate"].model_dump(mode="json"),
        "best_iteration": 0,
        "final_iteration_candidate_path": record["candidate"].candidate_path,
        "per_iteration_metrics": per_iteration_rows,
        "best_by_feature_set": [
            {
                "feature_set": "all_available",
                "candidate_id": record["candidate"].candidate_id,
                "model_family": "Ridge",
                "target_transform": "log10",
                "rmse": 120.0,
                "mae": 84.0,
                "r2": 0.8,
                "mape": 0.12,
                "test_error_pct": 12.0,
            }
        ],
        "best_by_target_transform": [
            {
                "target_transform": "log10",
                "candidate_id": record["candidate"].candidate_id,
                "feature_set": "all_available",
                "model_family": "Ridge",
                "rmse": 120.0,
                "mae": 84.0,
                "r2": 0.8,
                "mape": 0.12,
                "test_error_pct": 12.0,
            }
        ],
        "candidate_config_deduplication": {
            "n_candidate_ids_generated": 1,
            "n_unique_candidate_configs": 1,
            "n_duplicate_candidate_configs_pruned": 0,
            "final_batch9_top_k_unique": 1,
        },
        "label_source_summary": {
            "label_source": "synthetic",
            "source_dataset": "synthetic",
            "true_measured_cycle_life_labels": False,
            "author_model_prediction_pseudo_labels": False,
        },
        "validation_metrics": record["evaluation"].model_dump(mode="json"),
        "locked_batch9_validation": {"status": "not_run"},
        "final_batch9_metrics": {},
        "final_batch9_topk_rows": [],
        "artifact_index_path": "",
        "review_verdict": "pass",
        "best_review_status": "pass",
    }
    out = tmp_path / "role_agent_workflow.md"
    _write_report(out, report)
    text = out.read_text()
    assert "MAPE" in text, "MAPE column missing from per-iteration / best-by tables"
    assert "test_error_pct" in text, "test_error_pct missing from validation-metrics block"
    # The actual numeric row should also be rendered.
    assert "0.12" in text
    assert "12.0" in text


def test_evaluation_report_schema_round_trip_with_mape():
    payload = EvaluationReport(
        run_id="run_test",
        human_readable_summary="ok",
        candidate_id="c1",
        candidate_path="/tmp/x.py",
        split_mode="random",
        success=True,
        rmse=100.0,
        mae=70.0,
        r2=0.7,
        spearman=0.6,
        kendall=0.5,
        mape=0.15,
        test_error_pct=15.0,
        pgr=0.4,
    )
    serialized = payload.model_dump(mode="json")
    assert serialized["mape"] == 0.15
    assert serialized["test_error_pct"] == 15.0
    # And it round-trips back through pydantic.
    restored = EvaluationReport(**serialized)
    assert restored.mape == pytest.approx(0.15)
    assert restored.test_error_pct == pytest.approx(15.0)
