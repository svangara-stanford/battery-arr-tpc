import json

import pandas as pd

from battery_aar.agents.baseline_candidates import author_inspired_baseline_candidates
from battery_aar.agents.evaluator import HiddenEvaluator
from battery_aar.agents.orchestrator import make_splits, run_rediscovery, synthetic_dataset, weak_baseline_rmse


def test_author_inspired_baselines_run_through_candidate_evaluator(tmp_path):
    meta, cycles, labels = synthetic_dataset(seed=8, n_cells=32, max_cycle=110)
    train, val, test = make_splits(labels, seed=8)
    evaluator = HiddenEvaluator(
        meta,
        cycles,
        labels,
        train,
        val,
        test,
        weak_rmse=weak_baseline_rmse(labels, train, val),
        max_cycle=100,
    )

    for candidate in author_inspired_baseline_candidates():
        path = tmp_path / f"{candidate.name}.py"
        path.write_text(candidate.code)
        result = evaluator.evaluate_candidate(path)
        assert result["success"], result.get("failure_reason")
        assert result["metrics"]["rmse"] >= 0


def test_seed_with_author_inspired_baselines_adds_leaderboard_rows_and_feature_summary(tmp_path):
    report = run_rediscovery(
        processed_dir=tmp_path / "missing",
        reference_run=tmp_path / "missing_reference",
        out=tmp_path / "run",
        reports_dir=tmp_path / "reports",
        agents=1,
        iterations=1,
        offline=True,
        seed=9,
        seed_with_author_inspired_baselines=True,
    )
    leaderboard = pd.read_csv(tmp_path / "run" / "leaderboard.csv")
    summary = pd.read_csv(tmp_path / "run" / "candidate_feature_summary.csv")

    baseline_rows = leaderboard[leaderboard["agent_id"] == "author_inspired_baseline"]
    assert len(baseline_rows) == 3
    assert baseline_rows["success"].all()
    assert summary["used_toolbox"].any()
    assert report["seed_with_author_inspired_baselines"] is True
    assert report["candidate_feature_summary_rows"] == len(summary)


def _write_batch9_holdout(root):
    batch9 = root / "data" / "2019-01-24_batch9"
    batch9.mkdir(parents=True, exist_ok=True)
    for channel in range(1, 5):
        q = [1.12 - 0.001 * cycle - 0.0003 * channel for cycle in range(1, 121)]
        payload = {
            "barcode": f"B9{channel}",
            "channel_id": channel,
            "protocol": "OED\\20190124-4pt4_5pt6_5pt2_4pt252.sdu",
            "summary": {
                "cycle_index": list(range(1, 121)),
                "discharge_capacity": q,
                "cycle_life": [800 + 10 * channel] * 120,
            },
        }
        (batch9 / f"2019-01-24_batch9_CH{channel}_structure.json").write_text(json.dumps(payload))
    return batch9


def test_final_batch9_top_k_evaluates_after_search(tmp_path):
    root = tmp_path / "battery-fast-charging"
    batch9 = _write_batch9_holdout(root)
    report = run_rediscovery(
        processed_dir=tmp_path / "missing",
        reference_run=tmp_path / "missing_reference",
        out=tmp_path / "run_topk",
        reports_dir=tmp_path / "reports",
        agents=1,
        iterations=1,
        offline=True,
        seed=10,
        final_batch9_validation=True,
        final_batch9_top_k=2,
        battery_fast_charging_root=root,
        batch9_path=batch9,
        seed_with_author_inspired_baselines=True,
    )
    topk = pd.read_csv(tmp_path / "run_topk" / "final_batch9_topk_metrics.csv")
    predictions = sorted((tmp_path / "run_topk" / "final_batch9_topk_predictions").glob("*.csv"))
    summary = pd.read_csv(tmp_path / "run_topk" / "candidate_feature_summary.csv")

    assert len(topk) == 2
    assert len(predictions) == 2
    assert topk["rank_by_surrogate_rmse"].tolist() == [1, 2]
    assert summary["locked_batch9_rmse"].notna().sum() >= 2
    assert report["final_batch9_topk"]["status"] == "ok"
