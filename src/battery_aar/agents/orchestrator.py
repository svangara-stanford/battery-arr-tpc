from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .evaluator import HiddenEvaluator, regression_metrics
from .llm_client import make_agent
from .memory import append_event
from .prompts import rediscovery_prompt
from .reports import posthoc_feature_overlap, write_agent_reports


def synthetic_dataset(seed: int = 0, n_cells: int = 80, max_cycle: int = 120) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows_meta = []
    rows_cycles = []
    rows_labels = []
    for row_id in range(n_cells):
        q0 = rng.normal(1.08, 0.015)
        fade = rng.uniform(0.00005, 0.00045)
        curve = rng.normal(0, 0.001, max_cycle)
        lifetime = 350 + 1200 * (0.0005 - fade) / 0.00045 + rng.normal(0, 35)
        rows_meta.append({"row_id": row_id, "cell_id": f"synthetic_{row_id}", "batch_id": f"b{row_id % 4}", "cc1": rng.choice([3.6, 4.0, 4.8, 5.6])})
        rows_labels.append({"row_id": row_id, "y": max(100.0, lifetime)})
        for cycle in range(1, max_cycle + 1):
            rows_cycles.append({"row_id": row_id, "cycle_index": cycle, "discharge_capacity": q0 - fade * cycle + curve[cycle - 1]})
    return pd.DataFrame(rows_meta), pd.DataFrame(rows_cycles), pd.DataFrame(rows_labels)


def load_processed_or_synthetic(processed_dir: str | Path, seed: int, max_cycle: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    base = Path(processed_dir)
    meta_path = base / "metadata.csv"
    cycle_path = base / "cycle_summary.csv"
    labels_path = base / "labels.csv"
    if meta_path.exists() and cycle_path.exists() and labels_path.exists():
        return pd.read_csv(meta_path), pd.read_csv(cycle_path), pd.read_csv(labels_path), "processed"
    meta, cycles, labels = synthetic_dataset(seed=seed, max_cycle=max(120, max_cycle))
    return meta, cycles, labels, "synthetic_demo"


def make_splits(labels: pd.DataFrame, seed: int, split_mode: str = "random") -> tuple[list[int], list[int], list[int]]:
    rng = np.random.default_rng(seed)
    ids = labels["row_id"].to_numpy(int)
    rng.shuffle(ids)
    n = len(ids)
    n_train = max(4, int(0.6 * n))
    n_val = max(2, int(0.2 * n))
    return ids[:n_train].tolist(), ids[n_train : n_train + n_val].tolist(), ids[n_train + n_val :].tolist()


def weak_baseline_rmse(labels: pd.DataFrame, train_ids: list[int], val_ids: list[int]) -> float:
    train = labels[labels["row_id"].isin(train_ids)]["y"].to_numpy(float)
    val = labels[labels["row_id"].isin(val_ids)]["y"].to_numpy(float)
    pred = np.full_like(val, train.mean(), dtype=float)
    return regression_metrics(val, pred)["rmse"]


def read_reference_status(reference_run: str | Path | None) -> dict[str, Any]:
    if not reference_run:
        return {"author_model_predictions_available": False, "strong_rmse": None, "batch9_skipped": True}
    report_path = Path(reference_run).parent.parent / "reports" / "attia_reference_reproduction.json"
    alt = Path("reports/attia_reference_reproduction.json")
    if not report_path.exists() and alt.exists():
        report_path = alt
    if not report_path.exists():
        return {"author_model_predictions_available": False, "strong_rmse": None, "batch9_skipped": True}
    report = json.loads(report_path.read_text())
    return {
        "author_model_predictions_available": bool(report.get("author_model_predictions_available")),
        "strong_rmse": report.get("validation_metrics", {}).get("rmse"),
        "batch9_skipped": bool(report.get("author_model_validation_metrics_unavailable_batch9_skipped", True)),
    }


def run_rediscovery(
    processed_dir: str | Path,
    reference_run: str | Path | None,
    out: str | Path,
    reports_dir: str | Path,
    agents: int = 2,
    iterations: int = 3,
    offline: bool = False,
    model: str | None = None,
    split_mode: str = "random",
    allow_protocol_features: bool = False,
    max_cycle: int = 100,
    locked_test: bool = False,
    seed: int = 0,
) -> dict[str, Any]:
    out = Path(out)
    cand_dir = out / "candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)
    metadata, cycles, labels, data_source = load_processed_or_synthetic(processed_dir, seed, max_cycle)
    train_ids, val_ids, test_ids = make_splits(labels, seed, split_mode)
    weak_rmse = weak_baseline_rmse(labels, train_ids, val_ids)
    ref = read_reference_status(reference_run)
    evaluator = HiddenEvaluator(
        metadata=metadata,
        cycle_summary=cycles,
        labels=labels,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        weak_rmse=weak_rmse,
        strong_rmse=ref["strong_rmse"],
        allow_protocol_features=allow_protocol_features,
        max_cycle=max_cycle,
    )

    leaderboard_rows: list[dict[str, Any]] = []
    event_path = out / "events.jsonl"
    agent_objs = [make_agent(f"agent_{i}", offline=offline, model=model) for i in range(agents)]
    for iteration in range(iterations):
        leaderboard_text = pd.DataFrame(leaderboard_rows).tail(10).to_string(index=False) if leaderboard_rows else "No prior candidates."
        prompt = rediscovery_prompt(max_cycle=max_cycle, leaderboard=leaderboard_text)
        for agent in agent_objs:
            response = agent.propose(prompt, iteration)
            candidate_path = cand_dir / f"{agent.agent_id}_iter_{iteration}.py"
            candidate_path.write_text(response.code)
            result = evaluator.evaluate_candidate(candidate_path)
            row = {
                "agent_id": agent.agent_id,
                "iteration": iteration,
                "candidate_path": str(candidate_path),
                "success": bool(result.get("success")),
            }
            if result.get("success"):
                row.update(result["metrics"])
            else:
                row["error"] = result.get("error")
            leaderboard_rows.append(row)
            append_event(event_path, {**row, "prompt": prompt, "response": response.response_text})

    leaderboard = pd.DataFrame(leaderboard_rows)
    leaderboard.to_csv(out / "leaderboard.csv", index=False)
    successes = leaderboard[leaderboard["success"] == True]  # noqa: E712
    best_metrics: dict[str, Any] = {}
    best_candidate = None
    overlap: list[str] = []
    if not successes.empty:
        best = successes.sort_values("rmse").iloc[0]
        best_candidate = str(best["candidate_path"])
        shutil.copyfile(best_candidate, out / "best_candidate.py")
        best_metrics = {k: best[k] for k in ["rmse", "mae", "r2", "spearman", "kendall", "pgr_author_model"] if k in best}
        overlap = posthoc_feature_overlap(Path(best_candidate).read_text())
        if locked_test:
            test_result = evaluator.evaluate_candidate(best_candidate, split="test")
            append_event(event_path, {"locked_test": test_result, "candidate_path": best_candidate})

    report = {
        "mode": "offline" if offline or not any(hasattr(a, "api_key") for a in agent_objs) else "llm_driven",
        "data_source": data_source,
        "split_mode": split_mode,
        "labels_hidden": ["validation", "test"],
        "batch9_status": "skipped_not_required",
        "weak_baseline_rmse": weak_rmse,
        "exact_author_model_rmse": ref["strong_rmse"],
        "author_model_predictions_available": ref["author_model_predictions_available"],
        "author_model_validation_metrics_unavailable_batch9_skipped": ref["batch9_skipped"],
        "best_candidate": best_candidate,
        "best_metrics": best_metrics,
        "posthoc_feature_overlap": overlap,
        "caveats": [
            "Batch 9 is not required for this rediscovery run.",
            "Battery-PGR against the author model is undefined unless exact author validation RMSE is available.",
            "Synthetic/demo data are used when processed local data files are absent.",
        ],
    }
    write_agent_reports(report, reports_dir)
    return report
