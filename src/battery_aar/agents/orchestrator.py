from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .attia_data_bridge import load_author_validation_metrics, load_batch9_holdout
from .evaluator import HiddenEvaluator, battery_pgr, evaluate_candidate_train_test, regression_metrics
from .llm_client import llm_startup_summary, make_agent
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
        rows_meta.append({"row_id": row_id, "cell_id": f"synthetic_{row_id}", "batch_id": f"b{row_id % 4}", "cc1": rng.choice([3.6, 4.0, 4.8, 5.6]), "label_source": "synthetic_demo"})
        rows_labels.append({"row_id": row_id, "y": max(100.0, lifetime)})
        for cycle in range(1, max_cycle + 1):
            rows_cycles.append({"row_id": row_id, "cycle_index": cycle, "discharge_capacity": q0 - fade * cycle + curve[cycle - 1], "charge_capacity": np.nan})
    return pd.DataFrame(rows_meta), pd.DataFrame(rows_cycles), pd.DataFrame(rows_labels)


def _load_dataset_card(base: Path) -> dict[str, Any]:
    card_path = base / "dataset_card.json"
    if not card_path.exists():
        return {}
    return json.loads(card_path.read_text())


def _processed_paths(base: Path) -> tuple[Path, Path, Path | None]:
    cell_metadata = base / "cell_metadata.csv"
    metadata = base / "metadata.csv"
    cycle = base / "cycle_summary.csv"
    labels = base / "labels.csv"
    if cell_metadata.exists() and cycle.exists():
        return cell_metadata, cycle, labels if labels.exists() else None
    if metadata.exists() and cycle.exists() and labels.exists():
        return metadata, cycle, labels
    return metadata, cycle, labels


def load_processed_or_synthetic(
    processed_dir: str | Path,
    seed: int,
    max_cycle: int,
    require_real_data: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, dict[str, Any], pd.DataFrame | None]:
    base = Path(processed_dir)
    if require_real_data and not ((base / "cell_metadata.csv").exists() and (base / "cycle_summary.csv").exists()):
        raise FileNotFoundError(
            f"Real processed data required but missing at {base}. "
            "Expected cell_metadata.csv, cycle_summary.csv, and labels/cycle_life."
        )
    meta_path, cycle_path, labels_path = _processed_paths(base)
    splits_path = base / "splits.csv"
    if meta_path.exists() and cycle_path.exists():
        metadata = pd.read_csv(meta_path)
        cycles = pd.read_csv(cycle_path)
        if labels_path is not None and labels_path.exists():
            labels = pd.read_csv(labels_path)
        elif "cycle_life" in metadata.columns:
            labels = metadata[["row_id", "cycle_life"]].rename(columns={"cycle_life": "y"}).copy()
        else:
            raise FileNotFoundError(f"Processed dataset at {base} has no labels.csv or cycle_life column")
        labels["y"] = pd.to_numeric(labels["y"], errors="coerce")
        labels = labels[np.isfinite(labels["y"])].copy()
        if labels.empty:
            raise ValueError(f"Processed dataset at {base} has no finite labeled rows")
        card = _load_dataset_card(base)
        splits = pd.read_csv(splits_path) if splits_path.exists() else None
        return metadata, cycles, labels, "processed_real", card, splits
    if require_real_data:
        raise FileNotFoundError(
            f"Real processed data required but missing at {base}. "
            "Expected cell_metadata.csv, cycle_summary.csv, and labels/cycle_life."
        )
    meta, cycles, labels = synthetic_dataset(seed=seed, max_cycle=max(120, max_cycle))
    return meta, cycles, labels, "synthetic_demo", {"label_source": "synthetic_demo"}, None


def make_splits(labels: pd.DataFrame, seed: int, split_mode: str = "random") -> tuple[list[int], list[int], list[int]]:
    rng = np.random.default_rng(seed)
    ids = labels["row_id"].to_numpy(int)
    rng.shuffle(ids)
    n = len(ids)
    if n == 0:
        return [], [], []
    n_train = max(1, int(0.6 * n))
    n_val = max(1, int(0.2 * n)) if n >= 2 else 0
    if n_train + n_val > n:
        n_train = max(1, n - n_val)
    return ids[:n_train].tolist(), ids[n_train : n_train + n_val].tolist(), ids[n_train + n_val :].tolist()


def make_splits_from_table(labels: pd.DataFrame, splits: pd.DataFrame | None, seed: int, split_mode: str = "random") -> tuple[list[int], list[int], list[int]]:
    if splits is None or splits.empty or not {"row_id", "split"}.issubset(splits.columns):
        return make_splits(labels, seed, split_mode)
    labeled = set(labels["row_id"].astype(int).tolist())
    split_rows = splits[splits["row_id"].isin(labeled)].copy()
    train = split_rows.loc[split_rows["split"] == "train", "row_id"].astype(int).tolist()
    val = split_rows.loc[split_rows["split"] == "val", "row_id"].astype(int).tolist()
    test = split_rows.loc[split_rows["split"] == "test", "row_id"].astype(int).tolist()
    if not train or not val:
        return make_splits(labels, seed, split_mode)
    if not test:
        assigned = set(train) | set(val)
        test = [int(row_id) for row_id in labels["row_id"] if int(row_id) not in assigned]
    return train, val, test


def weak_baseline_rmse(labels: pd.DataFrame, train_ids: list[int], val_ids: list[int]) -> float:
    train = labels[labels["row_id"].isin(train_ids)]["y"].to_numpy(float)
    val = labels[labels["row_id"].isin(val_ids)]["y"].to_numpy(float)
    pred = np.full_like(val, train.mean(), dtype=float)
    return regression_metrics(val, pred)["rmse"]


def read_reference_status(reference_run: str | Path | None) -> dict[str, Any]:
    if not reference_run:
        report_path = Path("reports/attia_reference_reproduction.json")
        if not report_path.exists():
            return {"author_model_predictions_available": False, "strong_rmse": None, "batch9_skipped": True}
    else:
        report_path = Path(reference_run).parent.parent / "reports" / "attia_reference_reproduction.json"
        if not report_path.exists():
            return {"author_model_predictions_available": False, "strong_rmse": None, "batch9_skipped": True}
    report = json.loads(report_path.read_text())
    nested = report.get("validation_metrics", {})
    strong_rmse = nested.get("rmse")
    if strong_rmse is None and isinstance(nested, dict):
        strong_rmse = (nested.get("early_prediction_vs_observed") or {}).get("rmse")
    return {
        "author_model_predictions_available": bool(report.get("author_model_predictions_available")),
        "strong_rmse": strong_rmse,
        "batch9_skipped": bool(report.get("author_model_validation_metrics_unavailable_batch9_skipped", True)),
    }


def _protocol_metrics(predictions: pd.DataFrame, metadata: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    merged = predictions.merge(metadata, on="row_id", how="left", suffixes=("", "_meta"))
    if "protocol_readable" in merged and merged["protocol_readable"].notna().any():
        group_cols = ["protocol_readable"]
    elif {"C1", "C2", "C3", "C4"}.issubset(merged.columns):
        group_cols = ["C1", "C2", "C3", "C4"]
    else:
        return pd.DataFrame(), {"protocol_level_rmse": None, "protocol_level_mae": None, "n_protocols": 0}
    grouped = (
        merged.groupby(group_cols, dropna=False)
        .agg(
            n_cells=("row_id", "count"),
            observed_cycle_life_mean=("y", "mean"),
            predicted_cycle_life_mean=("y_pred", "mean"),
        )
        .reset_index()
    )
    if grouped.empty:
        return grouped, {"protocol_level_rmse": None, "protocol_level_mae": None, "n_protocols": 0}
    metrics = regression_metrics(grouped["observed_cycle_life_mean"].to_numpy(float), grouped["predicted_cycle_life_mean"].to_numpy(float))
    grouped["prediction_error"] = grouped["predicted_cycle_life_mean"] - grouped["observed_cycle_life_mean"]
    return grouped, {"protocol_level_rmse": metrics["rmse"], "protocol_level_mae": metrics["mae"], "n_protocols": int(len(grouped))}


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
    require_real_data: bool = False,
    final_batch9_validation: bool = False,
    battery_fast_charging_root: str | Path | None = None,
    batch9_path: str | Path | None = None,
) -> dict[str, Any]:
    out = Path(out)
    cand_dir = out / "candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)
    metadata, cycles, labels, data_source, dataset_card, split_table = load_processed_or_synthetic(processed_dir, seed, max_cycle, require_real_data=require_real_data)
    train_ids, val_ids, test_ids = make_splits_from_table(labels, split_table, seed, split_mode)
    weak_rmse = weak_baseline_rmse(labels, train_ids, val_ids)
    ref = read_reference_status(reference_run)
    author_validation = load_author_validation_metrics(reference_run)
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
                "candidate_syntax_status": result.get("candidate_syntax_status"),
                "stdout": result.get("stdout"),
                "stderr": result.get("stderr"),
            }
            if result.get("success"):
                row.update(result["metrics"])
            else:
                row["error"] = result.get("error")
                row["error_type"] = result.get("error_type")
                row["failure_reason"] = result.get("failure_reason") or result.get("error")
                row["traceback"] = result.get("traceback")
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
    failure_summary: list[dict[str, Any]] = []
    if not leaderboard.empty and "success" in leaderboard.columns:
        failures = leaderboard[leaderboard["success"] == False]  # noqa: E712
        if not failures.empty:
            grouped = failures.groupby(["error_type", "failure_reason"], dropna=False).size().reset_index(name="count")
            grouped = grouped.sort_values("count", ascending=False)
            failure_summary = grouped.head(10).to_dict(orient="records")

    final_batch9: dict[str, Any] | None = None
    final_batch9_metrics: dict[str, Any] | None = None
    if final_batch9_validation:
        final_batch9 = {"status": "not_run"}
        try:
            if best_candidate is None:
                raise RuntimeError("No successful candidate is available for locked Batch 9 validation")
            if battery_fast_charging_root is None and batch9_path is None:
                raise ValueError("--final-batch9-validation requires --battery-fast-charging-root or --batch9-path")
            holdout_meta, holdout_cycles, holdout_labels = load_batch9_holdout(
                battery_fast_charging_root=battery_fast_charging_root,
                batch9_path=batch9_path,
                first_n_cycles=max_cycle,
            )
            row_offset = int(pd.to_numeric(metadata["row_id"], errors="coerce").max()) + 1
            holdout_meta = holdout_meta.copy()
            holdout_cycles = holdout_cycles.copy()
            holdout_labels = holdout_labels.copy()
            holdout_meta["row_id"] = holdout_meta["row_id"].astype(int) + row_offset
            holdout_cycles["row_id"] = holdout_cycles["row_id"].astype(int) + row_offset
            holdout_labels["row_id"] = holdout_labels["row_id"].astype(int) + row_offset
            final_result = evaluate_candidate_train_test(
                best_candidate,
                metadata,
                cycles,
                labels,
                holdout_meta,
                holdout_cycles,
                holdout_labels,
                weak_rmse=weak_rmse,
                strong_rmse=author_validation.get("author_model_batch9_rmse"),
                allow_protocol_features=allow_protocol_features,
                max_cycle=max_cycle,
                return_predictions=True,
            )
            if not final_result.get("success"):
                raise RuntimeError(final_result.get("failure_reason") or final_result.get("error") or "Batch 9 candidate evaluation failed")
            predictions = final_result["predictions"].merge(holdout_meta, on="row_id", how="left")
            predictions.to_csv(out / "final_batch9_predictions.csv", index=False)
            protocol_df, protocol_metric_subset = _protocol_metrics(final_result["predictions"], holdout_meta)
            protocol_df.to_csv(out / "final_batch9_protocol_metrics.csv", index=False)
            final_batch9_metrics = dict(final_result["metrics"])
            final_batch9_metrics.update(protocol_metric_subset)
            final_batch9_metrics["author_model_batch9_rmse"] = author_validation.get("author_model_batch9_rmse")
            final_batch9_metrics["author_model_batch9_mae"] = author_validation.get("author_model_batch9_mae")
            final_batch9_metrics["battery_pgr_author_model_batch9"] = battery_pgr(
                weak_rmse,
                final_batch9_metrics.get("rmse"),
                author_validation.get("author_model_batch9_rmse"),
            )
            (out / "final_batch9_metrics.json").write_text(json.dumps(final_batch9_metrics, indent=2, default=str, sort_keys=True) + "\n")
            final_batch9 = {
                "status": "ok",
                "n_cells": int(len(holdout_labels)),
                "predictions_path": str(out / "final_batch9_predictions.csv"),
                "metrics_path": str(out / "final_batch9_metrics.json"),
                "protocol_metrics_path": str(out / "final_batch9_protocol_metrics.csv"),
            }
            append_event(event_path, {"final_batch9_validation": final_batch9, "metrics": final_batch9_metrics, "candidate_path": best_candidate})
        except Exception as exc:
            final_batch9 = {"status": "failed", "error": str(exc)}
            append_event(event_path, {"final_batch9_validation": final_batch9, "candidate_path": best_candidate})

    report = {
        "mode": "offline" if offline or not any(hasattr(a, "api_key") for a in agent_objs) else "llm_driven",
        "llm_client": llm_startup_summary(model=model),
        "data_source": data_source,
        "real_data_used": data_source == "processed_real",
        "synthetic_fallback_used": data_source == "synthetic_demo",
        "require_real_data": require_real_data,
        "label_source": dataset_card.get("label_source") or (metadata["label_source"].dropna().iloc[0] if "label_source" in metadata and not metadata["label_source"].dropna().empty else None),
        "split_mode": split_mode,
        "labels_hidden": ["validation", "test"],
        "batch9_status": "skipped_not_required",
        "weak_baseline_rmse": weak_rmse,
        "exact_author_model_rmse": author_validation.get("author_model_batch9_rmse") or ref["strong_rmse"],
        "author_literature_batch9_metrics": author_validation,
        "author_model_predictions_available": ref["author_model_predictions_available"],
        "author_model_validation_metrics_unavailable_batch9_skipped": ref["batch9_skipped"],
        "best_candidate": best_candidate,
        "best_metrics": best_metrics,
        "final_batch9_validation": final_batch9,
        "final_batch9_metrics": final_batch9_metrics,
        "candidate_failures": failure_summary,
        "posthoc_feature_overlap": overlap,
        "caveats": [
            "Batch 9 is not required for this rediscovery run.",
            "Battery-PGR against the author model is undefined unless exact author validation RMSE is available.",
            "Synthetic/demo data are used only when processed local data files are absent and --require-real-data is not set.",
            "Surrogate-label search performance on OED/CLO batches is distinct from locked Batch 9 final validation.",
        ],
    }
    write_agent_reports(report, reports_dir)
    return report
