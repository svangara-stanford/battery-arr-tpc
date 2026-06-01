from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .attia_data_bridge import load_author_validation_metrics, load_batch9_holdout
from .baseline_candidates import author_inspired_baseline_candidates
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


def _coerce_row_id_series(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="raise").astype(int)


def _is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and np.isnan(value):
        return False
    text = str(value).strip()
    return bool(text) and text.lower() != "nan"


def _protocol_key_from_row(row: pd.Series) -> str | None:
    if "protocol_readable" in row.index and _is_present(row.get("protocol_readable")):
        return str(row["protocol_readable"]).strip()
    current_cols = ["C1", "C2", "C3", "C4"]
    if all(col in row.index for col in current_cols):
        values = pd.to_numeric(pd.Series([row[col] for col in current_cols]), errors="coerce").to_numpy(float)
        if np.isfinite(values).all():
            return "-".join(f"{value:.6g}" for value in values)
    return None


def _choose_validation_groups(
    unique_groups: list[str],
    validation_fraction: float,
    split_seed: int,
    validation_group_id: str | None = None,
) -> set[str]:
    if len(unique_groups) < 2:
        raise ValueError("At least two groups are required to create a train/validation split")
    if validation_group_id:
        if validation_group_id not in unique_groups:
            raise ValueError(f"Requested validation group {validation_group_id!r} was not found")
        return {validation_group_id}
    fraction = float(validation_fraction)
    if not 0 < fraction < 1:
        raise ValueError("--validation-fraction must be between 0 and 1")
    n_val = max(1, int(round(len(unique_groups) * fraction)))
    n_val = min(n_val, len(unique_groups) - 1)
    rng = np.random.default_rng(split_seed)
    groups = np.array(sorted(unique_groups), dtype=object)
    rng.shuffle(groups)
    return set(str(group) for group in groups[:n_val])


def make_splits(labels: pd.DataFrame, seed: int, split_mode: str = "random", validation_fraction: float = 0.25) -> tuple[list[int], list[int], list[int]]:
    rng = np.random.default_rng(seed)
    ids = _coerce_row_id_series(labels["row_id"]).to_numpy(int)
    rng.shuffle(ids)
    n = len(ids)
    if n == 0:
        return [], [], []
    if n < 2:
        return ids.tolist(), [], []
    n_val = max(1, int(round(n * validation_fraction)))
    n_val = min(n_val, n - 1)
    val = set(int(row_id) for row_id in ids[:n_val])
    train = [int(row_id) for row_id in ids if int(row_id) not in val]
    return train, sorted(val), []


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


def make_search_split(
    metadata: pd.DataFrame,
    labels: pd.DataFrame,
    split_mode: str = "random",
    validation_fraction: float = 0.25,
    split_seed: int = 0,
    validation_batch_id: str | None = None,
) -> tuple[list[int], list[int], list[int], dict[str, Any], pd.DataFrame]:
    """Create the surrogate-search split at cell level.

    The returned assignments are intentionally cell-level. Candidate-facing
    cycle rows inherit the split through row_id inside HiddenEvaluator.
    """
    if "row_id" not in metadata.columns or "row_id" not in labels.columns:
        raise ValueError("metadata and labels must both contain row_id")
    labeled_ids = set(_coerce_row_id_series(labels["row_id"]).tolist())
    meta = metadata.copy()
    meta["row_id"] = _coerce_row_id_series(meta["row_id"])
    split_meta = meta[meta["row_id"].isin(labeled_ids)].copy()
    if split_meta.empty:
        raise ValueError("No labeled metadata rows are available for splitting")
    if split_meta["row_id"].duplicated().any():
        dupes = split_meta.loc[split_meta["row_id"].duplicated(), "row_id"].tolist()
        raise ValueError(f"Duplicate row_id values in metadata: {dupes[:5]}")
    if "cell_id" not in split_meta.columns:
        split_meta["cell_id"] = split_meta["row_id"].map(lambda value: f"row_{int(value)}")
    if "batch_id" in split_meta.columns:
        batch_ids = split_meta["batch_id"].dropna().astype(str)
        if batch_ids.str.contains("2019-01-24_batch9", regex=False).any():
            raise ValueError("Batch 9 must not be included in the surrogate search dataset")

    mode = "batch" if split_mode == "leave_one_batch_out" else split_mode
    if mode == "random":
        split_meta["group_key"] = split_meta["row_id"].map(lambda value: f"row:{int(value)}")
        group_type = "random_cell"
        unique_groups = split_meta["group_key"].astype(str).tolist()
        validation_groups = _choose_validation_groups(unique_groups, validation_fraction, split_seed)
    elif mode == "protocol":
        split_meta["group_key"] = split_meta.apply(_protocol_key_from_row, axis=1)
        if split_meta["group_key"].isna().any():
            missing = split_meta.loc[split_meta["group_key"].isna(), "row_id"].tolist()
            raise ValueError(f"Protocol split requires protocol_readable or finite C1/C2/C3/C4 for every row; missing row_ids={missing[:10]}")
        group_type = "protocol"
        unique_groups = sorted(split_meta["group_key"].astype(str).unique().tolist())
        validation_groups = _choose_validation_groups(unique_groups, validation_fraction, split_seed)
    elif mode == "batch":
        if "batch_id" not in split_meta.columns:
            raise ValueError("Batch split requires a batch_id column")
        if split_meta["batch_id"].isna().any():
            raise ValueError("Batch split requires non-missing batch_id values")
        split_meta["group_key"] = split_meta["batch_id"].astype(str)
        group_type = "batch"
        unique_groups = sorted(split_meta["group_key"].unique().tolist())
        validation_groups = _choose_validation_groups(unique_groups, validation_fraction, split_seed, validation_batch_id)
    else:
        raise ValueError(f"Unsupported split mode: {split_mode}")

    split_meta["split"] = np.where(split_meta["group_key"].astype(str).isin(validation_groups), "validation", "train")
    train_ids = split_meta.loc[split_meta["split"] == "train", "row_id"].astype(int).tolist()
    val_ids = split_meta.loc[split_meta["split"] == "validation", "row_id"].astype(int).tolist()
    if not train_ids or not val_ids:
        raise ValueError(f"Split mode {split_mode!r} produced train={len(train_ids)} validation={len(val_ids)} rows")

    train_set = set(train_ids)
    val_set = set(val_ids)
    assert train_set.isdisjoint(val_set), "train and validation row_id sets overlap"
    train_groups = set(split_meta.loc[split_meta["split"] == "train", "group_key"].astype(str))
    val_groups = set(split_meta.loc[split_meta["split"] == "validation", "group_key"].astype(str))
    if mode in {"protocol", "batch"}:
        assert train_groups.isdisjoint(val_groups), f"{mode} groups overlap between train and validation"

    columns = ["row_id", "cell_id", "split", "group_key"]
    for col in ["batch_id", "protocol_readable", "C1", "C2", "C3", "C4"]:
        if col in split_meta.columns:
            columns.append(col)
    assignments = split_meta[columns].copy()
    assignments.insert(3, "split_mode", mode)

    manifest = {
        "split_mode": mode,
        "requested_split_mode": split_mode,
        "validation_fraction": float(validation_fraction),
        "split_seed": int(split_seed),
        "validation_batch_id": validation_batch_id,
        "group_type": group_type,
        "n_labeled_cells": int(len(split_meta)),
        "n_train_cells": int(len(train_ids)),
        "n_validation_cells": int(len(val_ids)),
        "n_train_groups": int(len(train_groups)),
        "n_validation_groups": int(len(val_groups)),
        "heldout_groups": sorted(val_groups),
        "heldout_protocols": sorted(val_groups) if mode == "protocol" else [],
        "heldout_batch_ids": sorted(val_groups) if mode == "batch" else [],
        "anti_leakage": {
            "train_validation_row_ids_disjoint": train_set.isdisjoint(val_set),
            "train_validation_groups_disjoint": train_groups.isdisjoint(val_groups),
            "validation_labels_passed_to_fit": False,
            "batch9_in_surrogate_search": False,
        },
    }
    return train_ids, val_ids, [], manifest, assignments


def weak_baseline_rmse(labels: pd.DataFrame, train_ids: list[int], val_ids: list[int]) -> float:
    train = labels[labels["row_id"].isin(train_ids)]["y"].to_numpy(float)
    val = labels[labels["row_id"].isin(val_ids)]["y"].to_numpy(float)
    pred = np.full_like(val, train.mean(), dtype=float)
    return regression_metrics(val, pred)["rmse"]


def weak_baseline_rmse_against(train_labels: pd.DataFrame, eval_labels: pd.DataFrame) -> float:
    train = pd.to_numeric(train_labels["y"], errors="coerce").to_numpy(float)
    target = pd.to_numeric(eval_labels["y"], errors="coerce").to_numpy(float)
    train = train[np.isfinite(train)]
    target = target[np.isfinite(target)]
    if train.size == 0 or target.size == 0:
        return float("nan")
    pred = np.full_like(target, float(np.mean(train)), dtype=float)
    return regression_metrics(target, pred)["rmse"]


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


def _safe_stem(path: str | Path) -> str:
    text = Path(path).stem
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)


def _sha256_file(path: str | Path) -> str | None:
    source = Path(path)
    if not source.exists():
        return None
    return hashlib.sha256(source.read_bytes()).hexdigest()


def _candidate_model_class(code: str) -> str:
    lowered = code.lower()
    for label, token in [
        ("GradientBoostingRegressor", "gradientboostingregressor"),
        ("RandomForestRegressor", "randomforestregressor"),
        ("ElasticNet", "elasticnet"),
        ("Ridge", "ridge"),
        ("LinearRegression", "linearregression"),
    ]:
        if token in lowered:
            return label
    if "candidatemodel" in lowered:
        return "CandidateModel"
    return "unknown"


def _candidate_feature_families(code: str) -> list[str]:
    families = set(posthoc_feature_overlap(code))
    lowered = code.lower()
    if "build_all_battery_features" in lowered or "battery_lifetime_features" in lowered:
        families.update(
            {
                "early discharge capacity",
                "late-cycle capacity slope",
                "difference between cycle 10 and cycle N curves",
                "log-transformed curve statistics",
            }
        )
    if "protocol" in lowered or "c1" in lowered or "cc1" in lowered:
        families.add("protocol current features")
    return sorted(families)


def _write_candidate_feature_summary(
    leaderboard: pd.DataFrame,
    out: Path,
    locked_batch9_by_candidate: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    locked_batch9_by_candidate = locked_batch9_by_candidate or {}
    rows: list[dict[str, Any]] = []
    if leaderboard.empty or "candidate_path" not in leaderboard.columns:
        summary = pd.DataFrame()
        summary.to_csv(out / "candidate_feature_summary.csv", index=False)
        return summary
    for _, row in leaderboard.iterrows():
        path = Path(str(row.get("candidate_path")))
        code = path.read_text() if path.exists() else ""
        batch9_metrics = locked_batch9_by_candidate.get(str(path), {})
        rows.append(
            {
                "candidate_path": str(path),
                "agent_id": row.get("agent_id"),
                "iteration": row.get("iteration"),
                "success": bool(row.get("success")),
                "model_class": _candidate_model_class(code),
                "used_toolbox": "build_all_battery_features" in code or "battery_lifetime_features" in code,
                "feature_families": "; ".join(_candidate_feature_families(code)),
                "surrogate_rmse": row.get("rmse"),
                "locked_batch9_rmse": batch9_metrics.get("rmse"),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "candidate_feature_summary.csv", index=False)
    return summary


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
    validation_fraction: float = 0.25,
    split_seed: int = 0,
    validation_batch_id: str | None = None,
    allow_protocol_features: bool = False,
    max_cycle: int = 100,
    locked_test: bool = False,
    seed: int = 0,
    require_real_data: bool = False,
    final_batch9_validation: bool = False,
    battery_fast_charging_root: str | Path | None = None,
    batch9_path: str | Path | None = None,
    seed_with_author_inspired_baselines: bool = False,
    final_batch9_top_k: int = 0,
    emit_artifact_trace: bool = False,
) -> dict[str, Any]:
    out = Path(out)
    run_id = out.name or "open_battery_agents_run"
    cand_dir = out / "candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)
    artifact_store = None
    trace_logger = None
    artifact_classes: dict[str, Any] = {}
    run_manifest_artifact_id: str | None = None
    dataset_profile_artifact_id: str | None = None
    split_artifact_id: str | None = None
    candidate_artifact_ids_by_path: dict[str, str] = {}
    if emit_artifact_trace:
        from battery_aar.workflows.artifacts import ArtifactStore, build_dataset_profile_artifact, build_split_artifact
        from battery_aar.workflows.schemas import AgentRole, CandidateSpec, EvaluationReport, ExperimentState, RunManifest
        from battery_aar.workflows.trace import TraceLogger

        artifact_store = ArtifactStore(out, run_id=run_id)
        trace_logger = TraceLogger(artifact_store.artifact_dir, run_id=run_id)
        artifact_classes = {
            "AgentRole": AgentRole,
            "CandidateSpec": CandidateSpec,
            "EvaluationReport": EvaluationReport,
            "ExperimentState": ExperimentState,
            "RunManifest": RunManifest,
            "build_dataset_profile_artifact": build_dataset_profile_artifact,
            "build_split_artifact": build_split_artifact,
        }
        run_manifest = RunManifest(
            run_id=run_id,
            output_dir=str(out),
            human_readable_summary=f"Open Battery Agents rediscovery run {run_id}.",
            config={
                "processed_dir": str(processed_dir),
                "reference_run": str(reference_run) if reference_run is not None else None,
                "agents": agents,
                "iterations": iterations,
                "offline": offline,
                "model": model,
                "split_mode": split_mode,
                "validation_fraction": validation_fraction,
                "split_seed": split_seed,
                "validation_batch_id": validation_batch_id,
                "allow_protocol_features": allow_protocol_features,
                "max_cycle": max_cycle,
                "locked_test": locked_test,
                "seed": seed,
                "require_real_data": require_real_data,
                "final_batch9_validation": final_batch9_validation,
                "final_batch9_top_k": final_batch9_top_k,
                "seed_with_author_inspired_baselines": seed_with_author_inspired_baselines,
            },
        )
        artifact_store.write_artifact(run_manifest)
        run_manifest_artifact_id = run_manifest.artifact_id
        trace_logger.log_event(
            event_type="run_started",
            iteration=None,
            agent_role=AgentRole.ORCHESTRATOR,
            output_artifact_ids=[run_manifest.artifact_id],
            success=True,
        )
    metadata, cycles, labels, data_source, dataset_card, _split_table = load_processed_or_synthetic(processed_dir, seed, max_cycle, require_real_data=require_real_data)
    label_source = dataset_card.get("label_source") or (metadata["label_source"].dropna().iloc[0] if "label_source" in metadata and not metadata["label_source"].dropna().empty else None)
    if emit_artifact_trace and artifact_store is not None:
        dataset_profile = artifact_classes["build_dataset_profile_artifact"](
            run_id=run_id,
            metadata=metadata,
            cycle_summary=cycles,
            labels=labels,
            data_source=data_source,
            label_source=label_source,
            parent_artifact_ids=[run_manifest_artifact_id] if run_manifest_artifact_id else [],
        )
        artifact_store.write_artifact(dataset_profile)
        dataset_profile_artifact_id = dataset_profile.artifact_id
        trace_logger.log_event(
            event_type="dataset_profiled",
            agent_role=artifact_classes["AgentRole"].DATASET_PROFILER,
            input_artifact_ids=[run_manifest_artifact_id] if run_manifest_artifact_id else [],
            output_artifact_ids=[dataset_profile.artifact_id],
            success=True,
        )
    train_ids, val_ids, test_ids, split_manifest, split_assignments = make_search_split(
        metadata=metadata,
        labels=labels,
        split_mode=split_mode,
        validation_fraction=validation_fraction,
        split_seed=split_seed,
        validation_batch_id=validation_batch_id,
    )
    split_assignments.to_csv(out / "split_assignments.csv", index=False)
    (out / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2, default=str, sort_keys=True) + "\n")
    if emit_artifact_trace and artifact_store is not None:
        split_artifact = artifact_classes["build_split_artifact"](
            run_id=run_id,
            split_manifest=split_manifest,
            assignments_path=out / "split_assignments.csv",
            parent_artifact_ids=[artifact_id for artifact_id in [dataset_profile_artifact_id] if artifact_id],
        )
        artifact_store.write_artifact(split_artifact)
        split_artifact_id = split_artifact.artifact_id
        trace_logger.log_event(
            event_type="split_created",
            agent_role=artifact_classes["AgentRole"].SPLIT_MANAGER,
            input_artifact_ids=[dataset_profile_artifact_id] if dataset_profile_artifact_id else [],
            output_artifact_ids=[split_artifact.artifact_id],
            success=True,
        )
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

    def emit_candidate_artifacts(
        *,
        agent_id: str,
        iteration: int,
        candidate_path: Path,
        result: dict[str, Any],
        candidate_name: str | None = None,
        agent_role: Any | None = None,
        response_summary: str | None = None,
    ) -> tuple[str | None, str | None]:
        if not emit_artifact_trace or artifact_store is None:
            return None, None
        AgentRole = artifact_classes["AgentRole"]
        CandidateSpec = artifact_classes["CandidateSpec"]
        role = agent_role or AgentRole.LLM_CANDIDATE
        code = candidate_path.read_text() if candidate_path.exists() else ""
        candidate_id = f"{agent_id}_iter_{iteration}_{candidate_name or candidate_path.stem}"
        spec = CandidateSpec(
            run_id=run_id,
            parent_artifact_ids=[artifact_id for artifact_id in [split_artifact_id] if artifact_id],
            human_readable_summary=f"Candidate specification for {candidate_name or candidate_path.stem}.",
            candidate_id=candidate_id,
            agent_id=agent_id,
            agent_role=role,
            iteration=iteration,
            candidate_path=str(candidate_path),
            candidate_name=candidate_name,
            code_sha256=_sha256_file(candidate_path),
            uses_toolbox="build_all_battery_features" in code or "battery_lifetime_features" in code,
            declared_dependencies=[dep for dep in ["numpy", "pandas", "scipy", "sklearn", "battery_aar.features"] if dep in code],
        )
        artifact_store.write_artifact(spec)
        candidate_artifact_ids_by_path[str(candidate_path)] = spec.artifact_id
        eval_id = emit_evaluation_artifact(
            candidate_path=candidate_path,
            result=result,
            candidate_id=candidate_id,
            agent_id=agent_id,
            iteration=iteration,
            candidate_name=candidate_name,
            parent_artifact_ids=[spec.artifact_id],
            locked_batch9_validation_run=False,
        )
        trace_logger.log_event(
            event_type="candidate_evaluated",
            iteration=iteration,
            agent_role=role,
            input_artifact_ids=[split_artifact_id] if split_artifact_id else [],
            output_artifact_ids=[artifact_id for artifact_id in [spec.artifact_id, eval_id] if artifact_id],
            success=bool(result.get("success")),
            error_type=result.get("error_type"),
            error_message=result.get("failure_reason") or result.get("error"),
        )
        if response_summary is not None:
            trace_logger.log_agent_message(
                event_type="candidate_response",
                iteration=iteration,
                agent_role=role,
                agent_id=agent_id,
                input_artifact_ids=[split_artifact_id] if split_artifact_id else [],
                output_artifact_ids=[spec.artifact_id],
                success=True,
                message_summary=response_summary[:1000],
            )
        return spec.artifact_id, eval_id

    def emit_evaluation_artifact(
        *,
        candidate_path: str | Path,
        result: dict[str, Any],
        candidate_id: str,
        agent_id: str | None,
        iteration: int | None,
        candidate_name: str | None = None,
        parent_artifact_ids: list[str] | None = None,
        locked_batch9_validation_run: bool = False,
    ) -> str | None:
        if not emit_artifact_trace or artifact_store is None:
            return None
        EvaluationReport = artifact_classes["EvaluationReport"]
        metrics = result.get("metrics") or {}
        pgr_value = metrics.get("pgr_author_model")
        if pgr_value is None:
            pgr_value = metrics.get("battery_pgr_author_model_batch9")
        report = EvaluationReport(
            run_id=run_id,
            parent_artifact_ids=parent_artifact_ids or [],
            human_readable_summary=f"Evaluation report for {candidate_name or Path(candidate_path).stem}.",
            candidate_id=candidate_id,
            candidate_path=str(candidate_path),
            agent_id=agent_id,
            candidate_name=candidate_name,
            iteration=iteration,
            split_mode=split_mode,
            locked_batch9_validation_run=locked_batch9_validation_run,
            success=bool(result.get("success")),
            rmse=metrics.get("rmse"),
            mae=metrics.get("mae"),
            r2=metrics.get("r2"),
            spearman=metrics.get("spearman"),
            kendall=metrics.get("kendall"),
            pgr=pgr_value,
            failure_reason=result.get("failure_reason") or result.get("error"),
            traceback=result.get("traceback"),
            stdout_excerpt=(result.get("stdout") or "")[:2000],
            stderr_excerpt=(result.get("stderr") or "")[:2000],
            extra_metrics={k: v for k, v in metrics.items() if k not in {"rmse", "mae", "r2", "spearman", "kendall", "pgr_author_model"}},
        )
        artifact_store.write_artifact(report)
        return report.artifact_id

    leaderboard_rows: list[dict[str, Any]] = []
    event_path = out / "events.jsonl"
    if seed_with_author_inspired_baselines:
        for baseline in author_inspired_baseline_candidates():
            candidate_path = cand_dir / f"{baseline.name}.py"
            candidate_path.write_text(baseline.code)
            result = evaluator.evaluate_candidate(candidate_path)
            row = {
                "agent_id": "author_inspired_baseline",
                "iteration": -1,
                "candidate_name": baseline.name,
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
            append_event(event_path, {**row, "baseline_candidate": baseline.name})
            emit_candidate_artifacts(
                agent_id="author_inspired_baseline",
                iteration=-1,
                candidate_path=candidate_path,
                result=result,
                candidate_name=baseline.name,
                agent_role=artifact_classes["AgentRole"].BASELINE if emit_artifact_trace else None,
            )

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
            emit_candidate_artifacts(
                agent_id=agent.agent_id,
                iteration=iteration,
                candidate_path=candidate_path,
                result=result,
                candidate_name=None,
                agent_role=artifact_classes["AgentRole"].LLM_CANDIDATE if emit_artifact_trace else None,
                response_summary=response.response_text,
            )

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

    def candidate_name_for_path(candidate_path: str | Path) -> str | None:
        if leaderboard.empty or "candidate_path" not in leaderboard.columns:
            return Path(candidate_path).stem
        matches = leaderboard[leaderboard["candidate_path"].astype(str) == str(candidate_path)]
        if not matches.empty and "candidate_name" in matches.columns:
            value = matches.iloc[0].get("candidate_name")
            if pd.notna(value):
                return str(value)
        return Path(candidate_path).stem

    locked_batch9_by_candidate: dict[str, dict[str, Any]] = {}
    locked_batch9_predictions_by_candidate: dict[str, pd.DataFrame] = {}
    final_batch9: dict[str, Any] | None = None
    final_batch9_metrics: dict[str, Any] | None = None
    batch9_status = "skipped_not_required"
    final_batch9_topk: dict[str, Any] | None = None
    if final_batch9_validation or final_batch9_top_k > 0:
        final_batch9 = {"status": "not_run"}
        try:
            if best_candidate is None:
                raise RuntimeError("No successful candidate is available for locked Batch 9 validation")
            if battery_fast_charging_root is None and batch9_path is None:
                raise ValueError("locked Batch 9 validation requires --battery-fast-charging-root or --batch9-path")
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
            batch9_weak_rmse = weak_baseline_rmse_against(labels, holdout_labels)

            def evaluate_locked_candidate(candidate_path: str | Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
                result = evaluate_candidate_train_test(
                    candidate_path,
                    metadata,
                    cycles,
                    labels,
                    holdout_meta,
                    holdout_cycles,
                    holdout_labels,
                    weak_rmse=batch9_weak_rmse,
                    strong_rmse=author_validation.get("author_model_batch9_rmse"),
                    allow_protocol_features=allow_protocol_features,
                    max_cycle=max_cycle,
                    return_predictions=True,
                )
                if not result.get("success"):
                    raise RuntimeError(result.get("failure_reason") or result.get("error") or "Batch 9 candidate evaluation failed")
                protocol_df, protocol_metric_subset = _protocol_metrics(result["predictions"], holdout_meta)
                metrics = dict(result["metrics"])
                metrics.update(protocol_metric_subset)
                metrics["batch9_weak_baseline_rmse"] = batch9_weak_rmse
                metrics["author_model_batch9_rmse"] = author_validation.get("author_model_batch9_rmse")
                metrics["author_model_batch9_mae"] = author_validation.get("author_model_batch9_mae")
                metrics["battery_pgr_author_model_batch9"] = battery_pgr(
                    batch9_weak_rmse,
                    metrics.get("rmse"),
                    author_validation.get("author_model_batch9_rmse"),
                )
                candidate_path_text = str(candidate_path)
                candidate_name = candidate_name_for_path(candidate_path_text)
                parent_ids = [candidate_artifact_ids_by_path[candidate_path_text]] if candidate_path_text in candidate_artifact_ids_by_path else []
                eval_id = emit_evaluation_artifact(
                    candidate_path=candidate_path_text,
                    result={"success": True, "metrics": metrics},
                    candidate_id=f"locked_batch9_{_safe_stem(candidate_path_text)}",
                    agent_id=None,
                    iteration=None,
                    candidate_name=candidate_name,
                    parent_artifact_ids=parent_ids,
                    locked_batch9_validation_run=True,
                )
                if trace_logger is not None:
                    trace_logger.log_event(
                        event_type="locked_batch9_candidate_evaluated",
                        iteration=None,
                        agent_role=artifact_classes["AgentRole"].EVALUATOR,
                        input_artifact_ids=parent_ids,
                        output_artifact_ids=[eval_id] if eval_id else [],
                        success=True,
                    )
                return metrics, result["predictions"].merge(holdout_meta, on="row_id", how="left"), protocol_df

            if final_batch9_validation:
                final_batch9_metrics, predictions, protocol_df = evaluate_locked_candidate(best_candidate)
                locked_batch9_by_candidate[str(best_candidate)] = final_batch9_metrics
                locked_batch9_predictions_by_candidate[str(best_candidate)] = predictions
                predictions.to_csv(out / "final_batch9_predictions.csv", index=False)
                protocol_df.to_csv(out / "final_batch9_protocol_metrics.csv", index=False)
                (out / "final_batch9_metrics.json").write_text(json.dumps(final_batch9_metrics, indent=2, default=str, sort_keys=True) + "\n")
                final_batch9 = {
                    "status": "ok",
                    "n_cells": int(len(holdout_labels)),
                    "predictions_path": str(out / "final_batch9_predictions.csv"),
                    "metrics_path": str(out / "final_batch9_metrics.json"),
                    "protocol_metrics_path": str(out / "final_batch9_protocol_metrics.csv"),
                }
                batch9_status = "locked_validation_ok"
                append_event(event_path, {"final_batch9_validation": final_batch9, "metrics": final_batch9_metrics, "candidate_path": best_candidate})

            if final_batch9_top_k > 0:
                topk_dir = out / "final_batch9_topk_predictions"
                topk_dir.mkdir(parents=True, exist_ok=True)
                topk_rows: list[dict[str, Any]] = []
                top_successes = successes.sort_values("rmse").head(int(final_batch9_top_k))
                for rank, (_, candidate_row) in enumerate(top_successes.iterrows(), start=1):
                    candidate_path = str(candidate_row["candidate_path"])
                    if candidate_path in locked_batch9_by_candidate:
                        metrics = locked_batch9_by_candidate[candidate_path]
                        predictions = locked_batch9_predictions_by_candidate[candidate_path]
                    else:
                        metrics, predictions, _protocol_df = evaluate_locked_candidate(candidate_path)
                        locked_batch9_by_candidate[candidate_path] = metrics
                        locked_batch9_predictions_by_candidate[candidate_path] = predictions
                    predictions.to_csv(topk_dir / f"rank_{rank:02d}_{_safe_stem(candidate_path)}.csv", index=False)
                    topk_rows.append(
                        {
                            "rank_by_surrogate_rmse": rank,
                            "candidate_path": candidate_path,
                            "candidate_name": candidate_row.get("candidate_name"),
                            "surrogate_rmse": candidate_row.get("rmse"),
                            "batch9_rmse": metrics.get("rmse"),
                            "batch9_mae": metrics.get("mae"),
                            "batch9_pgr": metrics.get("battery_pgr_author_model_batch9"),
                        }
                    )
                topk_metrics = pd.DataFrame(topk_rows)
                topk_metrics.to_csv(out / "final_batch9_topk_metrics.csv", index=False)
                final_batch9_topk = {
                    "status": "ok",
                    "k": int(final_batch9_top_k),
                    "metrics_path": str(out / "final_batch9_topk_metrics.csv"),
                    "predictions_dir": str(topk_dir),
                }
                append_event(event_path, {"final_batch9_topk": final_batch9_topk})
            if not final_batch9_validation:
                final_batch9 = None
        except Exception as exc:
            final_batch9 = {"status": "failed", "error": str(exc)}
            batch9_status = "locked_validation_failed"
            append_event(event_path, {"final_batch9_validation": final_batch9, "candidate_path": best_candidate})

    candidate_feature_summary = _write_candidate_feature_summary(
        leaderboard,
        out,
        locked_batch9_by_candidate=locked_batch9_by_candidate,
    )
    experiment_state_artifact_id: str | None = None
    if emit_artifact_trace and artifact_store is not None:
        ExperimentState = artifact_classes["ExperimentState"]
        experiment_state = ExperimentState(
            run_id=run_id,
            parent_artifact_ids=[artifact_id for artifact_id in [run_manifest_artifact_id, dataset_profile_artifact_id, split_artifact_id] if artifact_id],
            human_readable_summary=f"Experiment state for run {run_id}: {int(len(successes))} successful candidates.",
            status="completed",
            completed_iterations=int(iterations),
            candidate_count=int(len(leaderboard)),
            successful_candidate_count=int(len(successes)),
            best_candidate_path=best_candidate,
            best_metrics=best_metrics,
            leaderboard_path=str(out / "leaderboard.csv"),
            artifact_index_path=str(artifact_store.index_path),
            output_paths={
                "leaderboard": str(out / "leaderboard.csv"),
                "candidate_feature_summary": str(out / "candidate_feature_summary.csv"),
                "final_batch9_predictions": str(out / "final_batch9_predictions.csv") if (out / "final_batch9_predictions.csv").exists() else "",
                "final_batch9_metrics": str(out / "final_batch9_metrics.json") if (out / "final_batch9_metrics.json").exists() else "",
            },
        )
        artifact_store.write_artifact(experiment_state)
        experiment_state_artifact_id = experiment_state.artifact_id
        trace_logger.log_event(
            event_type="run_completed",
            iteration=None,
            agent_role=artifact_classes["AgentRole"].ORCHESTRATOR,
            input_artifact_ids=[artifact_id for artifact_id in [run_manifest_artifact_id, dataset_profile_artifact_id, split_artifact_id] if artifact_id],
            output_artifact_ids=[experiment_state.artifact_id],
            success=True,
        )

    report = {
        "run_id": run_id,
        "mode": "offline" if offline or not any(hasattr(a, "api_key") for a in agent_objs) else "llm_driven",
        "llm_client": llm_startup_summary(model=model),
        "agents": agents,
        "iterations": iterations,
        "data_source": data_source,
        "real_data_used": data_source == "processed_real",
        "synthetic_fallback_used": data_source == "synthetic_demo",
        "require_real_data": require_real_data,
        "label_source": dataset_card.get("label_source") or (metadata["label_source"].dropna().iloc[0] if "label_source" in metadata and not metadata["label_source"].dropna().empty else None),
        "split_mode": split_mode,
        "validation_fraction": validation_fraction,
        "split_seed": split_seed,
        "validation_batch_id": validation_batch_id,
        "split_manifest": split_manifest,
        "split_assignments_path": str(out / "split_assignments.csv"),
        "labels_hidden": ["validation", "test"],
        "batch9_status": batch9_status,
        "weak_baseline_rmse": weak_rmse,
        "surrogate_search_weak_baseline_rmse": weak_rmse,
        "surrogate_search_validation_rmse": best_metrics.get("rmse"),
        "locked_batch9_rmse": final_batch9_metrics.get("rmse") if final_batch9_metrics else None,
        "locked_batch9_weak_baseline_rmse": final_batch9_metrics.get("batch9_weak_baseline_rmse") if final_batch9_metrics else None,
        "author_literature_batch9_rmse": author_validation.get("author_model_batch9_rmse"),
        "batch9_pgr": final_batch9_metrics.get("battery_pgr_author_model_batch9") if final_batch9_metrics else None,
        "exact_author_model_rmse": author_validation.get("author_model_batch9_rmse") or ref["strong_rmse"],
        "author_literature_batch9_metrics": author_validation,
        "author_model_predictions_available": ref["author_model_predictions_available"],
        "author_model_validation_metrics_unavailable_batch9_skipped": ref["batch9_skipped"],
        "best_candidate": best_candidate,
        "best_metrics": best_metrics,
        "final_batch9_validation": final_batch9,
        "final_batch9_metrics": final_batch9_metrics,
        "final_batch9_top_k": final_batch9_top_k,
        "final_batch9_topk": final_batch9_topk,
        "candidate_failures": failure_summary,
        "candidate_feature_summary_path": str(out / "candidate_feature_summary.csv"),
        "candidate_feature_summary_rows": int(len(candidate_feature_summary)),
        "seed_with_author_inspired_baselines": seed_with_author_inspired_baselines,
        "emit_artifact_trace": emit_artifact_trace,
        "artifact_index_path": str(artifact_store.index_path) if artifact_store is not None else None,
        "experiment_state_artifact_id": experiment_state_artifact_id,
        "posthoc_feature_overlap": overlap,
        "caveats": [
            "Batch 9 is not required for this rediscovery run.",
            "Batch 9 was not used during surrogate search; it was only used for locked final validation when requested.",
            "Battery-PGR against the author model is undefined unless exact author validation RMSE is available.",
            "Synthetic/demo data are used only when processed local data files are absent and --require-real-data is not set.",
            "Surrogate-label search performance on OED/CLO batches is distinct from locked Batch 9 final validation.",
        ],
    }
    write_agent_reports(report, reports_dir)
    return report
