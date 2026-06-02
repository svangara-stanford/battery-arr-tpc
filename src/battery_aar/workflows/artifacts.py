from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel

from .schemas import (
    ArtifactModel,
    CandidateSpec,
    CritiqueReport,
    DatasetProfileArtifact,
    EvaluationReport,
    ExperimentState,
    FeaturePlan,
    ModelPlan,
    ReviewReport,
    RunManifest,
    SplitArtifact,
)


def _sanitize(value: str | None) -> str:
    text = str(value or "unknown")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "unknown"


def _json_ready(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def artifact_default_relative_path(artifact: ArtifactModel) -> Path:
    if isinstance(artifact, RunManifest):
        return Path("run_manifest.json")
    if isinstance(artifact, DatasetProfileArtifact):
        return Path("dataset_profile.json")
    if isinstance(artifact, SplitArtifact):
        return Path("split_artifact.json")
    if isinstance(artifact, ExperimentState):
        return Path("experiment_state.json")

    iteration = getattr(artifact, "iteration", None)
    folder = Path(f"iteration_{int(iteration):03d}") if isinstance(iteration, int) and iteration >= 0 else Path("iteration_seed")
    agent_id = _sanitize(getattr(artifact, "agent_id", None) or getattr(artifact, "reviewer_id", None) or getattr(artifact, "critic_id", None))
    candidate_name = _sanitize(getattr(artifact, "candidate_name", None))
    suffix = candidate_name if candidate_name != "unknown" and agent_id in {"author_inspired_baseline", "baseline"} else agent_id
    if isinstance(artifact, FeaturePlan):
        return folder / f"feature_plan_{agent_id}.json"
    if isinstance(artifact, ModelPlan):
        return folder / f"model_plan_{agent_id}.json"
    if isinstance(artifact, CandidateSpec):
        if "variant_" in candidate_name:
            suffix = candidate_name
        return folder / f"candidate_spec_{suffix}.json"
    if isinstance(artifact, EvaluationReport):
        if "variant_" in candidate_name:
            suffix = candidate_name
        if artifact.locked_batch9_validation_run:
            return folder / f"evaluation_report_batch9_{suffix}.json"
        return folder / f"evaluation_report_{suffix}.json"
    if isinstance(artifact, ReviewReport):
        return folder / f"review_report_{agent_id}.json"
    if isinstance(artifact, CritiqueReport):
        return folder / f"critique_report_{agent_id}.json"
    return folder / f"{_sanitize(artifact.artifact_type).lower()}_{agent_id}.json"


class ArtifactStore:
    """Filesystem-backed JSON artifact store for one Open Battery Agents run."""

    def __init__(self, run_dir: str | Path, run_id: str | None = None):
        self.run_dir = Path(run_dir)
        self.run_id = run_id or self.run_dir.name
        self.artifact_dir = self.run_dir / "artifacts"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.artifact_dir / "index.json"
        self._index: list[dict[str, Any]] = self._load_index()

    def _load_index(self) -> list[dict[str, Any]]:
        if not self.index_path.exists():
            return []
        data = json.loads(self.index_path.read_text())
        if isinstance(data, dict) and isinstance(data.get("artifacts"), list):
            return data["artifacts"]
        if isinstance(data, list):
            return data
        return []

    @property
    def index(self) -> list[dict[str, Any]]:
        return list(self._index)

    def write_artifact(self, artifact: ArtifactModel, relative_path: str | Path | None = None) -> Path:
        self._index = self._load_index()
        rel = Path(relative_path) if relative_path is not None else artifact_default_relative_path(artifact)
        path = self.artifact_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = _json_ready(artifact.model_dump(mode="json"))
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        entry = {
            "artifact_id": artifact.artifact_id,
            "artifact_type": artifact.artifact_type,
            "path": str(path.relative_to(self.run_dir)),
            "created_at": artifact.created_at.isoformat(),
            "parent_artifact_ids": list(artifact.parent_artifact_ids),
            "summary": artifact.human_readable_summary,
        }
        self._index = [item for item in self._index if item.get("artifact_id") != artifact.artifact_id]
        self._index.append(entry)
        self._write_index()
        return path

    def _write_index(self) -> None:
        payload = {"run_id": self.run_id, "artifacts": _json_ready(self._index)}
        self.index_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _nan_counts_for_columns(df: pd.DataFrame, prefix: str, columns: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for col in columns:
        if col in df.columns:
            counts[f"{prefix}.{col}"] = int(df[col].isna().sum())
    return counts


def build_dataset_profile_artifact(
    run_id: str,
    metadata: pd.DataFrame,
    cycle_summary: pd.DataFrame,
    labels: pd.DataFrame,
    data_source: str,
    label_source: str | None,
    parent_artifact_ids: list[str] | None = None,
) -> DatasetProfileArtifact:
    major_metadata_cols = ["row_id", "cell_id", "batch_id", "protocol_readable", "C1", "C2", "C3", "C4", "cycle_life"]
    major_cycle_cols = ["row_id", "cell_id", "cycle_index", "discharge_capacity", "charge_capacity"]
    nan_counts = {
        **_nan_counts_for_columns(metadata, "metadata", major_metadata_cols),
        **_nan_counts_for_columns(cycle_summary, "cycle_summary", major_cycle_cols),
        **_nan_counts_for_columns(labels, "labels", ["row_id", "y", "cycle_life"]),
    }
    cycle_index = pd.to_numeric(cycle_summary.get("cycle_index", pd.Series(dtype=float)), errors="coerce")
    batch_counts = metadata["batch_id"].astype(str).value_counts(dropna=False).to_dict() if "batch_id" in metadata.columns else {}
    protocol_counts = metadata["protocol_readable"].astype(str).value_counts(dropna=False).to_dict() if "protocol_readable" in metadata.columns else {}
    return DatasetProfileArtifact(
        run_id=run_id,
        parent_artifact_ids=parent_artifact_ids or [],
        human_readable_summary=f"Dataset profile: {len(metadata)} metadata rows, {len(cycle_summary)} cycle rows, {len(labels)} labeled cells.",
        data_source=data_source,
        label_source=label_source,
        metadata_row_count=int(len(metadata)),
        cycle_summary_row_count=int(len(cycle_summary)),
        labeled_cell_count=int(len(labels)),
        metadata_columns=list(map(str, metadata.columns)),
        cycle_summary_columns=list(map(str, cycle_summary.columns)),
        label_columns=list(map(str, labels.columns)),
        nan_counts=nan_counts,
        cycle_index_min=float(cycle_index.min()) if not cycle_index.dropna().empty else None,
        cycle_index_max=float(cycle_index.max()) if not cycle_index.dropna().empty else None,
        batch_id_counts={str(k): int(v) for k, v in batch_counts.items()},
        protocol_counts={str(k): int(v) for k, v in protocol_counts.items()},
    )


def build_split_artifact(
    run_id: str,
    split_manifest: dict[str, Any],
    assignments_path: str | Path,
    parent_artifact_ids: list[str] | None = None,
) -> SplitArtifact:
    return SplitArtifact(
        run_id=run_id,
        parent_artifact_ids=parent_artifact_ids or [],
        human_readable_summary=(
            f"Split artifact: {split_manifest.get('split_mode')} split with "
            f"{split_manifest.get('n_train_cells')} train and {split_manifest.get('n_validation_cells')} validation cells."
        ),
        split_mode=str(split_manifest.get("split_mode")),
        validation_fraction=float(split_manifest.get("validation_fraction", 0.0)),
        split_seed=int(split_manifest.get("split_seed", 0)),
        train_cell_count=int(split_manifest.get("n_train_cells", 0)),
        validation_cell_count=int(split_manifest.get("n_validation_cells", 0)),
        train_group_count=split_manifest.get("n_train_groups"),
        validation_group_count=split_manifest.get("n_validation_groups"),
        group_type=split_manifest.get("group_type"),
        heldout_groups=list(map(str, split_manifest.get("heldout_groups", []))),
        assignments_path=str(assignments_path),
        split_manifest=_json_ready(split_manifest),
    )
