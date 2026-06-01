from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "open_battery_agents_artifacts.v1"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_artifact_id() -> str:
    return f"artifact_{uuid4().hex}"


class AgentRole(str, Enum):
    ORCHESTRATOR = "orchestrator"
    DATASET_PROFILER = "dataset_profiler"
    SPLIT_MANAGER = "split_manager"
    FEATURE_ENGINEER = "feature_engineer"
    MODEL_BUILDER = "model_builder"
    REVIEWER = "reviewer"
    CRITIC = "critic"
    EVALUATOR = "evaluator"
    LLM_CANDIDATE = "llm_candidate"
    BASELINE = "baseline"


class ArtifactBase(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    artifact_id: str = Field(default_factory=new_artifact_id)
    artifact_type: str
    run_id: str
    created_at: datetime = Field(default_factory=utc_now)
    schema_version: str = SCHEMA_VERSION
    parent_artifact_ids: list[str] = Field(default_factory=list)
    human_readable_summary: str


class RunManifest(ArtifactBase):
    artifact_type: Literal["RunManifest"] = "RunManifest"
    output_dir: str
    command: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    code_version: str | None = None
    tags: list[str] = Field(default_factory=list)


class DatasetProfileArtifact(ArtifactBase):
    artifact_type: Literal["DatasetProfileArtifact"] = "DatasetProfileArtifact"
    data_source: str
    label_source: str | None = None
    metadata_row_count: int
    cycle_summary_row_count: int
    labeled_cell_count: int
    metadata_columns: list[str] = Field(default_factory=list)
    cycle_summary_columns: list[str] = Field(default_factory=list)
    label_columns: list[str] = Field(default_factory=list)
    nan_counts: dict[str, int] = Field(default_factory=dict)
    cycle_index_min: float | None = None
    cycle_index_max: float | None = None
    batch_id_counts: dict[str, int] = Field(default_factory=dict)
    protocol_counts: dict[str, int] = Field(default_factory=dict)


class SplitArtifact(ArtifactBase):
    artifact_type: Literal["SplitArtifact"] = "SplitArtifact"
    split_mode: str
    validation_fraction: float
    split_seed: int
    train_cell_count: int
    validation_cell_count: int
    train_group_count: int | None = None
    validation_group_count: int | None = None
    group_type: str | None = None
    heldout_groups: list[str] = Field(default_factory=list)
    assignments_path: str | None = None
    split_manifest: dict[str, Any] = Field(default_factory=dict)


class FeaturePlan(ArtifactBase):
    artifact_type: Literal["FeaturePlan"] = "FeaturePlan"
    agent_id: str
    agent_role: AgentRole = AgentRole.FEATURE_ENGINEER
    iteration: int | None = None
    feature_families: list[str] = Field(default_factory=list)
    selected_columns: list[str] = Field(default_factory=list)
    include_protocol_features: bool = False
    max_cycle: int | None = None
    rationale: str | None = None
    constraints: list[str] = Field(default_factory=list)


class ModelPlan(ArtifactBase):
    artifact_type: Literal["ModelPlan"] = "ModelPlan"
    agent_id: str
    agent_role: AgentRole = AgentRole.MODEL_BUILDER
    iteration: int | None = None
    model_family: str
    estimator_name: str | None = None
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    preprocessing_steps: list[str] = Field(default_factory=list)
    rationale: str | None = None


class CandidateSpec(ArtifactBase):
    artifact_type: Literal["CandidateSpec"] = "CandidateSpec"
    candidate_id: str
    agent_id: str
    agent_role: AgentRole
    iteration: int
    candidate_path: str
    candidate_name: str | None = None
    code_sha256: str | None = None
    uses_toolbox: bool = False
    declared_dependencies: list[str] = Field(default_factory=list)
    feature_plan_artifact_id: str | None = None
    model_plan_artifact_id: str | None = None


class ReviewReport(ArtifactBase):
    artifact_type: Literal["ReviewReport"] = "ReviewReport"
    reviewer_id: str
    agent_role: AgentRole = AgentRole.REVIEWER
    iteration: int | None = None
    target_artifact_ids: list[str] = Field(default_factory=list)
    verdict: str
    issues: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


class EvaluationReport(ArtifactBase):
    artifact_type: Literal["EvaluationReport"] = "EvaluationReport"
    candidate_id: str
    candidate_path: str
    agent_id: str | None = None
    candidate_name: str | None = None
    iteration: int | None = None
    split_mode: str
    locked_batch9_validation_run: bool = False
    success: bool
    rmse: float | None = None
    mae: float | None = None
    r2: float | None = None
    spearman: float | None = None
    kendall: float | None = None
    pgr: float | None = None
    failure_reason: str | None = None
    traceback: str | None = None
    stdout_excerpt: str | None = None
    stderr_excerpt: str | None = None
    extra_metrics: dict[str, Any] = Field(default_factory=dict)


class CritiqueReport(ArtifactBase):
    artifact_type: Literal["CritiqueReport"] = "CritiqueReport"
    critic_id: str
    agent_role: AgentRole = AgentRole.CRITIC
    iteration: int | None = None
    target_artifact_ids: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    proposed_next_steps: list[str] = Field(default_factory=list)


class ToolCallRecord(ArtifactBase):
    artifact_type: Literal["ToolCallRecord"] = "ToolCallRecord"
    tool_name: str
    agent_role: AgentRole
    iteration: int | None = None
    input_artifact_ids: list[str] = Field(default_factory=list)
    output_artifact_ids: list[str] = Field(default_factory=list)
    arguments_summary: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float | None = None
    success: bool
    error_type: str | None = None
    error_message: str | None = None


class ExperimentState(ArtifactBase):
    artifact_type: Literal["ExperimentState"] = "ExperimentState"
    status: str
    completed_iterations: int
    candidate_count: int
    successful_candidate_count: int
    best_candidate_path: str | None = None
    best_metrics: dict[str, Any] = Field(default_factory=dict)
    leaderboard_path: str | None = None
    artifact_index_path: str | None = None
    output_paths: dict[str, str] = Field(default_factory=dict)


ArtifactModel = (
    RunManifest
    | DatasetProfileArtifact
    | SplitArtifact
    | FeaturePlan
    | ModelPlan
    | CandidateSpec
    | ReviewReport
    | EvaluationReport
    | CritiqueReport
    | ToolCallRecord
    | ExperimentState
)
