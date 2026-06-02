from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from battery_aar.features.schemas import FeatureOperatorSpec, FeatureProgram, FeatureProgramResult

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
    parent_artifact_ids: List[str] = Field(default_factory=list)
    human_readable_summary: str


class RunManifest(ArtifactBase):
    artifact_type: Literal["RunManifest"] = "RunManifest"
    output_dir: str
    command: Optional[str] = None
    config: Dict[str, Any] = Field(default_factory=dict)
    code_version: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class DatasetProfileArtifact(ArtifactBase):
    artifact_type: Literal["DatasetProfileArtifact"] = "DatasetProfileArtifact"
    data_source: str
    label_source: Optional[str] = None
    metadata_row_count: int
    cycle_summary_row_count: int
    labeled_cell_count: int
    metadata_columns: List[str] = Field(default_factory=list)
    cycle_summary_columns: List[str] = Field(default_factory=list)
    label_columns: List[str] = Field(default_factory=list)
    nan_counts: Dict[str, int] = Field(default_factory=dict)
    cycle_index_min: Optional[float] = None
    cycle_index_max: Optional[float] = None
    batch_id_counts: Dict[str, int] = Field(default_factory=dict)
    protocol_counts: Dict[str, int] = Field(default_factory=dict)


class SplitArtifact(ArtifactBase):
    artifact_type: Literal["SplitArtifact"] = "SplitArtifact"
    split_mode: str
    validation_fraction: float
    split_seed: int
    train_cell_count: int
    validation_cell_count: int
    train_group_count: Optional[int] = None
    validation_group_count: Optional[int] = None
    group_type: Optional[str] = None
    heldout_groups: List[str] = Field(default_factory=list)
    assignments_path: Optional[str] = None
    split_manifest: Dict[str, Any] = Field(default_factory=dict)


class FeaturePlan(ArtifactBase):
    artifact_type: Literal["FeaturePlan"] = "FeaturePlan"
    agent_id: str
    agent_role: AgentRole = AgentRole.FEATURE_ENGINEER
    iteration: Optional[int] = None
    feature_families: List[str] = Field(default_factory=list)
    selected_columns: List[str] = Field(default_factory=list)
    include_protocol_features: bool = False
    feature_program_ids: List[str] = Field(default_factory=list)
    feature_program_paths: List[str] = Field(default_factory=list)
    feature_program_recipe: Optional[str] = None
    feature_set: str = "all_available"
    max_cycle: Optional[int] = None
    rationale: Optional[str] = None
    constraints: List[str] = Field(default_factory=list)


class ModelPlan(ArtifactBase):
    artifact_type: Literal["ModelPlan"] = "ModelPlan"
    agent_id: str
    agent_role: AgentRole = AgentRole.MODEL_BUILDER
    iteration: Optional[int] = None
    model_family: str
    estimator_name: Optional[str] = None
    target_transform: str = "raw"
    feature_set: str = "all_available"
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    preprocessing_steps: List[str] = Field(default_factory=list)
    rationale: Optional[str] = None


class CandidateSpec(ArtifactBase):
    artifact_type: Literal["CandidateSpec"] = "CandidateSpec"
    candidate_id: str
    agent_id: str
    agent_role: AgentRole
    iteration: int
    candidate_path: str
    candidate_name: Optional[str] = None
    code_sha256: Optional[str] = None
    uses_toolbox: bool = False
    compiled_candidate: Optional[bool] = None
    declared_dependencies: List[str] = Field(default_factory=list)
    feature_plan_artifact_id: Optional[str] = None
    model_plan_artifact_id: Optional[str] = None
    feature_families: List[str] = Field(default_factory=list)
    include_protocol_features: bool = False
    model_family: Optional[str] = None
    target_transform: str = "raw"
    feature_set: str = "all_available"
    feature_program_paths: List[str] = Field(default_factory=list)
    feature_program_mode: str = "none"
    include_feature_programs: bool = False
    feature_family_filter: List[str] = Field(default_factory=list)
    preprocessing: List[str] = Field(default_factory=list)
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)


class ReviewReport(ArtifactBase):
    artifact_type: Literal["ReviewReport"] = "ReviewReport"
    reviewer_id: str
    agent_role: AgentRole = AgentRole.REVIEWER
    iteration: Optional[int] = None
    target_artifact_ids: List[str] = Field(default_factory=list)
    verdict: str
    issues: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class EvaluationReport(ArtifactBase):
    artifact_type: Literal["EvaluationReport"] = "EvaluationReport"
    candidate_id: str
    candidate_path: str
    agent_id: Optional[str] = None
    candidate_name: Optional[str] = None
    iteration: Optional[int] = None
    split_mode: str
    locked_batch9_validation_run: bool = False
    success: bool
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2: Optional[float] = None
    spearman: Optional[float] = None
    kendall: Optional[float] = None
    pgr: Optional[float] = None
    failure_reason: Optional[str] = None
    traceback: Optional[str] = None
    stdout_excerpt: Optional[str] = None
    stderr_excerpt: Optional[str] = None
    prediction_path: Optional[str] = None
    extra_metrics: Dict[str, Any] = Field(default_factory=dict)


class CritiqueReport(ArtifactBase):
    artifact_type: Literal["CritiqueReport"] = "CritiqueReport"
    critic_id: str
    agent_role: AgentRole = AgentRole.CRITIC
    iteration: Optional[int] = None
    target_artifact_ids: List[str] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    proposed_next_steps: List[str] = Field(default_factory=list)


class ToolCallRecord(ArtifactBase):
    artifact_type: Literal["ToolCallRecord"] = "ToolCallRecord"
    tool_name: str
    agent_role: AgentRole
    iteration: Optional[int] = None
    input_artifact_ids: List[str] = Field(default_factory=list)
    output_artifact_ids: List[str] = Field(default_factory=list)
    arguments_summary: Dict[str, Any] = Field(default_factory=dict)
    duration_ms: Optional[float] = None
    success: bool
    error_type: Optional[str] = None
    error_message: Optional[str] = None


class ExperimentState(ArtifactBase):
    artifact_type: Literal["ExperimentState"] = "ExperimentState"
    status: str
    completed_iterations: int
    candidate_count: int
    successful_candidate_count: int
    best_candidate_path: Optional[str] = None
    best_candidate_id: Optional[str] = None
    best_iteration: Optional[int] = None
    best_metrics: Dict[str, Any] = Field(default_factory=dict)
    all_candidate_metrics: List[Dict[str, Any]] = Field(default_factory=list)
    final_batch9_metrics_path: Optional[str] = None
    final_batch9_predictions_path: Optional[str] = None
    final_batch9_topk_metrics_path: Optional[str] = None
    best_locked_validation_candidate_path: Optional[str] = None
    leaderboard_path: Optional[str] = None
    artifact_index_path: Optional[str] = None
    output_paths: Dict[str, str] = Field(default_factory=dict)


ArtifactModel = Union[
    RunManifest,
    DatasetProfileArtifact,
    SplitArtifact,
    FeaturePlan,
    ModelPlan,
    FeatureProgram,
    FeatureProgramResult,
    CandidateSpec,
    ReviewReport,
    EvaluationReport,
    CritiqueReport,
    ToolCallRecord,
    ExperimentState,
]
