from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


def new_tool_call_id() -> str:
    return f"tool_call_{uuid4().hex}"


class ToolRequestBase(BaseModel):
    run_id: str = "tool_run"
    run_dir: Optional[str] = None
    tool_call_id: str = Field(default_factory=new_tool_call_id)
    iteration: Optional[int] = None
    agent_role: Optional[str] = None
    input_artifact_ids: List[str] = Field(default_factory=list)


class ToolResponseBase(BaseModel):
    tool_name: str
    tool_call_id: str
    run_id: str
    success: bool
    output_artifact_ids: List[str] = Field(default_factory=list)
    output_paths: Dict[str, str] = Field(default_factory=dict)
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    duration_ms: Optional[float] = None


class DatasetProfileRequest(ToolRequestBase):
    processed_dir: Optional[str] = None
    metadata_path: Optional[str] = None
    cycle_summary_path: Optional[str] = None
    labels_path: Optional[str] = None
    data_source: str = "processed"
    label_source: Optional[str] = None


class DatasetProfileResponse(ToolResponseBase):
    profile: Dict[str, Any] = Field(default_factory=dict)


class BuildFeaturesRequest(ToolRequestBase):
    metadata_path: str
    cycle_summary_path: str
    output_path: Optional[str] = None
    max_cycle: int = 100
    include_protocol: bool = True
    return_feature_metadata: bool = True
    feature_program_paths: List[str] = Field(default_factory=list)
    feature_program_mode: str = "none"
    include_feature_programs: bool = False
    feature_family_filter: List[str] = Field(default_factory=list)
    feature_program_recipe: Optional[str] = None
    feature_program_json: Optional[str] = None


class BuildFeaturesResponse(ToolResponseBase):
    n_rows: int = 0
    n_features: int = 0
    feature_columns: List[str] = Field(default_factory=list)
    feature_programs_used: List[str] = Field(default_factory=list)
    n_feature_program_columns: int = 0
    feature_family_counts: Dict[str, int] = Field(default_factory=dict)
    n_matched_rows: int = 0
    n_missing_rows: int = 0
    true_raw_curve_features_used: bool = False
    proxy_features_used: bool = False
    protocol_features_used: bool = False


class FeatureProgramsResponse(ToolResponseBase):
    recipes: List[str] = Field(default_factory=list)
    operators: List[Dict[str, Any]] = Field(default_factory=list)


class CandidateReviewRequest(ToolRequestBase):
    candidate_path: str


class CandidateReviewResponse(ToolResponseBase):
    verdict: Optional[str] = None
    issues: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    failure_reason: Optional[str] = None


class CandidateEvaluateRequest(ToolRequestBase):
    candidate_path: str
    metadata_path: str
    cycle_summary_path: str
    labels_path: Optional[str] = None
    split_assignments_path: Optional[str] = None
    split_mode: str = "tool_validation"
    max_cycle: int = 100
    allow_protocol_features: bool = False
    weak_rmse: Optional[float] = None
    strong_rmse: Optional[float] = None
    timeout_s: int = 30
    feature_program_paths: List[str] = Field(default_factory=list)
    feature_program_mode: str = "none"
    include_feature_programs: bool = False
    feature_family_filter: List[str] = Field(default_factory=list)


class CandidateEvaluateResponse(ToolResponseBase):
    metrics: Dict[str, Any] = Field(default_factory=dict)
    n_eval: Optional[int] = None
    prediction_path: Optional[str] = None
    failure_reason: Optional[str] = None
    traceback: Optional[str] = None


class RunCompareRequest(ToolRequestBase):
    reports_dir: str = "reports"
    summary_csv: Optional[str] = None
    run_ids: Optional[List[str]] = None
    output_path: Optional[str] = None


class RunCompareResponse(ToolResponseBase):
    comparison_rows: List[Dict[str, Any]] = Field(default_factory=list)


class ToolDescriptor(BaseModel):
    name: str
    endpoint: str
    description: str


class ToolListResponse(ToolResponseBase):
    tools: List[ToolDescriptor] = Field(default_factory=list)
