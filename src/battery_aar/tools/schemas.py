from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


def new_tool_call_id() -> str:
    return f"tool_call_{uuid4().hex}"


class ToolRequestBase(BaseModel):
    run_id: str = "tool_run"
    run_dir: str | None = None
    tool_call_id: str = Field(default_factory=new_tool_call_id)
    iteration: int | None = None
    agent_role: str | None = None
    input_artifact_ids: list[str] = Field(default_factory=list)


class ToolResponseBase(BaseModel):
    tool_name: str
    tool_call_id: str
    run_id: str
    success: bool
    output_artifact_ids: list[str] = Field(default_factory=list)
    output_paths: dict[str, str] = Field(default_factory=dict)
    error_type: str | None = None
    error_message: str | None = None
    duration_ms: float | None = None


class DatasetProfileRequest(ToolRequestBase):
    processed_dir: str | None = None
    metadata_path: str | None = None
    cycle_summary_path: str | None = None
    labels_path: str | None = None
    data_source: str = "processed"
    label_source: str | None = None


class DatasetProfileResponse(ToolResponseBase):
    profile: dict[str, Any] = Field(default_factory=dict)


class BuildFeaturesRequest(ToolRequestBase):
    metadata_path: str
    cycle_summary_path: str
    output_path: str | None = None
    max_cycle: int = 100
    include_protocol: bool = True
    return_feature_metadata: bool = True


class BuildFeaturesResponse(ToolResponseBase):
    n_rows: int = 0
    n_features: int = 0
    feature_columns: list[str] = Field(default_factory=list)


class CandidateReviewRequest(ToolRequestBase):
    candidate_path: str


class CandidateReviewResponse(ToolResponseBase):
    verdict: str | None = None
    issues: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


class CandidateEvaluateRequest(ToolRequestBase):
    candidate_path: str
    metadata_path: str
    cycle_summary_path: str
    labels_path: str | None = None
    split_assignments_path: str | None = None
    max_cycle: int = 100
    allow_protocol_features: bool = False
    weak_rmse: float | None = None
    strong_rmse: float | None = None
    timeout_s: int = 30


class CandidateEvaluateResponse(ToolResponseBase):
    metrics: dict[str, Any] = Field(default_factory=dict)
    n_eval: int | None = None
    failure_reason: str | None = None


class RunCompareRequest(ToolRequestBase):
    reports_dir: str = "reports"
    summary_csv: str | None = None
    run_ids: list[str] | None = None
    output_path: str | None = None


class RunCompareResponse(ToolResponseBase):
    comparison_rows: list[dict[str, Any]] = Field(default_factory=list)


class ToolDescriptor(BaseModel):
    name: str
    endpoint: str
    description: str


class ToolListResponse(ToolResponseBase):
    tools: list[ToolDescriptor] = Field(default_factory=list)
