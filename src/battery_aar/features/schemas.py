from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "open_battery_agents_artifacts.v1"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_artifact_id() -> str:
    return f"artifact_{uuid4().hex}"


class FeatureArtifactBase(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    artifact_id: str = Field(default_factory=new_artifact_id)
    artifact_type: str
    run_id: str
    created_at: datetime = Field(default_factory=utc_now)
    schema_version: str = SCHEMA_VERSION
    parent_artifact_ids: List[str] = Field(default_factory=list)
    human_readable_summary: str


class FeatureOperatorSpec(BaseModel):
    operator_name: str
    operator_type: str
    family: str
    enabled: bool = True
    params: Dict[str, Any] = Field(default_factory=dict)
    feature_prefix: Optional[str] = None
    description: Optional[str] = None
    hypothesis: Optional[str] = None


class FeatureProgram(FeatureArtifactBase):
    artifact_type: Literal["FeatureProgram"] = "FeatureProgram"
    program_id: str
    name: str
    description: str
    operators: List[FeatureOperatorSpec] = Field(default_factory=list)
    include_protocol_features: bool = False
    cycle_index_convention: str = "raw_zero_based"
    first_n_cycles: int = 100
    feature_selection_policy: Optional[Dict[str, Any]] = None
    proposed_by: Optional[str] = None
    rationale: Optional[str] = None


class FeatureProgramResult(FeatureArtifactBase):
    artifact_type: Literal["FeatureProgramResult"] = "FeatureProgramResult"
    program_id: str
    feature_table_path: str
    feature_metadata_path: str
    n_rows: int
    n_feature_columns: int
    n_numeric_feature_columns: int
    n_excluded_cells: int
    n_pruned_all_nan_features: int = 0
    feature_family_counts: Dict[str, int] = Field(default_factory=dict)
    operator_status: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
