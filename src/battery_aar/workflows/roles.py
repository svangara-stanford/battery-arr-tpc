from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from battery_aar.agents.llm_client import load_llm_client_config
from battery_aar.tools.client import HTTPToolClient, NativeToolClient
from battery_aar.workflows.artifacts import ArtifactStore
from battery_aar.workflows.candidate_compiler import candidate_spec_from_plans, compile_candidate_spec_to_python
from battery_aar.workflows.role_prompts import (
    ROLE_SYSTEM_PROMPTS,
    champion_adjudicator_prompt,
    champion_aggregator_prompt,
    code_generator_prompt,
    code_repair_prompt,
    feature_scientist_prompt,
    model_architect_prompt,
    schema_repair_prompt,
    scientist_critic_prompt,
)
from battery_aar.workflows.schemas import (
    AgentRole,
    CandidateSpec,
    ChampionDecision,
    ChampionShortlist,
    CritiqueReport,
    EvaluationReport,
    FeaturePlan,
    ModelPlan,
    ReviewReport,
)
from battery_aar.workflows.trace import TraceLogger


_CODE_FENCE_RE = re.compile(r"```[A-Za-z0-9_+-]*[ \t]*\n(.*?)(?:\n```|\Z)", re.DOTALL)


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    # Models often preface code with prose ("Here is the repaired code:").
    # Extract the first fenced block wherever it appears; fall back to the
    # raw text when the reply contains no fence at all.
    match = _CODE_FENCE_RE.search(stripped)
    if match:
        return match.group(1).strip()
    return stripped


def _extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = _strip_code_fences(stripped)
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("LLM response JSON must be an object")
    return parsed


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _as_str_or(value: Any, default: str | None) -> str | None:
    """Coerce malformed LLM payload values into a string or fall back to ``default``.

    Some role-prompted LLM payloads return dicts, lists, or nested JSON when the
    schema expects a string (e.g. ``feature_program_recipe`` or ``rationale``).
    This helper flattens such cases instead of letting Pydantic raise.
    """

    if value is None:
        return default
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else default
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple, set)):
        flat = [_as_str_or(v, None) for v in value]
        flat = [v for v in flat if v]
        return ", ".join(flat) if flat else default
    if isinstance(value, dict):
        try:
            return json.dumps(value, sort_keys=True, default=str)
        except Exception:
            return default
    return str(value)


def _as_int_or(value: Any, default: int | None) -> int | None:
    """Coerce a malformed LLM payload value to int or fall back to ``default``."""

    if value is None:
        return default
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return int(float(text))
        except (TypeError, ValueError):
            return default
    if isinstance(value, dict):
        for key in ("value", "max_cycle", "n", "count"):
            if key in value:
                return _as_int_or(value[key], default)
        return default
    return default


def _as_str_list(value: Any) -> list[str]:
    """Coerce a malformed LLM payload value into a list[str], discarding empties."""

    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, dict):
        items = []
        for v in value.values():
            items.extend(_as_str_list(v))
        return items
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, str):
                stripped = item.strip()
                if stripped:
                    out.append(stripped)
            elif isinstance(item, (int, float, bool)):
                out.append(str(item))
            elif isinstance(item, dict):
                inner = _as_str_or(item, None)
                if inner:
                    out.append(inner)
            else:
                out.append(str(item))
        return out
    return [str(value)]


def _as_dict_or_empty(value: Any) -> dict[str, Any]:
    """Coerce a malformed LLM payload value into a dict; return ``{}`` on failure."""

    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
            return dict(parsed) if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _call_llm(role_name: str, prompt: str, model: str | None = None) -> str:
    from openai import OpenAI

    config = load_llm_client_config(model=model)
    if not config.api_key:
        raise RuntimeError("No Open Battery Agents API key found for role workflow LLM mode")
    client_kwargs: dict[str, Any] = {"api_key": config.api_key, "base_url": config.base_url}
    if config.default_headers:
        client_kwargs["default_headers"] = config.default_headers
    client = OpenAI(**client_kwargs)
    response = client.chat.completions.create(
        model=config.model,
        messages=[
            {"role": "system", "content": ROLE_SYSTEM_PROMPTS[role_name]},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2 if role_name != "CodeGenerator" else 0.35,
    )
    return response.choices[0].message.content or ""


def _llm_json(role_name: str, prompt: str, schema_hint: str, model: str | None = None) -> dict[str, Any]:
    text = _call_llm(role_name, prompt, model=model)
    try:
        return _extract_json(text)
    except Exception:
        repaired = _call_llm(role_name, schema_repair_prompt(text, schema_hint), model=model)
        return _extract_json(repaired)


def _default_toolbox_candidate_code() -> str:
    return r'''
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from battery_aar.features.battery_lifetime_features import build_all_battery_features


def _feature_frame(metadata, cycle_summary, config, feature_cols=None):
    max_cycle = int(config.get("max_cycle", 100))
    include_protocol = bool(config.get("allow_protocol_features", False))
    X = build_all_battery_features(metadata, cycle_summary, max_cycle=max_cycle, include_protocol=include_protocol)
    key_name = X.index.name or "row_id"
    X = X.reset_index().rename(columns={key_name: "row_id"})
    if "row_id" not in X.columns:
        X.insert(0, "row_id", metadata["row_id"].to_numpy())
    X["row_id"] = pd.to_numeric(X["row_id"], errors="coerce").astype(int)
    feature_data = X.drop(columns=["row_id", "cell_id"], errors="ignore")
    feature_data = feature_data.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if feature_cols is None:
        feature_data = feature_data.dropna(axis=1, how="all")
        feature_cols = list(feature_data.columns)
    else:
        for col in feature_cols:
            if col not in feature_data.columns:
                feature_data[col] = np.nan
        feature_data = feature_data[feature_cols]
    return X[["row_id"]], feature_data, feature_cols


def fit(train_metadata, train_cycle_summary, train_labels, config):
    ids, X, feature_cols = _feature_frame(train_metadata, train_cycle_summary, config)
    y = train_labels.set_index("row_id").loc[ids["row_id"], "y"].to_numpy(float)
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
    model.fit(X, y)
    return {"model": model, "feature_cols": feature_cols}


def predict(model, test_metadata, test_cycle_summary, config):
    ids, X, _ = _feature_frame(test_metadata, test_cycle_summary, config, model["feature_cols"])
    y_pred = model["model"].predict(X)
    return pd.DataFrame({"row_id": ids["row_id"], "y_pred": y_pred})
'''


def offline_candidate_code() -> str:
    return _default_toolbox_candidate_code()


def offline_repair_candidate_code() -> str:
    return _default_toolbox_candidate_code()


@dataclass
class RoleContext:
    run_id: str
    run_dir: Path
    processed_dir: Path
    metadata_path: Path
    cycle_summary_path: Path
    labels_path: Path | None
    split_assignments_path: Path
    split_mode: str
    max_cycle: int
    allow_protocol_features: bool
    offline: bool
    model: str | None
    store: ArtifactStore
    trace: TraceLogger
    tool_client: NativeToolClient | HTTPToolClient
    allow_freeform_code: bool = False
    candidates_per_iteration: int = 1
    feature_program_paths: list[str] | None = None
    feature_program_mode: str = "none"
    include_feature_programs: bool = False
    feature_family_filter: list[str] | None = None
    feature_program_recipe: str | None = None
    rag_augment: bool = True


class DatasetProfiler:
    role_name = "DatasetProfiler"

    def run(self, ctx: RoleContext, parent_ids: list[str] | None = None) -> tuple[dict[str, Any], list[str]]:
        response = ctx.tool_client.profile_dataset(
            run_id=ctx.run_id,
            run_dir=str(ctx.run_dir),
            processed_dir=str(ctx.processed_dir),
            data_source=str(ctx.processed_dir),
            iteration=None,
            agent_role=AgentRole.DATASET_PROFILER.value,
            input_artifact_ids=parent_ids or [],
        )
        return response.profile, response.output_artifact_ids


class FeatureScientist:
    role_name = "FeatureScientist"

    @staticmethod
    def _augmented_prompt(
        ctx: RoleContext, dataset_profile: dict[str, Any], probe: dict[str, Any]
    ) -> tuple[str, str | None, str | None]:
        """Build the role prompt, RAG-augmented when enabled.

        Returns (prompt, rag_summary, rag_error). Any RAG failure (missing
        indexes, missing optional deps, retrieval errors) falls back to the
        plain prompt so the workflow never breaks on the augmentation path.
        """
        original = feature_scientist_prompt(dataset_profile, probe)
        if not ctx.rag_augment:
            return original, None, None
        try:
            from battery_aar.rag.scripts.augmented_prompt import (
                augment_feature_scientist_prompt,
            )

            augmented = augment_feature_scientist_prompt(dataset_profile, probe)
            summary = (
                f"RAG context: {len(augmented.chunks)} chunks "
                f"{[chunk['chunk_id'] for chunk in augmented.chunks]} "
                f"from queries {augmented.queries}; trace={augmented.trace_path}"
            )
            return augmented.prompt, summary, None
        except Exception as exc:
            return original, None, f"{type(exc).__name__}: {exc}"

    def run(self, ctx: RoleContext, dataset_profile: dict[str, Any], parent_ids: list[str], iteration: int) -> FeaturePlan:
        feature_response = ctx.tool_client.build_battery_features(
            run_id=ctx.run_id,
            run_dir=str(ctx.run_dir),
            metadata_path=str(ctx.metadata_path),
            cycle_summary_path=str(ctx.cycle_summary_path),
            output_path=str(ctx.run_dir / "artifacts" / f"iteration_{iteration:03d}" / "feature_probe.csv"),
            max_cycle=ctx.max_cycle,
            include_protocol=ctx.allow_protocol_features,
            return_feature_metadata=True,
            feature_program_paths=ctx.feature_program_paths or [],
            feature_program_mode=ctx.feature_program_mode,
            include_feature_programs=ctx.include_feature_programs,
            feature_family_filter=ctx.feature_family_filter or [],
            feature_program_recipe=ctx.feature_program_recipe,
            iteration=iteration,
            agent_role=AgentRole.FEATURE_ENGINEER.value,
            input_artifact_ids=parent_ids,
        )
        probe = {
            "success": feature_response.success,
            "n_rows": feature_response.n_rows,
            "n_features": feature_response.n_features,
            "feature_columns": feature_response.feature_columns[:80],
            "output_paths": feature_response.output_paths,
        }
        all_parent_ids = parent_ids + feature_response.output_artifact_ids
        if ctx.offline:
            plan = FeaturePlan(
                run_id=ctx.run_id,
                parent_artifact_ids=all_parent_ids,
                human_readable_summary="FeatureScientist selected author-inspired first-100-cycle feature families.",
                agent_id="feature_scientist",
                iteration=iteration,
                feature_families=[
                    "capacity_cycles_2_10_100",
                    "max_minus_cycle2",
                    "cycleN_minus_cycle10",
                    "early_capacity_slope",
                    "late_capacity_slope",
                    "log_difference_proxies",
                    "protocol_features" if ctx.allow_protocol_features else "protocol_features_disabled",
                ],
                selected_columns=feature_response.feature_columns,
                include_protocol_features=ctx.allow_protocol_features,
                feature_program_paths=list(ctx.feature_program_paths or []),
                feature_program_recipe=ctx.feature_program_recipe,
                feature_set="all_available",
                max_cycle=ctx.max_cycle,
                rationale="Offline deterministic plan uses the battery feature toolbox without author coefficients.",
                constraints=["row_id and cell_id are join keys only", "batch 9 is not used during surrogate search"],
            )
        else:
            prompt, rag_summary, rag_error = self._augmented_prompt(ctx, dataset_profile, probe)
            if ctx.rag_augment:
                ctx.trace.log_agent_message(
                    event_type="rag_prompt_augmentation",
                    iteration=iteration,
                    agent_role=AgentRole.FEATURE_ENGINEER,
                    agent_id="feature_scientist",
                    input_artifact_ids=all_parent_ids,
                    success=rag_error is None,
                    error_message=rag_error,
                    message_summary=rag_summary or "RAG augmentation failed; plain prompt used",
                )
            payload = _llm_json(
                self.role_name,
                prompt,
                "FeaturePlan keys: agent_id, feature_families, selected_columns, include_protocol_features, max_cycle, rationale, constraints",
                model=ctx.model,
            )
            rationale_text = _as_str_or(payload.get("rationale"), None)
            plan = FeaturePlan(
                run_id=ctx.run_id,
                parent_artifact_ids=all_parent_ids,
                human_readable_summary=_as_str_or(rationale_text, "LLM FeatureScientist plan.") or "LLM FeatureScientist plan.",
                agent_id=_as_str_or(payload.get("agent_id"), "feature_scientist") or "feature_scientist",
                iteration=iteration,
                feature_families=_as_str_list(payload.get("feature_families", [])),
                selected_columns=_as_str_list(payload.get("selected_columns", [])),
                include_protocol_features=bool(payload.get("include_protocol_features", ctx.allow_protocol_features)),
                feature_program_ids=_as_str_list(payload.get("feature_program_ids", [])),
                feature_program_paths=list(ctx.feature_program_paths or []),
                feature_program_recipe=_as_str_or(payload.get("feature_program_recipe"), ctx.feature_program_recipe),
                feature_set=_as_str_or(payload.get("feature_set"), "all_available") or "all_available",
                max_cycle=_as_int_or(payload.get("max_cycle"), ctx.max_cycle),
                rationale=rationale_text,
                constraints=_as_str_list(payload.get("constraints", [])),
            )
        path = ctx.store.write_artifact(plan)
        ctx.trace.log_agent_message(
            event_type="feature_plan_created",
            iteration=iteration,
            agent_role=AgentRole.FEATURE_ENGINEER,
            agent_id=plan.agent_id,
            input_artifact_ids=all_parent_ids,
            output_artifact_ids=[plan.artifact_id],
            message_summary=plan.human_readable_summary,
        )
        return plan


class ModelArchitect:
    role_name = "ModelArchitect"

    def run(self, ctx: RoleContext, feature_plan: FeaturePlan, dataset_profile: dict[str, Any], iteration: int) -> ModelPlan:
        if ctx.offline:
            plan = ModelPlan(
                run_id=ctx.run_id,
                parent_artifact_ids=[feature_plan.artifact_id],
                human_readable_summary="ModelArchitect selected a Ridge pipeline with median imputation and scaling.",
                agent_id="model_architect",
                iteration=iteration,
                model_family="linear_regularized",
                estimator_name="Ridge",
                target_transform="log10",
                feature_set="all_available",
                hyperparameters={"alpha": 1.0},
                preprocessing_steps=["drop_all_nan_columns", "SimpleImputer(strategy='median')", "StandardScaler"],
                rationale="Small surrogate datasets need a stable regularized baseline before more flexible models.",
            )
        else:
            payload = _llm_json(
                self.role_name,
                model_architect_prompt(feature_plan.model_dump(mode="json"), dataset_profile),
                "ModelPlan keys: agent_id, model_family, estimator_name, target_transform, hyperparameters, preprocessing_steps, rationale",
                model=ctx.model,
            )
            rationale_text = _as_str_or(payload.get("rationale"), None)
            plan = ModelPlan(
                run_id=ctx.run_id,
                parent_artifact_ids=[feature_plan.artifact_id],
                human_readable_summary=_as_str_or(rationale_text, "LLM ModelArchitect plan.") or "LLM ModelArchitect plan.",
                agent_id=_as_str_or(payload.get("agent_id"), "model_architect") or "model_architect",
                iteration=iteration,
                model_family=_as_str_or(payload.get("model_family"), "regularized_regression") or "regularized_regression",
                estimator_name=_as_str_or(payload.get("estimator_name"), None),
                target_transform=_as_str_or(payload.get("target_transform"), "log10") or "log10",
                feature_set=_as_str_or(payload.get("feature_set"), "all_available") or "all_available",
                hyperparameters=_as_dict_or_empty(payload.get("hyperparameters")),
                preprocessing_steps=_as_str_list(payload.get("preprocessing_steps", [])),
                rationale=rationale_text,
            )
        ctx.store.write_artifact(plan)
        ctx.trace.log_agent_message(
            event_type="model_plan_created",
            iteration=iteration,
            agent_role=AgentRole.MODEL_BUILDER,
            agent_id=plan.agent_id,
            input_artifact_ids=[feature_plan.artifact_id],
            output_artifact_ids=[plan.artifact_id],
            message_summary=plan.human_readable_summary,
        )
        return plan


class CodeGenerator:
    role_name = "CodeGenerator"

    def _variant_hyperparameters(self, model_family: str, iteration: int, variant_index: int) -> dict[str, Any]:
        seed = 1009 * int(iteration) + 37 * int(variant_index)
        selector = int(iteration) + int(variant_index)
        if model_family == "Ridge":
            alphas = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
            return {"alpha": alphas[selector % len(alphas)]}
        if model_family == "ElasticNetCV":
            l1_ratios = [[0.15, 0.5, 0.85], [0.25, 0.5, 0.75], [0.1, 0.3, 0.7, 0.9]]
            return {
                "l1_ratio": l1_ratios[selector % len(l1_ratios)],
                "n_alphas": [25, 50, 75][selector % 3],
                "eps": [1e-3, 3e-4, 1e-4][selector % 3],
                "cv": 3,
                "max_iter": 10000,
                "random_state": seed,
            }
        if model_family == "LassoCV":
            return {
                "n_alphas": [25, 50, 75][selector % 3],
                "eps": [1e-3, 3e-4, 1e-4][selector % 3],
                "cv": 3,
                "max_iter": 10000,
                "random_state": seed,
            }
        if model_family == "RandomForestRegressor":
            return {
                "n_estimators": [80, 120, 180][selector % 3],
                "max_depth": [None, 4, 7][selector % 3],
                "min_samples_leaf": [1, 2, 4][selector % 3],
                "random_state": seed,
            }
        if model_family == "GradientBoostingRegressor":
            return {
                "n_estimators": [80, 120, 180][selector % 3],
                "learning_rate": [0.03, 0.05, 0.08][selector % 3],
                "max_depth": [1, 2, 3][selector % 3],
                "min_samples_leaf": [1, 2, 4][selector % 3],
                "random_state": seed,
            }
        return {}

    def _write_spec(
        self,
        ctx: RoleContext,
        feature_plan: FeaturePlan,
        model_plan: ModelPlan,
        iteration: int,
        candidate_id: str,
        agent_id: str,
        code: str,
        started: float,
    ) -> CandidateSpec:
        candidate_dir = ctx.run_dir / "candidates"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        candidate_path = candidate_dir / f"{candidate_id}.py"
        candidate_path.write_text(code)
        spec = CandidateSpec(
            run_id=ctx.run_id,
            parent_artifact_ids=[feature_plan.artifact_id, model_plan.artifact_id],
            human_readable_summary="CodeGenerator produced a candidate Python file.",
            candidate_id=candidate_id,
            agent_id=agent_id,
            agent_role=AgentRole.LLM_CANDIDATE,
            iteration=iteration,
            candidate_path=str(candidate_path),
            candidate_name=candidate_id,
            code_sha256=_sha256_text(code),
            uses_toolbox="build_all_battery_features" in code,
            compiled_candidate=False,
            feature_plan_artifact_id=feature_plan.artifact_id,
            model_plan_artifact_id=model_plan.artifact_id,
            feature_families=list(feature_plan.feature_families),
            include_protocol_features=bool(feature_plan.include_protocol_features),
            model_family=model_plan.estimator_name or model_plan.model_family,
            target_transform=model_plan.target_transform,
            feature_set=model_plan.feature_set,
            feature_program_paths=list(feature_plan.feature_program_paths),
            feature_program_mode="table" if feature_plan.feature_program_paths else "none",
            include_feature_programs=bool(feature_plan.feature_program_paths),
            feature_family_filter=list(ctx.feature_family_filter or []),
            preprocessing=list(model_plan.preprocessing_steps),
            hyperparameters=dict(model_plan.hyperparameters),
        )
        ctx.store.write_artifact(spec)
        ctx.trace.log_agent_message(
            event_type="candidate_code_created",
            iteration=iteration,
            agent_role=AgentRole.LLM_CANDIDATE,
            agent_id=spec.agent_id,
            input_artifact_ids=[feature_plan.artifact_id, model_plan.artifact_id],
            output_artifact_ids=[spec.artifact_id],
            duration_ms=(time.perf_counter() - started) * 1000,
            message_summary=f"{spec.candidate_path} uses_toolbox={spec.uses_toolbox}",
        )
        return spec

    def _compile_spec(self, ctx: RoleContext, feature_plan: FeaturePlan, model_plan: ModelPlan, iteration: int, candidate_id: str, agent_id: str, started: float) -> CandidateSpec:
        candidate_dir = ctx.run_dir / "candidates"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        candidate_path = candidate_dir / f"{candidate_id}.py"
        spec = candidate_spec_from_plans(
            run_id=ctx.run_id,
            candidate_id=candidate_id,
            agent_id=agent_id,
            iteration=iteration,
            candidate_path=candidate_path,
            feature_plan=feature_plan,
            model_plan=model_plan,
        )
        code = compile_candidate_spec_to_python(spec, candidate_path)
        spec.code_sha256 = _sha256_text(code)
        ctx.store.write_artifact(spec)
        ctx.trace.log_agent_message(
            event_type="candidate_compiled",
            iteration=iteration,
            agent_role=AgentRole.LLM_CANDIDATE,
            agent_id=spec.agent_id,
            input_artifact_ids=[feature_plan.artifact_id, model_plan.artifact_id],
            output_artifact_ids=[spec.artifact_id],
            duration_ms=(time.perf_counter() - started) * 1000,
            message_summary=f"{spec.candidate_path} model_family={spec.model_family} feature_families={spec.feature_families}",
        )
        return spec

    def _compiled_variants(self, ctx: RoleContext, feature_plan: FeaturePlan, model_plan: ModelPlan, iteration: int) -> list[tuple[str, FeaturePlan, ModelPlan]]:
        n = max(1, int(ctx.candidates_per_iteration))
        if n == 1:
            model_family = model_plan.estimator_name or model_plan.model_family
            variant_model_plan = model_plan.model_copy(update={"hyperparameters": self._variant_hyperparameters(str(model_family), iteration, 0)})
            return [(f"role_graph_iter_{iteration:03d}", feature_plan, variant_model_plan)]
        model_families = ["Ridge", "ElasticNetCV", "LassoCV", "RandomForestRegressor", "GradientBoostingRegressor"]
        target_transforms = ["raw", "log10"]
        feature_sets = ["all_available"]
        if ctx.include_feature_programs and ctx.feature_program_paths:
            feature_sets = ["scalar_only", "curve_only", "scalar_plus_curve", "broad_physics", "all_available"]
        include_protocol_options = [bool(feature_plan.include_protocol_features)]
        if ctx.allow_protocol_features:
            include_protocol_options = [True, False]
        preferred_keys: list[tuple[str, str, bool, str]] = [
            ("Ridge", str(model_plan.target_transform or "raw"), bool(feature_plan.include_protocol_features), feature_sets[0]),
            ("Ridge", "log10", bool(feature_plan.include_protocol_features), feature_sets[min(1, len(feature_sets) - 1)]),
            ("ElasticNetCV", "raw", bool(feature_plan.include_protocol_features), feature_sets[min(2, len(feature_sets) - 1)]),
            ("RandomForestRegressor", "raw", False if ctx.allow_protocol_features else bool(feature_plan.include_protocol_features), feature_sets[min(3, len(feature_sets) - 1)]),
            ("GradientBoostingRegressor", "log10", bool(feature_plan.include_protocol_features), feature_sets[min(4, len(feature_sets) - 1)]),
            ("LassoCV", "log10", False if ctx.allow_protocol_features else bool(feature_plan.include_protocol_features), feature_sets[-1]),
        ]
        if preferred_keys:
            shift = int(iteration) % len(preferred_keys)
            preferred_keys = preferred_keys[shift:] + preferred_keys[:shift]
        all_keys = preferred_keys + [
            (model_family, target_transform, include_protocol, feature_set)
            for feature_set in feature_sets
            for target_transform in target_transforms
            for include_protocol in include_protocol_options
            for model_family in model_families
        ]
        variants: list[tuple[str, FeaturePlan, ModelPlan]] = []
        seen: set[tuple[str, str, bool, str]] = set()
        for model_family, target_transform, include_protocol, feature_set in all_keys:
            key = (model_family, target_transform, include_protocol, feature_set)
            if key in seen:
                continue
            seen.add(key)
            candidate_id = f"role_graph_iter_{iteration:03d}_variant_{len(variants):02d}"
            variant_feature_plan = feature_plan.model_copy(update={"include_protocol_features": include_protocol, "feature_set": feature_set})
            variant_model_plan = model_plan.model_copy(
                update={
                    "model_family": model_family,
                    "estimator_name": model_family,
                    "target_transform": target_transform,
                    "feature_set": feature_set,
                    "hyperparameters": self._variant_hyperparameters(model_family, iteration, len(variants)),
                }
            )
            variants.append((candidate_id, variant_feature_plan, variant_model_plan))
            if len(variants) >= n:
                return variants
        return variants

    def run(self, ctx: RoleContext, feature_plan: FeaturePlan, model_plan: ModelPlan, iteration: int) -> CandidateSpec:
        started = time.perf_counter()
        candidate_id = f"role_graph_iter_{iteration:03d}"
        if not ctx.allow_freeform_code:
            return self._compile_spec(ctx, feature_plan, model_plan, iteration, candidate_id, "code_generator", started)
        if ctx.offline:
            code = offline_candidate_code()
        else:
            response_text = _call_llm(
                self.role_name,
                code_generator_prompt(feature_plan.model_dump(mode="json"), model_plan.model_dump(mode="json")),
                model=ctx.model,
            )
            code = _strip_code_fences(response_text)
        return self._write_spec(ctx, feature_plan, model_plan, iteration, candidate_id, "code_generator", code, started)

    def run_many(self, ctx: RoleContext, feature_plan: FeaturePlan, model_plan: ModelPlan, iteration: int) -> list[CandidateSpec]:
        started = time.perf_counter()
        if ctx.allow_freeform_code:
            return [self.run(ctx, feature_plan, model_plan, iteration)]
        specs: list[CandidateSpec] = []
        for candidate_id, variant_feature_plan, variant_model_plan in self._compiled_variants(ctx, feature_plan, model_plan, iteration):
            specs.append(self._compile_spec(ctx, variant_feature_plan, variant_model_plan, iteration, candidate_id, "code_generator", started))
        return specs

    def repair(
        self,
        ctx: RoleContext,
        feature_plan: FeaturePlan,
        model_plan: ModelPlan,
        previous_candidate: CandidateSpec,
        error_message: str,
        traceback_text: str | None,
        iteration: int,
    ) -> CandidateSpec:
        started = time.perf_counter()
        candidate_id = f"role_graph_iter_{iteration:03d}_repair_1"
        if not ctx.allow_freeform_code:
            return self._compile_spec(ctx, feature_plan, model_plan, iteration, candidate_id, "code_generator_repair", started)
        previous_code = Path(previous_candidate.candidate_path).read_text()
        if ctx.offline:
            code = offline_repair_candidate_code()
        else:
            response_text = _call_llm(
                self.role_name,
                code_repair_prompt(
                    feature_plan.model_dump(mode="json"),
                    model_plan.model_dump(mode="json"),
                    previous_code,
                    error_message,
                    traceback_text,
                ),
                model=ctx.model,
            )
            code = _strip_code_fences(response_text)
        spec = self._write_spec(ctx, feature_plan, model_plan, iteration, candidate_id, "code_generator_repair", code, started)
        ctx.trace.log_event(
            event_type="candidate_repair_attempted",
            iteration=iteration,
            agent_role=AgentRole.LLM_CANDIDATE,
            input_artifact_ids=[previous_candidate.artifact_id],
            output_artifact_ids=[spec.artifact_id],
            success=True,
            extra={"error_message": error_message},
        )
        return spec


class CodeReviewer:
    role_name = "CodeReviewer"

    def run(self, ctx: RoleContext, candidate: CandidateSpec, iteration: int) -> ReviewReport:
        response = ctx.tool_client.review_candidate(
            run_id=ctx.run_id,
            run_dir=str(ctx.run_dir),
            candidate_path=candidate.candidate_path,
            iteration=iteration,
            agent_role=AgentRole.REVIEWER.value,
            input_artifact_ids=[candidate.artifact_id],
        )
        report = ReviewReport(
            run_id=ctx.run_id,
            parent_artifact_ids=[candidate.artifact_id, *response.output_artifact_ids],
            human_readable_summary=f"CodeReviewer verdict: {response.verdict}.",
            reviewer_id="code_reviewer",
            iteration=iteration,
            target_artifact_ids=[candidate.artifact_id],
            verdict=response.verdict or "unknown",
            issues=response.issues + ([f"failure_reason: {response.failure_reason}"] if response.failure_reason else []),
            recommendations=response.recommendations,
        )
        ctx.store.write_artifact(report)
        return report


class Evaluator:
    role_name = "Evaluator"

    def run(self, ctx: RoleContext, candidate: CandidateSpec, review: ReviewReport, iteration: int) -> EvaluationReport:
        response = ctx.tool_client.evaluate_candidate(
            run_id=ctx.run_id,
            run_dir=str(ctx.run_dir),
            candidate_path=candidate.candidate_path,
            metadata_path=str(ctx.metadata_path),
            cycle_summary_path=str(ctx.cycle_summary_path),
            labels_path=str(ctx.labels_path) if ctx.labels_path else None,
            split_assignments_path=str(ctx.split_assignments_path),
            split_mode=ctx.split_mode,
            max_cycle=ctx.max_cycle,
            allow_protocol_features=ctx.allow_protocol_features,
            feature_program_paths=ctx.feature_program_paths or [],
            feature_program_mode=ctx.feature_program_mode,
            include_feature_programs=ctx.include_feature_programs,
            feature_family_filter=ctx.feature_family_filter or [],
            iteration=iteration,
            agent_role=AgentRole.EVALUATOR.value,
            input_artifact_ids=[candidate.artifact_id, review.artifact_id],
        )
        metrics = response.metrics or {}
        report = EvaluationReport(
            run_id=ctx.run_id,
            parent_artifact_ids=[candidate.artifact_id, review.artifact_id, *response.output_artifact_ids],
            human_readable_summary=f"Evaluator completed with success={response.success}.",
            candidate_id=candidate.candidate_id,
            candidate_path=candidate.candidate_path,
            agent_id="evaluator",
            candidate_name=candidate.candidate_name,
            iteration=iteration,
            split_mode=ctx.split_mode,
            locked_batch9_validation_run=False,
            success=response.success,
            rmse=metrics.get("rmse"),
            mae=metrics.get("mae"),
            r2=metrics.get("r2"),
            spearman=metrics.get("spearman"),
            kendall=metrics.get("kendall"),
            mape=metrics.get("mape"),
            test_error_pct=metrics.get("test_error_pct"),
            pgr=metrics.get("pgr_author_model"),
            failure_reason=response.failure_reason or response.error_message,
            traceback=response.traceback,
            prediction_path=response.prediction_path,
            extra_metrics={k: v for k, v in metrics.items() if k not in {"rmse", "mae", "r2", "spearman", "kendall", "mape", "test_error_pct", "pgr_author_model"}},
        )
        ctx.store.write_artifact(report)
        return report


class ScientistCritic:
    role_name = "ScientistCritic"

    def run(self, ctx: RoleContext, review: ReviewReport, evaluation: EvaluationReport, iteration: int) -> CritiqueReport:
        if ctx.offline:
            strengths = ["The candidate used the shared author-inspired battery feature toolbox."]
            weaknesses = []
            if not evaluation.success:
                weaknesses.append(evaluation.failure_reason or "Candidate evaluation failed.")
            elif evaluation.rmse is not None:
                weaknesses.append("This is surrogate-search validation only; locked Batch 9 is not part of this role graph run.")
            report = CritiqueReport(
                run_id=ctx.run_id,
                parent_artifact_ids=[review.artifact_id, evaluation.artifact_id],
                human_readable_summary="ScientistCritic summarized the candidate evaluation.",
                critic_id="scientist_critic",
                iteration=iteration,
                target_artifact_ids=[review.artifact_id, evaluation.artifact_id],
                strengths=strengths,
                weaknesses=weaknesses,
                proposed_next_steps=[
                    "Compare protocol and batch splits before claiming generalization.",
                    "Run locked Batch 9 validation outside the search loop for final assessment.",
                ],
            )
        else:
            payload = _llm_json(
                self.role_name,
                scientist_critic_prompt(review.model_dump(mode="json"), evaluation.model_dump(mode="json")),
                "CritiqueReport keys: critic_id, strengths, weaknesses, proposed_next_steps",
                model=ctx.model,
            )
            report = CritiqueReport(
                run_id=ctx.run_id,
                parent_artifact_ids=[review.artifact_id, evaluation.artifact_id],
                human_readable_summary="ScientistCritic produced a critique report.",
                critic_id=_as_str_or(payload.get("critic_id"), "scientist_critic") or "scientist_critic",
                iteration=iteration,
                target_artifact_ids=[review.artifact_id, evaluation.artifact_id],
                strengths=_as_str_list(payload.get("strengths", [])),
                weaknesses=_as_str_list(payload.get("weaknesses", [])),
                proposed_next_steps=_as_str_list(payload.get("proposed_next_steps", [])),
            )
        ctx.store.write_artifact(report)
        ctx.trace.log_agent_message(
            event_type="critique_created",
            iteration=iteration,
            agent_role=AgentRole.CRITIC,
            agent_id=report.critic_id,
            input_artifact_ids=[review.artifact_id, evaluation.artifact_id],
            output_artifact_ids=[report.artifact_id],
            message_summary=report.human_readable_summary,
        )
        return report


_SHORTLIST_REQUIRED_KEYS: tuple[str, ...] = (
    "candidate_id",
    "source_run_id",
    "candidate_path",
    "feature_program_path",
    "processed_dir",
    "recipe",
    "split_mode",
    "split_seed",
    "model_family",
    "feature_set",
    "target_transform",
    "surrogate_rmse",
    "surrogate_mae",
    "surrogate_mape",
    "surrogate_spearman",
    "rank_in_shortlist",
)


def _coerce_shortlist_entry(raw: Any, rank_hint: int) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    entry: dict[str, Any] = dict(raw)
    for key in _SHORTLIST_REQUIRED_KEYS:
        if key not in entry:
            if key == "feature_program_path":
                entry[key] = None
            elif key == "rank_in_shortlist":
                entry[key] = rank_hint
            else:
                return None
    if not isinstance(entry.get("candidate_id"), str) or not entry["candidate_id"].strip():
        return None
    if not isinstance(entry.get("source_run_id"), str) or not entry["source_run_id"].strip():
        return None
    coerced_rank = _as_int_or(entry.get("rank_in_shortlist"), rank_hint)
    entry["rank_in_shortlist"] = coerced_rank if coerced_rank is not None else rank_hint
    coerced_seed = _as_int_or(entry.get("split_seed"), None)
    if coerced_seed is None:
        return None
    entry["split_seed"] = coerced_seed
    for metric_key in ("surrogate_rmse", "surrogate_mae", "surrogate_mape", "surrogate_spearman"):
        value = entry.get(metric_key)
        if value is None:
            return None
        try:
            entry[metric_key] = float(value)
        except (TypeError, ValueError):
            return None
    return entry


class ChampionAggregator:
    role_name = "ChampionAggregator"

    def run(
        self,
        ctx: RoleContext,
        severson_summary: dict[str, Any],
        k: int,
        selection_criteria: str,
        parent_artifact_ids: list[str] | None = None,
    ) -> ChampionShortlist:
        parent_ids = list(parent_artifact_ids or [])
        if int(k) <= 0:
            raise ValueError("ChampionAggregator requires k > 0")
        if ctx.offline:
            eligible = severson_summary.get("top_eligible_configs") or []
            if not isinstance(eligible, list):
                eligible = []
            scored: list[tuple[float, dict[str, Any]]] = []
            for raw in eligible:
                if not isinstance(raw, dict):
                    continue
                rmse_value = raw.get("surrogate_rmse")
                try:
                    rmse_float = float(rmse_value)
                except (TypeError, ValueError):
                    continue
                scored.append((rmse_float, raw))
            scored.sort(key=lambda item: item[0])
            shortlist: list[dict[str, Any]] = []
            for rank_idx, (_, raw) in enumerate(scored, start=1):
                entry = _coerce_shortlist_entry({**raw, "rank_in_shortlist": rank_idx}, rank_idx)
                if entry is None:
                    continue
                shortlist.append(entry)
                if len(shortlist) >= int(k):
                    break
            if len(shortlist) < int(k):
                raise RuntimeError(
                    f"ChampionAggregator offline fallback could not assemble {k} valid entries "
                    f"from top_eligible_configs (got {len(shortlist)})."
                )
            selection_criteria_used = "offline deterministic argmin(surrogate_rmse)"
            rationale_text = (
                f"Offline deterministic fallback selected the top {k} candidates by surrogate_rmse."
            )
            agent_id = "champion_aggregator"
        else:
            schema_hint = (
                "ChampionShortlist keys: agent_id, k, selection_criteria, rationale, "
                "shortlist (list of dicts with candidate_id, source_run_id, candidate_path, "
                "feature_program_path, processed_dir, recipe, split_mode, split_seed, "
                "model_family, feature_set, target_transform, surrogate_rmse, surrogate_mae, "
                "surrogate_mape, surrogate_spearman, rank_in_shortlist)"
            )
            payload = _llm_json(
                self.role_name,
                champion_aggregator_prompt(severson_summary, int(k), selection_criteria),
                schema_hint,
                model=ctx.model,
            )
            raw_shortlist = payload.get("shortlist", [])
            if not isinstance(raw_shortlist, list):
                raw_shortlist = []
            shortlist = []
            for rank_idx, raw in enumerate(raw_shortlist, start=1):
                entry = _coerce_shortlist_entry(raw, rank_idx)
                if entry is None:
                    ctx.trace.log_event(
                        event_type="champion_shortlist_entry_dropped",
                        agent_role=AgentRole.CRITIC,
                        success=False,
                        error_type="invalid_shortlist_entry",
                        error_message=f"Dropped shortlist entry at position {rank_idx}: missing or malformed required keys.",
                    )
                    continue
                shortlist.append(entry)
            if len(shortlist) > int(k):
                shortlist = shortlist[: int(k)]
            if len(shortlist) < int(k):
                raise RuntimeError(
                    f"ChampionAggregator returned only {len(shortlist)} valid entries; expected {k}."
                )
            selection_criteria_used = _as_str_or(payload.get("selection_criteria"), selection_criteria) or selection_criteria
            rationale_text = _as_str_or(payload.get("rationale"), None) or ""
            agent_id = _as_str_or(payload.get("agent_id"), "champion_aggregator") or "champion_aggregator"

        artifact = ChampionShortlist(
            run_id=ctx.run_id,
            parent_artifact_ids=parent_ids,
            human_readable_summary=(
                rationale_text
                if rationale_text
                else f"ChampionAggregator produced a top-{k} Severson-only shortlist."
            ),
            agent_id=agent_id,
            iteration=None,
            k=int(k),
            selection_criteria=selection_criteria_used,
            rationale=rationale_text or None,
            shortlist=shortlist,
        )
        ctx.store.write_artifact(artifact)
        ctx.trace.log_agent_message(
            event_type="champion_shortlist_created",
            iteration=None,
            agent_role=AgentRole.CRITIC,
            agent_id=artifact.agent_id,
            input_artifact_ids=parent_ids,
            output_artifact_ids=[artifact.artifact_id],
            message_summary=f"Severson-only top-{k} shortlist with {len(shortlist)} entries.",
        )
        return artifact


class ChampionAdjudicator:
    role_name = "ChampionAdjudicator"

    def run(
        self,
        ctx: RoleContext,
        shortlist_with_batch9: dict[str, Any],
        selection_criteria: str,
        parent_artifact_ids: list[str] | None = None,
    ) -> ChampionDecision:
        parent_ids = list(parent_artifact_ids or [])
        raw_input_shortlist = shortlist_with_batch9.get("shortlist", [])
        if not isinstance(raw_input_shortlist, list):
            raw_input_shortlist = []
        input_shortlist: list[dict[str, Any]] = [dict(item) for item in raw_input_shortlist if isinstance(item, dict)]

        if ctx.offline:
            scored: list[tuple[float, dict[str, Any]]] = []
            for entry in input_shortlist:
                value = entry.get("batch9_rmse")
                try:
                    rmse_float = float(value)
                except (TypeError, ValueError):
                    continue
                scored.append((rmse_float, entry))
            scored.sort(key=lambda item: item[0])
            if not scored:
                raise RuntimeError(
                    "ChampionAdjudicator offline fallback found no shortlist entry with a numeric batch9_rmse."
                )
            final_champion = dict(scored[0][1])
            runner_up = dict(scored[1][1]) if len(scored) > 1 else None
            selection_criteria_used = "offline deterministic argmin(batch9_rmse)"
            rationale_text = (
                "Offline deterministic fallback selected the shortlist entry with the lowest batch9_rmse; "
                "runner_up is the second-best."
            )
            agent_id = "champion_adjudicator"
            shortlist_evaluated = input_shortlist
        else:
            schema_hint = (
                "ChampionDecision keys: agent_id, selection_criteria, rationale, "
                "final_champion (dict with at least candidate_id and source_run_id, plus batch9_rmse, batch9_mae, batch9_mape, batch9_spearman, batch9_pgr, "
                "and the equivalent secondary_test_rmse, secondary_test_mae, secondary_test_mape, secondary_test_spearman, secondary_test_pgr aliases if available), "
                "runner_up (same shape or null), shortlist_evaluated (list of dicts)"
            )
            payload = _llm_json(
                self.role_name,
                champion_adjudicator_prompt(shortlist_with_batch9, selection_criteria),
                schema_hint,
                model=ctx.model,
            )
            raw_final = payload.get("final_champion")
            if not isinstance(raw_final, dict):
                raise RuntimeError("ChampionAdjudicator did not return a final_champion object.")
            final_champion = dict(raw_final)
            final_candidate_id = final_champion.get("candidate_id")
            final_source_run_id = final_champion.get("source_run_id")
            if not isinstance(final_candidate_id, str) or not final_candidate_id.strip():
                raise RuntimeError("ChampionAdjudicator final_champion missing candidate_id.")
            if not isinstance(final_source_run_id, str) or not final_source_run_id.strip():
                raise RuntimeError("ChampionAdjudicator final_champion missing source_run_id.")

            raw_runner_up = payload.get("runner_up")
            if isinstance(raw_runner_up, dict) and raw_runner_up:
                runner_up = dict(raw_runner_up)
            else:
                runner_up = None
                if raw_runner_up not in (None, {}, ""):
                    ctx.trace.log_event(
                        event_type="champion_decision_runner_up_dropped",
                        agent_role=AgentRole.CRITIC,
                        success=False,
                        error_type="invalid_runner_up",
                        error_message="runner_up was present but malformed; coerced to None.",
                    )

            raw_evaluated = payload.get("shortlist_evaluated", [])
            if isinstance(raw_evaluated, list) and raw_evaluated:
                shortlist_evaluated = [dict(item) for item in raw_evaluated if isinstance(item, dict)]
            else:
                shortlist_evaluated = input_shortlist
            selection_criteria_used = _as_str_or(payload.get("selection_criteria"), selection_criteria) or selection_criteria
            rationale_text = _as_str_or(payload.get("rationale"), None) or ""
            agent_id = _as_str_or(payload.get("agent_id"), "champion_adjudicator") or "champion_adjudicator"

        artifact = ChampionDecision(
            run_id=ctx.run_id,
            parent_artifact_ids=parent_ids,
            human_readable_summary=(
                rationale_text
                if rationale_text
                else f"ChampionAdjudicator selected final champion {final_champion.get('candidate_id')}."
            ),
            agent_id=agent_id,
            iteration=None,
            selection_criteria=selection_criteria_used,
            rationale=rationale_text or None,
            final_champion=final_champion,
            runner_up=runner_up,
            shortlist_evaluated=shortlist_evaluated,
        )
        ctx.store.write_artifact(artifact)
        ctx.trace.log_agent_message(
            event_type="champion_decision_created",
            iteration=None,
            agent_role=AgentRole.CRITIC,
            agent_id=artifact.agent_id,
            input_artifact_ids=parent_ids,
            output_artifact_ids=[artifact.artifact_id],
            message_summary=(
                f"Final champion candidate_id={final_champion.get('candidate_id')} "
                f"source_run_id={final_champion.get('source_run_id')}."
            ),
        )
        return artifact


ROLE_SEQUENCE = [
    "DatasetProfiler",
    "FeatureScientist",
    "ModelArchitect",
    "CodeGenerator",
    "CodeReviewer",
    "Evaluator",
    "ScientistCritic",
]
