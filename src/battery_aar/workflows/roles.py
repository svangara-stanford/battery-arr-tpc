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
from battery_aar.workflows.role_prompts import (
    ROLE_SYSTEM_PROMPTS,
    code_generator_prompt,
    feature_scientist_prompt,
    model_architect_prompt,
    schema_repair_prompt,
    scientist_critic_prompt,
)
from battery_aar.workflows.schemas import AgentRole, CandidateSpec, CritiqueReport, EvaluationReport, FeaturePlan, ModelPlan, ReviewReport
from battery_aar.workflows.trace import TraceLogger


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines)
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


def offline_candidate_code() -> str:
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
                max_cycle=ctx.max_cycle,
                rationale="Offline deterministic plan uses the battery feature toolbox without author coefficients.",
                constraints=["row_id and cell_id are join keys only", "batch 9 is not used during surrogate search"],
            )
        else:
            payload = _llm_json(
                self.role_name,
                feature_scientist_prompt(dataset_profile, probe),
                "FeaturePlan keys: agent_id, feature_families, selected_columns, include_protocol_features, max_cycle, rationale, constraints",
                model=ctx.model,
            )
            plan = FeaturePlan(
                run_id=ctx.run_id,
                parent_artifact_ids=all_parent_ids,
                human_readable_summary=str(payload.get("rationale") or "LLM FeatureScientist plan."),
                agent_id=str(payload.get("agent_id") or "feature_scientist"),
                iteration=iteration,
                feature_families=list(map(str, payload.get("feature_families", []))),
                selected_columns=list(map(str, payload.get("selected_columns", []))),
                include_protocol_features=bool(payload.get("include_protocol_features", ctx.allow_protocol_features)),
                max_cycle=int(payload.get("max_cycle") or ctx.max_cycle),
                rationale=payload.get("rationale"),
                constraints=list(map(str, payload.get("constraints", []))),
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
                hyperparameters={"alpha": 1.0},
                preprocessing_steps=["drop_all_nan_columns", "SimpleImputer(strategy='median')", "StandardScaler"],
                rationale="Small surrogate datasets need a stable regularized baseline before more flexible models.",
            )
        else:
            payload = _llm_json(
                self.role_name,
                model_architect_prompt(feature_plan.model_dump(mode="json"), dataset_profile),
                "ModelPlan keys: agent_id, model_family, estimator_name, hyperparameters, preprocessing_steps, rationale",
                model=ctx.model,
            )
            plan = ModelPlan(
                run_id=ctx.run_id,
                parent_artifact_ids=[feature_plan.artifact_id],
                human_readable_summary=str(payload.get("rationale") or "LLM ModelArchitect plan."),
                agent_id=str(payload.get("agent_id") or "model_architect"),
                iteration=iteration,
                model_family=str(payload.get("model_family") or "regularized_regression"),
                estimator_name=payload.get("estimator_name"),
                hyperparameters=payload.get("hyperparameters") if isinstance(payload.get("hyperparameters"), dict) else {},
                preprocessing_steps=list(map(str, payload.get("preprocessing_steps", []))),
                rationale=payload.get("rationale"),
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

    def run(self, ctx: RoleContext, feature_plan: FeaturePlan, model_plan: ModelPlan, iteration: int) -> CandidateSpec:
        started = time.perf_counter()
        candidate_id = f"role_graph_iter_{iteration:03d}"
        candidate_dir = ctx.run_dir / "candidates"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        candidate_path = candidate_dir / f"{candidate_id}.py"
        if ctx.offline:
            code = offline_candidate_code()
            response_text = "offline deterministic toolbox Ridge candidate"
        else:
            response_text = _call_llm(
                self.role_name,
                code_generator_prompt(feature_plan.model_dump(mode="json"), model_plan.model_dump(mode="json")),
                model=ctx.model,
            )
            code = _strip_code_fences(response_text)
        candidate_path.write_text(code)
        spec = CandidateSpec(
            run_id=ctx.run_id,
            parent_artifact_ids=[feature_plan.artifact_id, model_plan.artifact_id],
            human_readable_summary="CodeGenerator produced a candidate Python file.",
            candidate_id=candidate_id,
            agent_id="code_generator",
            agent_role=AgentRole.LLM_CANDIDATE,
            iteration=iteration,
            candidate_path=str(candidate_path),
            candidate_name=candidate_id,
            code_sha256=_sha256_text(code),
            uses_toolbox="build_all_battery_features" in code,
            feature_plan_artifact_id=feature_plan.artifact_id,
            model_plan_artifact_id=model_plan.artifact_id,
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
            issues=response.issues,
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
            max_cycle=ctx.max_cycle,
            allow_protocol_features=ctx.allow_protocol_features,
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
            pgr=metrics.get("pgr_author_model"),
            failure_reason=response.failure_reason or response.error_message,
            traceback=None,
            extra_metrics={k: v for k, v in metrics.items() if k not in {"rmse", "mae", "r2", "spearman", "kendall", "pgr_author_model"}},
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
                critic_id=str(payload.get("critic_id") or "scientist_critic"),
                iteration=iteration,
                target_artifact_ids=[review.artifact_id, evaluation.artifact_id],
                strengths=list(map(str, payload.get("strengths", []))),
                weaknesses=list(map(str, payload.get("weaknesses", []))),
                proposed_next_steps=list(map(str, payload.get("proposed_next_steps", []))),
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


ROLE_SEQUENCE = [
    "DatasetProfiler",
    "FeatureScientist",
    "ModelArchitect",
    "CodeGenerator",
    "CodeReviewer",
    "Evaluator",
    "ScientistCritic",
]
