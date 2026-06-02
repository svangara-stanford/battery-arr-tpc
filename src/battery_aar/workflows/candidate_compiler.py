from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from battery_aar.workflows.schemas import CandidateSpec, FeaturePlan, ModelPlan

SUPPORTED_MODEL_FAMILIES = {
    "Ridge",
    "ElasticNetCV",
    "LassoCV",
    "RandomForestRegressor",
    "GradientBoostingRegressor",
}

SUPPORTED_TARGET_TRANSFORMS = {"raw", "log10"}


def _model_name(value: str | None) -> str:
    text = str(value or "").strip()
    lowered = re.sub(r"[^a-z0-9]+", "", text.lower())
    aliases = {
        "ridge": "Ridge",
        "ridgeregression": "Ridge",
        "linearregularized": "Ridge",
        "regularizedregression": "Ridge",
        "elasticnet": "ElasticNetCV",
        "elasticnetcv": "ElasticNetCV",
        "lasso": "LassoCV",
        "lassocv": "LassoCV",
        "randomforest": "RandomForestRegressor",
        "randomforestregressor": "RandomForestRegressor",
        "forest": "RandomForestRegressor",
        "gradientboosting": "GradientBoostingRegressor",
        "gradientboostingregressor": "GradientBoostingRegressor",
        "gbr": "GradientBoostingRegressor",
    }
    if text in SUPPORTED_MODEL_FAMILIES:
        return text
    return aliases.get(lowered, "Ridge")


def normalize_model_family(model_family: str | None, estimator_name: str | None = None) -> str:
    family = _model_name(model_family)
    if family == "Ridge" and estimator_name:
        return _model_name(estimator_name)
    return family


def normalize_target_transform(value: str | None) -> str:
    text = str(value or "raw").strip().lower()
    aliases = {
        "": "raw",
        "none": "raw",
        "identity": "raw",
        "linear": "raw",
        "raw": "raw",
        "log": "log10",
        "log10": "log10",
        "log10_cycle_life": "log10",
    }
    normalized = aliases.get(text, text)
    if normalized not in SUPPORTED_TARGET_TRANSFORMS:
        raise ValueError(f"Unsupported target_transform={value!r}; expected one of {sorted(SUPPORTED_TARGET_TRANSFORMS)}")
    return normalized


def _clean_hyperparameters(model_family: str, hyperparameters: dict[str, Any] | None) -> dict[str, Any]:
    raw = hyperparameters if isinstance(hyperparameters, dict) else {}
    allowed = {
        "Ridge": {"alpha", "fit_intercept", "positive", "random_state"},
        "ElasticNetCV": {"l1_ratio", "eps", "n_alphas", "alphas", "fit_intercept", "max_iter", "cv", "random_state"},
        "LassoCV": {"eps", "n_alphas", "alphas", "fit_intercept", "max_iter", "cv", "random_state"},
        "RandomForestRegressor": {"n_estimators", "max_depth", "min_samples_leaf", "min_samples_split", "max_features", "random_state"},
        "GradientBoostingRegressor": {"n_estimators", "learning_rate", "max_depth", "min_samples_leaf", "subsample", "random_state"},
    }[model_family]
    cleaned: dict[str, Any] = {}
    for key, value in raw.items():
        if key in allowed and (isinstance(value, (str, int, float, bool, list, tuple)) or value is None):
            cleaned[key] = value
    if model_family in {"RandomForestRegressor", "GradientBoostingRegressor", "ElasticNetCV", "LassoCV"}:
        cleaned.setdefault("random_state", 0)
    if model_family == "RandomForestRegressor":
        cleaned.setdefault("n_estimators", 100)
        cleaned.setdefault("min_samples_leaf", 1)
    if model_family == "GradientBoostingRegressor":
        cleaned.setdefault("n_estimators", 100)
        cleaned.setdefault("learning_rate", 0.05)
        cleaned.setdefault("max_depth", 2)
    if model_family == "Ridge":
        cleaned.setdefault("alpha", 1.0)
    if model_family in {"ElasticNetCV", "LassoCV"}:
        cleaned.setdefault("cv", 3)
        cleaned.setdefault("max_iter", 10000)
    return cleaned


def candidate_spec_from_plans(
    *,
    run_id: str,
    candidate_id: str,
    agent_id: str,
    iteration: int,
    candidate_path: str | Path,
    feature_plan: FeaturePlan,
    model_plan: ModelPlan,
    parent_artifact_ids: list[str] | None = None,
) -> CandidateSpec:
    model_family = normalize_model_family(model_plan.model_family, model_plan.estimator_name)
    target_transform = normalize_target_transform(model_plan.target_transform)
    return CandidateSpec(
        run_id=run_id,
        parent_artifact_ids=parent_artifact_ids or [feature_plan.artifact_id, model_plan.artifact_id],
        human_readable_summary=f"Trusted compiled sklearn candidate using {model_family}.",
        candidate_id=candidate_id,
        agent_id=agent_id,
        agent_role="llm_candidate",
        iteration=iteration,
        candidate_path=str(candidate_path),
        candidate_name=candidate_id,
        uses_toolbox=True,
        compiled_candidate=True,
        declared_dependencies=["numpy", "pandas", "scikit-learn", "battery_aar"],
        feature_plan_artifact_id=feature_plan.artifact_id,
        model_plan_artifact_id=model_plan.artifact_id,
        feature_families=list(feature_plan.feature_families),
        include_protocol_features=bool(feature_plan.include_protocol_features),
        model_family=model_family,
        target_transform=target_transform,
        preprocessing=list(model_plan.preprocessing_steps) or ["drop_all_nan_columns", "SimpleImputer(strategy='median')"],
        hyperparameters=_clean_hyperparameters(model_family, model_plan.hyperparameters),
    )


def compile_candidate_spec_to_python(candidate_spec: CandidateSpec, output_path: str | Path | None = None) -> str:
    """Compile a declarative CandidateSpec into trusted executable candidate code."""
    model_family = normalize_model_family(candidate_spec.model_family)
    target_transform = normalize_target_transform(candidate_spec.target_transform)
    hyperparameters = _clean_hyperparameters(model_family, candidate_spec.hyperparameters)
    constants = {
        "feature_families": list(candidate_spec.feature_families),
        "include_protocol_features": bool(candidate_spec.include_protocol_features),
        "model_family": model_family,
        "target_transform": target_transform,
        "preprocessing": list(candidate_spec.preprocessing),
        "hyperparameters": hyperparameters,
    }
    code = f'''# Auto-generated by Open Battery Agents trusted candidate compiler.
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, LassoCV, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from battery_aar.features.battery_lifetime_features import build_all_battery_features

FEATURE_FAMILIES = {constants["feature_families"]!r}
INCLUDE_PROTOCOL_FEATURES = {constants["include_protocol_features"]!r}
MODEL_FAMILY = {constants["model_family"]!r}
TARGET_TRANSFORM = {constants["target_transform"]!r}
PREPROCESSING = {constants["preprocessing"]!r}
HYPERPARAMETERS = {constants["hyperparameters"]!r}
IDENTIFIER_COLUMNS = {{
    "row_id", "cell_id", "batch_id", "original_cell_id", "original_batch_id",
    "barcode", "source_path", "file_path", "filename", "path", "channel",
    "cycle_life", "Lifetime", "protocol_readable", "policy_readable", "y",
}}


def _make_estimator():
    params = dict(HYPERPARAMETERS)
    if MODEL_FAMILY == "Ridge":
        return Ridge(**params)
    if MODEL_FAMILY == "ElasticNetCV":
        return ElasticNetCV(**params)
    if MODEL_FAMILY == "LassoCV":
        return LassoCV(**params)
    if MODEL_FAMILY == "RandomForestRegressor":
        return RandomForestRegressor(**params)
    if MODEL_FAMILY == "GradientBoostingRegressor":
        return GradientBoostingRegressor(**params)
    return Ridge(alpha=1.0)


def _feature_frame(metadata, cycle_summary, config, feature_columns=None):
    max_cycle = int(config.get("max_cycle", 100))
    allow_protocol = bool(config.get("allow_protocol_features", INCLUDE_PROTOCOL_FEATURES))
    include_protocol = bool(INCLUDE_PROTOCOL_FEATURES and allow_protocol)
    ids = metadata[["row_id"]].drop_duplicates().copy()
    features = build_all_battery_features(
        metadata,
        cycle_summary,
        max_cycle=max_cycle,
        include_protocol=include_protocol,
    )
    key_name = features.index.name or "row_id"
    features = features.reset_index().rename(columns={{key_name: "row_id"}})
    if "row_id" not in features.columns:
        features.insert(0, "row_id", ids["row_id"].to_numpy())
    features = ids.merge(features, on="row_id", how="left")
    X = features.drop(columns=[col for col in IDENTIFIER_COLUMNS if col in features.columns], errors="ignore")
    for col in list(X.columns):
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    if feature_columns is None:
        X = X.dropna(axis=1, how="all")
        feature_columns = list(X.columns)
    else:
        for col in feature_columns:
            if col not in X.columns:
                X[col] = np.nan
        X = X[feature_columns]
    if X.shape[1] == 0:
        X["_constant_feature"] = 1.0
        feature_columns = ["_constant_feature"]
    return ids, X, list(feature_columns)


def fit(train_metadata, train_cycle_summary, train_labels, config):
    ids, X, feature_columns = _feature_frame(train_metadata, train_cycle_summary, config)
    labels = train_labels[["row_id", "y"]].copy()
    y = ids.merge(labels, on="row_id", how="left")["y"].to_numpy(float)
    if not np.isfinite(y).all():
        raise ValueError("compiled candidate could not align finite training labels by row_id")
    if TARGET_TRANSFORM == "log10":
        if np.any(y <= 0):
            raise ValueError("log10 target_transform requires positive finite y")
        train_target = np.log10(y)
    else:
        train_target = y
    estimator = _make_estimator()
    if MODEL_FAMILY in {{"Ridge", "ElasticNetCV", "LassoCV"}}:
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), estimator)
    else:
        model = make_pipeline(SimpleImputer(strategy="median"), estimator)
    model.fit(X, train_target)
    return {{"model": model, "feature_columns": feature_columns, "target_transform": TARGET_TRANSFORM}}


def predict(model, test_metadata, test_cycle_summary, config):
    ids, X, _ = _feature_frame(test_metadata, test_cycle_summary, config, model["feature_columns"])
    y_pred_model = model["model"].predict(X)
    if model.get("target_transform", TARGET_TRANSFORM) == "log10":
        y_pred = np.power(10.0, y_pred_model)
    else:
        y_pred = y_pred_model
    return pd.DataFrame({{"row_id": ids["row_id"].to_numpy(), "y_pred": y_pred}})
'''
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code)
    return code
