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
SUPPORTED_FEATURE_SETS = {
    "scalar_only",
    "curve_only",
    "scalar_plus_curve",
    "broad_physics",
    "protocol_only",
    "all_available",
}


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


def _collapse(text: str) -> str:
    return re.sub(r"[\s\-]+", "_", text.strip().lower())


def normalize_target_transform(value: str | None) -> str:
    raw_text = str(value or "raw")
    text = _collapse(raw_text)
    aliases = {
        "": "raw",
        "none": "raw",
        "null": "raw",
        "identity": "raw",
        "linear": "raw",
        "raw": "raw",
        "untransformed": "raw",
        "no_transform": "raw",
        "no_transformation": "raw",
        "cycle_life": "raw",
        "cyclelife": "raw",
        "y": "raw",
        "target": "raw",
        "log": "log10",
        "log10": "log10",
        "log_10": "log10",
        "log_target": "log10",
        "log10_target": "log10",
        "log10_cycle_life": "log10",
        "log10cyclelife": "log10",
        "log_cycle_life": "log10",
        "log_y": "log10",
        "log_transform": "log10",
        "logtransform": "log10",
        "log_transformed": "log10",
    }
    normalized = aliases.get(text, text)
    if normalized not in SUPPORTED_TARGET_TRANSFORMS:
        # Fail soft to "raw" for unknown wording during Track A discovery.
        return "raw"
    return normalized


def normalize_feature_set(value: str | None) -> str:
    raw_text = str(value or "all_available")
    text = _collapse(raw_text)
    aliases = {
        "": "all_available",
        "none": "all_available",
        "null": "all_available",
        "default": "all_available",
        "all": "all_available",
        "all_features": "all_available",
        "all_available": "all_available",
        "all_available_features": "all_available",
        "everything": "all_available",
        "full": "all_available",
        "scalar": "scalar_only",
        "scalars": "scalar_only",
        "scalar_only": "scalar_only",
        "scalaronly": "scalar_only",
        "scalar_features": "scalar_only",
        "scalar_features_only": "scalar_only",
        "summary": "scalar_only",
        "summary_only": "scalar_only",
        "curve": "curve_only",
        "curves": "curve_only",
        "curve_only": "curve_only",
        "curveonly": "curve_only",
        "curve_features": "curve_only",
        "curve_shape": "curve_only",
        "curve_shape_only": "curve_only",
        "scalar_curve": "scalar_plus_curve",
        "scalar_and_curve": "scalar_plus_curve",
        "scalars_and_curves": "scalar_plus_curve",
        "scalar_plus_curve": "scalar_plus_curve",
        "scalarpluscurve": "scalar_plus_curve",
        "scalar_curves": "scalar_plus_curve",
        "broad": "broad_physics",
        "broad_physics": "broad_physics",
        "broadphysics": "broad_physics",
        "physics": "broad_physics",
        "physics_aware": "broad_physics",
        "physicsaware": "broad_physics",
        "physics_based": "broad_physics",
        "physics_features": "broad_physics",
        "protocol": "protocol_only",
        "protocols": "protocol_only",
        "protocol_only": "protocol_only",
        "protocol_features": "protocol_only",
        "protocol_features_only": "protocol_only",
    }
    normalized = aliases.get(text, text)
    if normalized not in SUPPORTED_FEATURE_SETS:
        # Fail soft to all_available during Track A so unfamiliar LLM wording
        # does not crash the entire workflow.
        return "all_available"
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


def _feature_metadata_path_for(program_path: str | Path) -> Path:
    """Resolve a program-table dir/file to its feature_metadata.csv."""
    p = Path(program_path)
    if p.is_dir():
        return p / "feature_metadata.csv"
    if p.name == "feature_table.csv":
        return p.parent / "feature_metadata.csv"
    return p.parent / "feature_metadata.csv"


def _resolve_operator_column_allowlist(feature_plan: FeaturePlan) -> list[str]:
    """Compile the plan's operator specs into an explicit column allowlist.

    Tier B: map each operator spec to the columns it selects in the plan's
    materialized program table (by operator_type / y_signal / aggregation),
    honoring feature_budget. Returns [] when there are no operators, no program
    table, or the metadata can't be read -- callers then fall back to the
    existing family/feature_set behavior.
    """
    if not feature_plan.feature_operators or not feature_plan.feature_program_paths:
        return []
    try:
        import pandas as pd

        from battery_aar.features.operator_spec_selector import select_columns_for_specs
    except Exception:
        return []
    frames = []
    for program_path in feature_plan.feature_program_paths:
        meta_path = _feature_metadata_path_for(program_path)
        if Path(meta_path).exists():
            try:
                frames.append(pd.read_csv(meta_path))
            except Exception:
                continue
    if not frames:
        return []
    metadata = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
    specs = [
        {
            "operator_type": spec.operator_type,
            "params": dict(spec.params or {}),
        }
        for spec in feature_plan.feature_operators
    ]
    selection = select_columns_for_specs(metadata, specs, budget=feature_plan.feature_budget)
    return list(selection.columns)


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
    feature_set = normalize_feature_set(model_plan.feature_set)
    # Tier B: compile the FeatureScientist's operator specs into an explicit
    # column allowlist so the agent's budgeted choices actually drive the model.
    column_allowlist = _resolve_operator_column_allowlist(feature_plan)
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
        feature_set=feature_set,
        feature_program_paths=list(feature_plan.feature_program_paths),
        feature_program_mode="table" if feature_plan.feature_program_paths else "none",
        include_feature_programs=bool(feature_plan.feature_program_paths),
        feature_family_filter=[],
        feature_column_allowlist=column_allowlist,
        preprocessing=list(model_plan.preprocessing_steps) or ["drop_all_nan_columns", "SimpleImputer(strategy='median')"],
        hyperparameters=_clean_hyperparameters(model_family, model_plan.hyperparameters),
    )


def compile_candidate_spec_to_python(candidate_spec: CandidateSpec, output_path: str | Path | None = None) -> str:
    """Compile a declarative CandidateSpec into trusted executable candidate code."""
    model_family = normalize_model_family(candidate_spec.model_family)
    target_transform = normalize_target_transform(candidate_spec.target_transform)
    feature_set = normalize_feature_set(candidate_spec.feature_set)
    hyperparameters = _clean_hyperparameters(model_family, candidate_spec.hyperparameters)
    constants = {
        "feature_families": list(candidate_spec.feature_families),
        "include_protocol_features": bool(candidate_spec.include_protocol_features),
        "model_family": model_family,
        "target_transform": target_transform,
        "feature_set": feature_set,
        "feature_program_paths": list(candidate_spec.feature_program_paths),
        "feature_program_mode": candidate_spec.feature_program_mode,
        "include_feature_programs": bool(candidate_spec.include_feature_programs),
        "feature_family_filter": list(candidate_spec.feature_family_filter),
        "feature_column_allowlist": list(candidate_spec.feature_column_allowlist),
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
FEATURE_SET = {constants["feature_set"]!r}
FEATURE_PROGRAM_PATHS = {constants["feature_program_paths"]!r}
FEATURE_PROGRAM_MODE = {constants["feature_program_mode"]!r}
INCLUDE_FEATURE_PROGRAMS = {constants["include_feature_programs"]!r}
FEATURE_FAMILY_FILTER = {constants["feature_family_filter"]!r}
FEATURE_COLUMN_ALLOWLIST = {constants["feature_column_allowlist"]!r}
PREPROCESSING = {constants["preprocessing"]!r}
HYPERPARAMETERS = {constants["hyperparameters"]!r}
IDENTIFIER_COLUMNS = {{
    "row_id", "cell_id", "batch_id", "original_cell_id", "original_batch_id",
    "barcode", "source_path", "file_path", "filename", "path", "channel",
    "cycle_life", "Lifetime", "protocol_readable", "policy_readable", "y",
}}
SCALAR_FAMILIES = {{
    "capacity_summary", "resistance_summary", "thermal_summary", "energy_summary",
    "capacity_summary_delta", "resistance_summary_delta", "thermal_summary_delta",
    "energy_summary_delta", "scalar_delta",
}}
CURVE_FAMILIES = {{
    "true_curve_shape", "true_curve_difference", "curve_difference_proxy",
    "curve_difference_approximate", "curve_shape_proxy",
}}
# Families that the broad_physics recipe emits in addition to SCALAR + CURVE.
# Kept as a separate set so broad_physics is a strict superset of
# scalar_plus_curve (fix E3.1).
BROAD_PHYSICS_EXTRA_FAMILIES = {{
    "learned_embedding_placeholder", "generic_timeseries_placeholder",
}}
PROTOCOL_FAMILIES = {{"protocol"}}


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


def _families_for_feature_set(feature_set, allow_protocol):
    if FEATURE_FAMILY_FILTER:
        families = set(FEATURE_FAMILY_FILTER)
    elif feature_set == "scalar_only":
        families = set(SCALAR_FAMILIES)
    elif feature_set == "curve_only":
        families = set(CURVE_FAMILIES)
    elif feature_set == "scalar_plus_curve":
        families = set(SCALAR_FAMILIES) | set(CURVE_FAMILIES)
    elif feature_set == "broad_physics":
        # Fix E3.1: broad_physics is a STRICT SUPERSET of scalar_plus_curve.
        # The make_broad_physics_program recipe in program_library.py emits all
        # the scalar/curve_shape/curve_delta families plus placeholder families
        # reserved for future learned-embedding / generic-timeseries operators.
        # We allow those families through here so any feature_program table
        # built from the broad_physics recipe is not silently filtered down to
        # the scalar_plus_curve subset.
        families = set(SCALAR_FAMILIES) | set(CURVE_FAMILIES) | set(BROAD_PHYSICS_EXTRA_FAMILIES)
        # If the candidate declares feature_program_paths but none of them
        # looks like a broad_physics program table, emit a stdlib warning so
        # the missing recipe is visible in stderr_excerpt / report logs.
        _broad_physics_paths = [str(p) for p in FEATURE_PROGRAM_PATHS]
        if _broad_physics_paths and not any(
            "broad_physics" in p.lower() for p in _broad_physics_paths
        ):
            import warnings as _warnings
            _warnings.warn(
                "broad_physics_feature_program_missing: feature_set='broad_physics' "
                "was requested but no feature_program_paths entry references a "
                "broad_physics recipe; falling back to scalar_plus_curve coverage.",
                RuntimeWarning,
                stacklevel=2,
            )
        elif not _broad_physics_paths:
            import warnings as _warnings
            _warnings.warn(
                "broad_physics_feature_program_missing: feature_set='broad_physics' "
                "was requested but no feature_program_paths were provided; "
                "richer broad_physics features will be unavailable.",
                RuntimeWarning,
                stacklevel=2,
            )
    elif feature_set == "protocol_only":
        families = set(PROTOCOL_FAMILIES)
    else:
        families = set()
    if allow_protocol and feature_set in {{"all_available", "protocol_only"}}:
        families |= set(PROTOCOL_FAMILIES)
    if not allow_protocol:
        families -= set(PROTOCOL_FAMILIES)
    return families


def _select_feature_set_columns(features, feature_metadata, feature_set, allow_protocol):
    if feature_metadata is None or len(feature_metadata) == 0:
        if feature_set == "protocol_only":
            return [col for col in features.columns if col.startswith("protocol_")]
        if feature_set == "curve_only":
            return [col for col in features.columns if "curve" in col or "qdiff" in col]
        if feature_set == "scalar_only":
            return [col for col in features.columns if not col.startswith("protocol_") and "curve" not in col and "qdiff" not in col]
        return list(features.columns)
    meta = feature_metadata.copy()
    if "feature_family" not in meta.columns and "family" in meta.columns:
        meta["feature_family"] = meta["family"]
    if "family" not in meta.columns and "feature_family" in meta.columns:
        meta["family"] = meta["feature_family"]
    families = _families_for_feature_set(feature_set, allow_protocol)
    if families:
        selected = set(meta.loc[meta["feature_family"].astype(str).isin(families), "feature_name"].astype(str))
    else:
        selected = set(meta["feature_name"].astype(str))
        if not allow_protocol:
            protocol_cols = set(meta.loc[meta["feature_family"].astype(str).isin(PROTOCOL_FAMILIES), "feature_name"].astype(str))
            selected -= protocol_cols
    return [col for col in features.columns if col in selected]


def _feature_frame(metadata, cycle_summary, config, feature_columns=None):
    max_cycle = int(config.get("max_cycle", 100))
    allow_protocol = bool(config.get("allow_protocol_features", INCLUDE_PROTOCOL_FEATURES))
    include_protocol = bool(INCLUDE_PROTOCOL_FEATURES and allow_protocol)
    feature_program_paths = config.get("feature_program_paths", FEATURE_PROGRAM_PATHS)
    include_feature_programs = bool(config.get("include_feature_programs", INCLUDE_FEATURE_PROGRAMS))
    feature_program_mode = config.get("feature_program_mode", FEATURE_PROGRAM_MODE)
    feature_family_filter = config.get("feature_family_filter", FEATURE_FAMILY_FILTER)
    ids = metadata[["row_id"]].drop_duplicates().copy()
    features, feature_metadata = build_all_battery_features(
        metadata,
        cycle_summary,
        max_cycle=max_cycle,
        include_protocol=include_protocol,
        return_feature_metadata=True,
        feature_program_paths=feature_program_paths,
        feature_program_mode=feature_program_mode,
        include_feature_programs=include_feature_programs,
        feature_family_filter=feature_family_filter,
    )
    selected_columns = _select_feature_set_columns(features, feature_metadata, FEATURE_SET, allow_protocol)
    if FEATURE_COLUMN_ALLOWLIST:
        # Restrict to the agent's operator-selected columns (Tier B). Preserve
        # the feature_set ordering; keep only allowlisted columns that exist.
        allow = set(FEATURE_COLUMN_ALLOWLIST)
        base = selected_columns or list(features.columns)
        selected_columns = [col for col in base if col in allow]
    features = features[selected_columns] if selected_columns else features.iloc[:, 0:0]
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
