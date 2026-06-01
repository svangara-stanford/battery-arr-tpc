from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineCandidate:
    name: str
    code: str


def _common_candidate_code(estimator_import: str, estimator_expr: str) -> str:
    return f'''
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
{estimator_import}

from battery_aar.features.battery_lifetime_features import build_all_battery_features


def _make_features(metadata, cycle_summary, config):
    max_cycle = int(config.get("max_cycle", 100))
    include_protocol = bool(config.get("allow_protocol_features", False))
    X = build_all_battery_features(metadata, cycle_summary, max_cycle=max_cycle, include_protocol=include_protocol)
    X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    return X


def fit(train_metadata, train_cycle_summary, train_labels, config):
    X = _make_features(train_metadata, train_cycle_summary, config)
    labels = train_labels.set_index("row_id")["y"]
    X.index = pd.to_numeric(pd.Series(X.index), errors="coerce").to_numpy()
    X = X.loc[X.index[~pd.isna(X.index)]]
    common = [idx for idx in X.index if idx in labels.index]
    y = labels.loc[common].to_numpy(float)
    X = X.loc[common]
    if X.empty or y.size == 0:
        return {{"fallback": float(train_labels["y"].mean()), "columns": []}}
    model = {estimator_expr}
    model.fit(X, y)
    return {{"model": model, "columns": list(X.columns), "fallback": float(np.mean(y))}}


def predict(model, test_metadata, test_cycle_summary, config):
    row_ids = test_metadata["row_id"].to_numpy()
    if "model" not in model:
        return pd.DataFrame({{"row_id": row_ids, "y_pred": model["fallback"]}})
    X = _make_features(test_metadata, test_cycle_summary, config)
    X.index = pd.to_numeric(pd.Series(X.index), errors="coerce").to_numpy()
    X = X.reindex(row_ids)
    X = X.reindex(columns=model["columns"])
    y_pred = model["model"].predict(X)
    return pd.DataFrame({{"row_id": row_ids, "y_pred": y_pred}})
'''


def author_inspired_ridge_candidate() -> BaselineCandidate:
    return BaselineCandidate(
        name="author_inspired_ridge",
        code=_common_candidate_code(
            "from sklearn.linear_model import Ridge",
            "make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=3.0))",
        ),
    )


def author_inspired_random_forest_candidate() -> BaselineCandidate:
    return BaselineCandidate(
        name="author_inspired_random_forest",
        code=_common_candidate_code(
            "from sklearn.ensemble import RandomForestRegressor",
            "make_pipeline(SimpleImputer(strategy='median'), RandomForestRegressor(n_estimators=160, min_samples_leaf=2, random_state=19))",
        ),
    )


def author_inspired_gradient_boosting_candidate() -> BaselineCandidate:
    return BaselineCandidate(
        name="author_inspired_gradient_boosting",
        code=_common_candidate_code(
            "from sklearn.ensemble import GradientBoostingRegressor",
            "make_pipeline(SimpleImputer(strategy='median'), GradientBoostingRegressor(random_state=23, max_depth=2, learning_rate=0.05, n_estimators=120))",
        ),
    )


def author_inspired_baseline_candidates() -> list[BaselineCandidate]:
    return [
        author_inspired_ridge_candidate(),
        author_inspired_random_forest_candidate(),
        author_inspired_gradient_boosting_candidate(),
    ]
