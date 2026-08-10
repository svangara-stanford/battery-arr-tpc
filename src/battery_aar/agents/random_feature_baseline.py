"""Random-feature baseline candidate generator.

A control for the RAG+LLM FeatureScientist: instead of reasoning about which
early-cycle features predict lifetime, this baseline draws a random subset of
columns from the SAME ``build_all_battery_features`` pool and fits the SAME
fixed model (Ridge on a log10 target). Running it through the identical
``evaluate_candidate_train_test`` harness on the identical split isolates the
value of *feature selection*: any gap between this baseline and the agent is
attributable to how features are chosen, not to the model or the data split.

Each emitted candidate is a self-contained ``.py`` exposing the ``fit`` /
``predict`` contract the candidate evaluator expects. Selection is seeded so a
given ``(seed, n_features)`` is fully reproducible.

Use ``random_feature_baseline_candidate`` for a single control and
``random_feature_baseline_candidates`` to sweep several seeds (the comparison
script averages over the sweep so the baseline is not judged on one lucky draw).
"""

from __future__ import annotations

from dataclasses import dataclass

# Kept in sync with the fixed model used for the agent-features control in
# scripts/compare_random_feature_baseline.py so the only moving part is which
# columns are selected.
DEFAULT_N_FEATURES = 8
DEFAULT_SEEDS = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class BaselineCandidate:
    name: str
    code: str
    seed: int
    n_features: int


_CANDIDATE_TEMPLATE = '''# Auto-generated random-feature baseline (no RAG, no LLM).
# Selects {n_features} features at random (seed={seed}) from the
# build_all_battery_features pool (expanded by any supplied feature programs),
# then fits Ridge on a log10 target -- the same model held fixed for the
# agent-features control.
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from battery_aar.features.battery_lifetime_features import build_all_battery_features

RANDOM_SEED = {seed}
N_FEATURES = {n_features}
MAX_CYCLE = {max_cycle}
RIDGE_ALPHA = {ridge_alpha}
# Feature-program settings: when a program table is supplied the random draw
# happens over the SAME expanded pool the agent sees, so the comparison stays
# apples-to-apples with feature-programs-enabled runs.
FEATURE_PROGRAM_PATHS = {feature_program_paths!r}
FEATURE_PROGRAM_MODE = {feature_program_mode!r}
INCLUDE_FEATURE_PROGRAMS = {include_feature_programs!r}
IDENTIFIER_COLUMNS = {{
    "row_id", "cell_id", "batch_id", "original_cell_id", "original_batch_id",
    "barcode", "source_path", "file_path", "filename", "path", "channel",
    "cycle_life", "Lifetime", "protocol_readable", "policy_readable", "y",
}}


def _numeric_feature_frame(metadata, cycle_summary, config):
    max_cycle = int(config.get("max_cycle", MAX_CYCLE))
    ids = metadata[["row_id"]].drop_duplicates().copy()
    features = build_all_battery_features(
        metadata,
        cycle_summary,
        max_cycle=max_cycle,
        include_protocol=False,
        feature_program_paths=config.get("feature_program_paths", FEATURE_PROGRAM_PATHS),
        feature_program_mode=config.get("feature_program_mode", FEATURE_PROGRAM_MODE),
        include_feature_programs=config.get(
            "include_feature_programs", INCLUDE_FEATURE_PROGRAMS
        ),
    )
    key_name = features.index.name or "row_id"
    features = features.reset_index().rename(columns={{key_name: "row_id"}})
    if "row_id" not in features.columns:
        features.insert(0, "row_id", ids["row_id"].to_numpy())
    features = ids.merge(features, on="row_id", how="left")
    X = features.drop(
        columns=[c for c in IDENTIFIER_COLUMNS if c in features.columns],
        errors="ignore",
    )
    for col in list(X.columns):
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    return ids, X


def _select_random_columns(all_columns):
    # Deterministic draw: sort for a stable universe, then seeded permutation.
    universe = sorted(str(c) for c in all_columns)
    rng = np.random.default_rng(RANDOM_SEED)
    k = min(N_FEATURES, len(universe))
    idx = rng.choice(len(universe), size=k, replace=False)
    return [universe[i] for i in sorted(idx)]


def fit(train_metadata, train_cycle_summary, train_labels, config):
    ids, X = _numeric_feature_frame(train_metadata, train_cycle_summary, config)
    X = X.dropna(axis=1, how="all")
    selected = _select_random_columns(X.columns)
    if not selected:
        selected = list(X.columns)
    X = X[selected]
    labels = train_labels[["row_id", "y"]].copy()
    y = ids.merge(labels, on="row_id", how="left")["y"].to_numpy(float)
    if not np.isfinite(y).all():
        raise ValueError("random-feature baseline could not align finite training labels")
    if np.any(y <= 0):
        raise ValueError("log10 target requires positive finite y")
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(alpha=RIDGE_ALPHA),
    )
    model.fit(X, np.log10(y))
    return {{"model": model, "feature_columns": selected}}


def predict(model, test_metadata, test_cycle_summary, config):
    ids, X = _numeric_feature_frame(test_metadata, test_cycle_summary, config)
    for col in model["feature_columns"]:
        if col not in X.columns:
            X[col] = np.nan
    X = X[model["feature_columns"]]
    y_pred = np.power(10.0, model["model"].predict(X))
    return pd.DataFrame({{"row_id": ids["row_id"].to_numpy(), "y_pred": y_pred}})
'''


def random_feature_baseline_code(
    seed: int = 0,
    n_features: int = DEFAULT_N_FEATURES,
    max_cycle: int = 100,
    ridge_alpha: float = 1.0,
    feature_program_paths: list[str] | None = None,
    feature_program_mode: str = "none",
    include_feature_programs: bool = False,
) -> str:
    """Return self-contained candidate source for one random-feature draw.

    Pass ``feature_program_paths`` (with ``feature_program_mode="table"`` or
    ``include_feature_programs=True``) to draw from the same expanded pool an
    agent run with feature programs sees.
    """
    return _CANDIDATE_TEMPLATE.format(
        seed=int(seed),
        n_features=int(n_features),
        max_cycle=int(max_cycle),
        ridge_alpha=float(ridge_alpha),
        feature_program_paths=[str(p) for p in (feature_program_paths or [])],
        feature_program_mode=str(feature_program_mode),
        include_feature_programs=bool(include_feature_programs),
    )


def random_feature_baseline_candidate(
    seed: int = 0,
    n_features: int = DEFAULT_N_FEATURES,
    max_cycle: int = 100,
    ridge_alpha: float = 1.0,
    feature_program_paths: list[str] | None = None,
    feature_program_mode: str = "none",
    include_feature_programs: bool = False,
) -> BaselineCandidate:
    return BaselineCandidate(
        name=f"random_feature_baseline_seed{seed}_n{n_features}",
        code=random_feature_baseline_code(
            seed, n_features, max_cycle, ridge_alpha,
            feature_program_paths, feature_program_mode, include_feature_programs,
        ),
        seed=int(seed),
        n_features=int(n_features),
    )


def random_feature_baseline_candidates(
    seeds=DEFAULT_SEEDS,
    n_features: int = DEFAULT_N_FEATURES,
    max_cycle: int = 100,
    ridge_alpha: float = 1.0,
    feature_program_paths: list[str] | None = None,
    feature_program_mode: str = "none",
    include_feature_programs: bool = False,
) -> list[BaselineCandidate]:
    """A sweep of seeded random-feature baselines for averaging."""
    return [
        random_feature_baseline_candidate(
            seed, n_features, max_cycle, ridge_alpha,
            feature_program_paths, feature_program_mode, include_feature_programs,
        )
        for seed in seeds
    ]
