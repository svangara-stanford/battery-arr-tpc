import importlib.util
from pathlib import Path

import pytest
import pandas as pd

from battery_aar.workflows.candidate_compiler import candidate_spec_from_plans, compile_candidate_spec_to_python
from battery_aar.workflows.schemas import FeaturePlan, ModelPlan


def _toy_tables():
    metadata = pd.DataFrame(
        {
            "row_id": [0, 1, 2, 3],
            "cell_id": ["c0", "c1", "c2", "c3"],
            "batch_id": ["leak_a", "leak_a", "leak_b", "leak_b"],
            "protocol_readable": ["p0", "p1", "p0", "p1"],
            "C1": [4.0, 4.2, 4.4, 4.6],
            "C2": [4.4, 4.6, 4.8, 5.0],
            "C3": [4.8, 5.0, 5.2, 5.4],
            "C4": [3.5, 3.6, 3.7, 3.8],
            "cycle_life": [900.0, 860.0, 820.0, 780.0],
        }
    )
    rows = []
    for row_id in metadata["row_id"]:
        for cycle in [1, 2, 10, 50, 91, 100]:
            rows.append(
                {
                    "row_id": row_id,
                    "cell_id": f"c{row_id}",
                    "cycle_index": cycle,
                    "discharge_capacity": 1.12 - 0.001 * cycle - 0.0005 * row_id,
                    "charge_capacity": 1.14 - 0.001 * cycle,
                }
            )
    cycles = pd.DataFrame(rows)
    labels = metadata[["row_id", "cycle_life"]].rename(columns={"cycle_life": "y"})
    return metadata, cycles, labels


def _compiled_module(tmp_path: Path, model_family: str = "Ridge", target_transform: str = "raw"):
    feature_plan = FeaturePlan(
        run_id="compiler_test",
        human_readable_summary="features",
        agent_id="feature_scientist",
        feature_families=["capacity_summary", "curve_difference_approximate", "protocol"],
        include_protocol_features=True,
        max_cycle=100,
    )
    model_plan = ModelPlan(
        run_id="compiler_test",
        human_readable_summary="model",
        agent_id="model_architect",
        model_family=model_family,
        estimator_name=model_family,
        target_transform=target_transform,
        preprocessing_steps=["SimpleImputer", "StandardScaler"],
        hyperparameters={"alpha": 1.0, "unsafe_unused": "ignored"},
    )
    candidate_path = tmp_path / "compiled_candidate.py"
    spec_model = candidate_spec_from_plans(
        run_id="compiler_test",
        candidate_id="compiled_candidate",
        agent_id="code_generator",
        iteration=0,
        candidate_path=candidate_path,
        feature_plan=feature_plan,
        model_plan=model_plan,
    )
    compile_candidate_spec_to_python(spec_model, candidate_path)
    spec = importlib.util.spec_from_file_location("compiled_candidate", candidate_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return spec_model, module


def test_compiled_candidate_builds_numeric_only_nonleaky_features(tmp_path):
    _spec_model, module = _compiled_module(tmp_path)
    metadata, cycles, _labels = _toy_tables()
    ids, X, feature_cols = module._feature_frame(metadata, cycles, {"max_cycle": 100, "allow_protocol_features": True})

    assert ids["row_id"].tolist() == [0, 1, 2, 3]
    assert feature_cols
    assert all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns)
    assert not {"row_id", "cell_id", "batch_id", "protocol_readable", "cycle_life"}.intersection(X.columns)


def test_compiled_candidate_aligns_train_test_feature_columns(tmp_path):
    _spec_model, module = _compiled_module(tmp_path)
    metadata, cycles, _labels = _toy_tables()
    train_meta = metadata.iloc[:3].copy()
    test_meta = metadata.iloc[3:].drop(columns=["C1", "C2", "C3", "C4"]).copy()
    train_cycles = cycles[cycles["row_id"].isin(train_meta["row_id"])]
    test_cycles = cycles[cycles["row_id"].isin(test_meta["row_id"])]

    _ids, X_train, feature_cols = module._feature_frame(train_meta, train_cycles, {"max_cycle": 100, "allow_protocol_features": True})
    _ids, X_test, _ = module._feature_frame(test_meta, test_cycles, {"max_cycle": 100, "allow_protocol_features": True}, feature_cols)

    assert list(X_test.columns) == list(X_train.columns)


def test_compiled_candidate_predicts_every_test_row(tmp_path):
    spec_model, module = _compiled_module(tmp_path)
    metadata, cycles, labels = _toy_tables()
    train_meta = metadata.iloc[:3].copy()
    test_meta = metadata.iloc[3:].copy()
    model = module.fit(
        train_meta,
        cycles[cycles["row_id"].isin(train_meta["row_id"])],
        labels[labels["row_id"].isin(train_meta["row_id"])],
        {"max_cycle": 100, "allow_protocol_features": True},
    )
    pred = module.predict(
        model,
        test_meta,
        cycles[cycles["row_id"].isin(test_meta["row_id"])],
        {"max_cycle": 100, "allow_protocol_features": True},
    )

    assert pred.columns.tolist() == ["row_id", "y_pred"]
    assert pred["row_id"].tolist() == test_meta["row_id"].tolist()
    assert pred["y_pred"].notna().all()
    assert spec_model.target_transform == "raw"


def test_log10_target_compiled_candidate_returns_cycle_units(tmp_path):
    spec_model, module = _compiled_module(tmp_path, target_transform="log10")
    metadata, cycles, labels = _toy_tables()
    train_meta = metadata.iloc[:3].copy()
    test_meta = metadata.iloc[3:].copy()
    model = module.fit(
        train_meta,
        cycles[cycles["row_id"].isin(train_meta["row_id"])],
        labels[labels["row_id"].isin(train_meta["row_id"])],
        {"max_cycle": 100, "allow_protocol_features": True},
    )
    pred = module.predict(
        model,
        test_meta,
        cycles[cycles["row_id"].isin(test_meta["row_id"])],
        {"max_cycle": 100, "allow_protocol_features": True},
    )

    assert spec_model.target_transform == "log10"
    assert model["target_transform"] == "log10"
    assert pred["y_pred"].notna().all()
    assert pred["y_pred"].median() > 100.0


def test_log10_target_rejects_nonpositive_labels(tmp_path):
    _spec_model, module = _compiled_module(tmp_path, target_transform="log10")
    metadata, cycles, labels = _toy_tables()
    labels.loc[labels["row_id"] == 1, "y"] = 0.0

    with pytest.raises(ValueError, match="log10 target_transform requires positive finite y"):
        module.fit(
            metadata.iloc[:3].copy(),
            cycles[cycles["row_id"].isin([0, 1, 2])],
            labels[labels["row_id"].isin([0, 1, 2])],
            {"max_cycle": 100, "allow_protocol_features": True},
        )


def test_candidate_spec_contains_declarative_fields(tmp_path):
    spec_model, _module = _compiled_module(tmp_path, model_family="RandomForestRegressor")

    assert spec_model.compiled_candidate is True
    assert spec_model.model_family == "RandomForestRegressor"
    assert spec_model.target_transform == "raw"
    assert spec_model.feature_families
    assert spec_model.include_protocol_features is True
    assert "n_estimators" in spec_model.hyperparameters
