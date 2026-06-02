from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from battery_aar.features.battery_lifetime_features import build_all_battery_features
from battery_aar.features.feature_programs import build_feature_program_table
from battery_aar.features.program_library import make_attia_severson_like_program
from battery_aar.workflows.candidate_compiler import candidate_spec_from_plans, compile_candidate_spec_to_python
from battery_aar.workflows.role_graph import _combine_locked_feature_program_tables
from battery_aar.workflows.schemas import FeaturePlan, ModelPlan

from .feature_program_test_utils import write_toy_feature_manifest, write_toy_processed_from_manifest


def _load_compiled_candidate(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _feature_program_dataset(tmp_path: Path):
    manifest, raw_root = write_toy_feature_manifest(tmp_path, n_cells=5)
    processed = write_toy_processed_from_manifest(tmp_path, manifest)
    program_result = build_feature_program_table(
        make_attia_severson_like_program(cycle_early=9, cycle_late=99),
        manifest,
        raw_root=raw_root,
        out_dir=tmp_path / "program",
    )
    metadata = pd.read_csv(processed / "cell_metadata.csv")
    cycles = pd.read_csv(processed / "cycle_summary.csv")
    labels = metadata[["row_id", "cycle_life"]].rename(columns={"cycle_life": "y"})
    return metadata, cycles, labels, program_result


def test_feature_program_table_merges_with_existing_features_on_row_id(tmp_path):
    metadata, cycles, _labels, program_result = _feature_program_dataset(tmp_path)

    features, feature_metadata = build_all_battery_features(
        metadata,
        cycles,
        return_feature_metadata=True,
        include_protocol=False,
        feature_program_paths=[program_result.feature_table_path],
        feature_program_mode="table",
        include_feature_programs=True,
    )

    assert list(features.index) == [0, 1, 2, 3, 4]
    assert "row_id" not in features.columns
    assert "cell_id" not in features.columns
    assert "true_curve_difference" in set(feature_metadata["feature_family"].astype(str))
    assert feature_metadata["source_feature_program_table"].notna().any()


def test_feature_family_filter_selects_curve_columns_from_feature_program_table(tmp_path):
    metadata, cycles, _labels, program_result = _feature_program_dataset(tmp_path)

    features, feature_metadata = build_all_battery_features(
        metadata,
        cycles,
        return_feature_metadata=True,
        include_protocol=False,
        feature_program_paths=[program_result.feature_table_path],
        feature_program_mode="table",
        include_feature_programs=True,
        feature_family_filter=["true_curve_difference"],
    )

    selected = set(feature_metadata.loc[feature_metadata["source_feature_program_table"].notna(), "feature_family"].astype(str))
    assert selected == {"true_curve_difference"}
    assert any("curve_delta" in col for col in features.columns)


def test_compiled_candidate_runs_supported_feature_sets_with_feature_program_table(tmp_path):
    metadata, cycles, labels, program_result = _feature_program_dataset(tmp_path)
    train_meta = metadata.iloc[:4].copy()
    test_meta = metadata.iloc[4:].copy()
    train_cycles = cycles[cycles["row_id"].isin(train_meta["row_id"])]
    test_cycles = cycles[cycles["row_id"].isin(test_meta["row_id"])]

    for feature_set in ["scalar_only", "curve_only", "scalar_plus_curve", "broad_physics", "all_available"]:
        feature_plan = FeaturePlan(
            run_id="feature_program_candidate",
            human_readable_summary="feature program",
            agent_id="feature_scientist",
            feature_families=["capacity_summary", "true_curve_difference"],
            include_protocol_features=False,
            feature_program_paths=[program_result.feature_table_path],
            feature_set=feature_set,
            max_cycle=100,
        )
        model_plan = ModelPlan(
            run_id="feature_program_candidate",
            human_readable_summary="model",
            agent_id="model_architect",
            model_family="Ridge",
            estimator_name="Ridge",
            feature_set=feature_set,
            preprocessing_steps=["SimpleImputer", "StandardScaler"],
            hyperparameters={"alpha": 1.0},
        )
        candidate_path = tmp_path / f"candidate_{feature_set}.py"
        spec = candidate_spec_from_plans(
            run_id="feature_program_candidate",
            candidate_id=f"candidate_{feature_set}",
            agent_id="code_generator",
            iteration=0,
            candidate_path=candidate_path,
            feature_plan=feature_plan,
            model_plan=model_plan,
        )
        compile_candidate_spec_to_python(spec, candidate_path)
        module = _load_compiled_candidate(candidate_path)
        model = module.fit(train_meta, train_cycles, labels[labels["row_id"].isin(train_meta["row_id"])], {"max_cycle": 100})
        pred = module.predict(model, test_meta, test_cycles, {"max_cycle": 100})
        assert pred["row_id"].tolist() == test_meta["row_id"].tolist()
        assert pred["y_pred"].notna().all()


def test_batch9_feature_program_row_offset_merge_for_locked_validation(tmp_path):
    search_dir = tmp_path / "search_program"
    batch9_dir = tmp_path / "batch9_program"
    search_dir.mkdir()
    batch9_dir.mkdir()
    pd.DataFrame({"row_id": [0, 1], "cell_id": ["c0", "c1"], "curve_feature": [1.0, 2.0]}).to_csv(search_dir / "feature_table.csv", index=False)
    pd.DataFrame({"row_id": [0], "cell_id": ["b9_0"], "curve_feature": [3.0]}).to_csv(batch9_dir / "feature_table.csv", index=False)
    pd.DataFrame(
        {
            "feature_name": ["curve_feature"],
            "feature_family": ["true_curve_difference"],
            "family": ["true_curve_difference"],
            "operator_name": ["cross_cycle_curve_delta"],
            "operator_type": ["cross_cycle_curve_delta"],
        }
    ).to_csv(search_dir / "feature_metadata.csv", index=False)
    pd.DataFrame(
        {
            "feature_name": ["curve_feature"],
            "feature_family": ["true_curve_difference"],
            "family": ["true_curve_difference"],
            "operator_name": ["cross_cycle_curve_delta"],
            "operator_type": ["cross_cycle_curve_delta"],
        }
    ).to_csv(batch9_dir / "feature_metadata.csv", index=False)

    combined_paths = _combine_locked_feature_program_tables(
        search_paths=[search_dir],
        batch9_path=batch9_dir,
        out_dir=tmp_path / "combined",
        row_offset=10,
    )
    combined = pd.read_csv(combined_paths[0])

    assert combined["row_id"].tolist() == [0, 1, 10]
    assert combined.loc[combined["cell_id"] == "b9_0", "curve_feature"].iloc[0] == 3.0
