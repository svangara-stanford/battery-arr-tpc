from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd

from battery_aar.features.feature_programs import build_feature_program_table, compile_feature_program
from battery_aar.features.program_library import make_attia_severson_like_program, make_broad_physics_program, make_minimal_debug_program
from battery_aar.workflows.schemas import FeatureOperatorSpec, FeatureProgram

from .feature_program_test_utils import toy_raw_payload, write_toy_feature_manifest, write_toy_raw_cell


def _precision_loss_warnings(caught):
    return [
        warning
        for warning in caught
        if "Precision loss occurred in moment calculation" in str(warning.message)
    ]


def test_feature_program_json_round_trips_through_pydantic():
    program = make_minimal_debug_program()
    payload = program.model_dump(mode="json")
    restored = FeatureProgram.model_validate(payload)

    assert restored.program_id == program.program_id
    assert restored.operators[0].operator_name == "cycle_scalar"


def test_compile_feature_program_returns_features_metadata_and_warnings():
    program = make_attia_severson_like_program(cycle_early=9, cycle_late=99)
    features, metadata, warnings = compile_feature_program(
        program,
        toy_raw_payload(n_cycles=101, n_points=8),
        row_id=0,
        cell_id="cell_0",
        processed_metadata_row=pd.Series({"C1": 4.0, "C2": 4.4, "C3": 4.8, "C4": 3.5}),
    )

    assert features
    assert metadata
    assert any(row["feature_family"] == "true_curve_difference" for row in metadata)
    assert any("cycle_index_convention=raw_zero_based" in warning for warning in warnings)


def test_feature_program_compiler_builds_feature_table_and_metadata_for_toy_cells(tmp_path):
    manifest, raw_root = write_toy_feature_manifest(tmp_path, n_cells=3)
    program = make_attia_severson_like_program(cycle_early=9, cycle_late=99)
    result = build_feature_program_table(program, manifest, raw_root=raw_root, out_dir=tmp_path / "features")

    assert result.n_rows == 3
    assert result.n_feature_columns > 0
    assert result.n_numeric_feature_columns == result.n_feature_columns
    assert result.n_excluded_cells == 0
    table = pd.read_csv(result.feature_table_path)
    metadata = pd.read_csv(result.feature_metadata_path)
    card = json.loads((tmp_path / "features" / "dataset_card.json").read_text())

    assert {"row_id", "cell_id"}.issubset(table.columns)
    assert len(table) == 3
    assert set(table["row_id"]) == {0, 1, 2}
    assert "true_curve_difference" in set(metadata["feature_family"])
    assert metadata["is_true_curve_feature"].astype(bool).any()
    assert card["true_curve_features_used"] is True
    assert Path(tmp_path / "features" / "exclusions.csv").exists()


def test_feature_program_compiler_records_exclusion_for_bad_raw_file(tmp_path):
    manifest, raw_root = write_toy_feature_manifest(tmp_path, n_cells=2)
    bad_path = raw_root / "bad.json"
    bad_path.write_text("{not valid json")
    manifest.loc[1, "source_path"] = str(bad_path)

    result = build_feature_program_table(make_minimal_debug_program(), manifest, raw_root=raw_root, out_dir=tmp_path / "features")
    exclusions = pd.read_csv(tmp_path / "features" / "exclusions.csv")

    assert result.n_rows == 1
    assert result.n_excluded_cells == 1
    assert "JSONDecodeError" in exclusions.loc[0, "reason"]


def test_feature_program_rebases_stale_absolute_battery_fast_charging_source_path(tmp_path):
    raw_root = tmp_path / "battery-fast-charging"
    batch_id = "2018-08-28_oed_0"
    raw_file = write_toy_raw_cell(raw_root / "data" / batch_id / f"{batch_id}_CH17_structure.json")
    stale_source_path = (
        f"/other/machine/not-this-repo/battery-fast-charging/data/{batch_id}/{raw_file.name}"
    )
    manifest = pd.DataFrame(
        [
            {
                "row_id": 17,
                "cell_id": f"{batch_id}_CH17",
                "source_cell_id": f"{batch_id}_CH17",
                "batch_id": batch_id,
                "source_path": stale_source_path,
            }
        ]
    )

    result = build_feature_program_table(make_minimal_debug_program(), manifest, raw_root=raw_root, out_dir=tmp_path / "rebased")
    table = pd.read_csv(result.feature_table_path)
    card = json.loads((tmp_path / "rebased" / "dataset_card.json").read_text())
    source_audit = pd.read_csv(tmp_path / "rebased" / "source_path_resolution.csv")

    assert result.n_rows == 1
    assert result.n_excluded_cells == 0
    assert "resolved_source_path" not in table.columns
    assert card["source_paths_rebased"] == 1
    assert card["source_paths_used_directly"] == 0
    assert source_audit.loc[0, "source_path_resolution_method"] == "rebased"
    assert Path(source_audit.loc[0, "resolved_source_path"]) == raw_file


def test_feature_program_reconstructs_source_path_from_batch_and_source_cell_id(tmp_path):
    raw_root = tmp_path / "battery-fast-charging"
    batch_id = "2018-09-02_oed_1"
    raw_file = write_toy_raw_cell(raw_root / "data" / batch_id / f"{batch_id}_CH03_structure.json")
    manifest = pd.DataFrame(
        [
            {
                "row_id": 3,
                "cell_id": f"{batch_id}_CH03",
                "source_cell_id": f"{batch_id}_CH03",
                "batch_id": batch_id,
                "source_path": "/other/machine/no-marker/missing.json",
            }
        ]
    )

    result = build_feature_program_table(make_minimal_debug_program(), manifest, raw_root=raw_root, out_dir=tmp_path / "reconstructed")
    card = json.loads((tmp_path / "reconstructed" / "dataset_card.json").read_text())
    source_audit = pd.read_csv(tmp_path / "reconstructed" / "source_path_resolution.csv")

    assert result.n_rows == 1
    assert result.n_excluded_cells == 0
    assert card["source_paths_reconstructed"] == 1
    assert source_audit.loc[0, "source_path_resolution_method"] == "reconstructed"
    assert Path(source_audit.loc[0, "resolved_source_path"]) == raw_file


def test_broad_physics_feature_program_emits_no_skew_precision_loss_runtime_warnings(tmp_path):
    manifest, raw_root = write_toy_feature_manifest(tmp_path, n_cells=2)
    program = make_broad_physics_program(first_n_cycles=100)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = build_feature_program_table(program, manifest, raw_root=raw_root, out_dir=tmp_path / "broad_physics")

    assert result.n_rows == 2
    assert result.n_numeric_feature_columns > 0
    assert not _precision_loss_warnings(caught)


def test_feature_program_prunes_all_nan_features_and_matching_metadata(tmp_path):
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    payload = toy_raw_payload(n_cycles=12, n_points=4)
    payload["summary"].pop("internal_resistance", None)
    payload["cycles_interpolated"].pop("internal_resistance", None)
    rows = []
    for row_id in range(2):
        raw_path = raw_root / f"cell_{row_id}.json"
        raw_path.write_text(json.dumps(payload))
        rows.append({"row_id": row_id, "cell_id": f"cell_{row_id}", "source_path": str(raw_path)})
    manifest = pd.DataFrame(rows)
    program = FeatureProgram(
        run_id="prune_test",
        human_readable_summary="all nan pruning",
        program_id="all_nan_resistance",
        name="All-NaN Resistance",
        description="Requests a missing resistance signal.",
        operators=[
            FeatureOperatorSpec(
                operator_name="cycle_scalar",
                operator_type="cycle_scalar",
                family="resistance_summary",
                params={"signals": ["internal_resistance"], "cycle_indices": [9], "aggregations": ["last"]},
            )
        ],
    )

    result = build_feature_program_table(program, manifest, raw_root=raw_root, out_dir=tmp_path / "pruned")
    table = pd.read_csv(result.feature_table_path)
    metadata = pd.read_csv(result.feature_metadata_path)
    card = json.loads((tmp_path / "pruned" / "dataset_card.json").read_text())

    assert result.n_pruned_all_nan_features == 1
    assert card["n_pruned_all_nan_features"] == 1
    assert table.columns.tolist() == ["row_id", "cell_id"]
    assert metadata.empty
