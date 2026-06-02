from __future__ import annotations

import subprocess
import sys

import pandas as pd

from battery_aar.features.feature_programs import build_feature_program_table
from battery_aar.features.program_library import make_minimal_debug_program, make_scalar_baseline_program
from battery_aar.features.severson_matr import build_severson_true_life_dataset
from battery_aar.workflows.role_graph import run_role_workflow

from .feature_program_test_utils import write_toy_raw_cell
from .severson_test_utils import write_toy_severson_mat_dir


def _processed_severson_with_features(tmp_path):
    mat_dir = write_toy_severson_mat_dir(tmp_path, n_files=2, n_cells_per_file=3)
    processed = tmp_path / "severson_processed"
    build_severson_true_life_dataset(mat_dir=mat_dir, out_dir=processed, first_n_cycles=100)
    metadata = pd.read_csv(processed / "cell_metadata.csv")
    result = build_feature_program_table(
        make_minimal_debug_program(),
        metadata,
        raw_root=processed,
        out_dir=tmp_path / "severson_feature_program",
    )
    return processed, result


def test_feature_program_builder_consumes_canonical_raw_path(tmp_path):
    processed, result = _processed_severson_with_features(tmp_path)
    table = pd.read_csv(result.feature_table_path)
    source_audit = pd.read_csv(tmp_path / "severson_feature_program" / "source_path_resolution.csv")

    assert result.n_rows == 6
    assert result.n_excluded_cells == 0
    assert "cycle_life" not in table.columns
    assert "label_source" not in table.columns
    assert source_audit["source_column"].eq("canonical_raw_path").all()
    assert source_audit["resolved_source_path"].astype(str).str.endswith(".json.gz").all()
    assert (processed / "canonical_raw_cells").is_dir()


def test_build_battery_feature_program_script_supports_severson_batch(tmp_path):
    mat_dir = write_toy_severson_mat_dir(tmp_path, n_files=1, n_cells_per_file=2)
    processed = tmp_path / "processed"
    build_severson_true_life_dataset(mat_dir=mat_dir, out_dir=processed, first_n_cycles=100)
    out = tmp_path / "script_feature_program"

    subprocess.run(
        [
            sys.executable,
            "scripts/build_battery_feature_program.py",
            "--processed-dir",
            str(processed),
            "--recipe",
            "minimal_debug",
            "--batch",
            "severson",
            "--out",
            str(out),
            "--first-n-cycles",
            "100",
            "--cycle-early-index",
            "9",
            "--cycle-late-index",
            "99",
            "--include-protocol-features",
            "false",
        ],
        check=True,
    )

    table = pd.read_csv(out / "feature_table.csv")
    assert len(table) == 2
    assert "cycle_life" not in table.columns


def test_scalar_baseline_feature_program_builds_on_ragged_severson_summary(tmp_path):
    mat_dir = write_toy_severson_mat_dir(tmp_path, n_files=1, n_cells_per_file=2, ragged_cycles=True)
    processed = tmp_path / "ragged_processed"
    build_severson_true_life_dataset(mat_dir=mat_dir, out_dir=processed, first_n_cycles=100)
    metadata = pd.read_csv(processed / "cell_metadata.csv")

    result = build_feature_program_table(
        make_scalar_baseline_program(first_n_cycles=100),
        metadata,
        raw_root=processed,
        out_dir=tmp_path / "scalar_baseline",
    )
    table = pd.read_csv(result.feature_table_path)

    assert result.n_rows == 2
    assert result.n_excluded_cells == 0
    assert result.n_numeric_feature_columns > 0
    assert "cycle_life" not in table.columns
    assert any(col.startswith("discharge_capacity") for col in table.columns)


def test_role_agent_workflow_trains_on_toy_severson_true_life_dataset(tmp_path):
    processed, result = _processed_severson_with_features(tmp_path)
    out = tmp_path / "role_severson"
    report = run_role_workflow(
        processed_dir=processed,
        reference_run=None,
        out=out,
        reports_dir=tmp_path / "reports",
        offline=True,
        split_mode="batch",
        iterations=1,
        candidates_per_iteration=2,
        include_feature_programs=True,
        feature_program_mode="table",
        feature_program_paths=[result.feature_table_path],
    )

    assert report["label_source"] == "true_measured_cycle_life"
    assert report["label_source_summary"]["true_measured_cycle_life_labels"] is True
    assert report["split_mode"] == "batch"
    assert report["validation_metrics"]["success"] is True
    assert "true_measured_cycle_life_labels: `True`" in (tmp_path / "reports" / "role_agent_workflow.md").read_text()


def test_severson_trained_role_workflow_locked_batch9_uses_holdout_after_search(tmp_path):
    processed, result = _processed_severson_with_features(tmp_path)
    batch9 = tmp_path / "2019-01-24_batch9"
    batch9.mkdir()
    for idx in range(3):
        write_toy_raw_cell(batch9 / f"2019-01-24_batch9_CH{idx + 1}_structure.json", n_cycles=120, row_offset=0.001 * idx)
    out = tmp_path / "role_severson_batch9"

    report = run_role_workflow(
        processed_dir=processed,
        reference_run=None,
        out=out,
        reports_dir=tmp_path / "reports",
        offline=True,
        split_mode="batch",
        iterations=1,
        candidates_per_iteration=2,
        include_feature_programs=True,
        feature_program_mode="table",
        feature_program_paths=[result.feature_table_path],
        final_batch9_validation=True,
        final_batch9_top_k=1,
        batch9_path=batch9,
    )

    assignments = pd.read_csv(out / "split_assignments.csv")
    assert not assignments.astype(str).apply(lambda col: col.str.contains("2019-01-24_batch9", regex=False)).any().any()
    assert report["locked_batch9_validation"]["status"] == "ok"
    assert (out / "final_batch9_predictions.csv").exists()
