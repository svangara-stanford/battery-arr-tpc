from __future__ import annotations

import json
import subprocess
import sys

import pandas as pd

import battery_aar.features.severson_matr as severson_matr
from battery_aar.features.severson_matr import TRUE_LABEL_SOURCE, build_severson_true_life_dataset
from battery_aar.workflows.role_graph import _label_source_summary

from .severson_test_utils import write_toy_severson_h5_mat_dir, write_toy_severson_mat_dir


def test_build_severson_true_life_dataset_outputs_processed_tables(tmp_path):
    mat_dir = write_toy_severson_mat_dir(tmp_path, n_files=2, n_cells_per_file=2)
    out = tmp_path / "processed"

    card = build_severson_true_life_dataset(mat_dir=mat_dir, out_dir=out, first_n_cycles=100)

    for name in ["cell_metadata.csv", "cycle_summary.csv", "labels.csv", "splits.csv", "dataset_card.json", "dataset_card.md", "exclusions.csv"]:
        assert (out / name).exists()
    assert (out / "canonical_raw_cells").is_dir()
    metadata = pd.read_csv(out / "cell_metadata.csv")
    labels = pd.read_csv(out / "labels.csv")
    cycles = pd.read_csv(out / "cycle_summary.csv")
    splits = pd.read_csv(out / "splits.csv")

    assert len(metadata) == 4
    assert len(cycles) == 400
    assert set(labels["label_source"]) == {TRUE_LABEL_SOURCE}
    assert set(metadata["label_source"]) == {TRUE_LABEL_SOURCE}
    assert metadata["source_dataset"].eq("severson_2019_true_life").all()
    assert "batch_id" in metadata.columns
    assert "canonical_raw_path" in metadata.columns
    assert set(splits["split"]).issubset({"train", "validation", "test"})
    assert card["label_source"] == TRUE_LABEL_SOURCE


def test_build_severson_true_life_dataset_script(tmp_path):
    mat_dir = write_toy_severson_mat_dir(tmp_path, n_files=1, n_cells_per_file=1)
    out = tmp_path / "processed_script"

    subprocess.run(
        [
            sys.executable,
            "scripts/build_severson_true_life_dataset.py",
            "--mat-dir",
            str(mat_dir),
            "--out",
            str(out),
            "--first-n-cycles",
            "100",
        ],
        check=True,
    )

    card = json.loads((out / "dataset_card.json").read_text())
    assert card["included_cells"] == 1
    assert card["label_source"] == TRUE_LABEL_SOURCE


def test_build_severson_true_life_dataset_handles_ragged_cycles(tmp_path):
    mat_dir = write_toy_severson_mat_dir(tmp_path, n_files=1, n_cells_per_file=1, ragged_cycles=True)
    out = tmp_path / "ragged_processed"

    card = build_severson_true_life_dataset(mat_dir=mat_dir, out_dir=out, first_n_cycles=100)
    metadata = pd.read_csv(out / "cell_metadata.csv")
    labels = pd.read_csv(out / "labels.csv")
    cycles = pd.read_csv(out / "cycle_summary.csv")

    assert card["included_cells"] == 1
    assert len(metadata) == 1
    assert len(labels) == 1
    assert len(cycles) == 100
    assert labels.loc[0, "label_source"] == TRUE_LABEL_SOURCE


def test_build_severson_true_life_dataset_includes_h5_v73_cells(tmp_path):
    mat_dir = write_toy_severson_h5_mat_dir(tmp_path, n_files=1, n_cells_per_file=2)
    out = tmp_path / "h5_processed"

    card = build_severson_true_life_dataset(mat_dir=mat_dir, out_dir=out, first_n_cycles=100)
    metadata = pd.read_csv(out / "cell_metadata.csv")
    labels = pd.read_csv(out / "labels.csv")
    cycles = pd.read_csv(out / "cycle_summary.csv")

    assert card["included_cells"] == 2
    assert len(metadata) == 2
    assert len(labels) == 2
    assert len(cycles) == 200
    assert set(labels["label_source"]) == {TRUE_LABEL_SOURCE}
    assert cycles["discharge_capacity"].notna().any()
    assert cycles["dc_internal_resistance"].notna().any()


def test_cycle_parse_failure_is_nonfatal_when_summary_and_life_are_valid(tmp_path, monkeypatch):
    mat_dir = write_toy_severson_mat_dir(tmp_path, n_files=1, n_cells_per_file=1)
    out = tmp_path / "nonfatal_processed"

    def fail_cycles(*args, **kwargs):
        raise ValueError("synthetic ragged failure")

    monkeypatch.setattr(severson_matr, "_cycles_interpolated_from_cell", fail_cycles)
    card = build_severson_true_life_dataset(mat_dir=mat_dir, out_dir=out, first_n_cycles=100)
    exclusions = pd.read_csv(out / "exclusions.csv")

    assert card["included_cells"] == 1
    assert card["excluded_cells"] == 0
    assert card["nonfatal_warning_rows"] >= 1
    assert exclusions["is_exclusion"].eq(False).any()
    assert exclusions["reason"].astype(str).str.contains("cycles_parse_failed_nonfatal").any()


def test_label_source_summary_distinguishes_true_life_and_oed_pseudolabels():
    true_meta = pd.DataFrame({"row_id": [0], "source_dataset": ["severson_2019_true_life"], "label_source": [TRUE_LABEL_SOURCE]})
    true_labels = pd.DataFrame({"row_id": [0], "y": [700.0], "label_source": [TRUE_LABEL_SOURCE]})
    pseudo_meta = pd.DataFrame({"row_id": [0], "label_source": ["author_model_prediction"]})
    pseudo_labels = pd.DataFrame({"row_id": [0], "y": [700.0], "label_source": ["author_model_prediction"]})

    true_summary = _label_source_summary(true_meta, true_labels)
    pseudo_summary = _label_source_summary(pseudo_meta, pseudo_labels)

    assert true_summary["true_measured_cycle_life_labels"] is True
    assert true_summary["author_model_prediction_pseudo_labels"] is False
    assert pseudo_summary["author_model_prediction_pseudo_labels"] is True
    assert pseudo_summary["true_measured_cycle_life_labels"] is False
