from __future__ import annotations

import json
import subprocess
import sys

import h5py
import numpy as np

from battery_aar.features.severson_matr import (
    _h5_cell_string_field,
    _h5_decode_matlab_string,
    audit_severson_mat_dir,
    severson_cells_from_file,
)

from .severson_test_utils import write_toy_severson_h5_mat_dir, write_toy_severson_mat_dir


def test_severson_matr_loader_audits_toy_mat_files(tmp_path):
    mat_dir = write_toy_severson_mat_dir(tmp_path, n_files=1, n_cells_per_file=2)
    audit = audit_severson_mat_dir(mat_dir)

    assert audit["n_files"] == 1
    file_report = audit["files"][0]
    assert file_report["load_method"] == "scipy.io.loadmat"
    assert file_report["n_cells"] == 2
    assert file_report["true_cycle_life_present"] is True
    assert file_report["cells_with_first_100_cycles"] == 2
    assert file_report["field_availability"]["discharge_capacity"] is True
    assert file_report["field_availability"]["internal_resistance"] is True


def test_severson_cells_from_file_extracts_summary_curves_and_life(tmp_path):
    mat_dir = write_toy_severson_mat_dir(tmp_path, n_files=1, n_cells_per_file=1)
    cells, loaded = severson_cells_from_file(next(mat_dir.glob("*.mat")), first_n_cycles=100)

    assert loaded.keys == ["batch"]
    assert len(cells) == 1
    cell = cells[0]
    assert cell.cycle_life == 700.0
    assert len(cell.summary["cycle_index"]) == 100
    assert "discharge_capacity" in cell.summary
    assert "dc_internal_resistance" in cell.summary
    assert {"voltage", "current", "discharge_capacity", "step_type"}.issubset(cell.cycles_interpolated)


def test_severson_h5_v73_refs_parse_multiple_cells(tmp_path):
    mat_dir = write_toy_severson_h5_mat_dir(tmp_path, n_files=1, n_cells_per_file=3)
    cells, loaded = severson_cells_from_file(next(mat_dir.glob("*.mat")), first_n_cycles=100)

    assert loaded.load_method == "h5py.File"
    assert loaded.keys == ["#refs#", "batch", "batch_date"]
    assert len(cells) == 3
    assert [cell.cycle_life for cell in cells] == [700.0, 705.0, 710.0]
    assert all(len(cell.summary["cycle_index"]) == 100 for cell in cells)
    assert all("discharge_capacity" in cell.summary for cell in cells)
    assert all("dc_internal_resistance" in cell.summary for cell in cells)
    assert all("charge_duration" in cell.summary for cell in cells)
    assert {"voltage", "current", "discharge_capacity", "step_type"}.issubset(cells[0].cycles_interpolated)


def test_severson_h5_audit_counts_referenced_cells(tmp_path):
    mat_dir = write_toy_severson_h5_mat_dir(tmp_path, n_files=1, n_cells_per_file=2)
    audit = audit_severson_mat_dir(mat_dir)

    file_report = audit["files"][0]
    assert file_report["load_method"] == "h5py.File"
    assert file_report["n_cells"] == 2
    assert file_report["true_cycle_life_present"] is True
    assert file_report["cycle_life_min"] == 700.0
    assert file_report["cycle_life_max"] == 705.0
    assert file_report["cells_with_first_100_cycles"] == 2
    assert file_report["field_availability"]["discharge_capacity"] is True


def test_h5_decode_matlab_string_huge_integer_returns_none(tmp_path):
    path = tmp_path / "bad_string.mat"
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset("bad", data=np.asarray([[np.iinfo(np.uint64).max]], dtype=np.uint64))
        assert _h5_decode_matlab_string(handle, dataset) is None


def test_h5_cell_string_field_bad_barcode_is_nonfatal(tmp_path):
    mat_dir = write_toy_severson_h5_mat_dir(tmp_path, n_files=1, n_cells_per_file=1, invalid_barcode=True)
    path = next(mat_dir.glob("*.mat"))
    warnings = []
    with h5py.File(path, "r") as handle:
        text = _h5_cell_string_field(handle, handle["batch"], 0, ["barcode"], warnings)

    assert text is None
    assert any("barcode_decode_failed_nonfatal" in warning for warning in warnings)


def test_severson_h5_invalid_barcode_uses_fallback_cell_id(tmp_path):
    mat_dir = write_toy_severson_h5_mat_dir(tmp_path, n_files=1, n_cells_per_file=1, invalid_barcode=True)
    path = next(mat_dir.glob("*.mat"))
    cells, loaded = severson_cells_from_file(path, first_n_cycles=100)

    assert loaded.load_method == "h5py.File"
    assert len(cells) == 1
    cell = cells[0]
    assert cell.cycle_life == 700.0
    assert cell.cell_id == f"{path.stem}_cell_000"
    assert cell.metadata["barcode"] is None
    assert len(cell.summary["cycle_index"]) == 100
    assert any("barcode_decode_failed_nonfatal" in warning for warning in cell.warnings)


def test_audit_severson_matr_data_script_writes_reports(tmp_path):
    mat_dir = write_toy_severson_mat_dir(tmp_path, n_files=1, n_cells_per_file=1)
    out = tmp_path / "reports" / "severson_matr_audit"

    subprocess.run(
        [
            sys.executable,
            "scripts/audit_severson_matr_data.py",
            "--mat-dir",
            str(mat_dir),
            "--out",
            str(out),
        ],
        check=True,
    )

    payload = json.loads((tmp_path / "reports" / "severson_matr_audit.json").read_text())
    assert payload["files"][0]["n_cells"] == 1
    assert (tmp_path / "reports" / "severson_matr_audit.md").exists()
