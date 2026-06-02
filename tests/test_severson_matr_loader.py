from __future__ import annotations

import json
import subprocess
import sys

from battery_aar.features.severson_matr import audit_severson_mat_dir, severson_cells_from_file

from .severson_test_utils import write_toy_severson_mat_dir


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
