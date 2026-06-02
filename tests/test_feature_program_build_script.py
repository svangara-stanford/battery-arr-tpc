from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

from .feature_program_test_utils import write_toy_feature_manifest, write_toy_processed_from_manifest


def test_build_battery_feature_program_script_builds_minimal_debug_table(tmp_path):
    manifest, _raw_root = write_toy_feature_manifest(tmp_path, n_cells=2)
    processed = write_toy_processed_from_manifest(tmp_path, manifest)
    paper_root = tmp_path / "battery-fast-charging"
    paper_root.mkdir()
    out = tmp_path / "feature_program_out"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_battery_feature_program.py",
            "--battery-fast-charging-root",
            str(paper_root),
            "--processed-dir",
            str(processed),
            "--recipe",
            "minimal_debug",
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
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    table = pd.read_csv(out / "feature_table.csv")
    metadata = pd.read_csv(out / "feature_metadata.csv")
    assert len(table) == 2
    assert "capacity_summary" in set(metadata["feature_family"])
    assert (out / "dataset_card.md").exists()
