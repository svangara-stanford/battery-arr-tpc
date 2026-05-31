import json
import os
import subprocess
import sys

from battery_aar.paper_reproduction.paths import OED_BATCH_NAMES


def test_attia_reference_script_smoke_without_real_data(tmp_path):
    root = tmp_path / "battery-fast-charging"
    (root / "data").mkdir(parents=True)
    (root / "BMS-autoanalysis").mkdir()
    (root / "battery-fast-charging-optimization").mkdir()
    for name in OED_BATCH_NAMES:
        (root / "data" / name).mkdir()
    (root / "data" / "2019-01-24_batch9.zip").write_text("do not open")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{os.getcwd()}/src"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_attia_reference_reproduction.py",
            "--battery-fast-charging-root",
            str(root),
            "--out",
            str(tmp_path / "runs"),
            "--reports-dir",
            str(tmp_path / "reports"),
            "--smoke",
            "--allow-partial",
        ],
        check=False,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "reports" / "attia_reference_reproduction.json").read_text())
    assert report["validation_status"] == "skipped_batch9_zip_present"


def test_attia_reference_script_cleans_stale_owned_outputs(tmp_path):
    root = tmp_path / "battery-fast-charging"
    (root / "data").mkdir(parents=True)
    (root / "BMS-autoanalysis").mkdir()
    (root / "battery-fast-charging-optimization").mkdir()
    for name in OED_BATCH_NAMES:
        (root / "data" / name).mkdir()
    (root / "data" / "2019-01-24_batch9.zip").write_text("do not open")
    out = tmp_path / "runs"
    (out / "early_predictions").mkdir(parents=True)
    (out / "early_predictions" / "stale.csv").write_text("C1,C2,C3,C4,Prediction\n4,4,4,4,0.001\n")
    (out / "bayesgap").mkdir()
    (out / "bayesgap" / "0_next_batch.csv").write_text("stale\n")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{os.getcwd()}/src"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_attia_reference_reproduction.py",
            "--battery-fast-charging-root",
            str(root),
            "--out",
            str(out),
            "--reports-dir",
            str(tmp_path / "reports"),
            "--smoke",
            "--allow-partial",
        ],
        check=False,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert not (out / "early_predictions" / "stale.csv").exists()
    assert not (out / "bayesgap" / "0_next_batch.csv").exists()
