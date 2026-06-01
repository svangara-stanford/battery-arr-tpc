import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.io import savemat

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


def _write_toy_model(path):
    savemat(
        path,
        {
            "mu": np.zeros(15),
            "sigma": np.ones(15),
            "feat_ind": np.array([1]),
            "B1": np.array([0.0]),
            "y_mu": 3.0,
            "MSE": 0.001,
            "t_val": 1.0,
            "des_mat": np.eye(2),
        },
    )


def _write_toy_cell(path, channel=1, protocol="OED\\20190124-4pt4_5pt6_5pt2_4pt252.sdu"):
    cycles = []
    voltage = []
    qd = []
    step = []
    for cycle in [10, 95, 98, 100]:
        v = np.linspace(2.8, 3.5, 30)
        q = np.linspace(1.0 - cycle * 0.0005, 0.001, 30)
        cycles.extend([cycle] * len(v))
        voltage.extend(v.tolist())
        qd.extend(q.tolist())
        step.extend(["discharge"] * len(v))
    payload = {
        "barcode": f"EL{channel:04d}",
        "channel_id": channel,
        "protocol": protocol,
        "summary": {
            "cycle_index": list(range(0, 102)),
            "discharge_capacity": [1.5] + [1.1 - 0.0005 * i for i in range(1, 102)],
        },
        "cycles_interpolated": {
            "cycle_index": cycles,
            "voltage": voltage,
            "discharge_capacity": qd,
            "step_type": step,
        },
    }
    path.write_text(json.dumps(payload))


def _make_toy_paper_root(tmp_path):
    root = tmp_path / "battery-fast-charging"
    data = root / "data"
    bms = root / "BMS-autoanalysis"
    (root / "battery-fast-charging-optimization").mkdir(parents=True)
    bms.mkdir(parents=True)
    data.mkdir(parents=True)
    _write_toy_model(bms / "oed_model.mat")
    _write_toy_model(bms / "oed_model_batch1.mat")
    for idx, name in enumerate(OED_BATCH_NAMES, start=1):
        batch = data / name
        batch.mkdir()
        _write_toy_cell(batch / f"{name}_CH1_structure.json", channel=idx)
    validation = data / "2019-01-24_batch9"
    validation.mkdir()
    _write_toy_cell(validation / "2019-01-24_batch9_CH1_structure.json", channel=9)
    return root, validation


def _run_reference(root, out, reports, include_validation=False, validation_path=None):
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{os.getcwd()}/src"
    cmd = [
        sys.executable,
        "scripts/run_attia_reference_reproduction.py",
        "--battery-fast-charging-root",
        str(root),
        "--out",
        str(out),
        "--reports-dir",
        str(reports),
        "--allow-partial",
        "--overwrite",
    ]
    if include_validation:
        cmd.append("--include-validation-batch")
        cmd.extend(["--validation-batch-path", str(validation_path)])
    result = subprocess.run(cmd, check=False, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_include_validation_batch_does_not_create_bayesgap_round5(tmp_path):
    root, validation = _make_toy_paper_root(tmp_path)
    out_four = tmp_path / "runs_four"
    out_val = tmp_path / "runs_val"
    reports_four = tmp_path / "reports_four"
    reports_val = tmp_path / "reports_val"

    _run_reference(root, out_four, reports_four)
    _run_reference(root, out_val, reports_val, include_validation=True, validation_path=validation)

    four_ranking = pd.read_csv(out_four / "bayesgap" / "final_posterior_ranking.csv")
    val_ranking = pd.read_csv(out_val / "bayesgap" / "final_posterior_ranking.csv")
    pd.testing.assert_frame_equal(four_ranking, val_ranking)
    assert len(val_ranking) == 224
    assert not list((out_val / "bayesgap").glob("round_5*"))
    assert (out_val / "validation_protocol_ranking.csv").exists()
    assert (out_val / "validation_metrics.json").exists()
    assert not (out_val / "early_predictions" / "2019-01-24_batch9.csv").exists()

    report = json.loads((reports_val / "attia_reference_reproduction.json").read_text())
    assert report["validation_status"] == "included_validation_batch"
    assert len(report["bayesgap_rounds"]) == 5
    assert all("batch9" not in str(row["consumed_prediction_file"]) for row in report["bayesgap_rounds"])

    four_check = pd.read_csv(out_four / "bayesgap" / "final_paper_top_protocol_check.csv")
    val_check = pd.read_csv(out_val / "bayesgap" / "final_paper_top_protocol_check.csv")
    pd.testing.assert_frame_equal(four_check, val_check)
