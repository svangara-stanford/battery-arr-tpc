import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from battery_aar.agents.attia_data_bridge import build_agent_surrogate_dataset, load_author_validation_metrics
from battery_aar.agents.evaluator import battery_pgr
from battery_aar.agents.orchestrator import run_rediscovery
from battery_aar.paper_reproduction.paths import OED_BATCH_NAMES


def _write_toy_cell(path, channel=1, protocol="OED\\20190124-4pt4_5pt6_5pt2_4pt252.sdu", n_cycles=130):
    q = [1.10 - 0.001 * i for i in range(1, n_cycles + 1)]
    payload = {
        "barcode": f"EL{channel:04d}",
        "channel_id": channel,
        "protocol": protocol,
        "summary": {
            "cycle_index": list(range(1, n_cycles + 1)),
            "discharge_capacity": q,
        },
        "cycles_interpolated": {
            "cycle_index": [10],
            "voltage": [3.1],
            "discharge_capacity": [1.0],
            "step_type": ["discharge"],
        },
    }
    path.write_text(json.dumps(payload))


def _write_malformed_cell_without_summary(path, channel=1, protocol="OED\\20190124-4pt4_5pt6_5pt2_4pt252.sdu"):
    path.write_text(json.dumps({"barcode": f"EL{channel:04d}", "channel_id": channel, "protocol": protocol}))


def _make_toy_root_and_reference(tmp_path, n_cells_per_batch=2):
    root = tmp_path / "battery-fast-charging"
    data = root / "data"
    (root / "BMS-autoanalysis").mkdir(parents=True)
    (root / "battery-fast-charging-optimization").mkdir()
    data.mkdir()
    reference = tmp_path / "runs" / "attia_reference_reproduction_with_validation"
    rich_dir = reference / "early_predictions_rich"
    rich_dir.mkdir(parents=True)
    for batch_idx, batch_name in enumerate(OED_BATCH_NAMES):
        batch = data / batch_name
        batch.mkdir()
        rows = []
        for channel in range(1, n_cells_per_batch + 1):
            _write_toy_cell(batch / f"{batch_name}_CH{channel}_structure.json", channel=channel)
            rows.append(
                {
                    "cell_id": f"{batch_name}_CH{channel}",
                    "batch_id": batch_name,
                    "channel": channel,
                    "protocol_readable": "4.4-5.6-5.2-4.252",
                    "C1": 4.4,
                    "C2": 5.6,
                    "C3": 5.2,
                    "C4": 4.252,
                    "Prediction": 700.0 + 25 * batch_idx + channel,
                    "anomaly_flag": False,
                }
            )
        pd.DataFrame(rows).to_csv(rich_dir / f"{batch_name}.csv", index=False)
    batch9 = data / "2019-01-24_batch9"
    batch9.mkdir()
    for channel in range(1, 4):
        _write_toy_cell(batch9 / f"2019-01-24_batch9_CH{channel}_structure.json", channel=channel)
    return root, reference, batch9


def test_build_agent_dataset_from_attia_reference_creates_processed_tables(tmp_path):
    root, reference, _ = _make_toy_root_and_reference(tmp_path)
    out = tmp_path / "processed"
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{os.getcwd()}/src"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_agent_dataset_from_attia_reference.py",
            "--battery-fast-charging-root",
            str(root),
            "--reference-run",
            str(reference),
            "--out",
            str(out),
            "--first-n-cycles",
            "100",
            "--exclude-batch9",
        ],
        check=False,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["valid_labeled_cells"] == 8
    assert summary["cycle_summary_rows"] > 0
    assert (out / "cell_metadata.csv").exists()
    assert (out / "cycle_summary.csv").exists()
    assert (out / "splits.csv").exists()
    assert (out / "dataset_card.json").exists()
    assert (out / "exclusions.csv").exists()
    metadata = pd.read_csv(out / "cell_metadata.csv")
    cycle_summary = pd.read_csv(out / "cycle_summary.csv")
    card = json.loads((out / "dataset_card.json").read_text())
    assert len(metadata) == 8
    assert not cycle_summary.empty
    assert "charge_capacity" in cycle_summary.columns
    assert cycle_summary["charge_capacity"].isna().all()
    assert set(metadata["label_source"]) == {"author_model_prediction"}
    assert card["batch9_included"] is False
    assert card["charge_capacity_status"] == "all_nan"
    assert "2019-01-24_batch9" not in set(metadata["batch_id"])


def test_build_agent_dataset_skips_malformed_labeled_raw_cell_and_filters_bad_labels(tmp_path):
    root, reference, _ = _make_toy_root_and_reference(tmp_path, n_cells_per_batch=1)
    batch_name = OED_BATCH_NAMES[0]
    batch = root / "data" / batch_name
    _write_malformed_cell_without_summary(batch / f"{batch_name}_CH2_structure.json", channel=2)
    _write_toy_cell(batch / f"{batch_name}_CH3_structure.json", channel=3)
    _write_toy_cell(batch / f"{batch_name}_CH4_structure.json", channel=4)
    pred_path = reference / "early_predictions_rich" / f"{batch_name}.csv"
    pd.DataFrame(
        [
            {
                "cell_id": f"{batch_name}_CH1_structure",
                "batch_id": batch_name,
                "channel": 1,
                "protocol_readable": "4.4-5.6-5.2-4.252",
                "C1": 4.4,
                "C2": 5.6,
                "C3": 5.2,
                "C4": 4.252,
                "Prediction": 701.0,
                "anomaly_flag": False,
            },
            {
                "cell_id": f"{batch_name}_CH2",
                "batch_id": batch_name,
                "channel": 2,
                "protocol_readable": "4.4-5.6-5.2-4.252",
                "C1": 4.4,
                "C2": 5.6,
                "C3": 5.2,
                "C4": 4.252,
                "Prediction": 702.0,
                "anomaly_flag": False,
            },
            {
                "cell_id": f"{batch_name}_CH3",
                "batch_id": batch_name,
                "channel": 3,
                "protocol_readable": "4.4-5.6-5.2-4.252",
                "C1": 4.4,
                "C2": 5.6,
                "C3": 5.2,
                "C4": 4.252,
                "Prediction": -5.0,
                "anomaly_flag": False,
            },
            {
                "cell_id": f"{batch_name}_CH4",
                "batch_id": batch_name,
                "channel": 4,
                "protocol_readable": "4.4-5.6-5.2-4.252",
                "C1": 4.4,
                "C2": 5.6,
                "C3": 5.2,
                "C4": 4.252,
                "Prediction": 704.0,
                "anomaly_flag": True,
            },
        ]
    ).to_csv(pred_path, index=False)

    out = tmp_path / "processed"
    result = build_agent_surrogate_dataset(root, reference, out, first_n_cycles=100)
    metadata = pd.read_csv(out / "cell_metadata.csv")
    cycle_summary = pd.read_csv(out / "cycle_summary.csv")
    exclusions = pd.read_csv(out / "exclusions.csv")
    card = json.loads((out / "dataset_card.json").read_text())

    assert not metadata.empty
    assert not cycle_summary.empty
    assert 701.0 in metadata["cycle_life"].to_numpy(float)
    assert 702.0 not in metadata["cycle_life"].to_numpy(float)
    assert -5.0 not in metadata["cycle_life"].to_numpy(float)
    assert "2019-01-24_batch9" not in set(metadata["batch_id"])
    assert "raw_parse" in set(exclusions["exclusion_stage"])
    assert exclusions["reason"].str.contains("no summary table").any()
    assert "label_filter" in set(exclusions["exclusion_stage"])
    assert exclusions["reason"].str.contains("prediction_le_0").any()
    assert exclusions["reason"].str.contains("anomaly_flag_true").any()
    assert card["skipped_raw_cells"] >= 1
    assert card["skipped_label_rows"] >= 2
    assert result.dataset_card["n_cells"] == len(metadata)


def test_require_real_data_cli_fails_when_processed_data_missing(tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{os.getcwd()}/src"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_agentic_rediscovery.py",
            "--processed-dir",
            str(tmp_path / "missing"),
            "--out",
            str(tmp_path / "run"),
            "--reports-dir",
            str(tmp_path / "reports"),
            "--offline",
            "--agents",
            "1",
            "--iterations",
            "1",
            "--require-real-data",
        ],
        check=False,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Real processed data required" in result.stderr


def test_require_real_data_uses_processed_dataset_without_synthetic_fallback(tmp_path):
    root, reference, _ = _make_toy_root_and_reference(tmp_path)
    processed = tmp_path / "processed"
    build_agent_surrogate_dataset(root, reference, processed, first_n_cycles=100)

    report = run_rediscovery(
        processed_dir=processed,
        reference_run=reference,
        out=tmp_path / "run",
        reports_dir=tmp_path / "reports",
        agents=1,
        iterations=1,
        offline=True,
        require_real_data=True,
        seed=1,
    )
    assert report["real_data_used"] is True
    assert report["synthetic_fallback_used"] is False
    assert report["label_source"] == "author_model_prediction"


def test_final_batch9_validation_writes_outputs_and_does_not_change_leaderboard(tmp_path):
    root, reference, batch9 = _make_toy_root_and_reference(tmp_path)
    processed = tmp_path / "processed"
    build_agent_surrogate_dataset(root, reference, processed, first_n_cycles=100)
    (reference / "validation_metrics.json").write_text(
        json.dumps({"early_prediction_vs_observed": {"rmse": 150.0, "mae": 100.0}}) + "\n"
    )

    common = dict(
        processed_dir=processed,
        reference_run=reference,
        reports_dir=tmp_path / "reports",
        agents=1,
        iterations=1,
        offline=True,
        require_real_data=True,
        seed=2,
    )
    run_rediscovery(out=tmp_path / "run_no_final", **common)
    report = run_rediscovery(
        out=tmp_path / "run_final",
        final_batch9_validation=True,
        battery_fast_charging_root=root,
        batch9_path=batch9,
        **common,
    )

    base = pd.read_csv(tmp_path / "run_no_final" / "leaderboard.csv")
    final = pd.read_csv(tmp_path / "run_final" / "leaderboard.csv")
    pd.testing.assert_frame_equal(base.drop(columns=["candidate_path"]), final.drop(columns=["candidate_path"]))
    assert (tmp_path / "run_final" / "final_batch9_predictions.csv").exists()
    assert (tmp_path / "run_final" / "final_batch9_metrics.json").exists()
    assert (tmp_path / "run_final" / "final_batch9_protocol_metrics.csv").exists()
    split_assignments = pd.read_csv(tmp_path / "run_final" / "split_assignments.csv")
    assert "2019-01-24_batch9" not in set(split_assignments.get("batch_id", pd.Series(dtype=str)).astype(str))
    assert report["batch9_status"] == "locked_validation_ok"
    assert report["final_batch9_validation"]["status"] == "ok"
    assert report["locked_batch9_rmse"] == report["final_batch9_metrics"]["rmse"]
    assert report["locked_batch9_weak_baseline_rmse"] == report["final_batch9_metrics"]["batch9_weak_baseline_rmse"]
    assert report["author_literature_batch9_rmse"] == 150.0
    assert report["final_batch9_metrics"]["author_model_batch9_rmse"] == 150.0
    expected_pgr = battery_pgr(
        report["locked_batch9_weak_baseline_rmse"],
        report["locked_batch9_rmse"],
        report["author_literature_batch9_rmse"],
    )
    assert report["final_batch9_metrics"]["battery_pgr_author_model_batch9"] == expected_pgr
    assert report["batch9_pgr"] == expected_pgr

    summary = pd.read_csv(tmp_path / "reports" / "agent_rediscovery_runs_summary.csv")
    assert set(summary["run_id"]) == {"run_no_final", "run_final"}
    row = summary.set_index("run_id").loc["run_final"]
    assert row["split_mode"] == "random"
    assert row["agents"] == 1
    assert row["iterations"] == 1
    assert np.isfinite(row["surrogate_rmse"])
    assert row["batch9_rmse"] == report["locked_batch9_rmse"]
    assert row["batch9_weak_rmse"] == report["locked_batch9_weak_baseline_rmse"]
    assert row["author_rmse"] == 150.0
    assert row["pgr"] == expected_pgr

    report_md = (tmp_path / "reports" / "agent_rediscovery.md").read_text()
    assert "batch9_status: `locked_validation_ok`" in report_md
    assert "surrogate-search validation RMSE" in report_md
    assert "best agent locked Batch 9 RMSE" in report_md
    assert "author/literature model Batch 9 RMSE" in report_md
    assert "Batch 9 weak baseline RMSE" in report_md
    assert "Batch 9 was not used during surrogate search" in report_md
    assert "## Run Comparison" in report_md


def test_author_validation_metrics_load_and_missing_pgr_is_undefined(tmp_path):
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "validation_metrics.json").write_text(
        json.dumps({"early_prediction_vs_observed": {"rmse": 123.0, "mae": 45.0}}) + "\n"
    )
    metrics = load_author_validation_metrics(reference)
    assert metrics["available"] is True
    assert metrics["author_model_batch9_rmse"] == 123.0

    missing = load_author_validation_metrics(tmp_path / "missing_reference")
    assert missing["available"] is False
