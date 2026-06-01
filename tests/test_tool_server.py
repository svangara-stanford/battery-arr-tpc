import json
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from battery_aar.tools.client import NativeToolClient
from battery_aar.tools.server import create_app


def _write_toy_processed(base: Path):
    base.mkdir(parents=True, exist_ok=True)
    metadata = pd.DataFrame(
        {
            "row_id": [0, 1, 2, 3],
            "cell_id": ["c0", "c1", "c2", "c3"],
            "batch_id": ["b0", "b0", "b1", "b1"],
            "protocol_readable": ["p0", "p1", "p0", "p1"],
            "cycle_life": [700.0, 720.0, 760.0, 780.0],
            "label_source": ["toy"] * 4,
        }
    )
    cycles = []
    for row_id in metadata["row_id"]:
        for cycle in range(1, 12):
            cycles.append(
                {
                    "row_id": row_id,
                    "cell_id": f"c{row_id}",
                    "cycle_index": cycle,
                    "discharge_capacity": 1.1 - 0.001 * cycle - 0.0002 * row_id,
                    "charge_capacity": None,
                }
            )
    metadata.to_csv(base / "cell_metadata.csv", index=False)
    pd.DataFrame(cycles).to_csv(base / "cycle_summary.csv", index=False)
    return metadata, pd.DataFrame(cycles)


def test_fastapi_health_and_tools():
    client = TestClient(create_app())

    health = client.get("/health")
    tools = client.get("/tools")

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert tools.status_code == 200
    payload = tools.json()
    assert payload["success"] is True
    assert {tool["name"] for tool in payload["tools"]} == {
        "profile_dataset",
        "build_battery_features",
        "review_candidate",
        "evaluate_candidate",
        "compare_runs",
    }


def test_native_tool_client_profile_dataset_writes_trace(tmp_path):
    processed = tmp_path / "processed"
    _write_toy_processed(processed)
    run_dir = tmp_path / "run"
    response = NativeToolClient().profile_dataset(
        run_id="native_profile",
        run_dir=str(run_dir),
        processed_dir=str(processed),
        data_source="toy",
    )

    assert response.success is True
    assert response.profile["metadata_row_count"] == 4
    assert response.output_artifact_ids
    tool_calls = (run_dir / "artifacts" / "tool_calls.jsonl").read_text().splitlines()
    assert len(tool_calls) == 1
    record = json.loads(tool_calls[0])
    assert record["tool_name"] == "profile_dataset"
    assert record["tool_call_id"] == response.tool_call_id
    assert record["success"] is True


def test_fastapi_dataset_profile_works_on_toy_data_and_logs(tmp_path):
    processed = tmp_path / "processed"
    _write_toy_processed(processed)
    run_dir = tmp_path / "run_http"
    client = TestClient(create_app())

    response = client.post(
        "/dataset/profile",
        json={
            "run_id": "http_profile",
            "run_dir": str(run_dir),
            "processed_dir": str(processed),
            "data_source": "toy",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["profile"]["metadata_row_count"] == 4
    assert (run_dir / "artifacts" / "dataset_profile.json").exists()
    assert (run_dir / "artifacts" / "tool_calls.jsonl").exists()


def test_failed_tool_call_is_logged_with_error_type_and_message(tmp_path):
    run_dir = tmp_path / "run_fail"
    client = TestClient(create_app())

    response = client.post(
        "/dataset/profile",
        json={
            "run_id": "failed_profile",
            "run_dir": str(run_dir),
            "processed_dir": str(tmp_path / "missing"),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is False
    assert payload["error_type"]
    assert payload["error_message"]
    records = (run_dir / "artifacts" / "tool_calls.jsonl").read_text().splitlines()
    assert records
    record = json.loads(records[-1])
    assert record["success"] is False
    assert record["error_type"] == payload["error_type"]
    assert record["error_message"] == payload["error_message"]
