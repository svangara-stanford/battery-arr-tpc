from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def toy_raw_payload(n_cycles: int = 101, n_points: int = 6, identical_late_curve: bool = False) -> dict[str, Any]:
    cycles = list(range(n_cycles))
    discharge = [1.12 - 0.001 * cycle for cycle in cycles]
    charge = [value + 0.02 for value in discharge]
    summary = {
        "cycle_index": cycles,
        "discharge_capacity": discharge,
        "charge_capacity": charge,
        "internal_resistance": [0.015 + 0.00001 * cycle for cycle in cycles],
        "temperature": [25.0 + 0.01 * cycle for cycle in cycles],
        "discharge_energy": [3.6 * value for value in discharge],
        "charge_energy": [3.7 * value for value in charge],
    }
    arrays: dict[str, list[Any]] = {
        "cycle_index": [],
        "step_type": [],
        "test_time": [],
        "voltage": [],
        "current": [],
        "charge_capacity": [],
        "discharge_capacity": [],
        "charge_energy": [],
        "discharge_energy": [],
        "internal_resistance": [],
        "temperature": [],
    }
    for cycle in cycles:
        for step in ["charge", "discharge"]:
            for point in range(n_points):
                frac = point / max(1, n_points - 1)
                if step == "discharge":
                    voltage = 4.2 - frac
                    q_base = discharge[9] if identical_late_curve and cycle == 99 else discharge[cycle]
                    qd = q_base * frac
                    qc = 0.0
                    current = -1.0
                else:
                    voltage = 3.0 + frac
                    qd = 0.0
                    qc = charge[cycle] * frac
                    current = 1.0
                arrays["cycle_index"].append(cycle)
                arrays["step_type"].append(step)
                arrays["test_time"].append(cycle * 1000 + point)
                arrays["voltage"].append(voltage)
                arrays["current"].append(current)
                arrays["charge_capacity"].append(qc)
                arrays["discharge_capacity"].append(qd)
                arrays["charge_energy"].append(qc * voltage)
                arrays["discharge_energy"].append(qd * voltage)
                arrays["internal_resistance"].append(summary["internal_resistance"][cycle])
                arrays["temperature"].append(summary["temperature"][cycle])
    return {"summary": summary, "cycles_interpolated": arrays}


def write_toy_raw_cell(path: Path, *, n_cycles: int = 101, row_offset: float = 0.0) -> Path:
    payload = toy_raw_payload(n_cycles=n_cycles)
    if row_offset:
        payload["summary"]["discharge_capacity"] = [value - row_offset for value in payload["summary"]["discharge_capacity"]]
        payload["summary"]["charge_capacity"] = [value - row_offset for value in payload["summary"]["charge_capacity"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def write_toy_feature_manifest(base: Path, n_cells: int = 3) -> tuple[pd.DataFrame, Path]:
    raw_root = base / "raw"
    rows = []
    for row_id in range(n_cells):
        raw_file = write_toy_raw_cell(raw_root / f"cell_{row_id}.json", row_offset=0.0005 * row_id)
        rows.append(
            {
                "row_id": row_id,
                "cell_id": f"cell_{row_id}",
                "source_path": str(raw_file),
                "batch_id": "toy_oed",
                "protocol_readable": f"p{row_id % 2}",
                "C1": 4.0 + 0.2 * row_id,
                "C2": 4.4,
                "C3": 4.8,
                "C4": 3.5,
            }
        )
    return pd.DataFrame(rows), raw_root


def write_toy_processed_from_manifest(base: Path, manifest: pd.DataFrame) -> Path:
    processed = base / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    metadata = manifest.copy()
    metadata["cycle_life"] = [900.0 - 20.0 * idx for idx in range(len(metadata))]
    metadata["label_source"] = "toy"
    metadata.to_csv(processed / "cell_metadata.csv", index=False)
    cycle_rows = []
    for _, row in metadata.iterrows():
        payload = json.loads(Path(row["source_path"]).read_text())
        summary = payload["summary"]
        for cycle, qd, qc in zip(summary["cycle_index"], summary["discharge_capacity"], summary["charge_capacity"]):
            if cycle <= 100:
                cycle_rows.append(
                    {
                        "row_id": row["row_id"],
                        "cell_id": row["cell_id"],
                        "cycle_index": cycle,
                        "discharge_capacity": qd,
                        "charge_capacity": qc,
                    }
                )
    pd.DataFrame(cycle_rows).to_csv(processed / "cycle_summary.csv", index=False)
    return processed
