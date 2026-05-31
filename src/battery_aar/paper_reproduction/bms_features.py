from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.stats import skew


class CurveArrayError(ValueError):
    pass


@dataclass
class BatteryCell:
    cell_id: str
    batch_id: str
    channel: int | None
    barcode: str | None
    protocol_readable: str | None
    C1: float | None
    C2: float | None
    C3: float | None
    C4: float | None
    q_discharge: np.ndarray
    qdlin_by_cycle: dict[int, np.ndarray]
    vdlin: np.ndarray
    lifetime: float | None = None
    source_path: Path | None = None


def parse_protocol_readable(protocol: str | None) -> str | None:
    if not protocol:
        return None
    name = Path(protocol.replace("\\", "/")).name
    stem = name.removesuffix(".sdu")
    if "-" in stem:
        stem = stem.split("-", 1)[1]
    stem = stem.replace("pt", ".").replace("_", "-")
    parts = stem.split("-")
    if len(parts) >= 4:
        return "-".join(parts[:4])
    return None


def parse_policy_currents(protocol_readable: str | None) -> tuple[float | None, float | None, float | None, float | None]:
    if not protocol_readable:
        return (None, None, None, None)
    parts = protocol_readable.split("-")
    if len(parts) < 4:
        return (None, None, None, None)
    try:
        return tuple(float(part) for part in parts[:4])  # type: ignore[return-value]
    except ValueError:
        return (None, None, None, None)


def _array(values: Any, dtype: type = float) -> np.ndarray:
    return np.asarray(values, dtype=dtype)


def _extract_lifetime(summary: dict[str, Any]) -> float | None:
    q = np.asarray(summary.get("discharge_capacity", []), dtype=float)
    if q.size == 0:
        return None
    below = np.where(q < 0.8 * q[1] if q.size > 1 else q < 0.8 * q[0])[0]
    if below.size:
        return float(below[0] + 1)
    return float(q.size)


def _extract_qdlin(ci: dict[str, Any], cycle_index: int) -> tuple[np.ndarray, np.ndarray]:
    cycles = _array(ci.get("cycle_index", []), int)
    step_type = np.asarray(ci.get("step_type", []), dtype=object)
    voltage = _array(ci.get("voltage", []), float)
    q_dis = _array(ci.get("discharge_capacity", []), float)
    if cycles.size == 0 or step_type.size == 0:
        raise CurveArrayError("cycles_interpolated lacks cycle_index/step_type arrays")
    mask = (cycles == cycle_index) & (step_type == "discharge") & np.isfinite(voltage) & np.isfinite(q_dis)
    if not np.any(mask):
        raise CurveArrayError(f"Qdlin/Vdlin arrays not found for cycle_index={cycle_index}")
    return q_dis[mask].astype(float), voltage[mask].astype(float)


def load_cell_from_json(path: str | Path, required_cycles: tuple[int, ...] = (10,)) -> BatteryCell:
    source = Path(path)
    data = json.loads(source.read_text())
    summary = data.get("summary") or {}
    cycles_interpolated = data.get("cycles_interpolated") or {}
    if not summary or not cycles_interpolated:
        raise CurveArrayError(f"{source} does not contain summary and cycles_interpolated arrays")

    protocol = data.get("protocol") or data.get("metadata", {}).get("protocol")
    protocol_readable = parse_protocol_readable(protocol)
    c1, c2, c3, c4 = parse_policy_currents(protocol_readable)
    channel = data.get("channel_id", data.get("metadata", {}).get("channel_id"))
    try:
        channel_int = int(channel)
    except Exception:
        channel_int = None
    barcode = data.get("barcode") or data.get("metadata", {}).get("barcode")
    batch_id = source.parent.name
    cell_id = f"{batch_id}_CH{channel_int}" if channel_int is not None else source.stem

    q_discharge = _array(summary.get("discharge_capacity", []), float)
    if q_discharge.size == 0:
        raise CurveArrayError(f"{source} does not contain summary.discharge_capacity")

    qdlin_by_cycle: dict[int, np.ndarray] = {}
    q10, vdlin = _extract_qdlin(cycles_interpolated, 10)
    qdlin_by_cycle[10] = q10
    for cycle in sorted(set(required_cycles) - {10}):
        qdlin_by_cycle[cycle], _ = _extract_qdlin(cycles_interpolated, cycle)

    return BatteryCell(
        cell_id=cell_id,
        batch_id=batch_id,
        channel=channel_int,
        barcode=str(barcode) if barcode is not None else None,
        protocol_readable=protocol_readable,
        C1=c1,
        C2=c2,
        C3=c3,
        C4=c4,
        q_discharge=q_discharge,
        qdlin_by_cycle=qdlin_by_cycle,
        vdlin=vdlin,
        lifetime=_extract_lifetime(summary),
        source_path=source,
    )


def load_cells_from_batch(batch_path: str | Path, max_cells: int | None = None, required_cycles: tuple[int, ...] = (10,)) -> list[BatteryCell]:
    cells, _ = load_cells_from_batch_with_status(batch_path, max_cells=max_cells, required_cycles=required_cycles)
    return cells


def load_cells_from_batch_with_status(
    batch_path: str | Path,
    max_cells: int | None = None,
    required_cycles: tuple[int, ...] = (10,),
) -> tuple[list[BatteryCell], list[dict[str, str]]]:
    batch = Path(batch_path)
    cells: list[BatteryCell] = []
    excluded: list[dict[str, str]] = []
    for path in sorted(batch.glob("*_structure.json")):
        try:
            cells.append(load_cell_from_json(path, required_cycles=required_cycles))
        except Exception as exc:
            excluded.append({"path": str(path), "reason": str(exc)})
            continue
        if max_cells is not None and len(cells) >= max_cells:
            break
    return cells, excluded


def ensure_cutoff_cycle_loaded(cell: BatteryCell, cutoff_cycle: int) -> BatteryCell:
    if cutoff_cycle in cell.qdlin_by_cycle:
        return cell
    if cell.source_path is None:
        raise CurveArrayError(f"cell {cell.cell_id} lacks qdlin for cycle_index={cutoff_cycle}")
    data = json.loads(cell.source_path.read_text())
    qn, _ = _extract_qdlin(data.get("cycles_interpolated") or {}, cutoff_cycle)
    cell.qdlin_by_cycle[cutoff_cycle] = qn
    return cell


def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if x.size != y.size or x.size < 2:
        raise CurveArrayError("linear fit requires at least two matching points")
    design = np.column_stack([x, np.ones_like(x)])
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(slope), float(intercept)


def _log10_abs(value: float) -> float:
    value = abs(float(value))
    if not np.isfinite(value) or value == 0:
        return float("nan")
    return math.log10(value)


def build_feature_vector(cell: BatteryCell, cutoff_cycle: int) -> tuple[np.ndarray, str]:
    ensure_cutoff_cycle_loaded(cell, cutoff_cycle)
    q = np.asarray(cell.q_discharge, dtype=float)
    if q.size < cutoff_cycle:
        raise CurveArrayError(f"{cell.cell_id} has {q.size} summary cycles, needs {cutoff_cycle}")

    q10 = np.asarray(cell.qdlin_by_cycle[10], dtype=float)
    qn = np.asarray(cell.qdlin_by_cycle[cutoff_cycle], dtype=float)
    vdlin = np.asarray(cell.vdlin, dtype=float)
    n = min(q10.size, qn.size, vdlin.size)
    if n == 0:
        raise CurveArrayError(f"{cell.cell_id} has empty Qdlin/Vdlin arrays")
    q10 = q10[:n]
    qn = qn[:n]
    vdlin = vdlin[:n]
    if not (np.all(np.isfinite(q10)) and np.all(np.isfinite(qn)) and np.all(np.isfinite(vdlin))):
        raise CurveArrayError(f"{cell.cell_id} has non-finite Qdlin/Vdlin values")

    features = np.zeros(15, dtype=float)
    features[0] = q[1]
    features[1] = np.nanmax(q[:cutoff_cycle]) - q[1]
    features[2] = q[cutoff_cycle - 1]

    cycles_full = np.arange(2, cutoff_cycle + 1, dtype=float)
    slope, intercept = _linear_fit(cycles_full, q[1:cutoff_cycle])
    features[3] = slope
    features[4] = intercept

    late_start = 91
    cycles_late = np.arange(late_start, cutoff_cycle + 1, dtype=float)
    late_q = q[late_start - 1 : cutoff_cycle]
    slope, intercept = _linear_fit(cycles_late, late_q)
    features[5] = slope
    features[6] = intercept

    qdiff = qn - q10
    features[7] = _log10_abs(np.nanmin(qdiff))
    features[8] = _log10_abs(np.nanmean(qdiff))
    features[9] = _log10_abs(np.nanvar(qdiff, ddof=1))
    features[10] = _log10_abs(skew(qdiff, bias=True, nan_policy="omit"))
    features[11] = _log10_abs(qdiff[0])
    features[12] = _log10_abs(np.nansum(np.abs(qdiff)))
    features[13] = _log10_abs(np.nansum(qdiff**2))

    # MATLAB code calls trapz(Qdlin, Vdlin). Some BEEP JSON exports encode the
    # same curve orientation such that E10 - EN is negative; feature 15 is not
    # selected by the provided OED models, so preserve that as an unavailable
    # scalar instead of silently changing the author formula.
    e10 = float(trapezoid(vdlin, q10))
    en = float(trapezoid(vdlin, qn))
    energy_delta = e10 - en
    status = "ok"
    if energy_delta > 0 and np.isfinite(energy_delta):
        features[14] = math.log10(energy_delta)
    else:
        features[14] = float("nan")
        status = "ok_unselected_energy_feature_unavailable"
    return features, status


def build_feature_matrix(cells: list[BatteryCell], cutoff_cycle: int) -> tuple[np.ndarray, pd.DataFrame]:
    rows: list[np.ndarray] = []
    statuses: list[dict[str, Any]] = []
    for cell in cells:
        try:
            vec, status = build_feature_vector(cell, cutoff_cycle)
            rows.append(vec)
            statuses.append({"cell_id": cell.cell_id, "status": status, "error": ""})
        except Exception as exc:
            rows.append(np.full(15, np.nan))
            statuses.append({"cell_id": cell.cell_id, "status": "unavailable", "error": str(exc)})
    return np.vstack(rows) if rows else np.empty((0, 15)), pd.DataFrame(statuses)
