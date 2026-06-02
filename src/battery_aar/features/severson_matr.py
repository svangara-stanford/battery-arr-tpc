from __future__ import annotations

import gzip
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat


SEVERSON_DATASET_NAME = "severson_2019_true_life"
TRUE_LABEL_SOURCE = "true_measured_cycle_life"


SUMMARY_ALIASES = {
    "cycle_index": ["cycle_index", "cycle", "cycle_number", "Cycle", "cycles"],
    "discharge_capacity": ["discharge_capacity", "QDischarge", "Qd", "QD", "q_discharge"],
    "charge_capacity": ["charge_capacity", "QCharge", "Qc", "QC", "q_charge"],
    "discharge_energy": ["discharge_energy", "EDischarge", "Ed", "Edischarge"],
    "charge_energy": ["charge_energy", "ECharge", "Ec", "Echarge"],
    "dc_internal_resistance": ["dc_internal_resistance", "internal_resistance", "IR", "DCIR", "resistance"],
    "temperature_average": ["temperature_average", "Tavg", "Tmean", "temperature", "temp_avg"],
    "temperature_minimum": ["temperature_minimum", "Tmin", "temp_min"],
    "temperature_maximum": ["temperature_maximum", "Tmax", "temp_max"],
}


CYCLE_ALIASES = {
    "voltage": ["voltage", "V", "v"],
    "test_time": ["test_time", "time", "t"],
    "current": ["current", "I", "i"],
    "charge_capacity": ["charge_capacity", "Qc", "QCharge", "q_charge"],
    "discharge_capacity": ["discharge_capacity", "Qd", "QDischarge", "q_discharge"],
    "charge_energy": ["charge_energy", "Ec", "ECharge"],
    "discharge_energy": ["discharge_energy", "Ed", "EDischarge"],
    "internal_resistance": ["internal_resistance", "dc_internal_resistance", "IR", "resistance"],
    "temperature": ["temperature", "T", "temp"],
    "step_type": ["step_type", "step", "state"],
}


LIFE_KEYS = ["cycle_life", "cycleLife", "CycleLife", "life", "lifetime", "cycle_to_failure", "cycles_to_failure"]


@dataclass
class MatLoadResult:
    path: Path
    load_method: str
    keys: List[str]
    data: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


@dataclass
class SeversonCell:
    source_file: str
    source_batch: str
    source_cell_index: int
    cell_id: str
    cycle_life: Optional[float]
    metadata: Dict[str, Any]
    summary: Dict[str, Any]
    cycles_interpolated: Dict[str, Any]
    warnings: List[str]


def discover_matr_files(mat_dir: Union[str, Path]) -> List[Path]:
    return sorted(Path(mat_dir).glob("*.mat"))


def _mat_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _decode_bytes(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _mat_to_python(value: Any) -> Any:
    value = _decode_bytes(value)
    if hasattr(value, "_fieldnames"):
        return {field: _mat_to_python(getattr(value, field)) for field in value._fieldnames}
    if isinstance(value, np.void) and value.dtype.names:
        return {field: _mat_to_python(value[field]) for field in value.dtype.names}
    if isinstance(value, np.ndarray):
        if value.dtype.names:
            flat = [_mat_to_python(item) for item in value.ravel()]
            return flat[0] if value.size == 1 else flat
        if value.dtype.kind in {"U", "S"}:
            if value.ndim == 0:
                return str(_decode_bytes(value.item()))
            return "".join(str(_decode_bytes(item)) for item in value.ravel()).strip()
        if value.dtype == object:
            flat = [_mat_to_python(item) for item in value.ravel()]
            return flat[0] if value.size == 1 else flat
        if value.ndim == 0:
            return _mat_scalar(value.item())
        return np.asarray(value).squeeze()
    if isinstance(value, (list, tuple)):
        return [_mat_to_python(item) for item in value]
    return _mat_scalar(value)


def _h5_to_python(obj: Any, root: Optional[h5py.File] = None) -> Any:
    if isinstance(obj, h5py.Group):
        return {key: _h5_to_python(obj[key], root or obj.file) for key in obj.keys()}
    if isinstance(obj, h5py.Dataset):
        data = obj[()]
        if isinstance(data, bytes):
            return data.decode("utf-8", errors="replace")
        if getattr(data, "dtype", None) is not None and data.dtype.kind == "O" and root is not None:
            values = []
            for ref in np.asarray(data).ravel():
                try:
                    values.append(_h5_to_python(root[ref], root))
                except Exception:
                    values.append(None)
            return values
        if getattr(data, "dtype", None) is not None and data.dtype.kind in {"S", "U"}:
            return "".join(str(_decode_bytes(item)) for item in np.asarray(data).ravel()).strip()
        return np.asarray(data).squeeze()
    return obj


def load_matr_file(path: Union[str, Path]) -> MatLoadResult:
    mat_path = Path(path)
    try:
        raw = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        data = {key: _mat_to_python(value) for key, value in raw.items() if not key.startswith("__")}
        return MatLoadResult(path=mat_path, load_method="scipy.io.loadmat", keys=sorted(data), data=data)
    except NotImplementedError as exc:
        warnings = [f"scipy.io.loadmat requires h5py fallback: {exc}"]
    except Exception as exc:
        warnings = [f"scipy.io.loadmat failed: {type(exc).__name__}: {exc}"]
    try:
        with h5py.File(mat_path, "r") as handle:
            keys = sorted(handle.keys())
            data = {key: _h5_to_python(handle[key], handle) for key in keys}
        return MatLoadResult(path=mat_path, load_method="h5py.File", keys=keys, data=data, warnings=warnings)
    except Exception as exc:
        return MatLoadResult(
            path=mat_path,
            load_method="failed",
            keys=[],
            data={},
            warnings=warnings + [f"h5py.File failed: {type(exc).__name__}: {exc}"],
        )


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray):
        return [_mat_to_python(item) for item in value.ravel()]
    return [value]


def _looks_like_cell(value: Any) -> bool:
    return isinstance(value, dict) and any(key in value for key in ["summary", "cycles", "cycles_interpolated", *LIFE_KEYS])


def _same_length_list_fields(mapping: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    lengths = {}
    for key, value in mapping.items():
        if isinstance(value, list) and value:
            lengths[key] = len(value)
    if not lengths:
        return None
    n = Counter(lengths.values()).most_common(1)[0][0]
    if n <= 0:
        return None
    rows = []
    for idx in range(n):
        row = {}
        for key, value in mapping.items():
            if isinstance(value, list) and len(value) == n:
                row[key] = value[idx]
            else:
                row[key] = value
        rows.append(row)
    return rows


def _cell_objects_from_data(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ["batch", "batchdata", "batchData", "data"]:
        if key not in data:
            continue
        obj = data[key]
        if isinstance(obj, list):
            cells = []
            for item in obj:
                if _looks_like_cell(item):
                    cells.append(item)
                elif isinstance(item, dict):
                    rows = _same_length_list_fields(item)
                    if rows:
                        cells.extend(rows)
            if cells:
                return cells
        if _looks_like_cell(obj):
            return [obj]
        if isinstance(obj, dict):
            rows = _same_length_list_fields(obj)
            if rows:
                return rows
    cells = [value for value in data.values() if _looks_like_cell(value)]
    return cells


def _first_present(mapping: Dict[str, Any], keys: Iterable[str]) -> Tuple[Optional[str], Any]:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return key, mapping[key]
    return None, None


def _numeric_scalar(value: Any) -> Optional[float]:
    arr = np.asarray(value).ravel()
    if arr.size == 0:
        return None
    try:
        out = float(arr[0])
    except Exception:
        return None
    return out if np.isfinite(out) else None


def _as_numeric_array(value: Any) -> np.ndarray:
    arr = np.asarray(value).ravel()
    return pd.to_numeric(pd.Series(arr), errors="coerce").to_numpy(float)


def _field_array(mapping: Dict[str, Any], aliases: List[str]) -> Tuple[Optional[str], np.ndarray]:
    key, value = _first_present(mapping, aliases)
    if key is None:
        return None, np.asarray([], dtype=float)
    return key, _as_numeric_array(value)


def _source_batch_from_path(path: Path) -> str:
    stem = path.stem
    return stem.split("_batchdata")[0]


def _summary_from_cell(cell: Dict[str, Any], first_n_cycles: Optional[int] = None) -> Tuple[Dict[str, Any], pd.DataFrame, List[str]]:
    warnings: List[str] = []
    summary = cell.get("summary")
    if not isinstance(summary, dict):
        warnings.append("summary is missing or not a mapping")
        return {}, pd.DataFrame(), warnings
    arrays: Dict[str, np.ndarray] = {}
    resolved: Dict[str, str] = {}
    lengths = []
    for canonical, aliases in SUMMARY_ALIASES.items():
        key, arr = _field_array(summary, aliases)
        if key is not None and arr.size > 0:
            arrays[canonical] = arr
            resolved[canonical] = key
            lengths.append(arr.size)
    if not lengths:
        warnings.append("summary has no recognized numeric cycle fields")
        return {}, pd.DataFrame(), warnings
    n = min(lengths)
    out: Dict[str, Any] = {}
    if "cycle_index" not in arrays:
        out["cycle_index"] = np.arange(n, dtype=int)
        warnings.append("summary cycle_index missing; generated zero-based positional cycle_index")
    for key, arr in arrays.items():
        out[key] = arr[:n].tolist()
    if "cycle_index" in out:
        order = np.argsort(pd.to_numeric(pd.Series(out["cycle_index"]), errors="coerce").to_numpy(float))
        for key, values in list(out.items()):
            arr = np.asarray(values)
            if arr.size == n:
                out[key] = arr[order].tolist()
    frame = pd.DataFrame(out)
    if first_n_cycles is not None:
        frame = frame.head(int(first_n_cycles)).copy()
        out = {col: frame[col].tolist() for col in frame.columns}
    frame.attrs["resolved_signals"] = resolved
    return out, frame, warnings


def _cycle_records_from_mapping(cycles: Dict[str, Any]) -> List[Tuple[int, Dict[str, Any]]]:
    if "cycle_index" in cycles and any(key in cycles for key in CYCLE_ALIASES["voltage"]):
        return [(0, cycles)]
    records = []
    for idx, key in enumerate(sorted(cycles, key=lambda value: str(value))):
        value = cycles[key]
        if isinstance(value, dict):
            records.append((idx, value))
    return records


def _infer_step_type(current: np.ndarray, length: int) -> List[str]:
    if current.size == length and np.isfinite(current).any():
        labels = np.where(current < 0, "discharge", np.where(current > 0, "charge", "")).astype(object)
        return [str(value) for value in labels]
    return [""] * length


def _cycles_interpolated_from_cell(cell: Dict[str, Any], first_n_cycles: Optional[int] = None) -> Tuple[Dict[str, Any], List[str]]:
    warnings: List[str] = []
    source = cell.get("cycles_interpolated")
    if isinstance(source, dict):
        return {str(key): _as_numeric_array(value).tolist() if key != "step_type" else [str(v).lower() for v in np.asarray(value).ravel()] for key, value in source.items()}, warnings
    cycles = cell.get("cycles")
    if not isinstance(cycles, (dict, list)):
        warnings.append("cycles_interpolated/cycles are unavailable")
        return {}, warnings
    records = list(enumerate(cycles)) if isinstance(cycles, list) else _cycle_records_from_mapping(cycles)
    arrays: Dict[str, List[Any]] = defaultdict(list)
    for positional_index, record in records:
        if first_n_cycles is not None and positional_index >= int(first_n_cycles):
            break
        if not isinstance(record, dict):
            continue
        mapped: Dict[str, np.ndarray] = {}
        for canonical, aliases in CYCLE_ALIASES.items():
            if canonical == "step_type":
                continue
            _key, arr = _field_array(record, aliases)
            if arr.size > 0:
                mapped[canonical] = arr
        if not mapped:
            continue
        lengths = [arr.size for arr in mapped.values() if arr.size > 0]
        if not lengths:
            continue
        n = min(lengths)
        cycle_index = _numeric_scalar(record.get("cycle_index"))
        if cycle_index is None:
            cycle_index = positional_index
        arrays["cycle_index"].extend([cycle_index] * n)
        for canonical in ["voltage", "test_time", "current", "charge_capacity", "discharge_capacity", "charge_energy", "discharge_energy", "internal_resistance", "temperature"]:
            values = mapped.get(canonical)
            arrays[canonical].extend(values[:n].tolist() if values is not None and values.size else [np.nan] * n)
        step_key, step_values = _first_present(record, CYCLE_ALIASES["step_type"])
        if step_key is not None:
            step_arr = np.asarray(step_values).ravel()
            if step_arr.size == 1:
                arrays["step_type"].extend([str(step_arr[0]).lower()] * n)
            else:
                arrays["step_type"].extend([str(value).lower() for value in step_arr[:n]])
        else:
            arrays["step_type"].extend(_infer_step_type(mapped.get("current", np.asarray([])), n))
    return dict(arrays), warnings


def severson_cells_from_file(path: Union[str, Path], first_n_cycles: Optional[int] = None) -> Tuple[List[SeversonCell], MatLoadResult]:
    mat_path = Path(path)
    loaded = load_matr_file(mat_path)
    if loaded.load_method == "failed":
        return [], loaded
    raw_cells = _cell_objects_from_data(loaded.data)
    cells: List[SeversonCell] = []
    source_batch = _source_batch_from_path(mat_path)
    for idx, cell in enumerate(raw_cells):
        if not isinstance(cell, dict):
            continue
        life_key, life_value = _first_present(cell, LIFE_KEYS)
        cycle_life = _numeric_scalar(life_value)
        summary, summary_df, summary_warnings = _summary_from_cell(cell, first_n_cycles=first_n_cycles)
        cycles_interpolated, cycle_warnings = _cycles_interpolated_from_cell(cell, first_n_cycles=first_n_cycles)
        barcode = str(cell.get("barcode") or cell.get("cell_id") or cell.get("CellID") or "").strip()
        channel = str(cell.get("channel_id") or cell.get("channel") or "").strip()
        cell_id = f"severson_{source_batch}_cell_{idx:03d}"
        metadata = {
            "barcode": barcode or None,
            "channel": channel or None,
            "policy": cell.get("policy") or cell.get("protocol"),
            "life_key": life_key,
            "n_summary_cycles": int(len(summary_df)),
            "has_cycles_interpolated": bool(cycles_interpolated),
        }
        cells.append(
            SeversonCell(
                source_file=mat_path.name,
                source_batch=source_batch,
                source_cell_index=idx,
                cell_id=cell_id,
                cycle_life=cycle_life,
                metadata=metadata,
                summary=summary,
                cycles_interpolated=cycles_interpolated,
                warnings=loaded.warnings + summary_warnings + cycle_warnings,
            )
        )
    return cells, loaded


def _field_availability_from_frames(cycle_summary: pd.DataFrame, canonical_payloads: List[Dict[str, Any]]) -> Dict[str, bool]:
    fields = {}
    for col in [
        "discharge_capacity",
        "charge_capacity",
        "discharge_energy",
        "charge_energy",
        "dc_internal_resistance",
        "temperature_average",
        "temperature_minimum",
        "temperature_maximum",
    ]:
        fields[col] = bool(col in cycle_summary.columns and pd.to_numeric(cycle_summary[col], errors="coerce").notna().any())
    raw_keys = set()
    for payload in canonical_payloads:
        raw_keys.update((payload.get("cycles_interpolated") or {}).keys())
    for col in ["voltage", "test_time", "current", "temperature", "internal_resistance"]:
        fields[col] = col in raw_keys
    return fields


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json_gz(path: Union[str, Path], payload: Dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt", encoding="utf-8") as handle:
        json.dump(_json_ready(payload), handle, sort_keys=True)


def audit_severson_mat_dir(mat_dir: Union[str, Path]) -> Dict[str, Any]:
    files = discover_matr_files(mat_dir)
    file_reports = []
    warnings: List[str] = []
    for path in files:
        cells, loaded = severson_cells_from_file(path, first_n_cycles=None)
        life_values = np.asarray([cell.cycle_life for cell in cells if cell.cycle_life is not None], dtype=float)
        summary_fields = Counter()
        cycle_fields = Counter()
        first100 = 0
        conventions = Counter()
        for cell in cells:
            summary_fields.update((cell.summary or {}).keys())
            cycle_fields.update((cell.cycles_interpolated or {}).keys())
            cycle_idx = pd.to_numeric(pd.Series((cell.summary or {}).get("cycle_index", [])), errors="coerce").dropna()
            if len(cycle_idx) >= 100:
                first100 += 1
            if not cycle_idx.empty:
                min_cycle = float(cycle_idx.min())
                conventions["zero_based" if min_cycle == 0 else "one_based" if min_cycle == 1 else "unknown"] += 1
        file_reports.append(
            {
                "path": str(path),
                "file_name": path.name,
                "load_method": loaded.load_method,
                "top_level_keys": loaded.keys,
                "warnings": loaded.warnings,
                "n_cells": int(len(cells)),
                "summary_fields": sorted(summary_fields),
                "cycle_level_fields": sorted(cycle_fields),
                "true_cycle_life_present": bool(life_values.size),
                "cycle_life_min": float(np.min(life_values)) if life_values.size else None,
                "cycle_life_mean": float(np.mean(life_values)) if life_values.size else None,
                "cycle_life_max": float(np.max(life_values)) if life_values.size else None,
                "cycle_index_conventions": dict(conventions),
                "cells_with_first_100_cycles": int(first100),
                "field_availability": {
                    "discharge_capacity": "discharge_capacity" in summary_fields or "discharge_capacity" in cycle_fields,
                    "charge_capacity": "charge_capacity" in summary_fields or "charge_capacity" in cycle_fields,
                    "voltage": "voltage" in cycle_fields,
                    "current": "current" in cycle_fields,
                    "temperature": any(field in summary_fields or field in cycle_fields for field in ["temperature_average", "temperature", "temperature_minimum", "temperature_maximum"]),
                    "internal_resistance": "dc_internal_resistance" in summary_fields or "internal_resistance" in cycle_fields,
                    "time": "test_time" in cycle_fields,
                    "energy": any(field in summary_fields or field in cycle_fields for field in ["discharge_energy", "charge_energy"]),
                },
            }
        )
        warnings.extend(loaded.warnings)
        if not cells:
            warnings.append(f"{path.name}: no parseable cells found")
    return {
        "mat_dir": str(mat_dir),
        "n_files": int(len(files)),
        "files": file_reports,
        "warnings": warnings,
    }


def _write_dataset_card_md(card: Dict[str, Any]) -> str:
    lines = [
        "# Severson 2019 True-Life Dataset",
        "",
        f"source_dataset: `{card.get('source_dataset')}`",
        f"label_source: `{card.get('label_source')}`",
        f"first_n_cycles: `{card.get('first_n_cycles')}`",
        f"total_cells: `{card.get('total_cells')}`",
        f"included_cells: `{card.get('included_cells')}`",
        f"excluded_cells: `{card.get('excluded_cells')}`",
        f"cycle_rows: `{card.get('cycle_rows')}`",
        "",
        "These labels are true measured cycle-to-failure values, not Attia/Chueh OED author-model pseudo-labels.",
        "",
        "## Cycle Life",
        "",
        f"- min: `{card.get('cycle_life_min')}`",
        f"- mean: `{card.get('cycle_life_mean')}`",
        f"- max: `{card.get('cycle_life_max')}`",
        "",
        "## Source Files",
        "",
    ]
    for source, count in (card.get("source_files") or {}).items():
        lines.append(f"- `{source}`: {count}")
    lines.extend(["", "## Field Availability", ""])
    for field, available in sorted((card.get("field_availability") or {}).items()):
        lines.append(f"- `{field}`: {available}")
    lines.extend(["", "## Caveats", ""])
    for warning in card.get("warnings", [])[:25]:
        lines.append(f"- {warning}")
    return "\n".join(lines) + "\n"


def build_severson_true_life_dataset(
    *,
    mat_dir: Union[str, Path],
    out_dir: Union[str, Path],
    first_n_cycles: int = 100,
) -> Dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    canonical_dir = out / "canonical_raw_cells"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    metadata_rows: List[Dict[str, Any]] = []
    label_rows: List[Dict[str, Any]] = []
    cycle_rows: List[Dict[str, Any]] = []
    exclusions: List[Dict[str, Any]] = []
    canonical_payloads: List[Dict[str, Any]] = []
    source_counts: Counter = Counter()
    row_id = 0
    for mat_path in discover_matr_files(mat_dir):
        cells, loaded = severson_cells_from_file(mat_path, first_n_cycles=first_n_cycles)
        if loaded.load_method == "failed":
            exclusions.append({"source_file": mat_path.name, "source_cell_index": None, "exclusion_stage": "load_mat", "reason": "; ".join(loaded.warnings)})
            continue
        for cell in cells:
            if cell.cycle_life is None:
                exclusions.append({"source_file": cell.source_file, "source_cell_index": cell.source_cell_index, "cell_id": cell.cell_id, "exclusion_stage": "label", "reason": "missing finite true cycle_life"})
                continue
            if not cell.summary or "discharge_capacity" not in cell.summary:
                exclusions.append({"source_file": cell.source_file, "source_cell_index": cell.source_cell_index, "cell_id": cell.cell_id, "exclusion_stage": "summary", "reason": "missing discharge_capacity summary"})
                continue
            payload = {
                "source_dataset": SEVERSON_DATASET_NAME,
                "source_file": cell.source_file,
                "source_batch": cell.source_batch,
                "source_cell_index": cell.source_cell_index,
                "cycle_life": cell.cycle_life,
                "label_source": TRUE_LABEL_SOURCE,
                "summary": cell.summary,
                "cycles_interpolated": cell.cycles_interpolated,
                "warnings": list(cell.warnings),
            }
            canonical_rel = Path("canonical_raw_cells") / f"{cell.cell_id}.json.gz"
            write_json_gz(out / canonical_rel, payload)
            summary_df = pd.DataFrame(cell.summary).head(first_n_cycles).copy()
            summary_df.insert(0, "cell_id", cell.cell_id)
            summary_df.insert(0, "row_id", row_id)
            cycle_rows.extend(summary_df.to_dict("records"))
            metadata = {
                "row_id": row_id,
                "cell_id": cell.cell_id,
                "source_dataset": SEVERSON_DATASET_NAME,
                "source_file": cell.source_file,
                "source_batch": cell.source_batch,
                "batch_id": cell.source_batch,
                "source_cell_index": cell.source_cell_index,
                "cycle_life": cell.cycle_life,
                "label_source": TRUE_LABEL_SOURCE,
                "canonical_raw_path": str(canonical_rel),
                "source_path": str(Path(mat_dir) / cell.source_file),
            }
            for key in ["barcode", "channel", "policy"]:
                if cell.metadata.get(key) is not None:
                    metadata[key] = cell.metadata[key]
            metadata_rows.append(metadata)
            label_rows.append({"row_id": row_id, "cell_id": cell.cell_id, "cycle_life": cell.cycle_life, "label_source": TRUE_LABEL_SOURCE})
            canonical_payloads.append(payload)
            source_counts[cell.source_file] += 1
            row_id += 1
    metadata = pd.DataFrame(metadata_rows)
    labels = pd.DataFrame(label_rows)
    cycle_summary = pd.DataFrame(cycle_rows)
    if metadata.empty:
        raise ValueError(f"No Severson true-life cells were included from {mat_dir}")
    if cycle_summary.empty:
        raise ValueError("No cycle_summary rows were built")
    labels["y"] = labels["cycle_life"]
    source_files = sorted(metadata["source_file"].astype(str).unique().tolist())
    split_map = {}
    if len(source_files) == 1:
        split_map[source_files[0]] = "train"
    elif len(source_files) == 2:
        split_map = {source_files[0]: "train", source_files[1]: "validation"}
    else:
        split_map = {source_files[0]: "train", source_files[1]: "validation"}
        for source in source_files[2:]:
            split_map[source] = "test"
    splits = metadata[["row_id", "cell_id", "source_file", "source_batch", "batch_id"]].copy()
    splits["split"] = splits["source_file"].map(split_map)
    splits["split_source"] = "source_file_identity_default"
    metadata.to_csv(out / "cell_metadata.csv", index=False)
    cycle_summary.to_csv(out / "cycle_summary.csv", index=False)
    labels.to_csv(out / "labels.csv", index=False)
    splits.to_csv(out / "splits.csv", index=False)
    pd.DataFrame(exclusions, columns=["source_file", "source_cell_index", "cell_id", "exclusion_stage", "reason"]).to_csv(out / "exclusions.csv", index=False)
    life = pd.to_numeric(metadata["cycle_life"], errors="coerce").to_numpy(float)
    field_availability = _field_availability_from_frames(cycle_summary, canonical_payloads)
    card = {
        "source_dataset": SEVERSON_DATASET_NAME,
        "label_source": TRUE_LABEL_SOURCE,
        "label_source_description": "true measured cycle-to-failure labels from Severson/MatR data",
        "first_n_cycles": int(first_n_cycles),
        "total_cells": int(len(metadata) + len(exclusions)),
        "included_cells": int(len(metadata)),
        "excluded_cells": int(len(exclusions)),
        "cycle_rows": int(len(cycle_summary)),
        "cycle_life_min": float(np.nanmin(life)),
        "cycle_life_mean": float(np.nanmean(life)),
        "cycle_life_max": float(np.nanmax(life)),
        "source_files": {str(k): int(v) for k, v in source_counts.items()},
        "field_availability": field_availability,
        "canonical_raw_cells_dir": str(canonical_dir),
        "warnings": sorted({warning for payload in canonical_payloads for warning in payload.get("warnings", [])}),
        "caveat": "These are true measured lifetimes, unlike OED author_model_prediction pseudo-labels.",
    }
    (out / "dataset_card.json").write_text(json.dumps(_json_ready(card), indent=2, sort_keys=True) + "\n")
    (out / "dataset_card.md").write_text(_write_dataset_card_md(card))
    return card
