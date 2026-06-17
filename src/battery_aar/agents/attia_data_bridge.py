from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from battery_aar.paper_reproduction.bms_features import (
    _extract_lifetime,
    parse_policy_currents,
    parse_protocol_readable,
)
from battery_aar.paper_reproduction.paths import (
    OED_BATCH_NAMES,
    SEVERSON_B3_BATCH_DATE,
    VALIDATION_BATCH_NAME,
    resolve_paper_paths,
    severson_b3_mat_path,
)


@dataclass
class AgentDatasetBuildResult:
    metadata: pd.DataFrame
    cycle_summary: pd.DataFrame
    labels: pd.DataFrame
    splits: pd.DataFrame
    dataset_card: dict[str, Any]
    exclusions: list[dict[str, Any]]


EXCLUSION_COLUMNS = ("batch_id", "raw_file", "cell_id", "channel", "exclusion_stage", "reason")


def _exclusion(
    batch_id: str,
    raw_file: str | Path | None,
    cell_id: str | None,
    channel: int | str | None,
    exclusion_stage: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "batch_id": batch_id,
        "raw_file": str(raw_file) if raw_file is not None else "",
        "cell_id": cell_id or "",
        "channel": "" if channel is None else str(channel),
        "exclusion_stage": exclusion_stage,
        "reason": reason,
    }


def _decode_hdf5_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    arr = np.asarray(value)
    if arr.dtype.kind == "S":
        return np.char.decode(arr, "utf-8", errors="replace").tolist()
    if arr.ndim == 0:
        item = arr.item()
        if isinstance(item, bytes):
            return item.decode("utf-8", errors="replace")
        return item
    return arr.tolist()


def _hdf5_group_to_dict(group: h5py.Group) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in group.items():
        if isinstance(value, h5py.Group):
            out[key] = _hdf5_group_to_dict(value)
        else:
            out[key] = _decode_hdf5_value(value[()])
    for key, value in group.attrs.items():
        out.setdefault(key, _decode_hdf5_value(value))
    return out


def load_raw_cell_payload(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if source.suffix.lower() == ".json":
        return json.loads(source.read_text())
    if source.suffix.lower() in {".h5", ".hdf5"}:
        with h5py.File(source, "r") as handle:
            return _hdf5_group_to_dict(handle)
    raise ValueError(f"Unsupported raw cell file type: {source}")


def _raw_cell_files(batch_path: str | Path) -> list[Path]:
    batch = Path(batch_path)
    files = sorted(batch.glob("*_structure.json"))
    if files:
        return files
    return sorted([*batch.glob("*.json"), *batch.glob("*.h5"), *batch.glob("*.hdf5")])


def _channel_from_text(text: str | None) -> int | None:
    if not text:
        return None
    match = re.search(r"(?:^|[_-])CH\s*0*(\d+)(?:[_-]|$)", str(text), flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def normalize_cell_identity(value: str | Path | None, batch_name: str | None = None) -> str:
    if value is None:
        return ""
    text = Path(str(value).replace("\\", "/")).name
    for suffix in (".json", ".hdf5", ".h5", ".mat"):
        if text.lower().endswith(suffix):
            text = text[: -len(suffix)]
    text = re.sub(r"_structure$", "", text, flags=re.IGNORECASE)
    channel = _channel_from_text(text)
    if channel is not None and batch_name:
        return f"{batch_name}_CH{channel}".lower()
    return text.lower()


def _raw_file_index(batch_path: str | Path) -> dict[str, Path]:
    batch = Path(batch_path)
    indexed: dict[str, Path] = {}
    for path in _raw_cell_files(batch):
        indexed.setdefault(normalize_cell_identity(path, batch.name), path)
        indexed.setdefault(normalize_cell_identity(path.stem, batch.name), path)
    return indexed


def _cell_identity(batch_name: str, path: Path, payload: dict[str, Any]) -> tuple[str, int | None, str | None]:
    channel = payload.get("channel_id", payload.get("metadata", {}).get("channel_id"))
    try:
        channel_int = int(channel)
    except Exception:
        channel_int = _channel_from_text(path.stem)
    barcode = payload.get("barcode") or payload.get("metadata", {}).get("barcode")
    cell_id = f"{batch_name}_CH{channel_int}" if channel_int is not None else path.stem
    return cell_id, channel_int, str(barcode) if barcode is not None else None


def _numeric_summary_arrays(summary: dict[str, Any], cycle_index: np.ndarray) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for key, value in summary.items():
        if key == "cycle_index":
            continue
        arr = np.asarray(value)
        if arr.ndim != 1 or arr.size != cycle_index.size:
            continue
        try:
            arrays[key] = arr.astype(float)
        except (TypeError, ValueError):
            continue
    return arrays


def _summary_to_cycle_rows(payload: dict[str, Any], row_id: int, cell_id: str, first_n_cycles: int) -> list[dict[str, Any]]:
    summary = payload.get("summary") or {}
    if not summary:
        raise ValueError("raw cell payload has no summary table")
    discharge = np.asarray(summary.get("discharge_capacity", summary.get("QDischarge", [])), dtype=float)
    if discharge.size == 0:
        raise ValueError("raw cell summary has no discharge_capacity")
    cycle_index = np.asarray(summary.get("cycle_index", np.arange(1, discharge.size + 1)), dtype=float)
    if cycle_index.size != discharge.size:
        cycle_index = np.arange(1, discharge.size + 1, dtype=float)
    charge_raw = summary.get("charge_capacity", summary.get("QCharge", None))
    if charge_raw is None:
        charge = np.full(discharge.size, np.nan, dtype=float)
    else:
        charge = np.asarray(charge_raw, dtype=float)
        if charge.size != discharge.size:
            charge = np.full(discharge.size, np.nan, dtype=float)

    numeric_arrays = _numeric_summary_arrays(summary, cycle_index)
    rows: list[dict[str, Any]] = []
    for idx, cyc in enumerate(cycle_index):
        if not np.isfinite(cyc):
            continue
        cyc_int = int(cyc)
        if cyc_int < 1 or cyc_int > first_n_cycles:
            continue
        row = {
            "row_id": row_id,
            "cell_id": cell_id,
            "cycle_index": cyc_int,
            "discharge_capacity": float(discharge[idx]) if np.isfinite(discharge[idx]) else np.nan,
            "charge_capacity": float(charge[idx]) if np.isfinite(charge[idx]) else np.nan,
        }
        for key, arr in numeric_arrays.items():
            if key in {"discharge_capacity", "QDischarge", "charge_capacity", "QCharge"}:
                continue
            row[key] = float(arr[idx]) if np.isfinite(arr[idx]) else np.nan
        rows.append(row)
    if not rows:
        raise ValueError(f"raw cell summary has no cycles in 1..{first_n_cycles}")
    return rows


def _load_raw_cells_by_id(batch_path: str | Path, first_n_cycles: int) -> dict[str, dict[str, Any]]:
    batch = Path(batch_path)
    cells: dict[str, dict[str, Any]] = {}
    for path in _raw_cell_files(batch):
        try:
            raw = _load_one_raw_cell(path, batch.name, first_n_cycles)
        except ValueError:
            continue
        cells[raw["normalized_cell_id"]] = raw
    return cells


def _load_one_raw_cell(path: Path, batch_name: str, first_n_cycles: int) -> dict[str, Any]:
    try:
        payload = load_raw_cell_payload(path)
    except Exception as exc:
        raise ValueError(f"raw_cell_parse_error: {exc}") from exc
    cell_id, channel, barcode = _cell_identity(batch_name, path, payload)
    protocol = payload.get("protocol") or payload.get("metadata", {}).get("protocol")
    protocol_readable = parse_protocol_readable(protocol)
    c1, c2, c3, c4 = parse_policy_currents(protocol_readable)
    try:
        cycle_rows = _summary_to_cycle_rows(payload, row_id=-1, cell_id=cell_id, first_n_cycles=first_n_cycles)
    except Exception as exc:
        raise ValueError(str(exc)) from exc
    lifetime = _extract_lifetime(payload.get("summary") or {})
    return {
        "source_path": path,
        "payload": payload,
        "cell_id": cell_id,
        "normalized_cell_id": normalize_cell_identity(cell_id, batch_name),
        "channel": channel,
        "barcode": barcode,
        "protocol_readable": protocol_readable,
        "C1": c1,
        "C2": c2,
        "C3": c3,
        "C4": c4,
        "cycle_rows": cycle_rows,
        "lifetime": lifetime,
    }


def _prediction_rows_with_exclusions(predictions: pd.DataFrame, batch_name: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    required = ["cell_id", "Prediction"]
    missing = [col for col in required if col not in predictions.columns]
    if missing:
        raise ValueError(f"early prediction file missing required columns: {missing}")
    df = predictions.copy()
    for col in ["Prediction", "C1", "C2", "C3", "C4"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    anomaly = df["anomaly_flag"].fillna(False).astype(bool) if "anomaly_flag" in df else pd.Series(False, index=df.index)
    valid = np.isfinite(df["Prediction"]) & (df["Prediction"] > 0) & ~anomaly
    reason_by_idx: dict[int, list[str]] = {int(idx): [] for idx in df.index}
    for idx in df.index[anomaly]:
        reason_by_idx[int(idx)].append("anomaly_flag_true")
    nonfinite_prediction = ~np.isfinite(df["Prediction"])
    for idx in df.index[nonfinite_prediction]:
        reason_by_idx[int(idx)].append("nonfinite_prediction")
    prediction_le_zero = np.isfinite(df["Prediction"]) & (df["Prediction"] <= 0)
    for idx in df.index[prediction_le_zero]:
        reason_by_idx[int(idx)].append("prediction_le_0")
    for col in ["C1", "C2", "C3", "C4"]:
        if col in df:
            bad = ~np.isfinite(df[col])
            valid &= ~bad
            for idx in df.index[bad]:
                reason_by_idx[int(idx)].append(f"nonfinite_{col}")
    exclusions = []
    for idx, reasons in reason_by_idx.items():
        if reasons:
            row = df.loc[idx]
            exclusions.append(
                _exclusion(
                    batch_name,
                    None,
                    str(row.get("cell_id", "")),
                    row.get("channel", _channel_from_text(str(row.get("cell_id", "")))),
                    "label_filter",
                    ";".join(reasons),
                )
            )
    return df.loc[valid].copy(), exclusions


def _valid_prediction_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    valid, _ = _prediction_rows_with_exclusions(predictions, batch_name="")
    return valid


def _make_splits(row_ids: list[int], seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ids = np.asarray(row_ids, dtype=int)
    rng.shuffle(ids)
    n = len(ids)
    if n == 0:
        return pd.DataFrame(columns=["row_id", "split"])
    n_train = max(1, int(0.6 * n))
    n_val = max(1, int(0.2 * n)) if n >= 3 else max(0, n - n_train)
    if n_train + n_val >= n and n > 1:
        n_val = max(1, n - n_train)
    splits = []
    for idx, row_id in enumerate(ids):
        split = "train" if idx < n_train else "val" if idx < n_train + n_val else "test"
        splits.append({"row_id": int(row_id), "split": split})
    return pd.DataFrame(splits).sort_values("row_id").reset_index(drop=True)


def _write_dataset_card_markdown(card: dict[str, Any], path: Path) -> None:
    lines = [
        "# Open Battery Agents Attia Surrogate Dataset",
        "",
        f"label_source: `{card.get('label_source')}`",
        f"first_n_cycles: `{card.get('first_n_cycles')}`",
        f"batch9_included: `{card.get('batch9_included')}`",
        f"n_cells: `{card.get('n_cells')}`",
        f"n_cycle_rows: `{card.get('n_cycle_rows')}`",
        f"charge_capacity_status: `{card.get('charge_capacity_status')}`",
        f"skipped_raw_cells: `{card.get('skipped_raw_cells')}`",
        f"skipped_label_rows: `{card.get('skipped_label_rows')}`",
        "",
        "## Important Caveat",
        "The `cycle_life` labels for OED/CLO rows are author-model early predictions, not true final measured lifetimes.",
        "Batch 9 is excluded from this training/search dataset and remains locked for final validation.",
        "",
        "## Batches",
    ]
    for batch in card.get("batches", []):
        lines.append(
            f"- `{batch.get('batch_id')}`: valid labels `{batch.get('valid_label_rows')}`, "
            f"included `{batch.get('included_cells')}`, exclusions `{batch.get('excluded_cells')}`"
        )
    lines.extend(["", "## Exclusion Counts"])
    for stage, count in (card.get("exclusion_counts_by_stage") or {}).items():
        lines.append(f"- `{stage}`: `{count}`")
    path.write_text("\n".join(lines) + "\n")


def build_agent_surrogate_dataset(
    battery_fast_charging_root: str | Path,
    reference_run: str | Path,
    out: str | Path,
    label_source: str = "author_model_prediction",
    first_n_cycles: int = 100,
    exclude_batch9: bool = True,
    seed: int = 0,
) -> AgentDatasetBuildResult:
    if label_source != "author_model_prediction":
        raise ValueError("Only label_source='author_model_prediction' is currently supported")
    if not exclude_batch9:
        raise ValueError("Batch 9 must remain excluded from the surrogate search dataset")

    paths = resolve_paper_paths(battery_fast_charging_root)
    reference = Path(reference_run)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows: list[dict[str, Any]] = []
    cycle_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    batch_cards: list[dict[str, Any]] = []
    row_id = 0

    for batch_name in OED_BATCH_NAMES:
        pred_path = reference / "early_predictions_rich" / f"{batch_name}.csv"
        batch_path = paths.data_dir / batch_name
        batch_card = {
            "batch_id": batch_name,
            "prediction_file": str(pred_path),
            "raw_files_discovered": 0,
            "valid_label_rows": 0,
            "included_cells": 0,
            "excluded_cells": 0,
        }
        if not pred_path.exists():
            exclusions.append(_exclusion(batch_name, pred_path, None, None, "input_discovery", "missing_early_predictions_rich"))
            batch_card["excluded_cells"] += 1
            batch_cards.append(batch_card)
            continue
        if not batch_path.is_dir():
            exclusions.append(_exclusion(batch_name, batch_path, None, None, "input_discovery", "missing_raw_batch_dir"))
            batch_card["excluded_cells"] += 1
            batch_cards.append(batch_card)
            continue
        predictions, label_exclusions = _prediction_rows_with_exclusions(pd.read_csv(pred_path), batch_name)
        exclusions.extend(label_exclusions)
        batch_card["excluded_cells"] += len(label_exclusions)
        batch_card["valid_label_rows"] = int(len(predictions))
        raw_index = _raw_file_index(batch_path)
        batch_card["raw_files_discovered"] = int(len(set(raw_index.values())))
        for _, pred_row in predictions.iterrows():
            cell_id = str(pred_row["cell_id"])
            normalized_cell_id = normalize_cell_identity(cell_id, batch_name)
            raw_path = raw_index.get(normalized_cell_id)
            if raw_path is None:
                exclusions.append(
                    _exclusion(
                        batch_name,
                        None,
                        cell_id,
                        pred_row.get("channel", _channel_from_text(cell_id)),
                        "raw_match",
                        "raw_cell_not_found",
                    )
                )
                batch_card["excluded_cells"] += 1
                continue
            try:
                raw = _load_one_raw_cell(raw_path, batch_name, first_n_cycles)
            except ValueError as exc:
                exclusions.append(
                    _exclusion(
                        batch_name,
                        raw_path,
                        cell_id,
                        pred_row.get("channel", _channel_from_text(cell_id)),
                        "raw_parse",
                        str(exc),
                    )
                )
                batch_card["excluded_cells"] += 1
                continue
            protocol_values = [pred_row.get(col, raw.get(col)) for col in ["C1", "C2", "C3", "C4"]]
            if not all(np.isfinite(pd.to_numeric(value, errors="coerce")) for value in protocol_values):
                exclusions.append(
                    _exclusion(
                        batch_name,
                        raw_path,
                        cell_id,
                        raw.get("channel"),
                        "metadata_filter",
                        "missing_protocol_currents",
                    )
                )
                batch_card["excluded_cells"] += 1
                continue
            anon = f"surrogate_cell_{row_id:05d}"
            metadata_rows.append(
                {
                    "row_id": row_id,
                    "cell_id": anon,
                    "anonymized_cell_id": anon,
                    "batch_id": batch_name,
                    "protocol_readable": pred_row.get("protocol_readable") or raw.get("protocol_readable"),
                    "C1": float(protocol_values[0]),
                    "C2": float(protocol_values[1]),
                    "C3": float(protocol_values[2]),
                    "C4": float(protocol_values[3]),
                    "cycle_life": float(pred_row["Prediction"]),
                    "label_source": label_source,
                    "source_cell_id": cell_id,
                    "source_path": str(raw["source_path"]),
                }
            )
            label_rows.append({"row_id": row_id, "y": float(pred_row["Prediction"])})
            for cyc_row in raw["cycle_rows"]:
                new_row = dict(cyc_row)
                new_row["row_id"] = row_id
                new_row["cell_id"] = anon
                cycle_rows.append(new_row)
            row_id += 1
            batch_card["included_cells"] += 1
        batch_cards.append(batch_card)

    metadata = pd.DataFrame(metadata_rows)
    cycle_summary = pd.DataFrame(cycle_rows)
    labels = pd.DataFrame(label_rows)
    splits = _make_splits(metadata["row_id"].tolist() if not metadata.empty else [], seed=seed)
    if metadata.empty:
        raise RuntimeError("No labeled OED/CLO rows were built for the agent surrogate dataset")
    if cycle_summary.empty:
        raise RuntimeError("No cycle-summary rows were built for the agent surrogate dataset")
    if labels.empty:
        raise RuntimeError("No labeled rows were built for the agent surrogate dataset")

    charge = pd.to_numeric(cycle_summary.get("charge_capacity", pd.Series(dtype=float)), errors="coerce")
    if "charge_capacity" not in cycle_summary:
        charge_status = "missing"
    elif charge.notna().sum() == 0:
        charge_status = "all_nan"
    elif charge.notna().sum() < len(charge):
        charge_status = "partially_missing"
    else:
        charge_status = "available"

    stage_counts = Counter(str(item["exclusion_stage"]) for item in exclusions)
    reason_counts = Counter(str(item["reason"]) for item in exclusions)
    card = {
        "dataset_type": "attia_oed_author_model_surrogate",
        "label_source": label_source,
        "surrogate_label_notice": "cycle_life is the author-model early prediction for OED/CLO cells, not true final measured lifetime.",
        "batch9_included": False,
        "batch9_role": "locked_final_validation_only",
        "first_n_cycles": int(first_n_cycles),
        "n_cells": int(len(metadata)),
        "n_cycle_rows": int(len(cycle_summary)),
        "n_exclusions": int(len(exclusions)),
        "skipped_raw_cells": int(sum(count for stage, count in stage_counts.items() if stage in {"raw_match", "raw_parse"})),
        "skipped_label_rows": int(stage_counts.get("label_filter", 0)),
        "exclusion_counts_by_stage": dict(sorted(stage_counts.items())),
        "exclusion_counts_by_reason": dict(sorted(reason_counts.items())),
        "charge_capacity_status": charge_status,
        "batches": batch_cards,
        "columns": {
            "cell_metadata": list(metadata.columns),
            "cycle_summary": list(cycle_summary.columns),
            "splits": list(splits.columns),
        },
    }

    metadata.to_csv(out_dir / "cell_metadata.csv", index=False)
    cycle_summary.to_csv(out_dir / "cycle_summary.csv", index=False)
    labels.to_csv(out_dir / "labels.csv", index=False)
    splits.to_csv(out_dir / "splits.csv", index=False)
    (out_dir / "dataset_card.json").write_text(json.dumps(card, indent=2, default=str, sort_keys=True) + "\n")
    _write_dataset_card_markdown(card, out_dir / "dataset_card.md")
    pd.DataFrame(exclusions, columns=EXCLUSION_COLUMNS).to_csv(out_dir / "exclusions.csv", index=False)
    return AgentDatasetBuildResult(metadata, cycle_summary, labels, splits, card, exclusions)


def load_batch9_holdout(
    battery_fast_charging_root: str | Path | None = None,
    batch9_path: str | Path | None = None,
    first_n_cycles: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if batch9_path is not None:
        batch_path = Path(batch9_path)
    else:
        paths = resolve_paper_paths(battery_fast_charging_root)
        batch_path = paths.data_dir / VALIDATION_BATCH_NAME
    if not batch_path.is_dir():
        raise FileNotFoundError(f"Batch 9 validation directory not found: {batch_path}")
    metadata_rows: list[dict[str, Any]] = []
    cycle_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    raw_cells = _load_raw_cells_by_id(batch_path, first_n_cycles)
    for row_id, raw in enumerate(raw_cells.values()):
        if raw["lifetime"] is None or not np.isfinite(raw["lifetime"]):
            continue
        anon = f"batch9_cell_{row_id:05d}"
        metadata_rows.append(
            {
                "row_id": row_id,
                "cell_id": anon,
                "anonymized_cell_id": anon,
                "batch_id": VALIDATION_BATCH_NAME,
                "protocol_readable": raw.get("protocol_readable"),
                "C1": raw.get("C1"),
                "C2": raw.get("C2"),
                "C3": raw.get("C3"),
                "C4": raw.get("C4"),
                "source_cell_id": raw["cell_id"],
            }
        )
        label_rows.append({"row_id": row_id, "y": float(raw["lifetime"])})
        for cyc_row in raw["cycle_rows"]:
            new_row = dict(cyc_row)
            new_row["row_id"] = row_id
            new_row["cell_id"] = anon
            cycle_rows.append(new_row)
    if not metadata_rows:
        raise RuntimeError(f"No labeled Batch 9 cells were loaded from {batch_path}")
    return pd.DataFrame(metadata_rows), pd.DataFrame(cycle_rows), pd.DataFrame(label_rows)


def load_severson_b3_holdout(
    battery_fast_charging_root: str | Path | None = None,
    first_n_cycles: int = 100,
    exclude_cell_ids: list[str] | tuple[str, ...] | set[str] | None = None,
    mat_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the Severson b3 (2018-04-12) batch as a locked-test holdout.

    This is the patch-E4 default for the campaign's "locked secondary test".
    It mirrors the shape returned by :func:`load_batch9_holdout` so downstream
    orchestrator / role-graph code can treat the two holdouts interchangeably:
    a metadata DataFrame, a cycle_summary DataFrame, and a labels DataFrame.

    The returned `batch_id` for every metadata row is `severson_2018-04-12`,
    deliberately disjoint from any Severson-2019 training batch ids so the
    locked test never leaks into surrogate search.

    Parameters
    ----------
    battery_fast_charging_root
        Root of the battery-fast-charging local clone. Resolved via
        :func:`severson_b3_mat_path` to locate the b3 `.mat` file.
    first_n_cycles
        Window of cycles to retain per cell (same semantics as
        ``load_batch9_holdout``).
    exclude_cell_ids
        Optional list/tuple/set of normalized Severson cell_ids to drop. Useful
        if a Severson training run later turns out to overlap; today the b3
        batch is held out of training.
    mat_path
        Optional explicit override for the b3 `.mat` file path. When provided,
        ``battery_fast_charging_root`` is ignored.
    """

    # Imported lazily because severson_matr pulls scipy.io / h5py and the
    # other consumers of this module (Attia path) should not have to load it.
    from battery_aar.features.severson_matr import discover_matr_files, severson_cells_from_file

    if mat_path is not None:
        b3_path = Path(mat_path)
    else:
        if battery_fast_charging_root is None:
            raise ValueError(
                "load_severson_b3_holdout requires battery_fast_charging_root or mat_path"
            )
        b3_path = severson_b3_mat_path(battery_fast_charging_root)
        # Fallback: if the explicit b3 file isn't present, scan the directory
        # for any file containing the b3 batch date so we still pick the right
        # mat. This makes the loader robust to slight filename variations.
        if not b3_path.exists():
            mat_dir = b3_path.parent
            if mat_dir.is_dir():
                candidates = [
                    candidate
                    for candidate in discover_matr_files(mat_dir)
                    if SEVERSON_B3_BATCH_DATE in candidate.name
                ]
                if candidates:
                    b3_path = candidates[0]
    if not b3_path.exists():
        raise FileNotFoundError(f"Severson b3 .mat file not found: {b3_path}")

    exclude = {str(value) for value in (exclude_cell_ids or [])}
    cells, _loaded = severson_cells_from_file(b3_path, first_n_cycles=first_n_cycles)
    batch_id_value = f"severson_{SEVERSON_B3_BATCH_DATE}"

    metadata_rows: list[dict[str, Any]] = []
    cycle_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    row_id = 0
    for cell in cells:
        cell_id = str(cell.cell_id)
        if cell_id in exclude:
            continue
        if cell.cycle_life is None or not np.isfinite(float(cell.cycle_life)):
            continue
        if not cell.summary:
            continue
        discharge = np.asarray(cell.summary.get("discharge_capacity", []), dtype=float)
        if discharge.size == 0:
            continue
        cycle_index = np.asarray(
            cell.summary.get("cycle_index", np.arange(1, discharge.size + 1)), dtype=float
        )
        if cycle_index.size != discharge.size:
            cycle_index = np.arange(1, discharge.size + 1, dtype=float)
        charge_raw = cell.summary.get("charge_capacity")
        if charge_raw is None:
            charge = np.full(discharge.size, np.nan, dtype=float)
        else:
            charge = np.asarray(charge_raw, dtype=float)
            if charge.size != discharge.size:
                charge = np.full(discharge.size, np.nan, dtype=float)
        anon = f"severson_b3_cell_{row_id:05d}"
        metadata_rows.append(
            {
                "row_id": row_id,
                "cell_id": anon,
                "anonymized_cell_id": anon,
                "batch_id": batch_id_value,
                "source_batch": cell.source_batch,
                "source_file": cell.source_file,
                "source_cell_index": cell.source_cell_index,
                "source_cell_id": cell_id,
                "barcode": cell.metadata.get("barcode"),
                "channel": cell.metadata.get("channel"),
                "policy": cell.metadata.get("policy"),
                "protocol_readable": cell.metadata.get("policy"),
            }
        )
        label_rows.append({"row_id": row_id, "y": float(cell.cycle_life)})
        for idx, cyc in enumerate(cycle_index):
            if not np.isfinite(cyc):
                continue
            cyc_int = int(cyc)
            if cyc_int < 1 or cyc_int > first_n_cycles:
                continue
            cycle_rows.append(
                {
                    "row_id": row_id,
                    "cell_id": anon,
                    "cycle_index": cyc_int,
                    "discharge_capacity": float(discharge[idx]) if np.isfinite(discharge[idx]) else np.nan,
                    "charge_capacity": float(charge[idx]) if np.isfinite(charge[idx]) else np.nan,
                }
            )
        row_id += 1

    if not metadata_rows:
        raise RuntimeError(
            f"No labeled Severson b3 cells were loaded from {b3_path}"
        )
    return (
        pd.DataFrame(metadata_rows),
        pd.DataFrame(cycle_rows),
        pd.DataFrame(label_rows),
    )


def load_author_validation_metrics(reference_run: str | Path | None) -> dict[str, Any]:
    if reference_run is None:
        return {"available": False, "reason": "reference_run_missing"}
    reference = Path(reference_run)
    metrics_path = reference / "validation_metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        early = metrics.get("early_prediction_vs_observed") or {}
        posterior = metrics.get("posterior_vs_observed") or {}
        return {
            "available": early.get("rmse") is not None,
            "path": str(metrics_path),
            "author_model_batch9_rmse": early.get("rmse"),
            "author_model_batch9_mae": early.get("mae"),
            "posterior_batch9_rmse": posterior.get("rmse"),
            "raw": metrics,
        }
    ranking_path = reference / "validation_protocol_ranking.csv"
    if ranking_path.exists():
        ranking = pd.read_csv(ranking_path)
        if {"observed_cycle_life_mean", "author_early_prediction_mean"}.issubset(ranking.columns):
            observed = pd.to_numeric(ranking["observed_cycle_life_mean"], errors="coerce").to_numpy(float)
            predicted = pd.to_numeric(ranking["author_early_prediction_mean"], errors="coerce").to_numpy(float)
            mask = np.isfinite(observed) & np.isfinite(predicted)
            if mask.any():
                diff = predicted[mask] - observed[mask]
                return {
                    "available": True,
                    "path": str(ranking_path),
                    "author_model_batch9_rmse": float(np.sqrt(np.mean(diff**2))),
                    "author_model_batch9_mae": float(np.mean(np.abs(diff))),
                    "raw": {"inferred_from_validation_protocol_ranking": True, "n": int(mask.sum())},
                }
    return {"available": False, "reason": "validation_metrics_unavailable"}
