from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from battery_aar.features.operator_registry import FeatureOperatorRegistry, default_operator_registry
from battery_aar.features.raw_cycles import (
    CANONICAL_CYCLE_COLUMNS,
    canonicalize_cycles_interpolated,
    canonicalize_summary,
    infer_cycle_index_convention,
    load_attia_json_payload,
    validate_feature_metadata,
    validate_feature_table,
)
from battery_aar.features.schemas import FeatureProgram, FeatureProgramResult

RAW_CYCLE_OPERATOR_TYPES = {"curve_shape", "cross_cycle_curve_delta"}
RAW_ROOT_MARKER = "battery-fast-charging/"
SOURCE_PATH_COLUMNS = ["canonical_raw_path", "source_path", "raw_file", "raw_path", "file_path", "path"]
SOURCE_PATH_RESOLUTION_METHODS = ["direct", "rebased", "reconstructed", "unresolved"]


@dataclass
class RawPathResolution:
    path: Optional[Path]
    method: str
    source_column: Optional[str] = None
    original_source_path: Optional[str] = None
    resolved_source_path: Optional[str] = None
    attempted_paths: Optional[List[str]] = None

    def to_audit_row(self, row: pd.Series, row_id: Optional[int], cell_id: Optional[str]) -> Dict[str, Any]:
        return {
            "row_id": row_id,
            "cell_id": cell_id,
            "batch_id": _clean_row_value(row, "batch_id"),
            "source_cell_id": _clean_row_value(row, "source_cell_id"),
            "canonical_raw_path": _clean_row_value(row, "canonical_raw_path"),
            "source_column": self.source_column,
            "source_path": self.original_source_path,
            "resolved_source_path": self.resolved_source_path,
            "source_path_resolution_method": self.method,
            "attempted_paths": ";".join(self.attempted_paths or []),
        }


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n")


def _clean_row_value(row: pd.Series, col: str) -> Optional[str]:
    if col not in row.index or pd.isna(row[col]):
        return None
    value = str(row[col]).strip()
    return value or None


def _append_attempt(attempted: List[str], path: Path) -> None:
    value = str(path)
    if value not in attempted:
        attempted.append(value)


def _rebase_marker_path(path_text: str, raw_root: Path) -> Optional[Path]:
    normalized = path_text.replace("\\", "/")
    marker_idx = normalized.find(RAW_ROOT_MARKER)
    if marker_idx < 0:
        return None
    suffix = normalized[marker_idx + len(RAW_ROOT_MARKER) :].strip("/")
    if not suffix:
        return None
    return raw_root / Path(suffix)


def _raw_data_root(raw_root: Path) -> Path:
    return raw_root if raw_root.name == "data" else raw_root / "data"


def _source_cell_name_candidates(batch_id: str, source_cell_id: str) -> List[str]:
    raw = str(source_cell_id).strip()
    if not raw:
        return []
    stem = Path(raw).stem if raw.endswith(".json") else raw
    stems = [stem]
    if stem.endswith("_structure"):
        stems.append(stem[: -len("_structure")])
    else:
        stems.append(f"{stem}_structure")
    if stem.startswith("CH"):
        stems.extend([f"{batch_id}_{stem}", f"{batch_id}_{stem}_structure"])
    seen: set[str] = set()
    names: List[str] = []
    for candidate in stems:
        if not candidate:
            continue
        name = f"{candidate}.json"
        if name not in seen:
            names.append(name)
            seen.add(name)
    return names


def _reconstruct_raw_path(row: pd.Series, raw_root: Path, attempted: List[str]) -> Optional[Path]:
    batch_id = _clean_row_value(row, "batch_id")
    if not batch_id:
        return None
    source_ids = [
        _clean_row_value(row, "source_cell_id"),
        _clean_row_value(row, "cell_id"),
        _clean_row_value(row, "anonymized_cell_id"),
    ]
    source_ids = [value for value in source_ids if value]
    if not source_ids:
        return None
    batch_dir = _raw_data_root(raw_root) / batch_id
    for source_cell_id in source_ids:
        for name in _source_cell_name_candidates(batch_id, source_cell_id):
            candidate = batch_dir / name
            _append_attempt(attempted, candidate)
            if candidate.exists():
                return candidate
    return None


def _resolve_raw_path(row: pd.Series, raw_root: Union[str, Path]) -> RawPathResolution:
    root = Path(raw_root)
    attempted: List[str] = []
    first_source_column: Optional[str] = None
    first_source_path: Optional[str] = None

    for col in SOURCE_PATH_COLUMNS:
        path_text = _clean_row_value(row, col)
        if not path_text:
            continue
        if first_source_column is None:
            first_source_column = col
            first_source_path = path_text

        path = Path(path_text).expanduser()
        _append_attempt(attempted, path)
        if path.exists():
            return RawPathResolution(
                path=path,
                method="direct",
                source_column=col,
                original_source_path=path_text,
                resolved_source_path=str(path),
                attempted_paths=attempted,
            )

        rebased_from_marker = _rebase_marker_path(path_text, root)
        if rebased_from_marker is not None:
            _append_attempt(attempted, rebased_from_marker)
            if rebased_from_marker.exists():
                return RawPathResolution(
                    path=rebased_from_marker,
                    method="rebased",
                    source_column=col,
                    original_source_path=path_text,
                    resolved_source_path=str(rebased_from_marker),
                    attempted_paths=attempted,
                )

        if not path.is_absolute():
            rebased_relative = root / path
            _append_attempt(attempted, rebased_relative)
            if rebased_relative.exists():
                return RawPathResolution(
                    path=rebased_relative,
                    method="rebased",
                    source_column=col,
                    original_source_path=path_text,
                    resolved_source_path=str(rebased_relative),
                    attempted_paths=attempted,
                )

    reconstructed = _reconstruct_raw_path(row, root, attempted)
    if reconstructed is not None:
        return RawPathResolution(
            path=reconstructed,
            method="reconstructed",
            source_column=first_source_column,
            original_source_path=first_source_path,
            resolved_source_path=str(reconstructed),
            attempted_paths=attempted,
        )

    return RawPathResolution(
        path=None,
        method="unresolved",
        source_column=first_source_column,
        original_source_path=first_source_path,
        resolved_source_path=None,
        attempted_paths=attempted,
    )


def _operator_key(operator_name: str, operator_type: str) -> str:
    return operator_name or operator_type


def _program_requires_raw_cycles(program: FeatureProgram) -> bool:
    return any(spec.enabled and spec.operator_type in RAW_CYCLE_OPERATOR_TYPES for spec in program.operators)


def _compile_feature_program_detailed(
    program: FeatureProgram,
    raw_payload: Dict[str, Any],
    *,
    row_id: Optional[int],
    cell_id: Optional[str],
    processed_metadata_row: Optional[pd.Series] = None,
    registry: Optional[FeatureOperatorRegistry] = None,
) -> Tuple[Dict[str, float], List[Dict[str, Any]], List[str], List[Dict[str, Any]]]:
    registry = registry or default_operator_registry()
    summary_df = canonicalize_summary(raw_payload, cell_id=cell_id, row_id=row_id)
    if _program_requires_raw_cycles(program):
        cycles_df = canonicalize_cycles_interpolated(raw_payload, cell_id=cell_id, row_id=row_id)
    else:
        cycles_df = pd.DataFrame(columns=CANONICAL_CYCLE_COLUMNS)
        cycles_df.attrs["warnings"] = []
        cycles_df.attrs["resolved_signals"] = {}
    convention = infer_cycle_index_convention(cycles_df, summary_df)
    warnings = list(cycles_df.attrs.get("warnings", [])) + list(summary_df.attrs.get("warnings", []))
    warnings.append(f"cycle_index_convention={convention.get('cycle_index_convention')}")
    features: dict[str, float] = {}
    metadata: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []
    for spec in program.operators:
        if not spec.enabled:
            statuses.append({"operator_name": spec.operator_name, "status": "disabled", "n_features": 0, "warnings": []})
            continue
        if spec.operator_type == "protocol" and not program.include_protocol_features:
            message = "protocol operator skipped because include_protocol_features is false"
            warnings.append(message)
            statuses.append({"operator_name": spec.operator_name, "status": "skipped", "n_features": 0, "warnings": [message]})
            continue
        try:
            fn = registry.get(_operator_key(spec.operator_name, spec.operator_type))
            output = fn(spec, cycles_df, summary_df, processed_metadata_row)
            for key, value in output.features.items():
                try:
                    numeric = float(value)
                except Exception:
                    numeric = np.nan
                features[key] = numeric if np.isfinite(numeric) else np.nan
            for row in output.metadata:
                enriched = dict(row)
                enriched.setdefault("program_id", program.program_id)
                enriched.setdefault("cycle_index_convention", program.cycle_index_convention)
                metadata.append(enriched)
            warnings.extend(output.warnings)
            statuses.append(
                {
                    "operator_name": spec.operator_name,
                    "operator_type": spec.operator_type,
                    "status": output.status,
                    "n_features": int(len(output.features)),
                    "warnings": list(output.warnings),
                }
            )
        except Exception as exc:
            message = f"{spec.operator_name} failed: {type(exc).__name__}: {exc}"
            warnings.append(message)
            statuses.append(
                {
                    "operator_name": spec.operator_name,
                    "operator_type": spec.operator_type,
                    "status": "failed",
                    "n_features": 0,
                    "warnings": [message],
                }
            )
    return features, metadata, warnings, statuses


def compile_feature_program(
    program: FeatureProgram,
    raw_payload: Dict[str, Any],
    *,
    row_id: Optional[int],
    cell_id: Optional[str],
    processed_metadata_row: Optional[pd.Series] = None,
    registry: Optional[FeatureOperatorRegistry] = None,
) -> Tuple[Dict[str, float], List[Dict[str, Any]], List[str]]:
    features, metadata, warnings, _statuses = _compile_feature_program_detailed(
        program,
        raw_payload,
        row_id=row_id,
        cell_id=cell_id,
        processed_metadata_row=processed_metadata_row,
        registry=registry,
    )
    return features, metadata, warnings


def _dataset_card_md(card: dict[str, Any]) -> str:
    family_counts = card.get("feature_family_counts", {})
    path_counts = card.get("source_path_resolution_counts", {})
    lines = [
        "# Feature Program Dataset Card",
        "",
        f"program_id: `{card.get('program_id')}`",
        f"name: `{card.get('name')}`",
        f"raw_source_root: `{card.get('raw_source_root')}`",
        f"cells_requested: `{card.get('cells_requested')}`",
        f"cells_featurized: `{card.get('cells_featurized')}`",
        f"excluded_cells: `{card.get('n_excluded_cells')}`",
        f"pruned_all_nan_features: `{card.get('n_pruned_all_nan_features')}`",
        f"cycle_index_convention: `{card.get('cycle_index_convention')}`",
        f"true_curve_features_used: `{card.get('true_curve_features_used')}`",
        f"proxy_features_used: `{card.get('proxy_features_used')}`",
        f"protocol_features_used: `{card.get('protocol_features_used')}`",
        "",
        "## Source Path Resolution",
        "",
        f"- used_directly: `{card.get('source_paths_used_directly', path_counts.get('direct', 0))}`",
        f"- rebased: `{card.get('source_paths_rebased', path_counts.get('rebased', 0))}`",
        f"- reconstructed: `{card.get('source_paths_reconstructed', path_counts.get('reconstructed', 0))}`",
        f"- unresolved: `{card.get('source_paths_unresolved', path_counts.get('unresolved', 0))}`",
        f"- audit_file: `{card.get('source_path_resolution_path')}`",
        "",
        "## Feature Families",
        "",
    ]
    if family_counts:
        for family, count in sorted(family_counts.items()):
            lines.append(f"- `{family}`: {count}")
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def build_feature_program_table(
    program: FeatureProgram,
    cell_manifest: pd.DataFrame,
    *,
    raw_root: Union[str, Path],
    out_dir: Union[str, Path],
    registry: Optional[FeatureOperatorRegistry] = None,
    strict: bool = False,
) -> FeatureProgramResult:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    registry = registry or default_operator_registry()
    feature_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    source_path_rows: list[dict[str, Any]] = []
    source_path_resolution_counts: Counter = Counter({method: 0 for method in SOURCE_PATH_RESOLUTION_METHODS})
    warning_rows: list[str] = []
    operator_status_counts: dict[str, Counter] = defaultdict(Counter)
    operator_warnings: dict[str, list[str]] = defaultdict(list)
    for _, row in cell_manifest.iterrows():
        row_id = int(row["row_id"]) if "row_id" in row and pd.notna(row["row_id"]) else None
        cell_id = str(row["cell_id"]) if "cell_id" in row and pd.notna(row["cell_id"]) else (f"cell_{row_id}" if row_id is not None else None)
        resolution: Optional[RawPathResolution] = None
        try:
            resolution = _resolve_raw_path(row, raw_root)
            source_path_resolution_counts[resolution.method] += 1
            source_path_rows.append(resolution.to_audit_row(row, row_id, cell_id))
            if resolution.path is None:
                attempted = ", ".join(resolution.attempted_paths or [])
                raise FileNotFoundError(f"could not resolve raw source path; attempted: {attempted or 'none'}")
            raw_path = resolution.path
            payload = load_attia_json_payload(raw_path)
            features, metadata, warnings, statuses = _compile_feature_program_detailed(
                program,
                payload,
                row_id=row_id,
                cell_id=cell_id,
                processed_metadata_row=row,
                registry=registry,
            )
            feature_rows.append({"row_id": row_id, "cell_id": cell_id, **features})
            metadata_rows.extend(metadata)
            for status in statuses:
                key = str(status.get("operator_name"))
                operator_status_counts[key][str(status.get("status"))] += 1
                operator_warnings[key].extend(map(str, status.get("warnings", [])))
            warning_rows.extend([f"{cell_id}: {warning}" for warning in warnings])
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            if resolution is None:
                resolution = _resolve_raw_path(row, raw_root)
                source_path_resolution_counts[resolution.method] += 1
                source_path_rows.append(resolution.to_audit_row(row, row_id, cell_id))
            exclusions.append(
                {
                    "row_id": row_id,
                    "cell_id": cell_id,
                    "batch_id": _clean_row_value(row, "batch_id"),
                    "source_cell_id": _clean_row_value(row, "source_cell_id"),
                    "canonical_raw_path": _clean_row_value(row, "canonical_raw_path"),
                    "source_path": resolution.original_source_path,
                    "resolved_source_path": resolution.resolved_source_path,
                    "source_path_resolution_method": resolution.method,
                    "exclusion_stage": "feature_program_compile",
                    "reason": reason,
                }
            )
            if strict:
                raise
    feature_table = pd.DataFrame(feature_rows)
    if feature_table.empty:
        feature_table = pd.DataFrame(columns=["row_id", "cell_id"])
    for col in list(feature_table.columns):
        if col in {"row_id", "cell_id"}:
            continue
        feature_table[col] = pd.to_numeric(feature_table[col], errors="coerce")
    feature_cols = [col for col in feature_table.columns if col not in {"row_id", "cell_id"}]
    all_nan_cols = [col for col in feature_cols if feature_table[col].isna().all()]
    if all_nan_cols:
        feature_table = feature_table.drop(columns=all_nan_cols)
    n_pruned_all_nan_features = int(len(all_nan_cols))
    feature_metadata = pd.DataFrame(metadata_rows)
    if feature_metadata.empty:
        feature_metadata = pd.DataFrame(
            columns=[
                "feature_name",
                "feature_family",
                "family",
                "feature_source",
                "operator_name",
                "operator_type",
                "is_true_curve_feature",
                "is_proxy_feature",
                "uses_raw_cycles_interpolated",
                "uses_summary",
                "uses_protocol",
                "requested_signal",
                "resolved_signal",
            ]
        )
    else:
        feature_metadata = feature_metadata.drop_duplicates("feature_name")
        feature_metadata = feature_metadata[feature_metadata["feature_name"].isin([col for col in feature_table.columns if col not in {"row_id", "cell_id"}])]
    family_col = "feature_family" if "feature_family" in feature_metadata else "family"
    feature_family_counts = (
        {str(k): int(v) for k, v in feature_metadata[family_col].value_counts().to_dict().items()} if family_col in feature_metadata else {}
    )
    operator_status = [
        {
            "operator_name": name,
            "status_counts": dict(counts),
            "warning_count": len(operator_warnings.get(name, [])),
            "warnings_preview": operator_warnings.get(name, [])[:10],
        }
        for name, counts in sorted(operator_status_counts.items())
    ]
    feature_table_path = out / "feature_table.csv"
    feature_metadata_path = out / "feature_metadata.csv"
    program_path = out / "feature_program.json"
    result_path = out / "feature_program_result.json"
    exclusions_path = out / "exclusions.csv"
    source_path_resolution_path = out / "source_path_resolution.csv"
    dataset_card_path = out / "dataset_card.json"
    dataset_card_md_path = out / "dataset_card.md"
    feature_table.to_csv(feature_table_path, index=False)
    feature_metadata.to_csv(feature_metadata_path, index=False)
    pd.DataFrame(
        exclusions,
        columns=[
            "row_id",
            "cell_id",
            "batch_id",
            "source_cell_id",
            "canonical_raw_path",
            "source_path",
            "resolved_source_path",
            "source_path_resolution_method",
            "exclusion_stage",
            "reason",
        ],
    ).to_csv(exclusions_path, index=False)
    pd.DataFrame(
        source_path_rows,
        columns=[
            "row_id",
            "cell_id",
            "batch_id",
            "source_cell_id",
            "canonical_raw_path",
            "source_column",
            "source_path",
            "resolved_source_path",
            "source_path_resolution_method",
            "attempted_paths",
        ],
    ).to_csv(source_path_resolution_path, index=False)
    _write_json(program_path, program)
    table_warnings = validate_feature_table(feature_table)
    metadata_warnings = validate_feature_metadata(feature_metadata)
    warnings = warning_rows[:100] + table_warnings + metadata_warnings
    numeric_cols = [
        col
        for col in feature_table.columns
        if col not in {"row_id", "cell_id"} and pd.api.types.is_numeric_dtype(feature_table[col])
    ]
    result = FeatureProgramResult(
        run_id=program.run_id,
        parent_artifact_ids=[program.artifact_id],
        human_readable_summary=f"Feature program {program.program_id} built {len(numeric_cols)} numeric feature columns for {len(feature_table)} cells.",
        program_id=program.program_id,
        feature_table_path=str(feature_table_path),
        feature_metadata_path=str(feature_metadata_path),
        n_rows=int(len(feature_table)),
        n_feature_columns=int(max(0, feature_table.shape[1] - 2)),
        n_numeric_feature_columns=int(len(numeric_cols)),
        n_excluded_cells=int(len(exclusions)),
        n_pruned_all_nan_features=n_pruned_all_nan_features,
        feature_family_counts=feature_family_counts,
        operator_status=operator_status,
        warnings=warnings,
    )
    _write_json(result_path, result)
    dataset_card = {
        "program_id": program.program_id,
        "name": program.name,
        "description": program.description,
        "raw_source_root": str(raw_root),
        "cells_requested": int(len(cell_manifest)),
        "cells_featurized": int(len(feature_table)),
        "n_excluded_cells": int(len(exclusions)),
        "n_pruned_all_nan_features": n_pruned_all_nan_features,
        "pruned_all_nan_features_preview": all_nan_cols[:25],
        "feature_family_counts": feature_family_counts,
        "cycle_index_convention": program.cycle_index_convention,
        "true_curve_features_used": bool(feature_metadata.get("is_true_curve_feature", pd.Series(dtype=bool)).fillna(False).astype(bool).any()),
        "proxy_features_used": bool(feature_metadata.get("is_proxy_feature", pd.Series(dtype=bool)).fillna(False).astype(bool).any()),
        "protocol_features_used": bool(feature_metadata.get("uses_protocol", pd.Series(dtype=bool)).fillna(False).astype(bool).any()),
        "warnings_count": int(len(warnings)),
        "exclusions_count": int(len(exclusions)),
        "source_path_resolution_path": str(source_path_resolution_path),
        "source_path_resolution_counts": {method: int(source_path_resolution_counts.get(method, 0)) for method in SOURCE_PATH_RESOLUTION_METHODS},
        "source_paths_used_directly": int(source_path_resolution_counts.get("direct", 0)),
        "source_paths_rebased": int(source_path_resolution_counts.get("rebased", 0)),
        "source_paths_reconstructed": int(source_path_resolution_counts.get("reconstructed", 0)),
        "source_paths_unresolved": int(source_path_resolution_counts.get("unresolved", 0)),
    }
    _write_json(dataset_card_path, dataset_card)
    dataset_card_md_path.write_text(_dataset_card_md(dataset_card))
    return result
