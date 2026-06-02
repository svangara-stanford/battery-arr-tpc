from __future__ import annotations

import json
from collections import Counter, defaultdict
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
from battery_aar.workflows.schemas import FeatureProgram, FeatureProgramResult

RAW_CYCLE_OPERATOR_TYPES = {"curve_shape", "cross_cycle_curve_delta"}


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


def _resolve_raw_path(row: pd.Series, raw_root: Union[str, Path]) -> Path:
    for col in ["source_path", "raw_file", "raw_path", "file_path", "path"]:
        if col in row.index and pd.notna(row[col]) and str(row[col]).strip():
            path = Path(str(row[col]))
            if path.is_absolute():
                return path
            return Path(raw_root) / path
    raise ValueError("cell_manifest row requires source_path/raw_file/raw_path/file_path/path")


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
    warning_rows: list[str] = []
    operator_status_counts: dict[str, Counter] = defaultdict(Counter)
    operator_warnings: dict[str, list[str]] = defaultdict(list)
    for _, row in cell_manifest.iterrows():
        row_id = int(row["row_id"]) if "row_id" in row and pd.notna(row["row_id"]) else None
        cell_id = str(row["cell_id"]) if "cell_id" in row and pd.notna(row["cell_id"]) else (f"cell_{row_id}" if row_id is not None else None)
        try:
            raw_path = _resolve_raw_path(row, raw_root)
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
            exclusions.append({"row_id": row_id, "cell_id": cell_id, "exclusion_stage": "feature_program_compile", "reason": reason})
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
    dataset_card_path = out / "dataset_card.json"
    dataset_card_md_path = out / "dataset_card.md"
    feature_table.to_csv(feature_table_path, index=False)
    feature_metadata.to_csv(feature_metadata_path, index=False)
    pd.DataFrame(exclusions, columns=["row_id", "cell_id", "exclusion_stage", "reason"]).to_csv(exclusions_path, index=False)
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
    }
    _write_json(dataset_card_path, dataset_card)
    dataset_card_md_path.write_text(_dataset_card_md(dataset_card))
    return result
