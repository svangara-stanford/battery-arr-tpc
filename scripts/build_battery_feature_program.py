#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from battery_aar.features.feature_programs import build_feature_program_table
from battery_aar.features.program_library import (
    PROGRAM_RECIPES,
    make_attia_severson_like_program,
    make_broad_physics_program,
    make_curve_delta_program,
    make_minimal_debug_program,
    make_scalar_baseline_program,
)
from battery_aar.paper_reproduction.paths import VALIDATION_BATCH_NAME, resolve_paper_paths
from battery_aar.workflows.schemas import FeatureOperatorSpec, FeatureProgram


def _bool_arg(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def _load_program(args: argparse.Namespace) -> FeatureProgram:
    if args.feature_program_json:
        program = FeatureProgram.model_validate_json(Path(args.feature_program_json).read_text())
    elif args.recipe == "scalar_baseline":
        program = make_scalar_baseline_program(args.first_n_cycles)
    elif args.recipe == "curve_delta":
        program = make_curve_delta_program(args.cycle_early_index, args.cycle_late_index)
    elif args.recipe == "attia_severson_like":
        program = make_attia_severson_like_program(args.cycle_early_index, args.cycle_late_index)
    elif args.recipe == "broad_physics":
        program = make_broad_physics_program(args.first_n_cycles)
    elif args.recipe == "minimal_debug":
        program = make_minimal_debug_program()
    else:
        raise ValueError(f"unknown feature program recipe: {args.recipe}")
    program.include_protocol_features = bool(args.include_protocol_features)
    program.first_n_cycles = int(args.first_n_cycles)
    if program.include_protocol_features and not any(op.operator_type == "protocol" for op in program.operators):
        program.operators.append(
            FeatureOperatorSpec(
                operator_name="protocol",
                operator_type="protocol",
                family="protocol",
                params={"columns": ["C1", "C2", "C3", "C4", "cc1", "cc2", "cc3", "cc4"]},
            )
        )
    return program


def _oed_manifest(processed_dir: Path) -> pd.DataFrame:
    path = processed_dir / "cell_metadata.csv"
    if not path.exists():
        path = processed_dir / "metadata.csv"
    if not path.exists():
        raise FileNotFoundError(f"processed metadata not found in {processed_dir}")
    metadata = pd.read_csv(path)
    if "source_path" not in metadata.columns:
        raise ValueError("OED feature-program build requires source_path in processed metadata")
    if "batch_id" in metadata.columns:
        metadata = metadata[~metadata["batch_id"].astype(str).str.contains(VALIDATION_BATCH_NAME, regex=False)].copy()
    return metadata[metadata["source_path"].notna()].copy()


def _batch9_manifest(root: Path) -> pd.DataFrame:
    batch_path = root / "data" / VALIDATION_BATCH_NAME
    if not batch_path.is_dir():
        raise FileNotFoundError(f"expanded Batch 9 directory not found: {batch_path}")
    files = sorted(batch_path.glob("*_structure.json")) or sorted(batch_path.glob("*.json"))
    rows = []
    for row_id, path in enumerate(files):
        rows.append({"row_id": row_id, "cell_id": path.stem.replace("_structure", ""), "batch_id": VALIDATION_BATCH_NAME, "source_path": str(path)})
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build trusted Open Battery Agents feature-program tables.")
    parser.add_argument("--battery-fast-charging-root", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--recipe", choices=sorted(PROGRAM_RECIPES), default="minimal_debug")
    parser.add_argument("--feature-program-json", type=Path, default=None)
    parser.add_argument("--first-n-cycles", type=int, default=100)
    parser.add_argument("--cycle-early-index", type=int, default=9)
    parser.add_argument("--cycle-late-index", type=int, default=99)
    parser.add_argument("--include-protocol-features", type=_bool_arg, default=False)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--batch", choices=["oed", "batch9", "all"], default="oed")
    args = parser.parse_args()

    paths = resolve_paper_paths(args.battery_fast_charging_root)
    program = _load_program(args)
    if args.batch == "oed":
        manifest = _oed_manifest(args.processed_dir)
    elif args.batch == "batch9":
        manifest = _batch9_manifest(paths.root)
    else:
        oed = _oed_manifest(args.processed_dir)
        batch9 = _batch9_manifest(paths.root)
        if not oed.empty:
            batch9 = batch9.copy()
            batch9["row_id"] = pd.to_numeric(batch9["row_id"], errors="raise").astype(int) + int(pd.to_numeric(oed["row_id"], errors="coerce").max()) + 1
        manifest = pd.concat([oed, batch9], ignore_index=True)

    result = build_feature_program_table(
        program,
        manifest,
        raw_root=paths.root,
        out_dir=args.out,
        strict=args.strict,
    )
    print(
        f"built feature program {result.program_id}: rows={result.n_rows}, "
        f"numeric_features={result.n_numeric_feature_columns}, excluded={result.n_excluded_cells}"
    )
    print(f"feature_table: {result.feature_table_path}")
    print(f"feature_metadata: {result.feature_metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
