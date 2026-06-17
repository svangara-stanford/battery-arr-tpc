#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from battery_aar.features.severson_matr import (
    SEVERSON_CANONICAL_SEED_DEFAULT,
    build_severson_true_life_dataset,
)


DEFAULT_EXCLUSIONS_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "severson_canonical_exclusions.txt"
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a processed Severson 2019 true-life dataset from MatR .mat files.")
    parser.add_argument("--mat-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--first-n-cycles", type=int, default=100)
    parser.add_argument(
        "--split-mode",
        choices=["canonical_severson", "source_file_identity"],
        default="canonical_severson",
        help=(
            "Splitter: 'canonical_severson' = Severson 2019 train / primary_test "
            "(half of b1+b2) + secondary_test (b3); 'source_file_identity' = "
            "legacy b1->train, b2->validation, b3->test."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEVERSON_CANONICAL_SEED_DEFAULT,
        help="Deterministic seed for the canonical train/primary_test shuffle.",
    )
    parser.add_argument(
        "--exclusion-cell-ids",
        type=Path,
        default=None,
        help=(
            "Optional path to a text file of cell_ids to drop (one per line; "
            "lines starting with '#' are comments). Defaults to "
            f"{DEFAULT_EXCLUSIONS_PATH} when that file exists."
        ),
    )
    parser.add_argument(
        "--no-default-exclusions",
        action="store_true",
        help="Disable the default canonical exclusions file.",
    )
    args = parser.parse_args()

    exclusion_path = args.exclusion_cell_ids
    if exclusion_path is None and not args.no_default_exclusions and DEFAULT_EXCLUSIONS_PATH.exists():
        exclusion_path = DEFAULT_EXCLUSIONS_PATH

    card = build_severson_true_life_dataset(
        mat_dir=args.mat_dir,
        out_dir=args.out,
        first_n_cycles=args.first_n_cycles,
        split_mode=args.split_mode,
        split_seed=args.seed,
        exclusion_cell_ids=exclusion_path,
    )
    print(
        "built Severson true-life dataset: "
        f"included_cells={card['included_cells']} cycle_rows={card['cycle_rows']} "
        f"label_source={card['label_source']} split_mode={card.get('split_mode')}"
    )
    if card.get("split_mode") == "canonical_severson":
        print(
            "  canonical split: "
            f"train={card.get('train_cells')} "
            f"primary_test={card.get('primary_test_cells')} "
            f"secondary_test={card.get('secondary_test_cells')} "
            f"(seed={card.get('split_seed')})"
        )
    print(f"out: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
