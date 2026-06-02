#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from battery_aar.features.severson_matr import build_severson_true_life_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a processed Severson 2019 true-life dataset from MatR .mat files.")
    parser.add_argument("--mat-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--first-n-cycles", type=int, default=100)
    args = parser.parse_args()

    card = build_severson_true_life_dataset(
        mat_dir=args.mat_dir,
        out_dir=args.out,
        first_n_cycles=args.first_n_cycles,
    )
    print(
        "built Severson true-life dataset: "
        f"included_cells={card['included_cells']} cycle_rows={card['cycle_rows']} "
        f"label_source={card['label_source']}"
    )
    print(f"out: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
