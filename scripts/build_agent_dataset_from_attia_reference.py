#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from battery_aar.agents.attia_data_bridge import build_agent_surrogate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Open Battery Agents surrogate dataset from Attia OED/CLO reference replay.")
    parser.add_argument("--battery-fast-charging-root", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label-source", choices=["author_model_prediction"], default="author_model_prediction")
    parser.add_argument("--first-n-cycles", type=int, default=100)
    parser.add_argument("--exclude-batch9", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_agent_surrogate_dataset(
        battery_fast_charging_root=args.battery_fast_charging_root,
        reference_run=args.reference_run,
        out=args.out,
        label_source=args.label_source,
        first_n_cycles=args.first_n_cycles,
        exclude_batch9=args.exclude_batch9,
        seed=args.seed,
    )
    card = result.dataset_card
    print(
        json.dumps(
            {
                "out": str(args.out),
                "batches_included": [batch["batch_id"] for batch in card.get("batches", []) if batch.get("included_cells", 0) > 0],
                "valid_labeled_cells": len(result.metadata),
                "cycle_summary_rows": len(result.cycle_summary),
                "skipped_raw_cells": card.get("skipped_raw_cells", 0),
                "skipped_label_rows": card.get("skipped_label_rows", 0),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
