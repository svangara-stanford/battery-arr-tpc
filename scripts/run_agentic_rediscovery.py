#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from battery_aar.agents.orchestrator import run_rediscovery


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Open Battery Agents rediscovery loop.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/chueh_toyota_fast_charge"))
    parser.add_argument("--reference-run", type=Path, default=Path("runs/attia_reference_reproduction"))
    parser.add_argument("--out", type=Path, default=Path("runs/open_battery_agents/demo"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--agents", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--model", default=None)
    parser.add_argument("--split-mode", choices=["random", "protocol", "batch", "leave_one_batch_out"], default="random")
    parser.add_argument("--allow-protocol-features", action="store_true")
    parser.add_argument("--max-cycle", type=int, default=100)
    parser.add_argument("--locked-test", action="store_true")
    parser.add_argument("--require-real-data", action="store_true")
    parser.add_argument("--final-batch9-validation", action="store_true")
    parser.add_argument("--battery-fast-charging-root", type=Path, default=None)
    parser.add_argument("--batch9-path", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_rediscovery(
        processed_dir=args.processed_dir,
        reference_run=args.reference_run,
        out=args.out,
        reports_dir=args.reports_dir,
        agents=args.agents,
        iterations=args.iterations,
        offline=args.offline,
        model=args.model,
        split_mode=args.split_mode,
        allow_protocol_features=args.allow_protocol_features,
        max_cycle=args.max_cycle,
        locked_test=args.locked_test,
        seed=args.seed,
        require_real_data=args.require_real_data,
        final_batch9_validation=args.final_batch9_validation,
        battery_fast_charging_root=args.battery_fast_charging_root,
        batch9_path=args.batch9_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
