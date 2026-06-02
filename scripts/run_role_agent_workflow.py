#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from battery_aar.workflows.role_graph import run_role_workflow


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Open Battery Agents v2 role-specialized workflow graph.")
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("runs/open_battery_agents/v2_role_graph_smoke"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--split-mode", choices=["random", "protocol", "batch", "leave_one_batch_out"], default="random")
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--validation-batch-id", default=None)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--use-http-tools", action="store_true")
    parser.add_argument("--tool-server-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-cycle", type=int, default=100)
    parser.add_argument("--allow-protocol-features", action="store_true")
    parser.add_argument("--allow-freeform-code", action="store_true")
    parser.add_argument("--candidates-per-iteration", type=int, default=1)
    parser.add_argument("--final-batch9-validation", action="store_true")
    parser.add_argument("--locked-test", action="store_true", help="Alias for --final-batch9-validation.")
    parser.add_argument("--battery-fast-charging-root", type=Path, default=None)
    parser.add_argument("--batch9-path", type=Path, default=None)
    parser.add_argument("--final-batch9-top-k", type=int, default=0)
    args = parser.parse_args()

    run_role_workflow(
        processed_dir=args.processed_dir,
        reference_run=args.reference_run,
        out=args.out,
        reports_dir=args.reports_dir,
        split_mode=args.split_mode,
        validation_fraction=args.validation_fraction,
        split_seed=args.split_seed,
        validation_batch_id=args.validation_batch_id,
        offline=args.offline,
        iterations=args.iterations,
        use_http_tools=args.use_http_tools,
        tool_server_url=args.tool_server_url,
        model=args.model,
        max_cycle=args.max_cycle,
        allow_protocol_features=args.allow_protocol_features,
        allow_freeform_code=args.allow_freeform_code,
        candidates_per_iteration=args.candidates_per_iteration,
        final_batch9_validation=bool(args.final_batch9_validation or args.locked_test),
        battery_fast_charging_root=args.battery_fast_charging_root,
        batch9_path=args.batch9_path,
        final_batch9_top_k=args.final_batch9_top_k,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
