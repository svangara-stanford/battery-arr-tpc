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
    parser.add_argument("--final-batch9-validation", action="store_true",
                        help="DEPRECATED: alias for --final-secondary-test-validation when "
                        "--secondary-test-source=attia_batch9.")
    parser.add_argument("--locked-test", action="store_true",
                        help="Alias for --final-batch9-validation / --final-secondary-test-validation.")
    parser.add_argument("--battery-fast-charging-root", type=Path, default=None)
    parser.add_argument("--batch9-path", type=Path, default=None,
                        help="DEPRECATED: alias for --secondary-test-path when "
                        "--secondary-test-source=attia_batch9.")
    parser.add_argument("--final-batch9-top-k", type=int, default=0,
                        help="DEPRECATED: alias for --final-secondary-test-top-k.")
    parser.add_argument("--feature-program-path", dest="feature_program_path", action="append", type=Path, default=[])
    parser.add_argument("--feature-program-paths", nargs="*", type=Path, default=[])
    parser.add_argument("--batch9-feature-program-path", type=Path, default=None,
                        help="DEPRECATED: alias for --secondary-test-feature-program-path.")
    # Patch E4: source-agnostic secondary-test contract.
    parser.add_argument("--secondary-test-source", choices=["severson_b3", "attia_batch9"],
                        default="severson_b3",
                        help="Source for the locked secondary test. Default 'severson_b3' "
                        "switches the locked test from Attia Batch 9 (transfer-only) to "
                        "Severson b3 (in-distribution true-life). 'attia_batch9' preserves "
                        "the old transfer behavior.")
    parser.add_argument("--secondary-test-path", type=Path, default=None,
                        help="Explicit path to the secondary-test data. For severson_b3 this "
                        "should be the .mat file (or its directory). For attia_batch9 this "
                        "is the Batch 9 directory.")
    parser.add_argument("--secondary-test-feature-program-path", type=Path, default=None,
                        help="Optional feature-program table for the secondary test (replaces "
                        "--batch9-feature-program-path).")
    parser.add_argument("--final-secondary-test-validation", action="store_true",
                        help="Run a single locked secondary-test validation on the best "
                        "candidate (replaces --final-batch9-validation).")
    parser.add_argument("--final-secondary-test-top-k", type=int, default=0,
                        help="Top-K Severson-validated candidates to also evaluate on the "
                        "secondary test (replaces --final-batch9-top-k).")
    parser.add_argument("--include-feature-programs", action="store_true")
    parser.add_argument("--feature-program-mode", choices=["none", "table", "auto"], default="none")
    parser.add_argument("--feature-program-recipe", default=None)
    parser.add_argument("--feature-family-filter", nargs="*", default=[])
    parser.add_argument("--cycle-early-index", type=int, default=9)
    parser.add_argument("--cycle-late-index", type=int, default=99)
    parser.add_argument("--no-rag", dest="rag_augment", action="store_false", default=True,
                        help="Disable RAG augmentation of the FeatureScientist prompt "
                        "(LLM-only ablation isolating the retrieval contribution).")
    parser.add_argument("--feature-budget", type=int, default=None,
                        help="Max number of feature columns the FeatureScientist's "
                        "operator specs may select. A tight budget (e.g. 8) makes "
                        "operator choice a knowledge decision so RAG can matter.")
    args = parser.parse_args()
    feature_program_paths = list(args.feature_program_path or []) + list(args.feature_program_paths or [])

    # Patch E4: resolve deprecated aliases.
    import warnings as _warnings

    if args.batch9_path is not None:
        _warnings.warn(
            "--batch9-path is deprecated; use --secondary-test-path (with "
            "--secondary-test-source=attia_batch9 to preserve transfer behavior).",
            DeprecationWarning,
            stacklevel=2,
        )
    if args.batch9_feature_program_path is not None and args.secondary_test_feature_program_path is None:
        args.secondary_test_feature_program_path = args.batch9_feature_program_path
    final_secondary_test_validation = bool(
        args.final_secondary_test_validation or args.final_batch9_validation or args.locked_test
    )
    final_secondary_test_top_k = max(
        int(args.final_secondary_test_top_k or 0),
        int(args.final_batch9_top_k or 0),
    )
    # Legacy --batch9-path: only respected when explicitly opting in to attia.
    legacy_batch9_path = args.batch9_path if args.secondary_test_source == "attia_batch9" else None

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
        final_batch9_validation=final_secondary_test_validation,
        battery_fast_charging_root=args.battery_fast_charging_root,
        batch9_path=legacy_batch9_path,
        final_batch9_top_k=final_secondary_test_top_k,
        feature_program_paths=feature_program_paths,
        batch9_feature_program_path=args.batch9_feature_program_path,
        secondary_test_source=args.secondary_test_source,
        secondary_test_path=args.secondary_test_path,
        secondary_test_feature_program_path=args.secondary_test_feature_program_path,
        final_secondary_test_validation=final_secondary_test_validation,
        final_secondary_test_top_k=final_secondary_test_top_k,
        include_feature_programs=args.include_feature_programs,
        feature_program_mode=args.feature_program_mode,
        feature_program_recipe=args.feature_program_recipe,
        feature_family_filter=args.feature_family_filter,
        cycle_early_index=args.cycle_early_index,
        cycle_late_index=args.cycle_late_index,
        rag_augment=args.rag_augment,
        feature_budget=args.feature_budget,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
