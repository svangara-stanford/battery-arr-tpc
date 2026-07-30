#!/usr/bin/env python
"""Demo: show the FeatureScientist's column-level feature pick actually training the model.

Runs a real N-iteration role workflow (live LLM), then prints, per iteration, how many
columns the FeatureScientist proposed vs the columns the compiled candidate actually
trained on. `match=True` is the invariant that matters -- it proves FeaturePlan
.selected_columns now flows into the trained model (before this wiring the compiled count
was always the full toolbox bucket, regardless of the agent's pick).

The exact column counts / RMSE vary run-to-run because the LLM is non-deterministic; that
is expected. Also prints the question-framed retrieval queries the rewriter produced.

Usage:
    python scripts/demo_selected_columns.py
    python scripts/demo_selected_columns.py --iterations 3 \
        --processed-dir data/processed/chueh_toyota_fast_charge_agent_surrogate

Flags if you want to vary it: --iterations N, --processed-dir <path>, --split-seed N, --out <dir>.

Requires an API key in .env (OPEN_BATTERY_AGENTS_API_KEY or STANFORD_AI_API_KEY); the
roles LLM path does not auto-load .env, so this script loads it for you.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def _read_selected_columns(candidate_path: Path) -> list[str] | None:
    for line in candidate_path.read_text().splitlines():
        if line.startswith("SELECTED_COLUMNS ="):
            return ast.literal_eval(line.split("=", 1)[1].strip())
    return None


def _first(paths: list[Path]) -> Path | None:
    return paths[0] if paths else None


def _report(out: Path, iterations: int) -> None:
    print("\n=== Retrieval queries (question-framed) ===")
    messages = out / "artifacts" / "agent_messages.jsonl"
    if messages.exists():
        for line in messages.read_text().splitlines():
            event = json.loads(line)
            if event.get("event_type") != "rag_prompt_augmentation":
                continue
            match = re.search(r"from queries (\[.*?\]);", event.get("message_summary", ""))
            if match:
                print(f"iter {event.get('iteration')}: {match.group(1)}")

    print("\n=== FeatureScientist's pick vs what the model actually trained on ===")
    for i in range(iterations):
        it = f"{i:03d}"
        d = out / "artifacts" / f"iteration_{it}"
        plan_path = _first(sorted(d.glob("feature_plan_*.json")))
        eval_path = _first(sorted(d.glob("evaluation_report_*.json")))
        cand_path = _first(sorted(out.glob(f"candidates/role_graph_iter_{it}*.py")))
        if not (plan_path and eval_path and cand_path):
            print(f"iter {it}: (artifacts missing -- iteration may have failed)")
            continue
        plan = json.loads(plan_path.read_text())
        evaluation = json.loads(eval_path.read_text())
        plan_sel = plan.get("selected_columns") or []
        compiled_sel = _read_selected_columns(cand_path) or []
        rmse = evaluation.get("rmse")
        rmse_text = f"{rmse:.2f}" if isinstance(rmse, (int, float)) else str(rmse)
        print(
            f"iter {it}: feature_set={plan.get('feature_set')!r} | "
            f"agent proposed {len(plan_sel)} cols | "
            f"compiled SELECTED_COLUMNS {len(compiled_sel)} cols | "
            f"match={set(plan_sel) == set(compiled_sel)} | RMSE={rmse_text}"
        )
    print(
        "\nmatch=True means the agent's column-level pick reached the trained model. "
        "Varying column counts across iterations = the choice actually changes training."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed/chueh_toyota_fast_charge_agent_surrogate"),
    )
    parser.add_argument("--out", type=Path, default=Path("runs/rag_selcol_demo"))
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--split-mode", default="random")
    parser.add_argument("--split-seed", type=int, default=0)
    args = parser.parse_args()

    _load_dotenv()

    # Import after dotenv so any env-dependent config picks up the key.
    from battery_aar.workflows.role_graph import run_role_workflow

    run_role_workflow(
        processed_dir=args.processed_dir,
        reference_run=None,
        out=args.out,
        reports_dir=args.out / "reports",
        split_mode=args.split_mode,
        split_seed=args.split_seed,
        offline=False,  # must be False so the LLM + RAG + selected_columns path runs
        iterations=args.iterations,
    )
    _report(args.out, args.iterations)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
