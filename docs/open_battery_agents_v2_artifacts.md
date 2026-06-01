# Open Battery Agents v2 Artifacts

Open Battery Agents v2 adds typed artifacts and trace logs around the existing rediscovery loop. The goal is inspectability: a run should be auditable without re-running candidate code or reading a long unstructured event stream.

The artifact layer is additive. Existing runs behave the same way unless `--emit-artifact-trace` is passed to `scripts/run_agentic_rediscovery.py`.

## Where Artifacts Are Written

For a run at:

```text
runs/open_battery_agents/<run_id>/
```

trace-enabled artifacts are written under:

```text
runs/open_battery_agents/<run_id>/artifacts/
```

Common files include:

```text
artifacts/run_manifest.json
artifacts/dataset_profile.json
artifacts/split_artifact.json
artifacts/experiment_state.json
artifacts/index.json
artifacts/events.jsonl
artifacts/tool_calls.jsonl
artifacts/agent_messages.jsonl
artifacts/iteration_000/candidate_spec_agent_0.json
artifacts/iteration_000/evaluation_report_agent_0.json
```

`artifacts/index.json` is the easiest entry point. It lists each artifact ID, type, relative path, parent artifact IDs, creation time, and summary.

## How To Inspect A Run

Run with tracing enabled:

```bash
python scripts/run_agentic_rediscovery.py \
  --offline \
  --agents 1 \
  --iterations 1 \
  --out runs/open_battery_agents/trace_demo \
  --reports-dir reports \
  --emit-artifact-trace
```

Then inspect:

```bash
python -m json.tool runs/open_battery_agents/trace_demo/artifacts/index.json
python -m json.tool runs/open_battery_agents/trace_demo/artifacts/dataset_profile.json
python -m json.tool runs/open_battery_agents/trace_demo/artifacts/experiment_state.json
```

The ordinary run outputs, including `leaderboard.csv`, `events.jsonl`, reports, and optional locked Batch 9 files, are still written in their existing locations.

## Future Agent Roles

The schemas include placeholders for role-specialized agents without implementing those agents yet:

- `FeaturePlan`: proposed feature families, selected columns, protocol-use decisions, and rationale.
- `ModelPlan`: estimator family, preprocessing, hyperparameters, and modeling rationale.
- `ReviewReport`: reviewer verdicts, issues, and recommendations on candidate artifacts.
- `CritiqueReport`: post-evaluation critique and proposed next steps.

Future FastAPI tools, DSPy workflows, or multi-role agents should exchange these artifacts rather than passing large untyped dictionaries. That keeps candidate generation, review, evaluation, and critique auditable and reproducible.
