# Open Battery Agents v2 Tools

Open Battery Agents v2 tools provide explicit, discoverable operations for dataset profiling, feature building, candidate review, candidate evaluation, and run comparison.

The tools can be used in two ways:

- `NativeToolClient`: calls Python implementations in-process.
- `HTTPToolClient`: calls the FastAPI server endpoints.

Both clients expose the same method names:

```python
profile_dataset(...)
build_battery_features(...)
review_candidate(...)
evaluate_candidate(...)
compare_runs(...)
```

## FastAPI Server

Start the server:

```bash
python scripts/run_tool_server.py --host 127.0.0.1 --port 8000
```

Smoke checks:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/tools
```

FastAPI automatically exposes OpenAPI docs at:

```text
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/openapi.json
```

## Tool Schemas

Request and response models live in:

```text
src/battery_aar/tools/schemas.py
```

Every response includes:

- `tool_name`
- `tool_call_id`
- `run_id`
- `success`
- `output_artifact_ids`
- `output_paths`
- `error_type`
- `error_message`
- `duration_ms`

## Trace Logs

Every tool call writes a record to:

```text
runs/open_battery_agents/<run_id>/artifacts/tool_calls.jsonl
```

Each record includes the tool name, tool call ID, run ID, optional iteration and agent role, request hash, output artifact IDs, duration, and success or failure details.

The trace directory also initializes:

```text
events.jsonl
tool_calls.jsonl
agent_messages.jsonl
```

so downstream agents can rely on stable paths even before messages or tool calls are written.

## Role-Specialized Agents

Future role-specialized agents should call tools through `NativeToolClient` or `HTTPToolClient` instead of directly reading arbitrary files. This gives each agent a typed, traceable interface:

- dataset profilers call `profile_dataset`
- feature engineers call `build_battery_features`
- reviewers call `review_candidate`
- evaluators call `evaluate_candidate`
- orchestrators and critics call `compare_runs`

Tool responses return artifact IDs and paths that can be passed into `FeaturePlan`, `ModelPlan`, `ReviewReport`, `EvaluationReport`, and `CritiqueReport` artifacts.
