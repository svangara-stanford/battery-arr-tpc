# Open Battery Agents Role Workflow

run_id: `v2_role_graph_smoke`
split_mode: `random`
offline: `True`
iterations: `1`

## Role Sequence

DatasetProfiler, FeatureScientist, ModelArchitect, CodeGenerator, CodeReviewer, Evaluator, ScientistCritic

## Split

- train cells: 134
- validation cells: 45
- validation fraction: 0.25
- split seed: 0

Batch 9 was not used during surrogate search.

## Tool Calls

- profile_dataset success=True duration_ms=38.84808300063014
- build_battery_features success=True duration_ms=564.9014169975999
- review_candidate success=True duration_ms=13.534291996620595
- evaluate_candidate success=True duration_ms=836.4130000045407

## Candidate

- candidate path: `runs/open_battery_agents/v2_role_graph_smoke/candidates/role_graph_iter_000.py`
- review verdict: `pass`

## Validation Metrics

- rmse: 176.03634523076144
- mae: 151.16576478995836
- r2: 0.020527598105133382
- spearman: 0.2191040843214756
- kendall: 0.1515151515151515

## Critique

ScientistCritic summarized the candidate evaluation.

## Artifacts

- artifact index: `runs/open_battery_agents/v2_role_graph_smoke/artifacts/index.json`
