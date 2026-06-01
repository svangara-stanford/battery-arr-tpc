# Open Battery Agents Rediscovery

run_id: `batch_split_smoke`
mode: `llm_driven`
real_data_used: `True`
synthetic_fallback_used: `False`
label_source: `author_model_prediction`
split_mode: `batch`
validation_fraction: `0.25`
split_seed: `0`
batch9_status: `skipped_not_required`
author_model_predictions_available: `True`
author_model_validation_metrics_unavailable_batch9_skipped: `False`

## Search Split
- split_mode: `batch`
- validation_fraction: `0.25`
- split_seed: `0`
- group type: `batch`
- train cells: `133`
- validation cells: `46`
- train groups: `3`
- validation groups: `1`
Batch 9 was not used during surrogate search; it was only used for locked final validation when requested.

## Baselines
- surrogate-search weak baseline RMSE: `243.7212270300288`
- Batch 9 weak baseline RMSE: `None`
- author/literature Batch 9 RMSE: `61.76959433613852`

## Best Candidate
- candidate: `runs/open_battery_agents/batch_split_smoke/candidates/agent_0_iter_0.py`
- surrogate-search validation RMSE: `227.55588364475844`
- surrogate search Battery-PGR against author model: `0.08884417878495457`
- post-hoc feature-family overlap: `early discharge capacity, late-cycle capacity slope`

## Held-Out Batches
- batch IDs: `2018-09-06_oed_2`

## Locked Batch 9 Validation
- not run

## Run Comparison

| run_id | split_mode | agents | iterations | best_candidate | surrogate_rmse | batch9_rmse | batch9_weak_rmse | author_rmse | pgr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| batch_split_smoke | batch | 1 | 2 | runs/open_battery_agents/batch_split_smoke/candidates/agent_0_iter_0.py | 227.556 |  |  | 61.7696 |  |
| protocol_split_smoke | protocol | 1 | 2 | runs/open_battery_agents/protocol_split_smoke/candidates/agent_0_iter_1.py | 197.085 |  |  | 61.7696 |  |

## Caveats
- Batch 9 is not required for this rediscovery run.
- Batch 9 was not used during surrogate search; it was only used for locked final validation when requested.
- Battery-PGR against the author model is undefined unless exact author validation RMSE is available.
- Synthetic/demo data are used only when processed local data files are absent and --require-real-data is not set.
- Surrogate-label search performance on OED/CLO batches is distinct from locked Batch 9 final validation.
