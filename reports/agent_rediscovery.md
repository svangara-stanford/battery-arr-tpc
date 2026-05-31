# Open Battery Agents Rediscovery

mode: `offline`
split_mode: `random`
batch9_status: `skipped_not_required`
author_model_predictions_available: `True`
author_model_validation_metrics_unavailable_batch9_skipped: `True`

## Baselines
- weak baseline RMSE: `296.0292049921881`
- exact author-model RMSE: `None`

## Best Candidate
- candidate: `runs/open_battery_agents/offline_smoke/candidates/agent_0_iter_0.py`
- validation RMSE: `294.3715475592552`
- Battery-PGR against author model: `None`
- post-hoc feature-family overlap: `early discharge capacity, late-cycle capacity slope, max early capacity change`

## Caveats
- Batch 9 is not required for this rediscovery run.
- Battery-PGR against the author model is undefined unless exact author validation RMSE is available.
- Synthetic/demo data are used when processed local data files are absent.
