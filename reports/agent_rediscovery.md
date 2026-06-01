# Open Battery Agents Rediscovery

mode: `llm_driven`
real_data_used: `True`
synthetic_fallback_used: `False`
label_source: `author_model_prediction`
split_mode: `random`
batch9_status: `skipped_not_required`
author_model_predictions_available: `True`
author_model_validation_metrics_unavailable_batch9_skipped: `False`

## Baselines
- weak baseline RMSE: `178.50752274737178`
- exact author-model RMSE: `61.76959433613852`

## Best Candidate
- candidate: `runs/open_battery_agents/real_surrogate_llm_demo/candidates/agent_0_iter_2.py`
- surrogate search validation RMSE: `178.42658743072016`
- surrogate search Battery-PGR against author model: `0.000693307802812019`
- post-hoc feature-family overlap: `difference between cycle 10 and cycle N curves, early discharge capacity, late-cycle capacity slope`

## Locked Batch 9 Validation
- not run

## Caveats
- Batch 9 is not required for this rediscovery run.
- Battery-PGR against the author model is undefined unless exact author validation RMSE is available.
- Synthetic/demo data are used only when processed local data files are absent and --require-real-data is not set.
- Surrogate-label search performance on OED/CLO batches is distinct from locked Batch 9 final validation.
