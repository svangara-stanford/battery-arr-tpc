# DGX Cloud Hackathon Notes

- Do not run compute-heavy processes on the login node.
- Use `srun`, `salloc`, or `sbatch` for the hackathon runs.
- Home directories are not permanent; back up important results.
- Use environment variables for API keys:
  - `OPEN_BATTERY_AGENTS_API_KEY`
  - `OPEN_BATTERY_AGENTS_BASE_URL`
  - `OPEN_BATTERY_AGENTS_MODEL`
  - `STANFORD_AI_PLAYGROUND_API_KEY`
  - `STANFORD_AI_PLAYGROUND_BASE_URL`
- Do not commit `.env`, API keys, raw data, processed data, checkpoints, or run outputs.
- Do not unzip or expand `2019-01-24_batch9.zip` for the initial hackathon run.
