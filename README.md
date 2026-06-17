# Open Battery Agents (`battery-arr-tpc`)

Prototype for replaying the Attia/Chueh author-provided fast-charging model and
running offline or LLM-driven early-cycle lifetime-predictor rediscovery loops
over the Severson 2019 cycling dataset.

## Getting started

See **[SETUP.md](SETUP.md)** for the full setup guide: creating the Python
environment, obtaining and placing the large datasets (which are *not* in git),
configuring API keys, and verifying the install.

Quick version:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,agents,tools]"
# place datasets under literature_models_and_data/ — see SETUP.md §3
make hackathon-demo
```

Generated outputs are written under `runs/` and `reports/`.

## Layout

- `src/battery_aar/` — library (features, agents, workflows, paper reproduction)
- `scripts/` — CLI entry points (dataset builders, campaign runners, scorers)
- `nersc/` — Slurm batch scripts (Stanford Sherlock + DGX Cloud)
- `docs/` — design notes, the Sherlock Track A recipe, literature survey
- `data/processed/` — committed small derived tables
- `literature_models_and_data/` — raw datasets you provide (git-ignored)
