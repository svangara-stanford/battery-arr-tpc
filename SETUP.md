# Setup Guide — Open Battery Agents / `battery-arr-tpc`

This guide gets the repo running from a fresh clone, either on an HPC cluster
(Stanford Sherlock is the reference target) or on a local workstation. The
large raw datasets are **not** committed to git — this guide is the source of
truth for obtaining and placing them.

> TL;DR: (1) create a Python 3.9–3.12 environment and `pip install -e .`,
> (2) drop the datasets into `literature_models_and_data/` with the exact
> layout shown below, (3) copy `.env.example` to `.env` if you want the LLM
> paths. Then `make hackathon-demo` should pass.

---

## 1. Clone and create the Python environment

Supported Python: **3.9 – 3.12** (CI/dev runs on 3.12).

```bash
git clone <your-remote-url> battery-arr-tpc
cd battery-arr-tpc

python3 -m venv .venv          # any 3.9–3.12 interpreter
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,agents,tools]"
```

The optional extras:

| Extra     | Pulls in                              | Needed for                              |
|-----------|---------------------------------------|-----------------------------------------|
| (base)    | numpy, pandas, scipy, scikit-learn, h5py | All offline feature/dataset processing |
| `dev`     | pytest, fastapi, httpx, uvicorn       | Running the test suite, the tool server |
| `agents`  | openai, pydantic, tenacity, python-dotenv | The LLM-driven discovery loops       |
| `tools`   | fastapi, httpx, uvicorn               | The FastAPI tool server                 |

### On Stanford Sherlock

Do **not** build environments or run Python on the login node. Use the
bootstrap script, which loads `python/3.12.1` and creates a persistent venv on
group storage (not `$HOME`):

```bash
bash scripts/setup_sherlock_env.sh
# then, in any fresh shell:
module purge && module load python/3.12.1
source /home/groups/darve/svangara/battery-arr-venvs/oba/bin/activate
```

See `docs/sherlock_trackA.md` for the full Sherlock/Slurm campaign recipe.

---

## 2. Secrets / API keys (optional — only for LLM paths)

```bash
cp .env.example .env
# edit .env and fill in ONE provider's key + base URL + model
```

`.env` is git-ignored. Everything that does not call an LLM (the Attia
reference reproduction, the offline rediscovery loop, all dataset builders and
tests) runs with no key configured.

---

## 3. Datasets (the part that is NOT in git)

All large raw inputs live under `literature_models_and_data/`, which is
git-ignored (~19 GB total). You must create this tree yourself. The code
resolves it relative to the repo root by default
(`literature_models_and_data/battery-fast-charging`), and every entry point
also accepts a `--battery-fast-charging-root` flag / `BFC_ROOT` env var if you
keep the data elsewhere (e.g. on `$SCRATCH` or `$OAK`).

### Required directory layout

```
battery-arr-tpc/
└── literature_models_and_data/
    └── battery-fast-charging/
        └── data/
            ├── severson_2019_true_life_matr/
            │   ├── 2017-05-12_batchdata_updated_struct_errorcorrect.mat   # batch "b1", ~3.0 GB
            │   ├── 2017-06-30_batchdata_updated_struct_errorcorrect.mat   # batch "b2", ~2.0 GB
            │   └── 2018-04-12_batchdata_updated_struct_errorcorrect.mat   # batch "b3", ~3.2 GB
            └── 2019-01-24_batch9/
                └── *_structure.json                                       # 46 files (Attia closed-loop validation batch 9)
```

The exact filenames matter — `src/battery_aar/paper_reproduction/paths.py` and
`src/battery_aar/features/severson_matr.py` key off the `YYYY-MM-DD` prefixes
(`2017-05-12` = b1, `2017-06-30` = b2, `2018-04-12` = b3) and the
`_batchdata_updated_struct_errorcorrect.mat` suffix.

### Dataset A — Severson 2019 "true-life" MATR `.mat` files (required)

Three processed `.mat` batch files from the Stanford/MIT/Toyota cycling study:

> K. A. Severson et al., "Data-driven prediction of battery cycle life before
> capacity degradation," *Nature Energy* 4, 383–391 (2019).

These are the canonical training/test batches (b1, b2, b3). They provide the
*true measured* cycle-to-failure labels used for the canonical
train / primary_test / secondary_test (b3) split.

**Where to get them:**
- Primary source: the public dataset on `data.matr.io` accompanying the
  Severson 2019 paper (project "Data-driven prediction of battery cycle life").
  Download the three `*_batchdata_updated_struct_errorcorrect.mat` files dated
  2017-05-12, 2017-06-30, and 2018-04-12.
- These are the same files BatteryML and the Braatz-lab repo
  (`rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation`)
  consume; if you already have them from those projects, just symlink/copy them
  into the directory above — no reprocessing needed.

Place all three under `.../data/severson_2019_true_life_matr/`.

### Dataset B — Attia closed-loop "batch 9" (`2019-01-24_batch9`)

Per-cell `*_structure.json` files (46 of them) for the closed-loop fast-charging
validation batch:

> P. M. Attia et al., "Closed-loop optimization of fast-charging protocols for
> batteries with machine learning," *Nature* 578, 397–402 (2020).

This batch is the optional locked "secondary test" / transfer batch. It stays
sealed by default (the campaign locks it until a champion is selected) — most
workflows do **not** need it, but `score_candidates_on_batch9.py` and the
`--include-validation-batch` paths do.

**Where to get it:** the `battery-fast-charging` materials / `data.matr.io`
release associated with the Attia 2020 paper. Place the structure JSONs under
`.../data/2019-01-24_batch9/`.

> Note: the code is deliberately careful never to recursively scan or unzip a
> `2019-01-24_batch9.zip` while resolving paths. If you keep the batch as a zip,
> unpack it into the folder above; a bare `.zip` is treated as "validation
> skipped".

### Pre-built processed artifacts (already in git)

The small derived tables under `data/processed/` **are** committed, so you do
not need to rebuild them to inspect the pipeline:

- `data/processed/chueh_toyota_fast_charge_agent_surrogate/` — surrogate
  labels/splits/metadata CSVs + dataset card.
- `data/processed/chueh_toyota_fast_charge_feature_programs/` — four example
  feature-program outputs (minimal_debug, scalar_baseline, broad_physics,
  attia_severson_like).

To regenerate the Severson processed dataset from the raw `.mat` files once
Dataset A is in place:

```bash
python scripts/build_severson_true_life_dataset.py \
  --mat-dir literature_models_and_data/battery-fast-charging/data/severson_2019_true_life_matr \
  --out data/processed/severson_2019_true_life \
  --first-n-cycles 100
```

---

## 4. Verify the setup

```bash
# Import + lightweight checks (no datasets, no API key required):
make demo
pytest -q

# Author-model reproduction smoke (no API key; needs Dataset A absent is fine for --smoke):
make attia-reference-smoke

# Full offline rediscovery loop (no API key):
make agent-rediscovery-offline

# Everything wired together:
make hackathon-demo
```

Generated outputs are written under `runs/` and `reports/` (both git-ignored
except for a few committed reference reports).

---

## 5. Where things live

| Path                          | What                                                        |
|-------------------------------|-------------------------------------------------------------|
| `src/battery_aar/`            | Library: features, agents, workflows, paper reproduction    |
| `scripts/`                    | CLI entry points (dataset builders, campaign runners, scorers) |
| `nersc/`                      | Slurm batch scripts (Sherlock + DGX Cloud)                  |
| `data/processed/`             | Committed small derived tables (see §3)                     |
| `literature_models_and_data/` | **You provide this** — raw datasets (git-ignored, §3)       |
| `docs/`                       | Design notes, Sherlock recipe, literature survey            |
| `runs/`, `reports/`, `logs/`  | Generated outputs (git-ignored)                             |
