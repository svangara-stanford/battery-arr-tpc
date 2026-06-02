#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --job-name=oba-batch9
#SBATCH --output=runs/slurm/oba-batch9-%j.out

set -euo pipefail

module purge > /dev/null 2>&1 || true

SPLIT_MODE=${SPLIT_MODE:-random}
VALIDATION_FRACTION=${VALIDATION_FRACTION:-0.25}
SPLIT_SEED=${SPLIT_SEED:-0}
VALIDATION_BATCH_ID=${VALIDATION_BATCH_ID:-}

mkdir -p runs/slurm

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "eval-type-backport>=0.2,<1"
python -m pip install -e ".[dev,agents]"

pytest

python scripts/run_attia_reference_reproduction.py \
  --battery-fast-charging-root literature_models_and_data/battery-fast-charging \
  --out runs/attia_reference_reproduction_with_validation \
  --reports-dir reports \
  --require-exact-author-model \
  --include-validation-batch \
  --validation-batch-path literature_models_and_data/battery-fast-charging/data/2019-01-24_batch9 \
  --overwrite

cmd=(
  python scripts/run_agentic_rediscovery.py
  --processed-dir data/processed/chueh_toyota_fast_charge_agent_surrogate
  --reference-run runs/attia_reference_reproduction_with_validation
  --out "runs/open_battery_agents/dgxc_batch9_${SLURM_JOB_ID}"
  --reports-dir reports
  --agents 2
  --iterations 3
  --offline
  --require-real-data
  --final-batch9-validation
  --battery-fast-charging-root literature_models_and_data/battery-fast-charging
  --batch9-path literature_models_and_data/battery-fast-charging/data/2019-01-24_batch9
  --split-mode "${SPLIT_MODE}"
  --validation-fraction "${VALIDATION_FRACTION}"
  --split-seed "${SPLIT_SEED}"
)

if [[ -n "${VALIDATION_BATCH_ID}" ]]; then
  cmd+=(--validation-batch-id "${VALIDATION_BATCH_ID}")
fi

"${cmd[@]}"
