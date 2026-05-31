#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --job-name=oba-agent-demo
#SBATCH --output=runs/slurm/oba-agent-demo-%j.out

set -euo pipefail

module purge > /dev/null 2>&1 || true

mkdir -p runs/slurm

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,agents]"

pytest

python scripts/run_attia_reference_reproduction.py \
  --battery-fast-charging-root literature_models_and_data/battery-fast-charging \
  --out runs/attia_reference_reproduction \
  --reports-dir reports \
  --require-exact-author-model \
  --skip-validation-batch

python scripts/run_agentic_rediscovery.py \
  --processed-dir data/processed/chueh_toyota_fast_charge \
  --reference-run runs/attia_reference_reproduction \
  --out runs/open_battery_agents/dgxc_${SLURM_JOB_ID} \
  --reports-dir reports \
  --agents 2 \
  --iterations 3 \
  --offline
