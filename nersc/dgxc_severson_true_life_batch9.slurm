#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --job-name=oba-severson-b9
#SBATCH --output=runs/slurm/oba-severson-b9-%j.out

set -euo pipefail

module purge > /dev/null 2>&1 || true

FEATURE_RECIPE=${FEATURE_RECIPE:-broad_physics}
SPLIT_MODE=${SPLIT_MODE:-batch}
ITERATIONS=${ITERATIONS:-1}
CANDIDATES_PER_ITERATION=${CANDIDATES_PER_ITERATION:-8}
FINAL_BATCH9_TOP_K=${FINAL_BATCH9_TOP_K:-8}
FIRST_N_CYCLES=${FIRST_N_CYCLES:-100}
CYCLE_EARLY_INDEX=${CYCLE_EARLY_INDEX:-9}
CYCLE_LATE_INDEX=${CYCLE_LATE_INDEX:-99}

mkdir -p runs/slurm reports data/processed runs/open_battery_agents

SCRATCH_BASE="${SCRATCH:-${TMPDIR:-/tmp}}/oba_severson_${SLURM_JOB_ID}"
PYTHONUSERBASE="${SCRATCH_BASE}/python_user"
RAW_ROOT_LOCAL="${SCRATCH_BASE}/battery-fast-charging"
export PYTHONUSERBASE
export PATH="${PYTHONUSERBASE}/bin:${PATH}"

mkdir -p "${SCRATCH_BASE}" "${PYTHONUSERBASE}"

python -m pip install --user --upgrade pip
python -m pip install --user "eval-type-backport>=0.2,<1"
python -m pip install --user -e ".[dev,agents]"

if [[ -d literature_models_and_data/battery-fast-charging ]]; then
  rm -rf "${RAW_ROOT_LOCAL}"
  cp -a literature_models_and_data/battery-fast-charging "${RAW_ROOT_LOCAL}"
else
  echo "Missing literature_models_and_data/battery-fast-charging" >&2
  exit 1
fi

SEVERSON_MAT_DIR="${RAW_ROOT_LOCAL}/data/severson_2019_true_life_matr"
BATCH9_PATH="${RAW_ROOT_LOCAL}/data/2019-01-24_batch9"
PROCESSED_DIR="data/processed/severson_2019_true_life"
SEVERSON_FP_DIR="data/processed/severson_2019_true_life_feature_programs/${FEATURE_RECIPE}_idx${CYCLE_EARLY_INDEX}_${CYCLE_LATE_INDEX}"
BATCH9_FP_DIR="data/processed/chueh_toyota_fast_charge_feature_programs/batch9_${FEATURE_RECIPE}_idx${CYCLE_EARLY_INDEX}_${CYCLE_LATE_INDEX}"
RUN_DIR="runs/open_battery_agents/severson_true_life_batch9_${SLURM_JOB_ID}"

python scripts/audit_severson_matr_data.py \
  --mat-dir "${SEVERSON_MAT_DIR}" \
  --out reports/severson_matr_audit

python scripts/build_severson_true_life_dataset.py \
  --mat-dir "${SEVERSON_MAT_DIR}" \
  --out "${PROCESSED_DIR}" \
  --first-n-cycles "${FIRST_N_CYCLES}"

python scripts/build_battery_feature_program.py \
  --processed-dir "${PROCESSED_DIR}" \
  --recipe "${FEATURE_RECIPE}" \
  --batch severson \
  --out "${SEVERSON_FP_DIR}" \
  --first-n-cycles "${FIRST_N_CYCLES}" \
  --cycle-early-index "${CYCLE_EARLY_INDEX}" \
  --cycle-late-index "${CYCLE_LATE_INDEX}" \
  --include-protocol-features false

if [[ -d "${BATCH9_PATH}" ]]; then
  python scripts/build_battery_feature_program.py \
    --battery-fast-charging-root "${RAW_ROOT_LOCAL}" \
    --processed-dir data/processed/chueh_toyota_fast_charge_agent_surrogate \
    --recipe "${FEATURE_RECIPE}" \
    --batch batch9 \
    --out "${BATCH9_FP_DIR}" \
    --first-n-cycles "${FIRST_N_CYCLES}" \
    --cycle-early-index "${CYCLE_EARLY_INDEX}" \
    --cycle-late-index "${CYCLE_LATE_INDEX}" \
    --include-protocol-features false
else
  echo "Expanded Batch 9 path missing: ${BATCH9_PATH}" >&2
  exit 1
fi

python scripts/run_role_agent_workflow.py \
  --processed-dir "${PROCESSED_DIR}" \
  --out "${RUN_DIR}" \
  --reports-dir reports \
  --offline \
  --split-mode "${SPLIT_MODE}" \
  --iterations "${ITERATIONS}" \
  --candidates-per-iteration "${CANDIDATES_PER_ITERATION}" \
  --include-feature-programs \
  --feature-program-mode table \
  --feature-program-paths "${SEVERSON_FP_DIR}/feature_table.csv" \
  --battery-fast-charging-root "${RAW_ROOT_LOCAL}" \
  --batch9-path "${BATCH9_PATH}" \
  --batch9-feature-program-path "${BATCH9_FP_DIR}/feature_table.csv" \
  --final-batch9-validation \
  --locked-test \
  --final-batch9-top-k "${FINAL_BATCH9_TOP_K}"

tar -czf "runs/oba-severson-b9-${SLURM_JOB_ID}-artifacts.tgz" \
  reports/severson_matr_audit.md \
  reports/severson_matr_audit.json \
  reports/role_agent_workflow.md \
  reports/role_agent_workflow.json \
  "${RUN_DIR}" \
  "${PROCESSED_DIR}" \
  "${SEVERSON_FP_DIR}" \
  "${BATCH9_FP_DIR}"
