#!/bin/bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=3:00:00
#SBATCH --job-name=oba-champion
#SBATCH --output=runs/slurm/oba-champion-%j.out

# Run the agentic champion-selection pipeline end-to-end on Sherlock.
# Pre-req: 18-job Track A campaign already completed and summarized.
#
# Environment variables (all optional):
#   REPO_ROOT, BFC_ROOT, RUN_ROOT, VENV_DIR, PYTHON_MODULE
#   K                       Top-K shortlist size (default 3).
#   OFFLINE                 true to use deterministic offline fallback for both roles.
#   SUMMARY_ROOT            Default $SCRATCH/battery-arr/reports/trackA_summary.
#   RUNS_ROOT               Default $SCRATCH/battery-arr/runs.
#   REPORTS_ROOT            Default $SCRATCH/battery-arr/reports.
#   SECONDARY_TEST_SOURCE   Patch-E4: severson_b3 (default) | attia_batch9.
#                           Controls which scorer the champion pipeline invokes.
#   SCORER_SCRIPT           Optional explicit override for the scorer script.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/groups/darve/svangara/battery-arr-tpc}"
cd "${REPO_ROOT}"
mkdir -p runs/slurm

echo "=========================================="
echo "Open Battery Agents - Champion selection"
echo "=========================================="
echo "Job ID      : ${SLURM_JOB_ID:-NA}"
echo "Host        : $(hostname)"
echo "Start time  : $(date)"
echo "REPO_ROOT   : ${REPO_ROOT}"

PYTHON_MODULE="${PYTHON_MODULE:-python/3.12.1}"
VENV_DIR="${VENV_DIR:-/home/groups/darve/svangara/battery-arr-venvs/oba}"
module purge >/dev/null 2>&1 || true
module load "${PYTHON_MODULE}"
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
echo "Python      : $(command -v python3)"
python3 --version

if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export OPEN_BATTERY_AGENTS_API_KEY="${OPEN_BATTERY_AGENTS_API_KEY:-${STANFORD_AI_API_KEY:-}}"
export OPEN_BATTERY_AGENTS_BASE_URL="${OPEN_BATTERY_AGENTS_BASE_URL:-${STANFORD_AI_BASE_URL:-}}"
export OPEN_BATTERY_AGENTS_MODEL="${OPEN_BATTERY_AGENTS_MODEL:-gpt-4.1}"

OFFLINE="${OFFLINE:-false}"
if [ "${OFFLINE}" != "true" ] && [ -z "${OPEN_BATTERY_AGENTS_API_KEY:-}" ]; then
  echo "ERROR: OFFLINE=false but no OPEN_BATTERY_AGENTS_API_KEY configured." >&2
  exit 1
fi

echo "LLM check:"
echo "  offline          : ${OFFLINE}"
echo "  API key set      : $([ -n "${OPEN_BATTERY_AGENTS_API_KEY:-}" ] && echo yes || echo no)"
echo "  Model            : ${OPEN_BATTERY_AGENTS_MODEL}"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

RUN_ROOT="${RUN_ROOT:-${SCRATCH}/battery-arr}"
SUMMARY_ROOT="${SUMMARY_ROOT:-${RUN_ROOT}/reports/trackA_summary}"
RUNS_ROOT="${RUNS_ROOT:-${RUN_ROOT}/runs}"
REPORTS_ROOT="${REPORTS_ROOT:-${RUN_ROOT}/reports}"
BFC_ROOT="${BFC_ROOT:-${REPO_ROOT}/literature_models_and_data/battery-fast-charging}"
K="${K:-3}"
SECONDARY_TEST_SOURCE="${SECONDARY_TEST_SOURCE:-severson_b3}"
SCORER_SCRIPT="${SCORER_SCRIPT:-}"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${RUN_ROOT}/champion_selection/${TS}_${SLURM_JOB_ID}"
mkdir -p "${OUT_DIR}"

echo
echo "Output dir  : ${OUT_DIR}"
echo "Summary root: ${SUMMARY_ROOT}"
echo "Runs root   : ${RUNS_ROOT}"
echo "BFC root    : ${BFC_ROOT}"
echo "K           : ${K}"
echo "Sec test src: ${SECONDARY_TEST_SOURCE}"
echo

RUN_ARGS=(
  python3 scripts/run_agent_champion_selection.py
  --out "${OUT_DIR}"
  --k "${K}"
  --summary-root "${SUMMARY_ROOT}"
  --runs-root "${RUNS_ROOT}"
  --reports-root "${REPORTS_ROOT}"
  --bfc-root "${BFC_ROOT}"
  --secondary-test-source "${SECONDARY_TEST_SOURCE}"
)
if [ -n "${SCORER_SCRIPT}" ]; then
  RUN_ARGS+=(--scorer-script "${SCORER_SCRIPT}")
fi
if [ "${OFFLINE}" = "true" ]; then
  RUN_ARGS+=(--offline)
else
  RUN_ARGS+=(--model "${OPEN_BATTERY_AGENTS_MODEL}")
fi

printf '%q ' "${RUN_ARGS[@]}"; echo
echo

"${RUN_ARGS[@]}"

echo
echo "Champion artifacts:"
ls -lh "${OUT_DIR}" 2>/dev/null
echo
if [ -f "${OUT_DIR}/champion_decision.md" ]; then
  echo "===== champion_decision.md ====="
  cat "${OUT_DIR}/champion_decision.md"
fi
echo
echo "Done at: $(date)"
