#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --job-name=oba-trackA
#SBATCH --output=runs/slurm/oba-trackA-%j.out

set -euo pipefail

mkdir -p runs/slurm reports runs/open_battery_agents runs

echo "=========================================="
echo "Open Battery Agents Track A Broad Discovery"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo

module purge > /dev/null 2>&1 || true
module load python3 2>/dev/null || true

PYTHON_BIN="/cm/local/apps/python3/bin/python3"
if [ ! -x "${PYTHON_BIN}" ]; then
  PYTHON_BIN="$(command -v python3 || command -v python)"
fi

echo "Python executable: ${PYTHON_BIN}"
"${PYTHON_BIN}" --version
"${PYTHON_BIN}" -m pip --version || true
echo

# Load local environment, but never print secrets.
if [ -f ".env" ]; then
  set -a
  source .env
  set +a
else
  echo "WARNING: .env not found. Continuing if required environment variables are already exported."
fi

export OPEN_BATTERY_AGENTS_API_KEY="${OPEN_BATTERY_AGENTS_API_KEY:-${STANFORD_AI_API_KEY:-${STANFORD_AI_PLAYGROUND_API_KEY:-}}}"
export OPEN_BATTERY_AGENTS_BASE_URL="${OPEN_BATTERY_AGENTS_BASE_URL:-${STANFORD_AI_BASE_URL:-${STANFORD_AI_PLAYGROUND_BASE_URL:-}}}"
export OPEN_BATTERY_AGENTS_MODEL="${OPEN_BATTERY_AGENTS_MODEL:-${MAGQUERY_KG_MODEL:-gpt-4.1}}"
export OPEN_BATTERY_AGENTS_ALIAS="${OPEN_BATTERY_AGENTS_ALIAS:-${STANFORD_AI_ALIAS:-spectraquery}}"

OFFLINE="${OFFLINE:-false}"
if [ "${OFFLINE}" != "true" ] && [ -z "${OPEN_BATTERY_AGENTS_API_KEY:-}" ]; then
  echo "ERROR: OFFLINE=false but no Open Battery Agents / Stanford AI API key is configured."
  echo "Either export OPEN_BATTERY_AGENTS_API_KEY or run with OFFLINE=true."
  exit 1
fi

echo "LLM config check:"
echo "  offline: ${OFFLINE}"
echo "  API key set: $([ -n "${OPEN_BATTERY_AGENTS_API_KEY:-}" ] && echo yes || echo no)"
echo "  Base URL: ${OPEN_BATTERY_AGENTS_BASE_URL:-MISSING}"
echo "  Model: ${OPEN_BATTERY_AGENTS_MODEL:-MISSING}"
echo "  Alias configured: $([ -n "${OPEN_BATTERY_AGENTS_ALIAS:-}" ] && echo yes || echo no)"
echo

FEATURE_RECIPE="${FEATURE_RECIPE:-broad_physics}"
SPLIT_MODE="${SPLIT_MODE:-random}"
SPLIT_SEED="${SPLIT_SEED:-0}"
VALIDATION_FRACTION="${VALIDATION_FRACTION:-0.25}"
VALIDATION_BATCH_ID="${VALIDATION_BATCH_ID:-}"
ITERATIONS="${ITERATIONS:-8}"
CANDIDATES_PER_ITERATION="${CANDIDATES_PER_ITERATION:-20}"
FIRST_N_CYCLES="${FIRST_N_CYCLES:-100}"
CYCLE_EARLY_INDEX="${CYCLE_EARLY_INDEX:-9}"
CYCLE_LATE_INDEX="${CYCLE_LATE_INDEX:-99}"
INCLUDE_PROTOCOL_FEATURES="${INCLUDE_PROTOCOL_FEATURES:-false}"
ALLOW_FREEFORM_CODE="${ALLOW_FREEFORM_CODE:-false}"
PATCH_GENERIC_PROMPTS="${PATCH_GENERIC_PROMPTS:-false}"
DO_FINAL_BATCH9="${DO_FINAL_BATCH9:-false}"
FINAL_BATCH9_TOP_K="${FINAL_BATCH9_TOP_K:-0}"

RUN_ID="trackA_${FEATURE_RECIPE}_${SPLIT_MODE}_seed${SPLIT_SEED}_${SLURM_JOB_ID}"
OUT_DIR="runs/open_battery_agents/${RUN_ID}"
REPORTS_DIR="reports/${RUN_ID}"
ARCHIVE="runs/oba-trackA-${SLURM_JOB_ID}-artifacts.tgz"

SRC_ROOT="$(pwd)/literature_models_and_data/battery-fast-charging"
if [ ! -d "${SRC_ROOT}" ]; then
  echo "ERROR: Missing battery-fast-charging root: ${SRC_ROOT}"
  exit 1
fi

SCRATCH_ROOT="/tmp/${USER}/oba_trackA_${SLURM_JOB_ID}"
mkdir -p "${SCRATCH_ROOT}"

export TMPDIR="${SCRATCH_ROOT}/tmp"
export PIP_CACHE_DIR="${SCRATCH_ROOT}/pip_cache"
export PYTHONUSERBASE="${SCRATCH_ROOT}/pyuser"
export PATH="${PYTHONUSERBASE}/bin:${PATH}"
export PYTHONPATH=""

mkdir -p "${TMPDIR}" "${PIP_CACHE_DIR}" "${PYTHONUSERBASE}" "${OUT_DIR}" "${REPORTS_DIR}"

archive_on_exit() {
  status=$?
  echo
  echo "=========================================="
  echo "Archiving partial/final outputs, exit status: ${status}"
  echo "=========================================="

  mkdir -p "${OUT_DIR}/feature_programs" runs

  if [ -d "${SCRATCH_ROOT}/feature_programs" ]; then
    cp -a "${SCRATCH_ROOT}/feature_programs" "${OUT_DIR}/" || true
  fi
  if [ -d "${SCRATCH_ROOT}/processed" ]; then
    mkdir -p "${OUT_DIR}/processed"
    cp -a "${SCRATCH_ROOT}/processed/." "${OUT_DIR}/processed/" || true
  fi

  tar -czf "${ARCHIVE}" \
    "${REPORTS_DIR}" \
    "${OUT_DIR}" \
    2>/dev/null || true

  echo "Archive path:"
  ls -lh "${ARCHIVE}" 2>/dev/null || true
  echo "Done/archive time: $(date)"

  exit "${status}"
}
trap archive_on_exit EXIT

echo "Copying battery-fast-charging folder to compute-node scratch..."
cp -a "${SRC_ROOT}" "${SCRATCH_ROOT}/battery-fast-charging"
SCRATCH_BFC="${SCRATCH_ROOT}/battery-fast-charging"
SEVERSON_MAT_DIR="${SCRATCH_BFC}/data/severson_2019_true_life_matr"
BATCH9_PATH="${SCRATCH_BFC}/data/2019-01-24_batch9"

du -sh "${SCRATCH_BFC}" || true
echo

if [ ! -d "${SEVERSON_MAT_DIR}" ]; then
  echo "ERROR: Missing Severson MatR directory on scratch: ${SEVERSON_MAT_DIR}"
  echo "Expected the three Severson .mat files under literature_models_and_data/battery-fast-charging/data/severson_2019_true_life_matr/"
  exit 1
fi

if [ "${DO_FINAL_BATCH9}" = "true" ] || [ "${FINAL_BATCH9_TOP_K}" -gt 0 ]; then
  if [ ! -d "${BATCH9_PATH}" ]; then
    echo "ERROR: Batch 9 requested but expanded Batch 9 folder is missing: ${BATCH9_PATH}"
    exit 1
  fi
fi

echo "Installing Python dependencies into scratch-local user base..."
"${PYTHON_BIN}" -m pip install --user --upgrade pip setuptools wheel

"${PYTHON_BIN}" -m pip install --user --only-binary=:all: \
  "numpy==1.26.4" \
  "pandas==2.2.1" \
  "scipy==1.11.4" \
  "scikit-learn==1.4.2" \
  "h5py==3.10.0" \
  "PyYAML==6.0.2" \
  "joblib==1.4.2"

"${PYTHON_BIN}" -m pip install --user \
  "openai>=1.40,<2" \
  "pydantic>=2,<3" \
  "eval-type-backport>=0.2,<1" \
  "tenacity>=8,<9" \
  "python-dotenv>=1,<2" \
  "fastapi>=0.110" \
  "uvicorn>=0.27" \
  "httpx>=0.27"

"${PYTHON_BIN}" -m pip install --user --no-deps --ignore-requires-python -e .

echo
echo "Python import and LLM config check..."
"${PYTHON_BIN}" - <<'PY'
import os, sys
print("python:", sys.executable)
print("version:", sys.version)
import numpy, pandas, scipy, sklearn, h5py
import openai, pydantic, tenacity, dotenv
import battery_aar
from battery_aar.agents.llm_client import llm_startup_summary
print("imports OK")
print("llm:", llm_startup_summary())
PY

if [ "${PATCH_GENERIC_PROMPTS}" = "true" ]; then
  echo
  echo "Patching role prompts to generic Track A wording if needed..."
  "${PYTHON_BIN}" - <<'PY'
from pathlib import Path
p = Path("src/battery_aar/workflows/role_prompts.py")
s = p.read_text()
orig = s
s = s.replace(
'''FeatureProgram objects are compiled by trusted repo code and can expose scalar-only, curve-only,
scalar-plus-curve, broad-physics, and Attia/Severson-like feature sets.

Strong plans should consider author-inspired but coefficient-free feature families:
- discharge capacity at cycles 2, 10, and 100 when available
- maximum early capacity minus cycle-2 capacity
- cycle-N minus cycle-10 capacity differences
- early and late capacity slopes
- log-transformed difference-statistic proxies
- protocol-current features only when allowed

The literature early predictor modeled log10(cycle life). Feature plans can support either raw cycle-life
models or log10 target modeling; the ModelArchitect chooses the target transform.
''',
'''FeatureProgram objects are compiled by trusted repo code and can expose scalar-only, curve-only,
scalar-plus-curve, and broad-physics feature sets.

Strong plans should reason from general battery degradation principles and choose among:
- early capacity level and retention
- capacity-fade slopes, curvature, variance, and early-window statistics
- resistance, thermal, energy, and efficiency proxies when available
- charge/discharge curve-shape statistics and cross-cycle curve deltas
- conservative protocol-current features only when explicitly allowed
- raw or transformed target modeling when statistically and physically justified

Do not assume a particular paper feature set, coefficient vector, target transform, or model family. The goal is to discover transferable early-life predictors from the available early-cycle data and validation feedback.
'''
)
s = s.replace(
'''Use robust preprocessing. Prefer Ridge, ElasticNet, ElasticNetCV, RandomForest, or GradientBoosting over neural networks.
Candidates must drop all-NaN features or impute safely and must not use row_id/cell_id as predictors.
The literature early predictor modeled log10(cycle life). For positive lifetime targets, log10 target modeling
may improve stability under batch shifts. Choose target_transform as either "raw" or "log10" and justify the choice.
''',
'''Use robust preprocessing. Prefer small-data regressors such as Ridge, ElasticNetCV, LassoCV, RandomForest, or GradientBoosting before neural networks.
Candidates must drop all-NaN features or impute safely and must not use row_id/cell_id as predictors.
Choose target_transform as either "raw" or "log10" and justify it from the target distribution, error structure, and transfer-stability considerations.
Do not assume a paper-specific target transform or model family.
'''
)
if s != orig:
    p.write_text(s)
    print("Patched", p)
else:
    print("No prompt patch applied; text may already be patched or source wording changed.")
PY

  echo "Prompt paper-specific term check:"
  grep -R "Attia\|Severson-like\|author-inspired\|literature early predictor" -n src/battery_aar/workflows/role_prompts.py || true
fi

PROCESSED_DIR="${SCRATCH_ROOT}/processed/severson_2019_true_life"
SEVERSON_FP_DIR="${SCRATCH_ROOT}/feature_programs/severson_${FEATURE_RECIPE}_idx${CYCLE_EARLY_INDEX}_${CYCLE_LATE_INDEX}"
BATCH9_FP_DIR="${SCRATCH_ROOT}/feature_programs/batch9_${FEATURE_RECIPE}_idx${CYCLE_EARLY_INDEX}_${CYCLE_LATE_INDEX}"

echo
echo "=========================================="
echo "Building Severson true-life processed dataset"
echo "=========================================="
"${PYTHON_BIN}" scripts/audit_severson_matr_data.py \
  --mat-dir "${SEVERSON_MAT_DIR}" \
  --out "${REPORTS_DIR}/severson_matr_audit"

"${PYTHON_BIN}" scripts/build_severson_true_life_dataset.py \
  --mat-dir "${SEVERSON_MAT_DIR}" \
  --out "${PROCESSED_DIR}" \
  --first-n-cycles "${FIRST_N_CYCLES}"

echo
echo "=========================================="
echo "Building Track A feature-program table"
echo "=========================================="
echo "recipe: ${FEATURE_RECIPE}"
echo "cycle early index: ${CYCLE_EARLY_INDEX}"
echo "cycle late index: ${CYCLE_LATE_INDEX}"
echo "include protocol features: ${INCLUDE_PROTOCOL_FEATURES}"
echo

"${PYTHON_BIN}" scripts/build_battery_feature_program.py \
  --processed-dir "${PROCESSED_DIR}" \
  --recipe "${FEATURE_RECIPE}" \
  --batch severson \
  --out "${SEVERSON_FP_DIR}" \
  --first-n-cycles "${FIRST_N_CYCLES}" \
  --cycle-early-index "${CYCLE_EARLY_INDEX}" \
  --cycle-late-index "${CYCLE_LATE_INDEX}" \
  --include-protocol-features "${INCLUDE_PROTOCOL_FEATURES}"

echo
echo "=========================================="
echo "Feature-table sanity checks"
echo "=========================================="
"${PYTHON_BIN}" - <<PY
from pathlib import Path
import pandas as pd
import numpy as np
out = Path("${SEVERSON_FP_DIR}")
ft = pd.read_csv(out / "feature_table.csv")
meta = pd.read_csv(out / "feature_metadata.csv")
print("feature_table:", ft.shape)
print("feature_metadata:", meta.shape)
if "feature_family" in meta.columns:
    print("families:", meta["feature_family"].value_counts().to_dict())
assert "row_id" in ft.columns
assert ft["row_id"].is_unique
forbidden = ["cycle_life", "author_model_prediction", "target", "label", "source_path", "raw_path", "filename"]
for col in forbidden:
    assert col not in ft.columns, f"leakage column in feature table: {col}"
X = ft.drop(columns=["row_id", "cell_id"], errors="ignore").select_dtypes(include="number")
print("numeric feature cols:", X.shape[1])
assert X.shape[1] > 0, "no numeric feature columns"
assert np.isinf(X.to_numpy()).sum() == 0, "feature table has inf values"
all_nan = X.columns[X.isna().all()].tolist()
print("all-NaN feature columns:", len(all_nan))
if all_nan[:10]:
    print("examples:", all_nan[:10])
print("FEATURE TABLE CHECK PASS")
PY

RUN_ARGS=(
  "${PYTHON_BIN}" scripts/run_role_agent_workflow.py
  --processed-dir "${PROCESSED_DIR}"
  --out "${OUT_DIR}"
  --reports-dir "${REPORTS_DIR}"
  --split-mode "${SPLIT_MODE}"
  --validation-fraction "${VALIDATION_FRACTION}"
  --split-seed "${SPLIT_SEED}"
  --iterations "${ITERATIONS}"
  --candidates-per-iteration "${CANDIDATES_PER_ITERATION}"
  --include-feature-programs
  --feature-program-mode table
  --feature-program-recipe "${FEATURE_RECIPE}"
  --feature-program-paths "${SEVERSON_FP_DIR}/feature_table.csv"
  --battery-fast-charging-root "${SCRATCH_BFC}"
  --max-cycle "${FIRST_N_CYCLES}"
  --cycle-early-index "${CYCLE_EARLY_INDEX}"
  --cycle-late-index "${CYCLE_LATE_INDEX}"
)

if [ "${OFFLINE}" = "true" ]; then
  RUN_ARGS+=(--offline)
else
  RUN_ARGS+=(--model "${OPEN_BATTERY_AGENTS_MODEL}")
fi

if [ -n "${VALIDATION_BATCH_ID}" ]; then
  RUN_ARGS+=(--validation-batch-id "${VALIDATION_BATCH_ID}")
fi

if [ "${INCLUDE_PROTOCOL_FEATURES}" = "true" ]; then
  RUN_ARGS+=(--allow-protocol-features)
fi

if [ "${ALLOW_FREEFORM_CODE}" = "true" ]; then
  RUN_ARGS+=(--allow-freeform-code)
fi

if [ "${DO_FINAL_BATCH9}" = "true" ] || [ "${FINAL_BATCH9_TOP_K}" -gt 0 ]; then
  echo
  echo "=========================================="
  echo "Building optional Batch 9 feature table"
  echo "=========================================="
  "${PYTHON_BIN}" scripts/build_battery_feature_program.py \
    --battery-fast-charging-root "${SCRATCH_BFC}" \
    --processed-dir data/processed/chueh_toyota_fast_charge_agent_surrogate \
    --recipe "${FEATURE_RECIPE}" \
    --batch batch9 \
    --out "${BATCH9_FP_DIR}" \
    --first-n-cycles "${FIRST_N_CYCLES}" \
    --cycle-early-index "${CYCLE_EARLY_INDEX}" \
    --cycle-late-index "${CYCLE_LATE_INDEX}" \
    --include-protocol-features "${INCLUDE_PROTOCOL_FEATURES}"

  RUN_ARGS+=(--batch9-path "${BATCH9_PATH}")
  RUN_ARGS+=(--batch9-feature-program-path "${BATCH9_FP_DIR}/feature_table.csv")
  if [ "${DO_FINAL_BATCH9}" = "true" ]; then
    RUN_ARGS+=(--final-batch9-validation --locked-test)
  fi
  if [ "${FINAL_BATCH9_TOP_K}" -gt 0 ]; then
    RUN_ARGS+=(--final-batch9-top-k "${FINAL_BATCH9_TOP_K}")
  fi
fi

echo
echo "=========================================="
echo "Running Track A role-agent workflow"
echo "=========================================="
echo "run id: ${RUN_ID}"
echo "feature recipe: ${FEATURE_RECIPE}"
echo "split mode: ${SPLIT_MODE}"
echo "split seed: ${SPLIT_SEED}"
echo "iterations: ${ITERATIONS}"
echo "candidates per iteration: ${CANDIDATES_PER_ITERATION}"
echo "Batch 9 final validation: ${DO_FINAL_BATCH9}; top-k: ${FINAL_BATCH9_TOP_K}"
echo
printf '%q ' "${RUN_ARGS[@]}"
echo

"${RUN_ARGS[@]}"

echo
echo "Copying feature-program and processed data into run directory..."
mkdir -p "${OUT_DIR}/feature_programs" "${OUT_DIR}/processed"
cp -a "${SEVERSON_FP_DIR}" "${OUT_DIR}/feature_programs/" || true
cp -a "${PROCESSED_DIR}" "${OUT_DIR}/processed/" || true
if [ -d "${BATCH9_FP_DIR}" ]; then
  cp -a "${BATCH9_FP_DIR}" "${OUT_DIR}/feature_programs/" || true
fi

echo
echo "Key result grep:"
grep -RniE "rmse|mae|spearman|kendall|best|error|traceback|failed|locked|batch9" "${OUT_DIR}" "${REPORTS_DIR}" | head -300 || true

echo
echo "Report preview:"
cat "${REPORTS_DIR}/role_agent_workflow.md" | head -340 || true

echo
echo "Creating result archive..."
tar -czf "${ARCHIVE}" \
  "${REPORTS_DIR}" \
  "${OUT_DIR}"

echo
echo "Result archive:"
ls -lh "${ARCHIVE}"

echo
echo "Done at: $(date)"
