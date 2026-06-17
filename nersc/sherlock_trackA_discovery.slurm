#!/bin/bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --job-name=oba-trackA
#SBATCH --output=/scratch/users/svangara/battery-arr/runs/slurm/oba-trackA-%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue

# Open Battery Agents / Track A sealed broad-discovery campaign — Sherlock.
#
# Environment variables (all optional):
#   REPO_ROOT                  Path to battery-arr-tpc checkout.
#   BFC_ROOT                   battery-fast-charging root.
#   RUN_ROOT                   Root for run outputs / archives ($SCRATCH/battery-arr).
#   VENV_DIR                   Persistent venv from setup_sherlock_env.sh.
#   PYTHON_MODULE              Sherlock module to load (python/3.12.1).
#   FEATURE_RECIPE             scalar_baseline|curve_delta|broad_physics|minimal_debug.
#   SPLIT_MODE                 random|batch (also accepts protocol|leave_one_batch_out).
#   SPLIT_SEED                 Integer.
#   VALIDATION_FRACTION        Float (0.25 default).
#   VALIDATION_BATCH_ID        Optional explicit validation batch id.
#   ITERATIONS                 Outer iterations.
#   CANDIDATES_PER_ITERATION   Trusted-compiled variants per iteration.
#   FIRST_N_CYCLES             100 default.
#   CYCLE_EARLY_INDEX          9 default.
#   CYCLE_LATE_INDEX           99 default.
#   INCLUDE_PROTOCOL_FEATURES  true|false.
#   ALLOW_FREEFORM_CODE        true|false.
#   PATCH_GENERIC_PROMPTS      true to re-run scripts/patch_trackA_generic_prompts.py.
#   SECONDARY_TEST_SOURCE      severson_b3 (default) | attia_batch9. Patch E4
#                              switched the locked secondary test from Attia
#                              Batch 9 (transfer-only) to Severson b3
#                              (in-distribution true-life).
#   DO_FINAL_SECONDARY_TEST    true|false; default false. Set true to run a
#                              single locked secondary-test pass on the best
#                              candidate post-discovery.
#   FINAL_SECONDARY_TEST_TOP_K 0 default; >0 to also evaluate top-K Severson
#                              candidates on the secondary test.
#   DO_FINAL_BATCH9            DEPRECATED alias for DO_FINAL_SECONDARY_TEST.
#                              When set without SECONDARY_TEST_SOURCE, also
#                              forces SECONDARY_TEST_SOURCE=attia_batch9.
#   FINAL_BATCH9_TOP_K         DEPRECATED alias for FINAL_SECONDARY_TEST_TOP_K.
#   OFFLINE                    true|false; offline disables the LLM stack.

# ---------- EARLY HEARTBEAT (runs before set -euo pipefail and any GPFS touch) ----------
# Detects slurmstepd hangs / stale-GPFS conditions on the assigned node within
# ~1 second of job start, before the relative-path Oak `--output` write or the
# `cd ${REPO_ROOT}` stat call. If this file appears on the node but the slurm
# `--output` log does not, the monitor can classify the job as PHANTOM and
# auto-resubmit with --exclude=<bad_node>. See diagnostic Dx5/Dx9 (2026-06-09).
HEARTBEAT_LOCAL="/tmp/oba-trackA-${SLURM_JOB_ID:-nojob}.heartbeat"
HEARTBEAT_SCRATCH_DIR="/scratch/users/svangara/battery-arr/heartbeats"
HEARTBEAT_SCRATCH="${HEARTBEAT_SCRATCH_DIR}/oba-trackA-${SLURM_JOB_ID:-nojob}.heartbeat"
_write_heartbeat() {
  printf 'jobid=%s host=%s ts=%s\n' \
    "${SLURM_JOB_ID:-nojob}" "$(hostname 2>/dev/null || echo unknown)" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${HEARTBEAT_LOCAL}" 2>/dev/null || true
  mkdir -p "${HEARTBEAT_SCRATCH_DIR}" 2>/dev/null || true
  cp -f "${HEARTBEAT_LOCAL}" "${HEARTBEAT_SCRATCH}" 2>/dev/null || true
}
_write_heartbeat
( while true; do _write_heartbeat || true; sleep 300; done ) &
HEARTBEAT_PID=$!
disown "${HEARTBEAT_PID}" 2>/dev/null || true

export PYTHONUNBUFFERED=1

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/groups/darve/svangara/battery-arr-tpc}"
# Wrap cd with timeout so a stale Oak mount on the node fails loudly instead
# of wedging silently for 8 hours (cf. Dx2/Dx5 root-cause analysis 2026-06-09).
if ! timeout 60 bash -c "cd '${REPO_ROOT}' && pwd >/dev/null"; then
  echo "FATAL: cd ${REPO_ROOT} timed out on $(hostname) -- likely stale GPFS. Aborting." >&2
  exit 42
fi
cd "${REPO_ROOT}"

mkdir -p runs/slurm /scratch/users/svangara/battery-arr/runs/slurm

echo "=========================================="
echo "Open Battery Agents Track A — Sherlock"
echo "=========================================="
echo "Job ID         : ${SLURM_JOB_ID:-NA}"
echo "Host           : $(hostname)"
echo "Start time     : $(date)"
echo "Working dir    : $(pwd)"
echo "REPO_ROOT      : ${REPO_ROOT}"
echo

# ---------------------------------------------------------------------------
# Python module + persistent venv
# ---------------------------------------------------------------------------
PYTHON_MODULE="${PYTHON_MODULE:-python/3.12.1}"
VENV_DIR="${VENV_DIR:-/home/groups/darve/svangara/battery-arr-venvs/oba}"

if ! command -v module >/dev/null 2>&1; then
  echo "ERROR: 'module' command unavailable on this node." >&2
  exit 1
fi

module purge >/dev/null 2>&1 || true
module load "${PYTHON_MODULE}" 2>&1
if [ ! -d "${VENV_DIR}" ]; then
  echo "ERROR: persistent venv missing at ${VENV_DIR}." >&2
  echo "Bootstrap it once with: bash scripts/setup_sherlock_env.sh" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

PYTHON_BIN="$(command -v python3)"
echo "Python module  : ${PYTHON_MODULE}"
echo "Venv           : ${VENV_DIR}"
echo "Python binary  : ${PYTHON_BIN}"
"${PYTHON_BIN}" --version

PY_OK="$(${PYTHON_BIN} -c 'import sys; v=sys.version_info; print("ok" if (3,9)<=(v.major,v.minor)<=(3,12) else "bad")')"
if [ "${PY_OK}" != "ok" ]; then
  echo "ERROR: venv Python is not in supported 3.9-3.12 range." >&2
  "${PYTHON_BIN}" -c 'import sys; print(sys.version)' >&2
  exit 1
fi
echo

# ---------------------------------------------------------------------------
# Single-thread BLAS to play well with --cpus-per-task and parallel candidates.
# ---------------------------------------------------------------------------
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# ---------------------------------------------------------------------------
# Load .env (never print secrets)
# ---------------------------------------------------------------------------
if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
else
  echo "WARNING: .env not found at ${REPO_ROOT}/.env"
fi

export OPEN_BATTERY_AGENTS_API_KEY="${OPEN_BATTERY_AGENTS_API_KEY:-${STANFORD_AI_API_KEY:-${STANFORD_AI_PLAYGROUND_API_KEY:-}}}"
export OPEN_BATTERY_AGENTS_BASE_URL="${OPEN_BATTERY_AGENTS_BASE_URL:-${STANFORD_AI_BASE_URL:-${STANFORD_AI_PLAYGROUND_BASE_URL:-}}}"
export OPEN_BATTERY_AGENTS_MODEL="${OPEN_BATTERY_AGENTS_MODEL:-${MAGQUERY_KG_MODEL:-gpt-4.1}}"
export OPEN_BATTERY_AGENTS_ALIAS="${OPEN_BATTERY_AGENTS_ALIAS:-${STANFORD_AI_ALIAS:-spectraquery}}"

OFFLINE="${OFFLINE:-false}"
if [ "${OFFLINE}" != "true" ] && [ -z "${OPEN_BATTERY_AGENTS_API_KEY:-}" ]; then
  echo "ERROR: OFFLINE=false but no Open Battery Agents API key is configured." >&2
  echo "Either export OPEN_BATTERY_AGENTS_API_KEY or run with OFFLINE=true." >&2
  exit 1
fi

echo "LLM config check:"
echo "  offline          : ${OFFLINE}"
echo "  API key set      : $([ -n "${OPEN_BATTERY_AGENTS_API_KEY:-}" ] && echo yes || echo no)"
echo "  Base URL         : ${OPEN_BATTERY_AGENTS_BASE_URL:-MISSING}"
echo "  Model            : ${OPEN_BATTERY_AGENTS_MODEL:-MISSING}"
echo "  Alias configured : $([ -n "${OPEN_BATTERY_AGENTS_ALIAS:-}" ] && echo yes || echo no)"
echo

# ---------------------------------------------------------------------------
# Track A configuration
# ---------------------------------------------------------------------------
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

# Patch E4: new source-agnostic secondary-test contract.
# SECONDARY_TEST_SOURCE chooses the locked secondary test (severson_b3 default).
# DO_FINAL_SECONDARY_TEST / FINAL_SECONDARY_TEST_TOP_K are the new gating vars.
# Legacy DO_FINAL_BATCH9 / FINAL_BATCH9_TOP_K still work and, when set, force
# SECONDARY_TEST_SOURCE=attia_batch9 unless the new var is also provided.
SECONDARY_TEST_SOURCE="${SECONDARY_TEST_SOURCE:-severson_b3}"
DO_FINAL_SECONDARY_TEST="${DO_FINAL_SECONDARY_TEST:-${DO_FINAL_BATCH9}}"
FINAL_SECONDARY_TEST_TOP_K="${FINAL_SECONDARY_TEST_TOP_K:-${FINAL_BATCH9_TOP_K}}"
if [ "${DO_FINAL_BATCH9}" = "true" ] && [ "${SECONDARY_TEST_SOURCE}" = "severson_b3" ] && [ -z "${SECONDARY_TEST_SOURCE_EXPLICIT:-}" ]; then
  # Backwards compat: if the caller set DO_FINAL_BATCH9=true without an explicit
  # SECONDARY_TEST_SOURCE, preserve the old Attia transfer semantics.
  echo "NOTE: DO_FINAL_BATCH9=true forces SECONDARY_TEST_SOURCE=attia_batch9 (legacy alias)."
  SECONDARY_TEST_SOURCE="attia_batch9"
fi
case "${SECONDARY_TEST_SOURCE}" in
  severson_b3|attia_batch9) : ;;
  *)
    echo "ERROR: SECONDARY_TEST_SOURCE must be 'severson_b3' or 'attia_batch9', got '${SECONDARY_TEST_SOURCE}'." >&2
    exit 1
    ;;
esac

# ---------------------------------------------------------------------------
# Output / scratch layout
# ---------------------------------------------------------------------------
RUN_ROOT="${RUN_ROOT:-${SCRATCH}/battery-arr}"
RUN_ID="trackA_${FEATURE_RECIPE}_${SPLIT_MODE}_seed${SPLIT_SEED}_${SLURM_JOB_ID}"
OUT_DIR="${RUN_ROOT}/runs/${RUN_ID}"
REPORTS_DIR="${RUN_ROOT}/reports/${RUN_ID}"
ARCHIVE_DIR="${RUN_ROOT}/archives"
ARCHIVE="${ARCHIVE_DIR}/oba-trackA-${SLURM_JOB_ID}-artifacts.tgz"

mkdir -p "${OUT_DIR}" "${REPORTS_DIR}" "${ARCHIVE_DIR}"

BFC_ROOT="${BFC_ROOT:-${REPO_ROOT}/literature_models_and_data/battery-fast-charging}"
if [ ! -d "${BFC_ROOT}" ]; then
  echo "ERROR: Missing battery-fast-charging root: ${BFC_ROOT}" >&2
  exit 1
fi

# Per-job local scratch (compute-node local). Fall back to $SCRATCH if unset.
SCRATCH_LOCAL_ROOT="${L_SCRATCH:-${SCRATCH}}"
SCRATCH_ROOT="${SCRATCH_LOCAL_ROOT}/oba_trackA_${SLURM_JOB_ID}"
mkdir -p "${SCRATCH_ROOT}"

export TMPDIR="${SCRATCH_ROOT}/tmp"
mkdir -p "${TMPDIR}"

# ---------------------------------------------------------------------------
# Archive on exit (success or failure)
# ---------------------------------------------------------------------------
archive_on_exit() {
  status=$?
  # Stop heartbeat updater and clean its files so monitor sees CLEAN exit.
  kill "${HEARTBEAT_PID:-0}" 2>/dev/null || true
  rm -f "${HEARTBEAT_LOCAL:-/dev/null}" "${HEARTBEAT_SCRATCH:-/dev/null}" 2>/dev/null || true
  echo
  echo "=========================================="
  echo "Archiving partial/final outputs, exit status: ${status}"
  echo "=========================================="
  mkdir -p "${OUT_DIR}/feature_programs" "${ARCHIVE_DIR}"
  if [ -d "${SCRATCH_ROOT}/feature_programs" ]; then
    cp -a "${SCRATCH_ROOT}/feature_programs/." "${OUT_DIR}/feature_programs/" || true
  fi
  if [ -d "${SCRATCH_ROOT}/processed" ]; then
    mkdir -p "${OUT_DIR}/processed"
    cp -a "${SCRATCH_ROOT}/processed/." "${OUT_DIR}/processed/" || true
  fi
  tar -czf "${ARCHIVE}" \
    -C "${RUN_ROOT}" \
    "reports/${RUN_ID}" \
    "runs/${RUN_ID}" \
    2>/dev/null || true
  echo "Archive path:"
  ls -lh "${ARCHIVE}" 2>/dev/null || true
  echo "Done/archive time: $(date)"
  exit "${status}"
}
trap archive_on_exit EXIT

# ---------------------------------------------------------------------------
# Stage data into node-local scratch (Severson always; Batch 9 only on demand).
# ---------------------------------------------------------------------------
SCRATCH_BFC="${SCRATCH_ROOT}/battery-fast-charging"
mkdir -p "${SCRATCH_BFC}/data"

SEVERSON_SRC_DIR="${BFC_ROOT}/data/severson_2019_true_life_matr"
SEVERSON_MAT_DIR="${SCRATCH_BFC}/data/severson_2019_true_life_matr"
if [ ! -d "${SEVERSON_SRC_DIR}" ]; then
  echo "ERROR: Missing Severson MatR directory in repo: ${SEVERSON_SRC_DIR}" >&2
  exit 1
fi
echo "Staging Severson .mat files to node-local scratch..."
mkdir -p "${SEVERSON_MAT_DIR}"
cp -a "${SEVERSON_SRC_DIR}/." "${SEVERSON_MAT_DIR}/"

BATCH9_SRC_DIR="${BFC_ROOT}/data/2019-01-24_batch9"
BATCH9_PATH="${SCRATCH_BFC}/data/2019-01-24_batch9"

# Resolve the Severson b3 .mat path inside the already-staged severson dir.
SEVERSON_B3_BATCH_DATE="2018-04-12"
SEVERSON_B3_MAT_FILE="${SEVERSON_MAT_DIR}/${SEVERSON_B3_BATCH_DATE}_batchdata_updated_struct_errorcorrect.mat"
SEVERSON_B3_PATH="${SEVERSON_B3_MAT_FILE}"

NEED_SECONDARY=false
if [ "${DO_FINAL_SECONDARY_TEST}" = "true" ] || [ "${FINAL_SECONDARY_TEST_TOP_K}" -gt 0 ]; then
  NEED_SECONDARY=true
fi

if [ "${SECONDARY_TEST_SOURCE}" = "attia_batch9" ]; then
  if [ "${NEED_SECONDARY}" = "true" ]; then
    if [ ! -d "${BATCH9_SRC_DIR}" ]; then
      echo "ERROR: Batch 9 (Attia transfer) requested but expanded folder missing: ${BATCH9_SRC_DIR}" >&2
      exit 1
    fi
    echo "Staging Batch 9 (Attia transfer; final secondary-test validation enabled)..."
    mkdir -p "${BATCH9_PATH}"
    cp -a "${BATCH9_SRC_DIR}/." "${BATCH9_PATH}/"
  else
    echo "Batch 9 NOT staged (SECONDARY_TEST_SOURCE=attia_batch9 but no final secondary test)."
  fi
else
  # severson_b3 routing: do NOT stage Attia. The b3 .mat is already in the
  # staged Severson directory, so we only need to confirm it exists.
  if [ "${NEED_SECONDARY}" = "true" ]; then
    if [ ! -f "${SEVERSON_B3_MAT_FILE}" ]; then
      echo "ERROR: Severson b3 .mat missing at ${SEVERSON_B3_MAT_FILE}" >&2
      exit 1
    fi
    echo "Severson b3 secondary test staged at ${SEVERSON_B3_MAT_FILE}"
  else
    echo "Severson b3 secondary test NOT run (DO_FINAL_SECONDARY_TEST=false, FINAL_SECONDARY_TEST_TOP_K=0)."
  fi
  echo "Attia Batch 9 NOT staged (default secondary test is severson_b3)."
fi
du -sh "${SCRATCH_BFC}" || true
echo

# ---------------------------------------------------------------------------
# Verify imports + LLM config
# ---------------------------------------------------------------------------
echo "Python import and LLM config check..."
"${PYTHON_BIN}" - <<'PY'
import sys
print("python :", sys.executable)
print("version:", sys.version)
import numpy, pandas, scipy, sklearn, h5py
import openai, pydantic, tenacity, dotenv
import battery_aar
from battery_aar.agents.llm_client import llm_startup_summary
print("imports OK")
print("llm   :", llm_startup_summary())
PY
echo

# ---------------------------------------------------------------------------
# Optional one-shot prompt patch (idempotent; already applied in the repo).
# ---------------------------------------------------------------------------
if [ "${PATCH_GENERIC_PROMPTS}" = "true" ]; then
  echo "Re-applying generic Track A prompt patch (idempotent)..."
  "${PYTHON_BIN}" scripts/patch_trackA_generic_prompts.py || true
  echo "Prompt paper-specific term check:"
  grep -RnE "Attia|Severson-like|author-inspired|literature early predictor" \
    src/battery_aar/workflows/role_prompts.py || echo "  (no paper-specific terms found)"
  echo
fi

# ---------------------------------------------------------------------------
# Build dataset
# ---------------------------------------------------------------------------
PROCESSED_DIR="${SCRATCH_ROOT}/processed/severson_2019_true_life"
SEVERSON_FP_DIR="${SCRATCH_ROOT}/feature_programs/severson_${FEATURE_RECIPE}_idx${CYCLE_EARLY_INDEX}_${CYCLE_LATE_INDEX}"
BATCH9_FP_DIR="${SCRATCH_ROOT}/feature_programs/batch9_${FEATURE_RECIPE}_idx${CYCLE_EARLY_INDEX}_${CYCLE_LATE_INDEX}"

echo "=========================================="
echo "Auditing Severson MatR data"
echo "=========================================="
"${PYTHON_BIN}" scripts/audit_severson_matr_data.py \
  --mat-dir "${SEVERSON_MAT_DIR}" \
  --out "${REPORTS_DIR}/severson_matr_audit"

echo
echo "=========================================="
echo "Building Severson true-life processed dataset"
echo "=========================================="
"${PYTHON_BIN}" scripts/build_severson_true_life_dataset.py \
  --mat-dir "${SEVERSON_MAT_DIR}" \
  --out "${PROCESSED_DIR}" \
  --first-n-cycles "${FIRST_N_CYCLES}"

echo
echo "=========================================="
echo "Building Track A feature-program table"
echo "=========================================="
echo "recipe                   : ${FEATURE_RECIPE}"
echo "cycle early index        : ${CYCLE_EARLY_INDEX}"
echo "cycle late index         : ${CYCLE_LATE_INDEX}"
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
SEVERSON_FP_DIR="${SEVERSON_FP_DIR}" "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path
import numpy as np
import pandas as pd

out = Path(os.environ["SEVERSON_FP_DIR"])
ft = pd.read_csv(out / "feature_table.csv")
meta = pd.read_csv(out / "feature_metadata.csv")
print("feature_table   :", ft.shape)
print("feature_metadata:", meta.shape)
if "feature_family" in meta.columns:
    print("families        :", meta["feature_family"].value_counts().to_dict())
assert "row_id" in ft.columns, "missing row_id column"
assert ft["row_id"].is_unique, "row_id not unique"
forbidden = ["cycle_life", "author_model_prediction", "target", "label",
             "source_path", "raw_path", "filename"]
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

# ---------------------------------------------------------------------------
# Optional Attia Batch 9 feature table (only if final validation enabled AND
# the caller has explicitly opted in to the Attia transfer source).
# Severson b3 reuses the search-time Severson feature programs.
# ---------------------------------------------------------------------------
if [ "${SECONDARY_TEST_SOURCE}" = "attia_batch9" ] && [ "${NEED_SECONDARY}" = "true" ]; then
  echo
  echo "=========================================="
  echo "Building optional Batch 9 feature table (Attia transfer)"
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
fi

# ---------------------------------------------------------------------------
# Assemble run command
# ---------------------------------------------------------------------------
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

RUN_ARGS+=(--secondary-test-source "${SECONDARY_TEST_SOURCE}")
if [ "${NEED_SECONDARY}" = "true" ]; then
  if [ "${SECONDARY_TEST_SOURCE}" = "attia_batch9" ]; then
    RUN_ARGS+=(--secondary-test-path "${BATCH9_PATH}")
    RUN_ARGS+=(--secondary-test-feature-program-path "${BATCH9_FP_DIR}/feature_table.csv")
  else
    RUN_ARGS+=(--secondary-test-path "${SEVERSON_B3_PATH}")
    # Reuse the Severson search-time feature table for the secondary test.
    RUN_ARGS+=(--secondary-test-feature-program-path "${SEVERSON_FP_DIR}/feature_table.csv")
  fi
  if [ "${DO_FINAL_SECONDARY_TEST}" = "true" ]; then
    RUN_ARGS+=(--final-secondary-test-validation --locked-test)
  fi
  if [ "${FINAL_SECONDARY_TEST_TOP_K}" -gt 0 ]; then
    RUN_ARGS+=(--final-secondary-test-top-k "${FINAL_SECONDARY_TEST_TOP_K}")
  fi
fi

echo
echo "=========================================="
echo "Running Track A role-agent workflow"
echo "=========================================="
echo "run id                  : ${RUN_ID}"
echo "feature recipe          : ${FEATURE_RECIPE}"
echo "split mode              : ${SPLIT_MODE}"
echo "split seed              : ${SPLIT_SEED}"
echo "iterations              : ${ITERATIONS}"
echo "candidates / iteration  : ${CANDIDATES_PER_ITERATION}"
echo "secondary test source   : ${SECONDARY_TEST_SOURCE}"
echo "secondary final / top-K : ${DO_FINAL_SECONDARY_TEST}; top-k=${FINAL_SECONDARY_TEST_TOP_K}"
echo
printf '%q ' "${RUN_ARGS[@]}"
echo

"${RUN_ARGS[@]}"

# ---------------------------------------------------------------------------
# Copy node-local feature programs / processed data back into the run dir.
# ---------------------------------------------------------------------------
echo
echo "Copying feature-program and processed data into run directory..."
mkdir -p "${OUT_DIR}/feature_programs" "${OUT_DIR}/processed"
cp -a "${SEVERSON_FP_DIR}" "${OUT_DIR}/feature_programs/" 2>/dev/null || true
cp -a "${PROCESSED_DIR}" "${OUT_DIR}/processed/" 2>/dev/null || true
if [ -d "${BATCH9_FP_DIR}" ]; then
  cp -a "${BATCH9_FP_DIR}" "${OUT_DIR}/feature_programs/" 2>/dev/null || true
fi

echo
echo "Key result grep (truncated):"
grep -RniE "rmse|mae|spearman|kendall|best|error|traceback|failed|locked|batch9" \
  "${OUT_DIR}" "${REPORTS_DIR}" 2>/dev/null | head -200 || true

echo
echo "Report preview:"
if [ -f "${REPORTS_DIR}/role_agent_workflow.md" ]; then
  head -240 "${REPORTS_DIR}/role_agent_workflow.md"
fi

echo
echo "Done at: $(date)"
