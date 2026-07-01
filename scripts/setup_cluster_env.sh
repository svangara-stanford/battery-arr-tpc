#!/bin/bash
# Bootstrap the Battery AAR Python environment.
# Idempotent: re-running upgrades pip wheels but never touches the repo source tree.
# Works on any cluster or local machine with Python 3.9+.
#
# Usage:
#   bash scripts/setup_env.sh
#
# Optional environment variables:
#   VENV_DIR       Absolute path to the persistent venv
#                  (default: ~/.venvs/battery-arr)
#   REPO_ROOT      Path to the battery-arr-tpc checkout
#                  (default: derived from this script's location)
#   PYTHON_BIN     Python 3.9+ executable to use
#                  (default: python3 from PATH)
#   FORCE_RECREATE If "true", delete and recreate the venv from scratch.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
VENV_DIR="${VENV_DIR:-${HOME}/.venvs/battery-arr}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
FORCE_RECREATE="${FORCE_RECREATE:-false}"

echo "=========================================="
echo "Battery AAR env setup"
echo "=========================================="
echo "REPO_ROOT     : ${REPO_ROOT}"
echo "VENV_DIR      : ${VENV_DIR}"
echo "PYTHON_BIN    : ${PYTHON_BIN}"
echo "FORCE_RECREATE: ${FORCE_RECREATE}"
echo

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: Python executable '${PYTHON_BIN}' not found on PATH." >&2
  echo "Ensure Python 3.9+ is available, or set PYTHON_BIN=/path/to/python3" >&2
  exit 1
fi

echo "Python executable: ${PYTHON_BIN}"
"${PYTHON_BIN}" --version

PY_VERSION_OK="$(${PYTHON_BIN} -c 'import sys; v=sys.version_info; print("ok" if (3,9) <= (v.major, v.minor) <= (3,12) else "bad")')"
if [ "${PY_VERSION_OK}" != "ok" ]; then
  echo "ERROR: Loaded interpreter is not in the supported 3.9-3.12 range." >&2
  "${PYTHON_BIN}" -c 'import sys; print(sys.version)' >&2
  exit 1
fi

if [ "${FORCE_RECREATE}" = "true" ] && [ -d "${VENV_DIR}" ]; then
  echo "FORCE_RECREATE=true: removing existing venv at ${VENV_DIR}"
  rm -rf "${VENV_DIR}"
fi

# If a stale Python 3.6 venv is present, replace it.
if [ -d "${VENV_DIR}" ]; then
  if "${VENV_DIR}/bin/python3" - <<'PY' 2>/dev/null
import sys
v = sys.version_info
sys.exit(0 if (3, 9) <= (v.major, v.minor) <= (3, 12) else 1)
PY
  then
    echo "Existing venv at ${VENV_DIR} uses a supported Python; reusing."
  else
    echo "Existing venv at ${VENV_DIR} is the wrong Python version; recreating."
    rm -rf "${VENV_DIR}"
  fi
fi

if [ ! -d "${VENV_DIR}" ]; then
  mkdir -p "$(dirname "${VENV_DIR}")"
  echo "Creating venv at ${VENV_DIR} using ${PYTHON_BIN}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# Activate the venv for the rest of this script.
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
echo "Activated venv python: $(command -v python3)"
python3 --version

python3 -m pip install --upgrade pip setuptools wheel

python3 -m pip install --only-binary=:all: \
  "numpy==1.26.4" \
  "pandas==2.2.1" \
  "scipy==1.11.4" \
  "scikit-learn==1.4.2" \
  "h5py==3.10.0" \
  "PyYAML==6.0.2" \
  "joblib==1.4.2"

python3 -m pip install \
  "openai>=1.40,<2" \
  "pydantic>=2,<3" \
  "eval-type-backport>=0.2,<1" \
  "tenacity>=8,<9" \
  "python-dotenv>=1,<2" \
  "fastapi>=0.110" \
  "uvicorn>=0.27" \
  "httpx>=0.27"

python3 -m pip install --no-deps --no-build-isolation --ignore-requires-python \
  -e "${REPO_ROOT}"

echo
echo "Final import / config check..."
python3 - <<'PY'
import sys
print("python:", sys.executable)
print("version:", sys.version)
import numpy, pandas, scipy, sklearn, h5py
import openai, pydantic, tenacity, dotenv
import battery_aar
from battery_aar.workflows.candidate_compiler import normalize_feature_set, normalize_target_transform
from battery_aar.workflows.roles import _as_str_or, _as_int_or, _as_str_list, _as_dict_or_empty
assert normalize_feature_set("scalar-only") == "scalar_only"
assert normalize_target_transform("log target") == "log10"
assert _as_str_or({"x": 1}, None)
print("imports + sanity OK")
PY

echo
echo "Setup complete."
echo "Activate with:"
echo "  source ${VENV_DIR}/bin/activate"
