#!/bin/bash
# Launch the full Open Battery Agents / Track A sealed broad-discovery campaign
# on Stanford Sherlock. Submits the 3 x 2 x 3 = 18-job matrix without launching
# Batch 9 (Batch 9 is reserved for post-discovery locked validation).
#
# Submit count: 18 jobs (3 recipes x 2 split modes x 3 seeds).
# Each job: 8 iterations, 20 candidates/iteration.
#
# Usage:
#   bash scripts/launch_sherlock_trackA_campaign.sh
#
# Optional environment variables:
#   ITERATIONS                Outer iterations per job (default 8).
#   CANDIDATES_PER_ITERATION  Trusted candidates per iteration (default 20).
#   SLURM_TIME                Wall time per job (default 8:00:00).
#   SLURM_MEM                 Memory per job (default 64G).
#   SLURM_CPUS                CPUs per task (default 16).
#   LAUNCH_LOG                Override the CSV launch log path.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/groups/darve/svangara/battery-arr-tpc}"
cd "${REPO_ROOT}"

ITERATIONS="${ITERATIONS:-8}"
CANDIDATES_PER_ITERATION="${CANDIDATES_PER_ITERATION:-20}"
SLURM_TIME="${SLURM_TIME:-8:00:00}"
SLURM_MEM="${SLURM_MEM:-64G}"
SLURM_CPUS="${SLURM_CPUS:-16}"

if [ -z "${SCRATCH:-}" ]; then
  echo "ERROR: \$SCRATCH is unset. Run this from a Sherlock login or compute node." >&2
  exit 1
fi

mkdir -p "${SCRATCH}/battery-arr/launch_logs"
LAUNCH_LOG="${LAUNCH_LOG:-${SCRATCH}/battery-arr/launch_logs/trackA_sherlock_$(date +%Y%m%d_%H%M%S).csv}"

echo "Launching sealed Track A campaign"
echo "  iterations              : ${ITERATIONS}"
echo "  candidates / iteration  : ${CANDIDATES_PER_ITERATION}"
echo "  sbatch resources        : cpus=${SLURM_CPUS} mem=${SLURM_MEM} time=${SLURM_TIME}"
echo "  launch log              : ${LAUNCH_LOG}"
echo

echo "job_id,recipe,split_mode,split_seed,iterations,candidates_per_iteration" > "${LAUNCH_LOG}"

for recipe in scalar_baseline curve_delta broad_physics; do
  for mode in random batch; do
    for seed in 0 1 2; do
      J=$(sbatch \
        --cpus-per-task="${SLURM_CPUS}" \
        --mem="${SLURM_MEM}" \
        --time="${SLURM_TIME}" \
        --export=ALL,OFFLINE=false,FEATURE_RECIPE="${recipe}",SPLIT_MODE="${mode}",SPLIT_SEED="${seed}",ITERATIONS="${ITERATIONS}",CANDIDATES_PER_ITERATION="${CANDIDATES_PER_ITERATION}",DO_FINAL_BATCH9=false,FINAL_BATCH9_TOP_K=0 \
        nersc/sherlock_trackA_discovery.slurm | awk '{print $4}')
      echo "${J},${recipe},${mode},${seed},${ITERATIONS},${CANDIDATES_PER_ITERATION}" | tee -a "${LAUNCH_LOG}"
    done
  done
done

echo
echo "Launch log:"
cat "${LAUNCH_LOG}"
