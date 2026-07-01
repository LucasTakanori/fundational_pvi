#!/bin/bash
#SBATCH --job-name=cache-main-disjoint
#SBATCH --output=logs/build-cache-main-disjoint_%j.out
#SBATCH --error=logs/build-cache-main-disjoint_%j.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250GB
#SBATCH --time=1-00:00:00

# PD protocol: branch=main, split_mode=disjoint (subject holdout from train pool).
# Use with: --cache-root ${PVI_CACHE_ROOT_IMPEDANCE}_main_disjoint

set -euo pipefail

REPO_ROOT="/mmfs1/projects/ece_bst/lsanc68/fundational_pvi"
cd "${REPO_ROOT}"

# shellcheck source=/dev/null
source "${REPO_ROOT}/env/cluster.env"
source "${REPO_ROOT}/.venv/bin/activate"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

INPUT_MODE="${INPUT_MODE:-impedance}"
CACHE_BASE="${PVI_CACHE_ROOT_IMPEDANCE:-/mmfs1/scratch/${USER}/pvi_cache/v1_impedance}"
CACHE_ROOT="${CACHE_BASE}_main_disjoint"

mkdir -p "${CACHE_ROOT}"

echo "PVI_DATA_ROOT=${PVI_DATA_ROOT}"
echo "CACHE_ROOT=${CACHE_ROOT}"
echo "branch=main  split_mode=disjoint  input_mode=${INPUT_MODE}"

python -u -m src.scripts.build_pvi_cache \
  --cache-root "${CACHE_ROOT}" \
  --input-mode "${INPUT_MODE}" \
  --output-mode waveform \
  --mask-key mask05 \
  --branch main \
  --split-mode disjoint \
  --force

python -u -m src.scripts.check_cache \
  --cache-root "${CACHE_ROOT}" \
  --input-mode "${INPUT_MODE}" \
  --branch main \
  --split-mode disjoint

deactivate
