#!/bin/bash
#SBATCH --job-name=build-pvi-cache-imp
#SBATCH --output=build-pvi-cache-imp_%j.out
#SBATCH --error=build-pvi-cache-imp_%j.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250GB
#SBATCH --time=1-00:00:00

# One-time build: HDF5 → Parquet for impedance (64-ch R+X). ~1–3 h for 216 files.
# After this completes, CRT / MAE jobs use PVI_CACHE_ROOT_IMPEDANCE automatically.

set -euo pipefail

REPO_ROOT="/mmfs1/projects/ece_bst/lsanc68/fundational_pvi"
cd "${REPO_ROOT}"

# shellcheck source=/dev/null
source "${REPO_ROOT}/env/cluster.env"

source "${REPO_ROOT}/.venv/bin/activate"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

INPUT_MODE="${INPUT_MODE:-impedance}"
export PVI_CACHE_ROOT="${PVI_CACHE_ROOT_IMPEDANCE:-/mmfs1/scratch/${USER}/pvi_cache/v1_impedance}"

mkdir -p "${PVI_CACHE_ROOT}"

echo "PVI_DATA_ROOT=${PVI_DATA_ROOT}"
echo "PVI_CACHE_ROOT=${PVI_CACHE_ROOT}"
echo "Building Parquet cache (${INPUT_MODE})..."

python -u -m src.scripts.build_pvi_cache \
  --cache-root "${PVI_CACHE_ROOT}" \
  --input-mode "${INPUT_MODE}" \
  --output-mode waveform \
  --mask-key mask05 \
  --force

python -u -m src.scripts.check_cache \
  --cache-root "${PVI_CACHE_ROOT}" \
  --input-mode "${INPUT_MODE}"

deactivate
