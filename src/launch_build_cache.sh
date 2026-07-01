#!/bin/bash
#SBATCH --job-name=build-pvi-cache
#SBATCH --output=logs/build-pvi-cache_%j.out
#SBATCH --error=logs/build-pvi-cache_%j.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250GB
#SBATCH --time=1-00:00:00

set -euo pipefail

REPO_ROOT="/mmfs1/projects/ece_bst/lsanc68/fundational_pvi"
cd "${REPO_ROOT}"

# shellcheck source=/dev/null
source "${REPO_ROOT}/env/cluster.env"

source "${REPO_ROOT}/.venv/bin/activate"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

mkdir -p "${PVI_CACHE_ROOT}"

echo "PVI_DATA_ROOT=${PVI_DATA_ROOT}"
echo "PVI_CACHE_ROOT=${PVI_CACHE_ROOT}"
echo "Building Parquet cache (one-time, ~1–3 h for 216 files)..."

python -u -m src.scripts.build_pvi_cache \
  --cache-root "${PVI_CACHE_ROOT}" \
  --input-mode signal \
  --output-mode waveform \
  --mask-key mask05 \
  "$@"

deactivate
