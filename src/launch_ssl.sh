#!/bin/bash
#SBATCH --job-name=foundation-ssl-pretrain
#SBATCH --output=foundation-ssl-pretrain_%j.out
#SBATCH --error=foundation-ssl-pretrain_%j.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250GB
#SBATCH --time=10-00:00:00
#SBATCH --gres=gpu:1

set -euo pipefail

module load CUDA/12.9.0
module load gcc/11.2.0

REPO_ROOT="/mmfs1/projects/ece_bst/lsanc68/fundational_pvi"
cd "${REPO_ROOT}"

# shellcheck source=/dev/null
source "${REPO_ROOT}/env/cluster.env"

source "${REPO_ROOT}/.venv/bin/activate"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if [[ ! -f "${PVI_CACHE_ROOT}/manifest.json" ]]; then
  echo "Parquet cache missing at ${PVI_CACHE_ROOT}; building now..."
  python -u -m src.scripts.build_pvi_cache --cache-root "${PVI_CACHE_ROOT}" \
    --input-mode signal --output-mode waveform --mask-key mask05
fi

# Core U: SSL cohort pretrain (masked reconstruction + forecasting).
# Output: artifacts/foundation-ssl-pretrain/main/foundation_core_U.pt
JOB_SCRIPT="${JOB_SCRIPT:-python -u -m src.foundation.ssl_pretrain --input-mode signal --max-epochs 500 --batch-size 256 --cache-num-workers 8}"

echo "PVIPROJECT_ROOT=${PVIPROJECT_ROOT}"
echo "PVI_DATA_ROOT=${PVI_DATA_ROOT}"
echo "PVI_CACHE_ROOT=${PVI_CACHE_ROOT}"
echo "Running: ${JOB_SCRIPT}"
eval "${JOB_SCRIPT}"

deactivate
