#!/bin/bash
#SBATCH --job-name=foundation-crt-pretrain
#SBATCH --output=logs/foundation-crt-pretrain_%j.out
#SBATCH --error=logs/foundation-crt-pretrain_%j.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250GB
#SBATCH --time=10-00:00:00
#SBATCH --gres=gpu:1

# --- edit before each CRT run ---
CACHE_ROOT="/mmfs1/scratch/lsanc68/pvi_cache/v1_impedance_main_within"
SPLIT_MODE="within"              # disjoint = PD  |  within = PW
LOGDIR="foundation-pretrain-crt-pw"
# --------------------------------

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
unset PVI_CACHE_ROOT

if ! python -u -m src.scripts.check_cache \
  --cache-root "${CACHE_ROOT}" \
  --input-mode impedance \
  --split-mode "${SPLIT_MODE}"; then
  echo "ERROR: cache check failed for ${CACHE_ROOT}"
  exit 1
fi

echo "CACHE_ROOT=${CACHE_ROOT}  split_mode=${SPLIT_MODE}  logdir=${LOGDIR}"

python -u -m src.foundation.pretrain \
  --input-mode impedance \
  --output-mode waveform \
  --arch crt \
  --batch-size 256 \
  --eval-every 5 \
  --max-epochs 500 \
  --cache-root "${CACHE_ROOT}" \
  --split-mode "${SPLIT_MODE}" \
  --cache-num-workers 8 \
  --branch main \
  --logdir "${LOGDIR}"

deactivate
