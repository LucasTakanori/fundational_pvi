#!/bin/bash
#SBATCH --job-name=ssl-scale-post
#SBATCH --output=logs/ssl-scale-post_%j.out
#SBATCH --error=logs/ssl-scale-post_%j.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250GB
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1

# Phase 2 post-SSL: subject transfer + prediction-scale inspection (GPU node).
# Run after launch_ssl_scale_check.sh completes and foundation_core_U.pt exists.

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

# Transfer uses per-subject HDF5, not the Parquet cache.
unset PVI_CACHE_ROOT

SUBJECT="${SUBJECT:-subject013}"
CORE="${CORE:-artifacts/debug-ssl-scale-check/main/foundation_core_U.pt}"
INPUT_MODE="${INPUT_MODE:-impedance}"
BRANCH="${BRANCH:-main}"
MAX_EPOCHS="${MAX_EPOCHS:-20}"
LOGDIR="${LOGDIR:-debug-ssl-scale-check-transfer}"

if [[ ! -f "${CORE}" ]]; then
  echo "ERROR: core checkpoint not found: ${CORE}" >&2
  echo "Run sbatch src/launch_ssl_scale_check.sh first (or set CORE=...)." >&2
  exit 1
fi

echo "[postcheck] subject=${SUBJECT}  core=${CORE}  max_epochs=${MAX_EPOCHS}"
echo "[postcheck] logdir=${LOGDIR}"

python -u -m src.foundation.transfer \
  --subject "${SUBJECT}" \
  --core "${CORE}" \
  --input-mode "${INPUT_MODE}" \
  --branch "${BRANCH}" \
  --max-epochs "${MAX_EPOCHS}" \
  --logdir "${LOGDIR}"

python -u -m src.scripts.inspect_prediction_scale \
  --subject "${SUBJECT}" \
  --core "${CORE}" \
  --input-mode "${INPUT_MODE}" \
  --branch "${BRANCH}" \
  --transfer-logdir "${LOGDIR}"

echo "[postcheck] done — check prediction min/max/mean above (~40–200 mmHg pass)."

deactivate
