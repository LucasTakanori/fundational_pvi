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

INPUT_MODE="${INPUT_MODE:-impedance}"
ARCH="${ARCH:-crt}"
LOGDIR="${LOGDIR:-foundation-pretrain-crt}"
export PVI_CACHE_ROOT="${PVI_CACHE_ROOT_IMPEDANCE:-/mmfs1/scratch/${USER}/pvi_cache/v1_impedance}"

# Fail fast if cache missing (submit launch_build_cache_impedance.sh first).
if ! python -u -m src.scripts.check_cache --cache-root "${PVI_CACHE_ROOT}" --input-mode "${INPUT_MODE}"; then
  echo "ERROR: No valid impedance Parquet cache at ${PVI_CACHE_ROOT}"
  echo "Run: sbatch src/launch_build_cache_impedance.sh"
  echo "Then: sbatch --dependency=afterok:<JOBID> src/launch_crt_pretrain.sh"
  exit 1
fi

JOB_SCRIPT="${JOB_SCRIPT:-python -u -m src.foundation.pretrain \
  --input-mode ${INPUT_MODE} --output-mode waveform --arch ${ARCH} \
  --batch-size 256 --eval-every 5 --max-epochs 500 \
  --cache-root ${PVI_CACHE_ROOT} --cache-num-workers 8 \
  --logdir ${LOGDIR}}"

echo "PVI_CACHE_ROOT=${PVI_CACHE_ROOT}"
echo "Running: ${JOB_SCRIPT}"
eval "${JOB_SCRIPT}"

deactivate
