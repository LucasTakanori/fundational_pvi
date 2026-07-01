#!/bin/bash
#SBATCH --job-name=foundation-mae-ssl
#SBATCH --output=logs/foundation-mae-ssl_%j.out
#SBATCH --error=logs/foundation-mae-ssl_%j.err
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
SSL_ARCH="${SSL_ARCH:-mae}"
LOGDIR="${LOGDIR:-foundation-ssl-pretrain-mae}"
export PVI_CACHE_ROOT="${PVI_CACHE_ROOT_IMPEDANCE:-/mmfs1/scratch/${USER}/pvi_cache/v1_impedance}"

if ! python -u -m src.scripts.check_cache --cache-root "${PVI_CACHE_ROOT}" --input-mode "${INPUT_MODE}"; then
  echo "ERROR: No valid impedance Parquet cache at ${PVI_CACHE_ROOT}"
  echo "Run: sbatch src/launch_build_cache_impedance.sh"
  echo "Then: sbatch --dependency=afterok:<JOBID> src/launch_mae_ssl.sh"
  exit 1
fi

JOB_SCRIPT="${JOB_SCRIPT:-python -u -m src.foundation.ssl_pretrain \
  --input-mode ${INPUT_MODE} --arch ${SSL_ARCH} --ssl-arch ${SSL_ARCH} \
  --batch-size 256 --max-epochs 500 \
  --cache-root ${PVI_CACHE_ROOT} --cache-num-workers 8 \
  --logdir ${LOGDIR}}"

echo "PVI_CACHE_ROOT=${PVI_CACHE_ROOT}"
echo "Running: ${JOB_SCRIPT}"
eval "${JOB_SCRIPT}"

deactivate
