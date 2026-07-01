#!/bin/bash
#SBATCH --job-name=foundation-transfer
#SBATCH --output=foundation-transfer_%j.out
#SBATCH --error=foundation-transfer_%j.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250GB
#SBATCH --time=2-00:00:00
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

# Transfer does NOT use PVI_CACHE_ROOT (per-subject HDF5 + within-subject split).
unset PVI_CACHE_ROOT

SUBJECT="${SUBJECT:-subject013}"
CORE="${CORE:-artifacts/foundation-pretrain/main/foundation_core.pt}"
CORE_TAG="${CORE_TAG:-}"

TAG_ARGS=()
if [[ -n "${CORE_TAG}" ]]; then
  TAG_ARGS=(--core-tag "${CORE_TAG}")
fi

JOB_SCRIPT="${JOB_SCRIPT:-python -u -m src.foundation.transfer --subject ${SUBJECT} --core ${CORE} --batch-size 32 --max-epochs 500 ${TAG_ARGS[*]}}"

echo "PVIPROJECT_ROOT=${PVIPROJECT_ROOT}"
echo "PVI_DATA_ROOT=${PVI_DATA_ROOT}"
echo "Running: ${JOB_SCRIPT}"
eval "${JOB_SCRIPT}"

deactivate
