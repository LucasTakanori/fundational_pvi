#!/bin/bash
#SBATCH --job-name=exp-b-budget
#SBATCH --output=exp-b-budget_%j.out
#SBATCH --error=exp-b-budget_%j.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250GB
#SBATCH --time=1-00:00:00
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
unset PVI_CACHE_ROOT

SUBJECT="${SUBJECT:-subject013}"
CORE_S="${CORE_S:-artifacts/foundation-pretrain/main/foundation_core.pt}"
CORE_U="${CORE_U:-artifacts/foundation-ssl-pretrain/main/foundation_core_U.pt}"
BUDGETS_MIN="${BUDGETS_MIN:-4,8,16,32,64}"

JOB_SCRIPT="${JOB_SCRIPT:-python -u -m src.foundation.budget_exp --subject ${SUBJECT} \
  --core-s ${CORE_S} --core-u ${CORE_U} --budgets-min ${BUDGETS_MIN} \
  --epochs-per-budget 50 --batch-size 32}"

echo "PVIPROJECT_ROOT=${PVIPROJECT_ROOT}"
echo "PVI_DATA_ROOT=${PVI_DATA_ROOT}"
echo "Running: ${JOB_SCRIPT}"
eval "${JOB_SCRIPT}"

deactivate
