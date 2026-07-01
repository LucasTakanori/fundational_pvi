#!/bin/bash
#SBATCH --job-name=ssl-scale-check
#SBATCH --output=logs/ssl-scale-check_%j.out
#SBATCH --error=logs/ssl-scale-check_%j.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250GB
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1

# Phase 2 diagnostic: short SSL on PD cache.
# Then: sbatch src/launch_ssl_scale_postcheck.sh

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

mkdir -p "${HF_DATASETS_CACHE:-/mmfs1/scratch/${USER}/hf_datasets_cache}"
mkdir -p "${HF_HOME:-/mmfs1/scratch/${USER}/hf_home}"

CACHE_ROOT="${PVI_CACHE_ROOT_IMPEDANCE:-/mmfs1/scratch/${USER}/pvi_cache/v1_impedance}_main_disjoint"
LOGDIR="${LOGDIR:-debug-ssl-scale-check}"

python -u -m src.scripts.check_cache \
  --cache-root "${CACHE_ROOT}" \
  --input-mode impedance \
  --branch main \
  --split-mode disjoint

python -u -m src.foundation.ssl_pretrain \
  --input-mode impedance \
  --arch mae \
  --ssl-arch mae \
  --branch main \
  --split-mode disjoint \
  --max-epochs 5 \
  --probe-every 0 \
  --batch-size 256 \
  --cache-root "${CACHE_ROOT}" \
  --cache-num-workers 8 \
  --logdir "${LOGDIR}"

deactivate
