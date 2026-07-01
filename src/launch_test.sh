#!/bin/bash
#SBATCH --job-name=pvi-pytest
#SBATCH --output=pytest_%j.out
#SBATCH --error=pytest_%j.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=00:30:00

set -euo pipefail

REPO_ROOT="/mmfs1/projects/ece_bst/lsanc68/fundational_pvi"
cd "${REPO_ROOT}"

# shellcheck source=/dev/null
source "${REPO_ROOT}/env/cluster.env"
source "${REPO_ROOT}/.venv/bin/activate"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Focused suite: model factory + foundation milestones (fast CPU tests).
# Full suite: pytest tests/ -q
TEST_TARGETS="${TEST_TARGETS:-tests/test_model_factory.py tests/test_smoke.py tests/test_milestone1.py tests/test_milestone3.py tests/test_mae_transformer.py tests/test_experiments.py}"

echo "Running: pytest ${TEST_TARGETS} -q"
pytest ${TEST_TARGETS} -q --tb=short

deactivate
