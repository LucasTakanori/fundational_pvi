#!/bin/bash
# Fast local test runner (no SLURM). Same focused targets as launch_test.sh.
set -euo pipefail

REPO_ROOT="/mmfs1/projects/ece_bst/lsanc68/fundational_pvi"
cd "${REPO_ROOT}"

source "${REPO_ROOT}/env/cluster.env"
source "${REPO_ROOT}/.venv/bin/activate"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

TEST_TARGETS="${TEST_TARGETS:-tests/test_subject_batch.py tests/test_rope.py tests/test_adversarial.py tests/test_decomposition.py tests/test_pvi_cache.py tests/test_model_factory.py tests/test_smoke.py tests/test_milestone1.py tests/test_milestone3.py tests/test_mae_transformer.py tests/test_experiments.py}"

echo "Running: pytest ${TEST_TARGETS} -q --tb=short"
pytest ${TEST_TARGETS} -q --tb=short
