#!/bin/bash
# Require a valid Parquet cache for INPUT_MODE; build if missing.
# Usage: ensure_parquet_cache.sh [input_mode]
set -euo pipefail

INPUT_MODE="${1:-impedance}"
REPO_ROOT="${REPO_ROOT:-/mmfs1/projects/ece_bst/lsanc68/fundational_pvi}"
# shellcheck source=/dev/null
source "${REPO_ROOT}/env/cluster.env"

if [[ "${INPUT_MODE}" == "impedance" ]]; then
  export PVI_CACHE_ROOT="${PVI_CACHE_ROOT_IMPEDANCE:-/mmfs1/scratch/${USER}/pvi_cache/v1_impedance}"
else
  export PVI_CACHE_ROOT="${PVI_CACHE_ROOT:-/mmfs1/scratch/${USER}/pvi_cache/v1}"
fi

if python -u -m src.scripts.check_cache --cache-root "${PVI_CACHE_ROOT}" --input-mode "${INPUT_MODE}"; then
  return 0 2>/dev/null || exit 0
fi

echo "Parquet cache invalid; building ${INPUT_MODE} cache at ${PVI_CACHE_ROOT}..."
mkdir -p "${PVI_CACHE_ROOT}"
python -u -m src.scripts.build_pvi_cache \
  --cache-root "${PVI_CACHE_ROOT}" \
  --input-mode "${INPUT_MODE}" \
  --output-mode waveform \
  --mask-key mask05 \
  --force

python -u -m src.scripts.check_cache --cache-root "${PVI_CACHE_ROOT}" --input-mode "${INPUT_MODE}"
