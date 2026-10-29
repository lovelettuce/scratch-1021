#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
# Use the owner of grid, rather than an unrelated conda earlier on PATH.
GRID_PYTHON="${GRID_PYTHON:-/home/yangyujie/data/miniconda3/envs/grid/bin/python}"
if [[ ! -x "$GRID_PYTHON" ]]; then
  echo "grid Python is missing: $GRID_PYTHON" >&2
  exit 2
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 TF_CPP_MIN_LOG_LEVEL=3
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
exec "$GRID_PYTHON" -m src.experiments.rq_cross_validation "$@"
