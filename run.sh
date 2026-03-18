#!/usr/bin/env bash
# run.sh – Build and execute the GPU-Accelerated Iris ML Pipeline.
#
# Usage:
#   ./run.sh [OPTIONS]
#
# All arguments are forwarded to iris_gpu.  When called with no arguments
# the full pipeline (kmeans + knn + pca) runs on iris/iris.data with
# default hyperparameters.
#
# Examples:
#   ./run.sh                                      # full pipeline, defaults
#   ./run.sh --algorithm kmeans --k 3             # only K-Means
#   ./run.sh --algorithm pca    --components 3    # only PCA (3 components)
#   ./run.sh --algorithm knn    --knn-k 7         # only KNN with k=7
#   ./run.sh --data iris/bezdekIris.data          # use alternate data file

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --------------------------------------------------------------------------
# 1. Ensure output directory exists
# --------------------------------------------------------------------------
mkdir -p output

# --------------------------------------------------------------------------
# 2. Build
# --------------------------------------------------------------------------
echo "=== Building iris_gpu ==="
make -j"$(nproc)" all
echo ""

# --------------------------------------------------------------------------
# 3. Run
# --------------------------------------------------------------------------
echo "=== Running GPU Iris ML Pipeline ==="

if [ "$#" -eq 0 ]; then
  ./iris_gpu \
    --data       iris/iris.data \
    --algorithm  all \
    --k          3 \
    --knn-k      5 \
    --components 2 \
    --iterations 300 \
    --seed       42 \
    --output     output
else
  ./iris_gpu "$@"
fi

echo ""
echo "=== Output files ==="
ls -lh output/
