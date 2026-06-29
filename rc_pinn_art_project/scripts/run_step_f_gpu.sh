#!/usr/bin/env bash
# Launch Step F (§13.10-A sparse forcing) training on GPU.
#
# Usage: in IDE terminal (NOT sandbox), then `tail -f $LOG` to follow.
#
# Prerequisites (one-time per session):
#   1. Make sure no CPU/GPU instance is still running: pgrep -af "tag f5k"
#      If you find one, kill it: kill <pid>
#   2. CUDA is visible to your shell:
#        nvidia-smi -L    # should list GPU(s)
#        echo $CUDA_VISIBLE_DEVICES
#   3. jax[cuda12] is installed in the active venv (matching the CUDA driver):
#        python -c "import jax; print(jax.devices())"   # should show CudaDevice(id=0)

set -euo pipefail

cd "$(dirname "$0")/.."  # rc_pinn_art_project

EPOCHS="${1:-5000}"
TAG="${2:-f5k_gpu}"

# Force JAX to use CUDA.  Without this, jax[cuda] may still pick CPU if
# the platform list isn't set in some installs.
export JAX_PLATFORMS=cuda

LOG_ROOT="$(pwd)/logs/v3_stage_a_laguerre_basis_f"
mkdir -p "$LOG_ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_ROOT/train_${TAG}_${STAMP}.log"

CKPT_DIR="$(pwd)/checkpoints/v3_stage_a_laguerre_basis_f"
mkdir -p "$CKPT_DIR"

echo "[$(date '+%F %T')] Launching Step F (§13.10-A sparse forcing) training on GPU:"
echo "  epochs    = $EPOCHS"
echo "  tag       = $TAG"
echo "  log       = $LOG"
echo "  ckpt dir  = $CKPT_DIR"
echo "  config    = configs/v3_stage_a_laguerre_basis_f.yaml"
echo "  devices   = $(python -c 'import jax; print(jax.devices())')"

nohup python scripts/v3_train_stage_a_laguerre_basis.py \
    --config configs/v3_stage_a_laguerre_basis_f.yaml \
    --epochs "$EPOCHS" \
    --tag "$TAG" \
    > "$LOG" 2>&1 &

PID=$!
echo "[$(date '+%F %T')] Launched PID=$PID"
echo "  tail -f $LOG"
echo "  kill when done: kill $PID"
