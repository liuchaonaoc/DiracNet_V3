#!/usr/bin/env bash
# Launch Step H (§13.C coeff anchor + §13.B analytic dQ) training on GPU.
#
# Step H reverts the coeff head to the legacy LaguerreCoeffHead
# (use_hybrid_head=false) but keeps §13.1 dual trunk + §13.2 log-r + §13.9
# branch capacity, and adds:
#   * Direction C: coeff_anchor loss (pull n>=8 coeffs toward analytic init)
#   * Direction B: analytic kinetic-balance dQ (no jnp.gradient(Q) noise)
#
# Usage (in IDE terminal, NOT sandbox):
#   bash scripts/run_step_h_gpu.sh 5000 h5k_gpu
#   tail -f logs/v3_stage_a_laguerre_basis_h/train_h5k_gpu_*.log
#
# Prerequisites:
#   1. No stale run:           pgrep -af "tag h5k"
#   2. CUDA visible:           nvidia-smi -L ; python -c "import jax; print(jax.devices())"
#   (No node table needed — Step H uses the legacy head, not the Hybrid head.)

set -euo pipefail

cd "$(dirname "$0")/.."  # rc_pinn_art_project

EPOCHS="${1:-5000}"
TAG="${2:-h5k_gpu}"

export JAX_PLATFORMS=cuda

LOG_ROOT="$(pwd)/logs/v3_stage_a_laguerre_basis_h"
mkdir -p "$LOG_ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_ROOT/train_${TAG}_${STAMP}.log"

CKPT_DIR="$(pwd)/checkpoints/v3_stage_a_laguerre_basis_h"
mkdir -p "$CKPT_DIR"

echo "[$(date '+%F %T')] Launching Step H (§13.C anchor + §13.B analytic dQ) on GPU:"
echo "  epochs    = $EPOCHS"
echo "  tag       = $TAG"
echo "  log       = $LOG"
echo "  ckpt dir  = $CKPT_DIR"
echo "  config    = configs/v3_stage_a_laguerre_basis_h.yaml"
echo "  devices   = $(python -c 'import jax; print(jax.devices())')"

nohup python scripts/v3_train_stage_a_laguerre_basis.py \
    --config configs/v3_stage_a_laguerre_basis_h.yaml \
    --epochs "$EPOCHS" \
    --tag "$TAG" \
    > "$LOG" 2>&1 &

PID=$!
echo "[$(date '+%F %T')] Launched PID=$PID"
echo "  tail -f $LOG"
echo "  kill when done: kill $PID"
