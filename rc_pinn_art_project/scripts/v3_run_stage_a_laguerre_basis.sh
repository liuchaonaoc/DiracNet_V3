#!/usr/bin/env bash
# Launch the full Stage-A Laguerre training run on GPU.
#
# The previous manual nohup attempt failed because the log dir
# `logs/v3_stage_a_laguerre_basis/` did not exist; bash's stdout
# redirection died before the training started, leaving no log file
# and an `Exit 1` from the foregrounded job.  This wrapper pre-creates
# the directory and writes to an absolute, time-stamped log path.
#
# Usage:
#   bash scripts/v3_run_stage_a_laguerre_basis.sh              # default 2000 epochs
#   bash scripts/v3_run_stage_a_laguerre_basis.sh 500          # override epochs
#   bash scripts/v3_run_stage_a_laguerre_basis.sh 200 v1       # tag=v1
#
# After the script prints the PID you can `tail -f $LOG` to follow it.

set -euo pipefail

EPOCHS="${1:-2000}"
TAG="${2:-}"

cd "$(dirname "$0")/.."  # rc_pinn_art_project

LOG_ROOT="$(pwd)/logs/v3_stage_a_laguerre_basis"
mkdir -p "$LOG_ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_ROOT/train_${STAMP}.log"

if [[ -n "${TAG:-}" ]]; then
    CKPT_DIR="$(pwd)/checkpoints/v3_stage_a_laguerre_basis_${TAG}"
    LOG_DIR="${LOG_ROOT}_${TAG}"
else
    CKPT_DIR="$(pwd)/checkpoints/v3_stage_a_laguerre_basis"
    LOG_DIR="$LOG_ROOT"
fi
mkdir -p "$LOG_DIR"

echo "[$(date '+%F %T')] Launching Laguerre training:"
echo "  epochs = $EPOCHS"
echo "  log    = $LOG"
echo "  ckpt   = $CKPT_DIR"
echo "  config = configs/v3_stage_a_laguerre_basis.yaml"

# `cd` into project; nohup with `&` so the shell returns immediately.
# The `--tag` flag is forwarded to v3_train_stage_a_laguerre_basis.py so
# the trainer writes ckpts/logs into the tag-suffixed subdirs.
nohup python scripts/v3_train_stage_a_laguerre_basis.py \
    --config configs/v3_stage_a_laguerre_basis.yaml \
    --epochs "$EPOCHS" \
    --tag "${TAG:-}" \
    > "$LOG" 2>&1 &

PID=$!
echo "[$(date '+%F %T')] Launched PID=$PID"
echo "  tail -f $LOG"