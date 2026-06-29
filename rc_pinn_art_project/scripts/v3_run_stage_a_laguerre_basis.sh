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
#   bash scripts/v3_run_stage_a_laguerre_basis.sh              # default 2000 epochs, _b config
#   bash scripts/v3_run_stage_a_laguerre_basis.sh 500          # override epochs (still _b)
#   bash scripts/v3_run_stage_a_laguerre_basis.sh 200 v1       # tag=v1 (still _b)
#   bash scripts/v3_run_stage_a_laguerre_basis.sh 5000 c       # config=_c (Step C, 260 rows)
#   bash scripts/v3_run_stage_a_laguerre_basis.sh 5000 d d5k   # config=_d (dual trunk + log-r), tag=d5k
#   bash scripts/v3_run_stage_a_laguerre_basis.sh 5000 b myrun # full path to any config (no _ suffix needed)
#
# Config selection rules (3rd positional arg "CONFIG"):
#   * unset / "b"  → configs/v3_stage_a_laguerre_basis.yaml       (15-row Step B baseline)
#   * "c"          → configs/v3_stage_a_laguerre_basis_c.yaml     (260-row Step C sweep)
#   * "d"          → configs/v3_stage_a_laguerre_basis_d.yaml     (dual trunk + log-r — §13.1 / §13.2)
#   * anything else → used as the literal config path (e.g. configs/my_custom.yaml)
#
# After the script prints the PID you can `tail -f $LOG` to follow it.

set -euo pipefail

EPOCHS="${1:-2000}"
TAG="${2:-}"
CONFIG_KEY="${3:-b}"  # 'b' (default), 'c', 'd', or any literal config path

cd "$(dirname "$0")/.."  # rc_pinn_art_project

# Resolve the config path from CONFIG_KEY.
case "$CONFIG_KEY" in
    b|"")
        CONFIG_PATH="configs/v3_stage_a_laguerre_basis.yaml"
        ;;
    c)
        CONFIG_PATH="configs/v3_stage_a_laguerre_basis_c.yaml"
        ;;
    d)
        CONFIG_PATH="configs/v3_stage_a_laguerre_basis_d.yaml"
        ;;
    configs/*)
        CONFIG_PATH="$CONFIG_KEY"
        ;;
    *)
        echo "[arg] unknown CONFIG_KEY: $CONFIG_KEY (use 'b', 'c', 'd', or a configs/*.yaml path)" >&2
        exit 1
        ;;
esac

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "[arg] config not found: $CONFIG_PATH" >&2
    exit 1
fi

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
echo "  config = $CONFIG_PATH"

# `cd` into project; nohup with `&` so the shell returns immediately.
# The `--tag` flag is forwarded to v3_train_stage_a_laguerre_basis.py so
# the trainer writes ckpts/logs into the tag-suffixed subdirs.
nohup python scripts/v3_train_stage_a_laguerre_basis.py \
    --config "$CONFIG_PATH" \
    --epochs "$EPOCHS" \
    --tag "${TAG:-}" \
    > "$LOG" 2>&1 &

PID=$!
echo "[$(date '+%F %T')] Launched PID=$PID"
echo "  tail -f $LOG"