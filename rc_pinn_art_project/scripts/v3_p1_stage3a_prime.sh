#!/usr/bin/env bash
# Stage 3a' — 防遗忘重训 Z=3–4（低 Z 混合 + 早停）
#
# 用法:
#   cd DiracNet_V3/rc_pinn_art_project && export PYTHONPATH=.
#   bash scripts/v3_p1_stage3a_prime.sh          # ground → full
#   bash scripts/v3_p1_stage3a_prime.sh full-only  # 仅从 p1lowz 跑 full（跳过 ground）
#
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

CFG=configs/v3_stage_a_z1_26_n10.yaml
RESUME=checkpoints/v3_stage_a_z1_26_n10_p1lowz/stage_a_last.msgpack
TAG=p1z3_4_prime_v2
PRIME=scripts/v3_train_stage_a_prime.py

_check_gpu() {
  python - <<'PY' || { echo "WARN: JAX not on GPU"; return 0; }
import jax
print("backend:", jax.default_backend(), jax.devices())
PY
}

_run_eval() {
  local ckpt="$1"
  local out="$2"
  echo ""
  echo "========== 快速评估: $ckpt =========="
  python scripts/v3_diag_h_potential.py --config "$CFG" --ckpt "$ckpt" | tee "$out/diag_h.txt"
  python scripts/v3_eval_excitation_vs_nist.py \
    --config "$CFG" --ckpt "$ckpt" \
    --manifest data_cache/manifest_z_3_4.parquet \
    --out-dir "$out/layer2b_z_3_4" || true
  python scripts/v3_eval_excitation_vs_nist.py \
    --config "$CFG" --ckpt "$ckpt" \
    --manifest data_cache/manifest_z_le_2.parquet \
    --out-dir "$out/layer2b_z_le_2" || true
}

MODE="${1:-all}"

if [[ ! -f "$RESUME" ]]; then
  echo "ERROR: 需要 p1lowz 权重: $RESUME"
  exit 1
fi

_check_gpu

# 确保 scope manifest 存在
python - <<'PY'
import pandas as pd
from pathlib import Path
df = pd.read_parquet("data_cache/manifest_nist_z1_26_n10.parquet")
for name, m in [
    ("manifest_z_le_2.parquet", df.Z <= 2),
    ("manifest_z_3_4.parquet", (df.Z >= 3) & (df.Z <= 4)),
]:
    Path("data_cache").mkdir(exist_ok=True)
    df[m].reset_index(drop=True).to_parquet(f"data_cache/{name}", index=False)
    print(f"  data_cache/{name}: {m.sum()} rows")
PY

if [[ "$MODE" != "full-only" ]]; then
  echo ""
  echo "========== Stage 3a' Phase 1: Z=3–4 闭壳基态 (600 ep) =========="
  python "$PRIME" --config "$CFG" --resume "$RESUME" \
    --phase ground --tag "$TAG" \
    --anchor-frac 0.30 --lr-mult 0.3 --check-every 10 \
    --warmup-epochs 3 --max-h-exc-meV 200 --max-v-r1-diff 0.10 --patience 6

  GROUND_CKPT="checkpoints/v3_stage_a_z1_26_n10_${TAG}_ground/best_anchor.msgpack"
  if [[ ! -f "$GROUND_CKPT" ]]; then
    GROUND_CKPT="checkpoints/v3_stage_a_z1_26_n10_${TAG}_ground/stage_a_last.msgpack"
  fi
  _run_eval "$GROUND_CKPT" "logs/${TAG}_ground_eval"
  RESUME="$GROUND_CKPT"
fi

echo ""
echo "========== Stage 3a' Phase 2: Z=3–4 全组态 (800 ep) =========="
python "$PRIME" --config "$CFG" --resume "$RESUME" \
  --phase full --tag "$TAG" \
  --anchor-frac 0.30 --lr-mult 0.3 --check-every 10 \
  --warmup-epochs 3 --max-h-exc-meV 200 --max-v-r1-diff 0.10 --patience 6

FULL_DIR="checkpoints/v3_stage_a_z1_26_n10_${TAG}_full"
BEST="$FULL_DIR/best_anchor.msgpack"
LAST="$FULL_DIR/stage_a_last.msgpack"
CKPT="$BEST"
[[ -f "$CKPT" ]] || CKPT="$LAST"

_run_eval "$CKPT" "logs/${TAG}_full_eval"

echo ""
echo "=============================================="
echo "Stage 3a' 完成"
echo "  推荐评估 ckpt: $CKPT"
echo "  完整套件: bash scripts/v3_eval_stage3a_suite.sh"
echo "    (改 CKPT 路径为 $CKPT)"
echo "  训练日志: logs/v3_stage_a_z1_26_n10_${TAG}_*/"
echo "  早停记录: .../anchor_guard.jsonl"
echo "=============================================="
