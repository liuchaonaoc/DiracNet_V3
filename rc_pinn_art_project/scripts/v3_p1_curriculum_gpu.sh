#!/usr/bin/env bash
# P1 课程式 Stage 2（GPU）：Z≤8 基态 → Z≤8 全组态 → 评估
# 从 p1lowz（Z≤2 已训好）热启动，避免再次混训 Z=1..26。
#
# 用法：
#   cd DiracNet_V3/rc_pinn_art_project && export PYTHONPATH=.
#   bash scripts/v3_p1_curriculum_gpu.sh          # 训练 + 评估全流程
#   bash scripts/v3_p1_curriculum_gpu.sh train  # 仅训练
#   bash scripts/v3_p1_curriculum_gpu.sh eval   # 仅评估（需已有 ckpt）
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

CFG=configs/v3_stage_a_z1_26_n10.yaml
RESUME_Z2=checkpoints/v3_stage_a_z1_26_n10_p1lowz/stage_a_last.msgpack
TAG_A=p1z8_ground
TAG_B=p1z8_full
CKPT_A=checkpoints/v3_stage_a_z1_26_n10_${TAG_A}/stage_a_last.msgpack
CKPT_B=checkpoints/v3_stage_a_z1_26_n10_${TAG_B}/stage_a_last.msgpack
LOG_EVAL=logs/v3_stage_a_z1_26_n10_${TAG_B}_eval

_mode="${1:-all}"

_check_gpu() {
  python - <<'PY'
import jax
print("JAX backend:", jax.default_backend())
print("Devices:", jax.devices())
if jax.default_backend() != "gpu":
    raise SystemExit(
        "ERROR: JAX 未使用 GPU。请设置 CUDA 可见并安装 jax[cuda12]，"
        "或 export JAX_PLATFORMS=cuda 后重试。"
    )
PY
}

_train() {
  if [[ ! -f "${RESUME_Z2}" ]]; then
    echo "ERROR: 缺少 Stage-1 权重: ${RESUME_Z2}"
    echo "请先完成 Z≤2 短训 (--tag p1lowz)。"
    exit 1
  fi

  echo "========== Stage 2a: Z≤8 闭壳基态 only (40 rows) =========="
  python scripts/v3_train_stage_a.py \
    --config "${CFG}" \
    --resume "${RESUME_Z2}" \
    --z-max 8 --ground-only \
    --epochs 800 --batch-size 32 \
    --tag "${TAG_A}"

  echo ""
  echo "========== Stage 2b: Z≤8 全组态 (2435 rows) =========="
  python scripts/v3_train_stage_a.py \
    --config "${CFG}" \
    --resume "${CKPT_A}" \
    --z-max 8 \
    --epochs 2000 --batch-size 32 \
    --tag "${TAG_B}"

  echo ""
  echo "训练完成。最终 ckpt: ${CKPT_B}"
}

_eval() {
  if [[ ! -f "${CKPT_B}" ]]; then
    echo "ERROR: 缺少 Stage-2b ckpt: ${CKPT_B}"
    exit 1
  fi

  mkdir -p "${LOG_EVAL}"

  echo "========== 诊断：H / He 价区势与激发能 =========="
  python scripts/v3_diag_h_potential.py \
    --config "${CFG}" \
    --ckpt "${CKPT_B}" | tee "${LOG_EVAL}/diag_h_he.txt"

  echo ""
  echo "========== Gate A（manifest 全量，前向较慢）=========="
  python scripts/v3_gate_analytic.py \
    --config "${CFG}" \
    --ckpt "${CKPT_B}" \
    --manifest data_cache/manifest_nist_z1_26_n10.parquet

  # Gate 默认写到 config 的 log_dir；复制一份到 eval 目录便于你发我分析
  if [[ -f logs/v3_stage_a_z1_26_n10/GATE_A_REPORT.md ]]; then
    cp -f logs/v3_stage_a_z1_26_n10/GATE_A_REPORT.md "${LOG_EVAL}/" 2>/dev/null || true
    cp -f logs/v3_stage_a_z1_26_n10/gate_a_detailed.csv "${LOG_EVAL}/" 2>/dev/null || true
  fi

  echo ""
  echo "========== Layer-2b：激发能 vs NIST（全 manifest，耗时最长）=========="
  python scripts/v3_eval_excitation_vs_nist.py \
    --config "${CFG}" \
    --ckpt "${CKPT_B}" \
    --out-dir "${LOG_EVAL}"

  echo ""
  echo "评估产物目录: ${LOG_EVAL}/"
  echo "  - EXCITATION_VS_NIST.md, excitation_vs_nist.csv, metrics.json"
  echo "  - diag_h_he.txt"
  ls -la "${LOG_EVAL}/" || true
}

case "${_mode}" in
  train) _check_gpu; _train ;;
  eval)  _eval ;;
  all)   _check_gpu; _train; _eval ;;
  *)
    echo "Usage: $0 [train|eval|all]"
    exit 1
    ;;
esac
