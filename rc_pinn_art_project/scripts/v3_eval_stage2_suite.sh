#!/usr/bin/env bash
# Stage 2 分层评估套件 — 见 docs/STAGE2_EVALUATION_PLAN.md
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

CFG=configs/v3_stage_a_z1_26_n10.yaml
CKPT=checkpoints/v3_stage_a_z1_26_n10_p1z8_full/stage_a_last.msgpack
MAN_FULL=data_cache/manifest_nist_z1_26_n10.parquet
SUITE=logs/v3_stage_a_z1_26_n10_p1z8_full_eval/stage2_suite

mkdir -p "${SUITE}" data_cache

echo "==> 0) 生成 scope 子集 manifest"
python - <<'PY'
import pandas as pd
from pathlib import Path
df = pd.read_parquet("data_cache/manifest_nist_z1_26_n10.parquet")
for name, m in [
    ("manifest_z_le_2.parquet", df.Z <= 2),
    ("manifest_z_le_8.parquet", df.Z <= 8),
    ("manifest_z_gt_8.parquet", df.Z > 8),
]:
    out = Path("data_cache") / name
    sub = df[m].reset_index(drop=True)
    sub.to_parquet(out, index=False)
    print(f"  {out}: {len(sub)} rows")
PY

if [[ ! -f "${CKPT}" ]]; then
  echo "ERROR: missing ${CKPT}"
  exit 1
fi

echo ""
echo "==> 1) H 势诊断 (target)"
python scripts/v3_diag_h_potential.py --config "${CFG}" --ckpt "${CKPT}" \
  | tee "${SUITE}/diag_h.txt"

echo ""
echo "==> 2) 基线对比 (氢锚点)"
python scripts/v3_compare_stage2_baselines.py \
  --target "${CKPT}" \
  --baselines p1lowz,p1z8_ground,round2,round1 \
  --out "${SUITE}/BASELINE_COMPARISON.md"

echo ""
echo "==> 3) Gate A — S0 (Z<=2)"
python scripts/v3_gate_analytic.py --config "${CFG}" --ckpt "${CKPT}" \
  --manifest data_cache/manifest_z_le_2.parquet
mkdir -p "${SUITE}/gate_z_le_2"
cp -f logs/v3_stage_a_z1_26_n10/GATE_A_REPORT.md "${SUITE}/gate_z_le_2/"
cp -f logs/v3_stage_a_z1_26_n10/gate_a_detailed.csv "${SUITE}/gate_z_le_2/"

echo ""
echo "==> 4) Gate A — S1 (Z<=8) **主验收**"
python scripts/v3_gate_analytic.py --config "${CFG}" --ckpt "${CKPT}" \
  --manifest data_cache/manifest_z_le_8.parquet
mkdir -p "${SUITE}/gate_z_le_8"
cp -f logs/v3_stage_a_z1_26_n10/GATE_A_REPORT.md "${SUITE}/gate_z_le_8/"
cp -f logs/v3_stage_a_z1_26_n10/gate_a_detailed.csv "${SUITE}/gate_z_le_8/"

echo ""
echo "==> 5) Layer-2b — S0 (Z<=2)"
python scripts/v3_eval_excitation_vs_nist.py \
  --config "${CFG}" --ckpt "${CKPT}" \
  --manifest data_cache/manifest_z_le_2.parquet \
  --out-dir "${SUITE}/layer2b_z_le_2"

echo ""
echo "==> 6) Layer-2b — S1 (Z<=8) **主验收**"
python scripts/v3_eval_excitation_vs_nist.py \
  --config "${CFG}" --ckpt "${CKPT}" \
  --manifest data_cache/manifest_z_le_8.parquet \
  --out-dir "${SUITE}/layer2b_z_le_8"

echo ""
echo "==> 7) 按 Z / l 拆解 (S1)"
python scripts/v3_summarize_excitation_by_Zl.py \
  --csv "${SUITE}/layer2b_z_le_8/excitation_vs_nist.csv" \
  --out "${SUITE}/summary_z_l.md"

echo ""
echo "==> 8) 训练曲线摘要"
for tag in p1lowz p1z8_ground p1z8_full; do
  h="logs/v3_stage_a_z1_26_n10_${tag}/history.csv"
  if [[ -f "${h}" ]]; then
    echo "--- ${tag} (first / mid / last epoch) ---"
    awk -F, 'NR==1{print; next} NR==2 || NR==int(NR/2) || NR==NR' "${h}" 2>/dev/null | tail -4
  fi
done | tee "${SUITE}/training_summary.txt"

echo ""
echo "=============================================="
echo "Stage 2 评估套件完成。"
echo "  主报告目录: ${SUITE}/"
echo "  必读: layer2b_z_le_8/EXCITATION_VS_NIST.md"
echo "        gate_z_le_8/GATE_A_REPORT.md"
echo "        BASELINE_COMPARISON.md"
echo "  全 manifest 已有: logs/v3_stage_a_z1_26_n10_p1z8_full_eval/ (S4 外推)"
echo "  交 agent 分析时说: Stage 2 suite 跑完了"
echo "=============================================="
