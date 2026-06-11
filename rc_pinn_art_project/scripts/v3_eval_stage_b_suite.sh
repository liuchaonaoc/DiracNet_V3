#!/usr/bin/env bash
# Stage B 评估套件（与 Stage 3a 对比：检查 l=0 瓶颈是否改善）
# 用法: bash scripts/v3_eval_stage_b_suite.sh [CKPT路径]
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

CFG=configs/v3_stage_b_z1_26_n10.yaml
_DEFAULT_B1="checkpoints/v3_stage_b_z1_26_n10/stage_b_last.msgpack"
CKPT="${1:-${_DEFAULT_B1}}"
SUITE=logs/stage_b_z1_26_n10_eval/suite

mkdir -p "${SUITE}" data_cache

if [[ ! -f "${CKPT}" ]]; then
  echo "ERROR: checkpoint not found: ${CKPT}"
  exit 1
fi

echo "==> Stage B Checkpoint: ${CKPT}"

# 确保子 manifest 存在
python - <<'PY'
import pandas as pd
from pathlib import Path
df = pd.read_parquet("data_cache/manifest_nist_z1_26_n10.parquet")
for name, m in [
    ("manifest_z_le_2.parquet", df.Z <= 2),
    ("manifest_z_3_4.parquet", (df.Z >= 3) & (df.Z <= 4)),
    ("manifest_z_le_4.parquet", df.Z <= 4),
]:
    out = Path("data_cache") / name
    sub = df[m].reset_index(drop=True)
    sub.to_parquet(out, index=False)
    print(f"  {out}: {len(sub)} rows")
PY

echo ""
echo "==> 1) 氢锚点 (防回退)"
python scripts/v3_diag_h_potential.py --config "${CFG}" --ckpt "${CKPT}" \
  | tee "${SUITE}/diag_h.txt"

echo ""
echo "==> 2) Gate — Z<=2 锚点"
python scripts/v3_gate_analytic.py --config "${CFG}" --ckpt "${CKPT}" \
  --manifest data_cache/manifest_z_le_2.parquet
mkdir -p "${SUITE}/gate_z_le_2"
cp -f logs/v3_stage_b_z1_26_n10/GATE_A_REPORT.md "${SUITE}/gate_z_le_2/" 2>/dev/null || true
cp -f logs/v3_stage_b_z1_26_n10/gate_a_detailed.csv "${SUITE}/gate_z_le_2/" 2>/dev/null || true

echo ""
echo "==> 3) Gate — Z=3-4 主验收 (l=0 瓶颈检查)"
python scripts/v3_gate_analytic.py --config "${CFG}" --ckpt "${CKPT}" \
  --manifest data_cache/manifest_z_3_4.parquet
mkdir -p "${SUITE}/gate_z_3_4"
cp -f logs/v3_stage_b_z1_26_n10/GATE_A_REPORT.md "${SUITE}/gate_z_3_4/" 2>/dev/null || true
cp -f logs/v3_stage_b_z1_26_n10/gate_a_detailed.csv "${SUITE}/gate_z_3_4/" 2>/dev/null || true

echo ""
echo "==> 4) Gate — Z<=4 累计"
python scripts/v3_gate_analytic.py --config "${CFG}" --ckpt "${CKPT}" \
  --manifest data_cache/manifest_z_le_4.parquet
mkdir -p "${SUITE}/gate_z_le_4"
cp -f logs/v3_stage_b_z1_26_n10/GATE_A_REPORT.md "${SUITE}/gate_z_le_4/" 2>/dev/null || true
cp -f logs/v3_stage_b_z1_26_n10/gate_a_detailed.csv "${SUITE}/gate_z_le_4/" 2>/dev/null || true

echo ""
echo "==> 5) Layer-2b — Z=3-4"
python scripts/v3_eval_excitation_vs_nist.py \
  --config "${CFG}" --ckpt "${CKPT}" \
  --manifest data_cache/manifest_z_3_4.parquet \
  --out-dir "${SUITE}/layer2b_z_3_4"

echo ""
echo "==> 6) Layer-2b — Z<=4 累计"
python scripts/v3_eval_excitation_vs_nist.py \
  --config "${CFG}" --ckpt "${CKPT}" \
  --manifest data_cache/manifest_z_le_4.parquet \
  --out-dir "${SUITE}/layer2b_z_le_4"

echo ""
echo "==> 7) Layer-2b — Z<=2 锚点"
python scripts/v3_eval_excitation_vs_nist.py \
  --config "${CFG}" --ckpt "${CKPT}" \
  --manifest data_cache/manifest_z_le_2.parquet \
  --out-dir "${SUITE}/layer2b_z_le_2"

echo ""
echo "==> 8) Z=3-4 按 l 拆解 (关键 l=0 检查)"
python scripts/v3_summarize_excitation_by_Zl.py \
  --csv "${SUITE}/layer2b_z_3_4/excitation_vs_nist.csv" \
  --out "${SUITE}/summary_z3_4_z_l.md"

echo ""
echo "==> 9) Z<=4 按 l 拆解"
python scripts/v3_summarize_excitation_by_Zl.py \
  --csv "${SUITE}/layer2b_z_le_4/excitation_vs_nist.csv" \
  --out "${SUITE}/summary_z_le_4_z_l.md"

echo ""
echo "DONE. Results in ${SUITE}"
