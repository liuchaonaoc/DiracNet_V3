#!/usr/bin/env bash
# 路径 B (anchor_vprior_zeff) 评估套件
# 用法: bash scripts/v3_eval_zeff_suite.sh [CKPT路径]
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

CFG=configs/v3_stage_a_z1_26_n10_zeff.yaml
_DEFAULT_ZEFF="checkpoints/v3_stage_a_z1_26_n10_zeff/stage_a_last.msgpack"
CKPT="${1:-${_DEFAULT_ZEFF}}"
SUITE=logs/stage_a_z1_26_n10_zeff_eval/suite

mkdir -p "${SUITE}" data_cache

if [[ ! -f "${CKPT}" ]]; then
  echo "ERROR: checkpoint not found: ${CKPT}"
  exit 1
fi

echo "==> 路径 B Checkpoint: ${CKPT}"

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
echo "==> 2) Gate — Z=3-4 **l=0 关键检查**"
python scripts/v3_gate_analytic.py --config "${CFG}" --ckpt "${CKPT}" \
  --manifest data_cache/manifest_z_3_4.parquet
mkdir -p "${SUITE}/gate_z_3_4"
cp -f logs/v3_stage_a_z1_26_n10_zeff/GATE_A_REPORT.md "${SUITE}/gate_z_3_4/" 2>/dev/null || true
cp -f logs/v3_stage_a_z1_26_n10_zeff/gate_a_detailed.csv "${SUITE}/gate_z_3_4/" 2>/dev/null || true

echo ""
echo "==> 3) Gate — Z<=2 锚点"
python scripts/v3_gate_analytic.py --config "${CFG}" --ckpt "${CKPT}" \
  --manifest data_cache/manifest_z_le_2.parquet
mkdir -p "${SUITE}/gate_z_le_2"
cp -f logs/v3_stage_a_z1_26_n10_zeff/GATE_A_REPORT.md "${SUITE}/gate_z_le_2/" 2>/dev/null || true
cp -f logs/v3_stage_a_z1_26_n10_zeff/gate_a_detailed.csv "${SUITE}/gate_z_le_2/" 2>/dev/null || true

echo ""
echo "==> 4) Gate — Z<=4 累计"
python scripts/v3_gate_analytic.py --config "${CFG}" --ckpt "${CKPT}" \
  --manifest data_cache/manifest_z_le_4.parquet
mkdir -p "${SUITE}/gate_z_le_4"
cp -f logs/v3_stage_a_z1_26_n10_zeff/GATE_A_REPORT.md "${SUITE}/gate_z_le_4/" 2>/dev/null || true
cp -f logs/v3_stage_a_z1_26_n10_zeff/gate_a_detailed.csv "${SUITE}/gate_z_le_4/" 2>/dev/null || true

echo ""
echo "==> 5) Layer-2b — Z=3-4 (NIST 激发能)"
python scripts/v3_eval_excitation_vs_nist.py \
  --config "${CFG}" --ckpt "${CKPT}" \
  --manifest data_cache/manifest_z_3_4.parquet \
  --out-dir "${SUITE}/layer2b_z_3_4"

echo ""
echo "==> 6) Layer-2b — Z<=2"
python scripts/v3_eval_excitation_vs_nist.py \
  --config "${CFG}" --ckpt "${CKPT}" \
  --manifest data_cache/manifest_z_le_2.parquet \
  --out-dir "${SUITE}/layer2b_z_le_2"

echo ""
echo "==> 7) Z=3-4 按 l 拆解"
python scripts/v3_summarize_excitation_by_Zl.py \
  --csv "${SUITE}/layer2b_z_3_4/excitation_vs_nist.csv" \
  --out "${SUITE}/summary_z3_4_z_l.md"

echo ""
echo "DONE. Results in ${SUITE}"
