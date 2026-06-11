#!/usr/bin/env bash
# Stage 3a（方案 A：Z=3–4 全组态）评估套件
# 用法: bash scripts/v3_eval_stage3a_suite.sh [CKPT路径]
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

CFG=configs/v3_stage_a_z1_26_n10.yaml
# 默认：E-prime（Stage 3a 推荐，H 锚点 -3.4 meV）
_DEFAULT_EPRIME="checkpoints/v3_stage_a_z1_26_n10_p1z3_4_eprime_full/best_anchor.msgpack"
CKPT="${1:-${_DEFAULT_EPRIME}}"
if [[ ! -f "${CKPT}" ]]; then
  CKPT="${1:-checkpoints/v3_stage_a_z1_26_n10_p1z3_4_prime_v2_full/best_anchor.msgpack}"
fi
SUITE=logs/stage3a_p1z3_4_eprime_eval/suite

mkdir -p "${SUITE}" data_cache

if [[ ! -f "${CKPT}" ]]; then
  echo "ERROR: checkpoint not found: ${CKPT}"
  echo "常见路径:"
  echo "  checkpoints/v3_stage_a_z1_26_n10_p1z3_4_eprime_full/best_anchor.msgpack"
  echo "  checkpoints/v3_stage_a_z1_26_n10_p1z3_4_prime_v2_full/best_anchor.msgpack"
  echo "  checkpoints/v3_phase1_stage_a_z1_8_p1z3_4_full/stage_a_last.msgpack"
  exit 1
fi

echo "==> Checkpoint: ${CKPT}"

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
echo "==> 1) 氢锚点（防回退，必过）"
python scripts/v3_diag_h_potential.py --config "${CFG}" --ckpt "${CKPT}" \
  | tee "${SUITE}/diag_h.txt"

echo ""
echo "==> 2) 基线对比（氢）"
python scripts/v3_compare_stage2_baselines.py \
  --target "${CKPT}" \
  --baselines p1lowz,p1z8_full \
  --out "${SUITE}/BASELINE_COMPARISON.md"

echo ""
echo "==> 3) Gate — Z<=2 锚点"
python scripts/v3_gate_analytic.py --config "${CFG}" --ckpt "${CKPT}" \
  --manifest data_cache/manifest_z_le_2.parquet
mkdir -p "${SUITE}/gate_z_le_2"
cp -f logs/v3_stage_a_z1_26_n10/GATE_A_REPORT.md "${SUITE}/gate_z_le_2/"
cp -f logs/v3_stage_a_z1_26_n10/gate_a_detailed.csv "${SUITE}/gate_z_le_2/"

echo ""
echo "==> 4) Gate — Z=3–4 **主验收**"
python scripts/v3_gate_analytic.py --config "${CFG}" --ckpt "${CKPT}" \
  --manifest data_cache/manifest_z_3_4.parquet
mkdir -p "${SUITE}/gate_z_3_4"
cp -f logs/v3_stage_a_z1_26_n10/GATE_A_REPORT.md "${SUITE}/gate_z_3_4/"
cp -f logs/v3_stage_a_z1_26_n10/gate_a_detailed.csv "${SUITE}/gate_z_3_4/"

echo ""
echo "==> 5) Gate — Z<=4 累计训练区"
python scripts/v3_gate_analytic.py --config "${CFG}" --ckpt "${CKPT}" \
  --manifest data_cache/manifest_z_le_4.parquet
mkdir -p "${SUITE}/gate_z_le_4"
cp -f logs/v3_stage_a_z1_26_n10/GATE_A_REPORT.md "${SUITE}/gate_z_le_4/"
cp -f logs/v3_stage_a_z1_26_n10/gate_a_detailed.csv "${SUITE}/gate_z_le_4/"

echo ""
echo "==> 6) Layer-2b — Z<=2（锚点）"
python scripts/v3_eval_excitation_vs_nist.py \
  --config "${CFG}" --ckpt "${CKPT}" \
  --manifest data_cache/manifest_z_le_2.parquet \
  --out-dir "${SUITE}/layer2b_z_le_2"

echo ""
echo "==> 7) Layer-2b — Z=3–4 **主验收**"
python scripts/v3_eval_excitation_vs_nist.py \
  --config "${CFG}" --ckpt "${CKPT}" \
  --manifest data_cache/manifest_z_3_4.parquet \
  --out-dir "${SUITE}/layer2b_z_3_4"

echo ""
echo "==> 8) Layer-2b — Z<=4 累计"
python scripts/v3_eval_excitation_vs_nist.py \
  --config "${CFG}" --ckpt "${CKPT}" \
  --manifest data_cache/manifest_z_le_4.parquet \
  --out-dir "${SUITE}/layer2b_z_le_4"

echo ""
echo "==> 9) 按 Z/l 拆解（Z=3–4）"
python scripts/v3_summarize_excitation_by_Zl.py \
  --csv "${SUITE}/layer2b_z_3_4/excitation_vs_nist.csv" \
  --out "${SUITE}/summary_z3_4_z_l.md"

echo ""
echo "==> 10) 与 Stage 2 对比（Li/Be 是否改善）"
python - <<'PY'
import json
from pathlib import Path

def load_metrics(p):
    if not p.exists():
        return None
    return json.loads(p.read_text())["layer2b"]

s2 = Path("logs/v3_stage_a_z1_26_n10_p1z8_full_eval/stage2_suite/layer2b_z_le_8/metrics.json")
s3 = Path("logs/stage3a_p1z3_4_full_eval/suite/layer2b_z_3_4/metrics.json")
if not s3.exists():
    print("(skip: run suite first)")
    raise SystemExit(0)
m2, m3 = load_metrics(s2), load_metrics(s3)
lines = ["# Stage 2 (p1z8) vs Stage 3a (p1z3_4) — Z=3–4 训练区", ""]
lines.append("| 组 | Stage2 p1z8 (Z<=8子集) | Stage3a p1z3_4 (Z=3–4) |")
lines.append("|----|----------------------|------------------------|")
for g in ("single_valence", "multi_electron"):
    if g in m2 and g in m3:
        lines.append(
            f"| {g} | MAE {m2[g]['mae_meV']/1e3:.2f} eV, pass {m2[g]['pass_fraction']*100:.1f}% "
            f"| MAE {m3[g]['mae_meV']/1e3:.2f} eV, pass {m3[g]['pass_fraction']*100:.1f}% |"
        )
Path("logs/stage3a_p1z3_4_full_eval/suite/STAGE2_VS_STAGE3A.md").write_text("\n".join(lines)+"\n")
print("Wrote logs/stage3a_p1z3_4_full_eval/suite/STAGE2_VS_STAGE3A.md")
PY

echo ""
echo "==> 11) 训练曲线（ground + full）"
for tag in p1z3_4_ground p1z3_4_full; do
  for base in logs/v3_stage_a_z1_26_n10_${tag} logs/v3_phase1_stage_a_z1_8_${tag}; do
    h="${base}/history.csv"
    if [[ -f "${h}" ]]; then
      echo "--- ${h} ---"
      awk -F, 'NR==1 || NR==2 || NR==int(NR/2) || NR==NR' "${h}"
    fi
  done
done | tee "${SUITE}/training_summary.txt"

echo ""
echo "=============================================="
echo "Stage 3a 评估完成 → ${SUITE}/"
echo "  必读: diag_h.txt, layer2b_z_3_4/, gate_z_3_4/"
echo "  防回退: layer2b_z_le_2/ (H/He 应仍 ~100 meV 级)"
echo "跑完后对我说: Stage 3a suite 跑完了"
echo "=============================================="
