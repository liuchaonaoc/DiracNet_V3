#!/usr/bin/env bash
# Phase 2: resume from Phase 1 + relaxed Gate + 1000 epochs
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

CFG=configs/v3_phase1_stage_a_z1_8_phase2.yaml
RESUME=checkpoints/v3_phase1_stage_a_z1_8/stage_a_last.msgpack

if [[ ! -f "${RESUME}" ]]; then
  echo "ERROR: Phase 1 checkpoint not found: ${RESUME}"
  echo "Run Phase 1 first: python scripts/v3_train_stage_a.py --config configs/v3_phase1_stage_a_z1_8.yaml"
  exit 1
fi

echo "==> Phase 2 training (1000 ep, resume from Phase 1)"
python scripts/v3_train_stage_a.py --config "${CFG}" --resume "${RESUME}"

echo ""
echo "==> Gate A (relaxed: cos>=0.95, pde<=1.0)"
python scripts/v3_gate_analytic.py \
  --config "${CFG}" \
  --ckpt checkpoints/v3_phase1_stage_a_z1_8_phase2/stage_a_last.msgpack

if python scripts/v3_gate_analytic.py --help 2>&1 | grep -q gate-mode; then
  echo ""
  echo "==> Gate A (strict, reference only)"
  python scripts/v3_gate_analytic.py \
    --config "${CFG}" \
    --ckpt checkpoints/v3_phase1_stage_a_z1_8_phase2/stage_a_last.msgpack \
    --gate-mode strict
fi

echo ""
echo "Done. Logs: logs/v3_phase1_stage_a_z1_8_phase2/"
