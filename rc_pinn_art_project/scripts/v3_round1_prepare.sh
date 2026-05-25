#!/usr/bin/env bash
# Round-1 data prep: H..O (Z=1..8) hydrogenic manifest + Racah cache
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

MANIFEST=data_cache/manifest_hydrogenic_z1_8.parquet
RACAH=data_cache/racah_cache_z1_8.npz
CFG=configs/v3_phase1_stage_a_z1_8.yaml

echo "==> Hydrogenic manifest Z=1..8 (H..O), n=1..6"
python scripts/v3_prepare_hydrogenic.py \
  --z-min 1 --z-max 8 --n-levels 6 \
  --out "${MANIFEST}"

echo "==> Racah cache"
python scripts/v3_build_racah_cache.py \
  --manifest "${MANIFEST}" \
  --out "${RACAH}" \
  --config "${CFG}"

echo "==> Summary"
python -c "
import pandas as pd
df = pd.read_parquet('${MANIFEST}')
print(df.groupby(['Z','element']).size())
print('total rows', len(df))
"

echo "Done. Next:"
echo "  PYTHONPATH=. python scripts/v3_train_stage_a.py --config ${CFG}"
