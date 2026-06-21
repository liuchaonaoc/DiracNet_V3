#!/usr/bin/env bash
# Generate 260 single-electron cFAC .sf scripts for Z=1..26 × n=1..10
# and run them all in sequence.

set -euo pipefail

cd /home/chaos/workspace2/DiracNet_V3/cfac_jobs/energy_batch
mkdir -p logs

ELEM=(
    "" H  He Li Be B  C  N  O  F  Ne
    Na Mg Al Si P  S  Cl Ar K  Ca Sc
    Ti V  Cr Mn Fe
)

for Z in $(seq 1 26); do
    ELEM_SYM=${ELEM[$Z]}
    for N in $(seq 1 10); do
        OUT_PQ="cf_Z${Z}_n${N}_PQ.dat"
        SF="cf_Z${Z}_n${N}.sf"
        if [[ -f "$OUT_PQ" ]]; then
            echo "[skip] $SF -> $OUT_PQ exists"
            continue
        fi
        # Choose rmax: bigger grid for higher n or higher Z
        if [[ $Z -le 4 ]]; then
            RMAX_R1=8.0
        else
            RMAX_R1=10.0
        fi
        NG=400
        R0=1.05
        RMIN=1.0e-6
        cat > "$SF" <<EOF
# cFAC single-electron ns at Z=$Z, n=$N

SetAtom('${ELEM_SYM}', ${Z})

SetRadialGrid(${NG}, ${R0}, ${RMAX_R1}, ${RMIN})

Config('g', '${N}s1')

ConfigEnergy(0)
OptimizeRadial(['g'])
ConfigEnergy(1)

GetPotential('cf_Z${Z}_n${N}_V.dat')
WaveFuncTable('${OUT_PQ}', ${N}, -1)

Print('done: Z=${Z}, n=${N}')
EOF
        sfac "$SF" > "logs/cf_Z${Z}_n${N}.log" 2>&1 || echo "[FAIL] $SF"
    done
done
echo "[batch done]"