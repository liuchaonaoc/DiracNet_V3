#!/usr/bin/env bash
# Generate and run cFAC .sf scripts for single-electron ns orbitals at
# (Z, n) over a sweep, writing
#     cf_Z<Z>_n<N>.sf      (cFAC script)
#     cf_Z<Z>_n<N>_PQ.dat  (radial wavefunctions from WaveFuncTable)
#     cf_Z<Z>_n<N>_V.dat   (central potential from GetPotential)
#
# Each .sf is one single-electron ns^1 configuration, the smallest unit
# cFAC supports.  Total wall time: ~30 s per .sf (most of it JIT
# initialization); for the full Z=1..26 × n=1..10 sweep this is ~3 min.
#
# Usage:
#     bash gen_cfac_batch.sh                            # full 260-row sweep
#     bash gen_cfac_batch.sh 1 5                        # one (Z=1, n=5)
#     bash gen_cfac_batch.sh 1 5 26 10                  # all (Z=1..26, n=5..10)
#     bash gen_cfac_batch.sh --no-run                   # only emit .sf, do not invoke sfac
#     bash gen_cfac_batch.sh --rebuild                  # ignore existing PQ.dat, force re-run
#
# After this finishes, run compare_energy_3way.py to drive the
# three-way energy comparison (NIST / cFAC / PINN).

set -euo pipefail

cd "$(dirname "$0")"  # energy_batch
mkdir -p logs

# Element symbols (Z=1..26).  Use Element table from cFAC demo.
ELEM=(
    "" H  He Li Be B  C  N  O  F  Ne
    Na Mg Al Si P  S  Cl Ar K  Ca Sc
    Ti V  Cr Mn Fe
)

# Default sweep
Z_LO=1; Z_HI=26
N_LO=1; N_HI=10
DO_RUN=1
REBUILD=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-run)  DO_RUN=0; shift ;;
        --rebuild) REBUILD=1; shift ;;
        [0-9]*)    Z_LO="${1}"; Z_HI="${2:-$Z_LO}"; N_LO="${3:-1}"; N_HI="${4:-10}"
                   shift; [[ "${2:-}" =~ ^[0-9]+$ ]] && shift
                   [[ "${2:-}" =~ ^[0-9]+$ ]] && shift
                   [[ "${2:-}" =~ ^[0-9]+$ ]] && shift
                   ;;
        *) echo "[arg] unknown: $1" >&2; exit 1 ;;
    esac
done

echo "[batch] Z=${Z_LO}..${Z_HI}, n=${N_LO}..${N_HI}  run=${DO_RUN}  rebuild=${REBUILD}"

# Grid parameters per (Z, n) range.
# For Z=1, n>=5: H high-n orbitals span r ~ n^2 (e.g. n=8 -> r ~ 64)
# So H, He need larger r1 (pts/wavelength) for high n.
# Other Z have n^2/Z < 4 a.u. for n<=10, so a smaller r1 suffices.
n_done=0; n_fail=0; n_skip=0
for Z in $(seq "$Z_LO" "$Z_HI"); do
    ELEM_SYM=${ELEM[$Z]}
    for N in $(seq "$N_LO" "$N_HI"); do
        SF="cf_Z${Z}_n${N}.sf"
        OUT_PQ="cf_Z${Z}_n${N}_PQ.dat"
        OUT_V="cf_Z${Z}_n${N}_V.dat"

        # Skip if already computed (unless --rebuild)
        if [[ -s "$OUT_PQ" && $REBUILD -eq 0 ]]; then
            n_skip=$((n_skip+1))
            continue
        fi

        # Grid: pick rmax-per-wavelength and grid size based on orbital extent
        NG=400
        R0=1.05
        RMIN=1.0e-6
        if [[ $Z -le 2 && $N -ge 5 ]]; then
            R1=14.0; NG=800
        elif [[ $Z -le 4 ]]; then
            R1=8.0
        else
            R1=10.0
        fi

        cat > "$SF" <<EOF
# cFAC single-electron ns at Z=${Z}, n=${N}

SetAtom('${ELEM_SYM}', ${Z})

SetRadialGrid(${NG}, ${R0}, ${R1}, ${RMIN})

Config('g', '${N}s1')

ConfigEnergy(0)
OptimizeRadial(['g'])
ConfigEnergy(1)

GetPotential('${OUT_V}')
WaveFuncTable('${OUT_PQ}', ${N}, -1)

Print('done: Z=${Z}, n=${N}')
EOF

        if [[ $DO_RUN -eq 1 ]]; then
            if sfac "$SF" > "logs/cf_Z${Z}_n${N}.log" 2>&1; then
                n_done=$((n_done+1))
            else
                n_fail=$((n_fail+1))
                echo "[FAIL] $SF (see logs/cf_Z${Z}_n${N}.log)"
            fi
        fi
    done
done

echo "[batch] done=${n_done}  fail=${n_fail}  skipped=${n_skip}"