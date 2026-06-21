# PINN-ART external comparison scripts

This folder bundles the two rounds of external-comparison artifacts used
to validate `PINN-ART` against `cFAC` (FAC's C port) and analytic
hydrogenic reference:

| Round | Script | Purpose |
|---|---|---|
| 1 | `compare_vpq_cases.py` | Plot V(r), P(r), Q(r) for a list of (Z, n) cases; one 3-panel figure per case plus one stacked grid |
| 2 | `compare_energy_3way.py` | Compute E_orb on every manifest row from PINN, cFAC and analytic hydrogenic; produce per-row CSV, per-Z statistics, and 1:1 plots |

All scripts default to the Step C checkpoint (`checkpoints/v3_stage_a_laguerre_basis_c_c2`)
and the corresponding config / manifest, but accept overrides so they
can be re-run after any future Stage-A retrain.

## Prerequisites

* `sfac` (cFAC binary) must be on `$PATH`.  Built from `cfac-1.7.1/`
  via `make` in `/home/chaos/cfac-1.7.1`.  Source dir:
  `/home/chaos/cfac-1.7.1/`.
* `python` (≥ 3.13) with `numpy`, `pandas`, `jax`, `flax`, `matplotlib`
  installed.
* The PINN-ART project is at `/home/chaos/workspace2/DiracNet_V3/rc_pinn_art_project/`.
* The hydrogenic manifest at `data_cache/manifest_hydrogenic_z1_26_n10.parquet`
  (260 rows: Z=1..26 × n=1..10).

## Step 1 — Generate cFAC reference data (only needed once)

```bash
cd /home/chaos/workspace2/DiracNet_V3/cfac_jobs/energy_batch

# Full 260-row sweep (~3 minutes)
bash gen_cfac_batch.sh

# Or, after the manifest is updated (e.g. adding new Z or n):
bash gen_cfac_batch.sh --rebuild

# Just one (Z=1, n=5)
bash gen_cfac_batch.sh 1 5

# Re-run only the failures
bash gen_cfac_batch.sh --rebuild 1 6 1 7 1 8 1 9 1 10
```

Outputs:
* `cf_Z<Z>_n<N>.sf`      — cFAC script
* `cf_Z<Z>_n<N>_PQ.dat`  — radial wavefunctions (r, Vc·r, U·r, P, Q)
* `cf_Z<Z>_n<N>_V.dat`   — central potential parameters
* `logs/cf_Z<Z>_n<N>.log` — sfac stderr/stdout

## Step 2 — V/P/Q morphology comparison

```bash
python compare_vpq_cases.py
# default: 14 cases (H/Fe {1,2,5,8} + Li/C/O {1,5})

# Custom (Z, n) list
python compare_vpq_cases.py --cases 1 1 8 1 26 1

# After a new checkpoint
python compare_vpq_cases.py --ckpt /path/to/new_ckpt.msgpack
```

Outputs (default `--out-dir` is `vpq_compare/`):
* `VPQ_Z<Z>_n<N>.png`    — one figure per case (V, P, Q panels)
* `VPQ_grid.png`          — all cases stacked (N rows × 3 cols)

## Step 3 — Three-way energy comparison

```bash
python compare_energy_3way.py
```

Defaults: full 260-row manifest, Step C checkpoint.

Outputs (default `--out-dir` is `energy_compare/`):
* `energy_comparison.csv`     — full row-level table with NIST/FAC/PINN E_orb
* `per_Z_stats.csv`            — per-Z RMSE / max |ΔE| statistics
* `energy_1to1.png`            — three 1:1 plots (PINN vs NIST, FAC vs NIST, PINN vs FAC)
* `per_Z_RMSE.png`             — per-Z RMSE / max |ΔE| bar charts
* `error_histograms.png`       — |ΔE| and signed-ΔE histograms
* `top10_worst_PINN.csv`       — top-10 worst rows for PINN vs NIST

## Conventions

* **NIST ground truth** = analytic hydrogenic E_n = -Z²/(2n²) Hartree
  (no Dirac / Breit / QED corrections).
* **cFAC reference** = single-electron Dirac from `WaveFuncTable`:
  the `# energy = X` comment (in eV) is parsed and converted to Hartree.
* **PINN** E_orb = `out["E_orb"][slot]` from `model.apply(...)` on the
  manifest row, in Hartree.  Slot selection matches `(n, l)` from
  `batch["shell_table"]`.
* All energies and errors are reported in meV (`hartree_to_meV(1.0)`).

## Re-running after a PINN-ART retrain

```bash
# After a new Stage-A training run lands in a new directory, e.g.
#   checkpoints/v3_stage_a_laguerre_basis_c_c3/stage_a_last.msgpack

python compare_vpq_cases.py \
    --ckpt ../rc_pinn_art_project/checkpoints/v3_stage_a_laguerre_basis_c_c3/stage_a_last.msgpack \
    --out-dir vpq_compare_c3

python compare_energy_3way.py \
    --ckpt ../rc_pinn_art_project/checkpoints/v3_stage_a_laguerre_basis_c_c3/stage_a_last.msgpack \
    --out-dir energy_compare_c3
```

The cFAC reference does not need to be re-generated unless the
manifest's `(Z, n)` set changes.