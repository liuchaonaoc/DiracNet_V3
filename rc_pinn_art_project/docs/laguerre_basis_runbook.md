# Stage A Round 2 — Laguerre Basis End-to-End Runbook

This document lists the **terminal commands you can copy-paste on the GPU box**
to take the Laguerre-basis implementation from "code complete" to "trained
and evaluated".  Each step has a one-line summary, the command, and the
expected log signature.

> **Working directory**: every command assumes
> `cd /home/chaos/workspace2/DiracNet_V3/rc_pinn_art_project`.
> **Backend**: all steps auto-detect GPU; if `JAX backend=cpu` is logged,
> install `jax[cuda12]` (see Step 0).

---

## Step 0 — verify environment

```bash
cd /home/chaos/workspace2/DiracNet_V3/rc_pinn_art_project
nvidia-smi | head -20                                  # GPU present?
python -c "import jax, jaxlib; print(jax.default_backend(), jax.devices())"
# Expected (GPU): gpu [CudaDevice(id=0)]
# Expected (CPU fallback): cpu [CpuDevice(id=0)]
```

If `cpu` and a GPU is available:

```bash
pip install --upgrade "jax[cuda12]==0.4.*"
```

---

## Step 1 — run unit tests (sanity, ~30 s)

```bash
cd /home/chaos/workspace2/DiracNet_V3/rc_pinn_art_project
python -m pytest tests/test_laguerre_basis.py \
                 tests/test_deeponet_forward.py \
                 tests/test_hydrogenic_P_jax.py \
                 tests/test_orthogonalizer.py -q
```

**Expected**: `42 passed in ~20s`.  All R0 + R2a tests; failure ⇒ code broken.

---

## Step 2 — short smoke training run (~2 min on CPU, ~30 s on GPU)

Trains one full epoch on the prepared hydrogenic manifest (Z ≤ 8).  Use
this as a pre-flight check before launching the long run.

```bash
cd /home/chaos/workspace2/DiracNet_V3/rc_pinn_art_project
python scripts/v3_train_stage_a_laguerre_basis.py \
    --epochs 1 --batch-size 4 --tag _smoke
```

**Expected log**:
```
JAX backend=gpu devices=[CudaDevice(id=0)]
Laguerre config: use=True K_max=9 learn_lambda=True perturb_P=0.05 perturb_Q=0.05
DFS: enabled=False ...
Laguerre loss weights: coeff_decay=0.001 lambda_prior=0.001 q_residual=1.0
Stage A Laguerre: 15 samples × 1 epochs × 5 steps × batch=4
JIT compile done in <X>s
epoch    0  loss=10.xxxx  pde=0.0000  ortho=...  norm=...  cd=... λ=... q=...
Laguerre training done.  ckpt=.../v3_stage_a_laguerre_basis_smoke/stage_a_last.msgpack
```

---

## Step 3 — full Laguerre Stage-A training (~1 h GPU for 2000 epochs)

> **The manual `nohup` you tried earlier failed because
> `logs/v3_stage_a_laguerre_basis/` did not exist yet** — bash's stdout
> redirection (`> logs/.../train_…log`) bailed before the training started,
> leaving no log file and `Exit 1` from the foregrounded job.  Use the
> wrapper script which `mkdir -p`'s the directory first:

```bash
cd /home/chaos/workspace2/DiracNet_V3/rc_pinn_art_project
bash scripts/v3_run_stage_a_laguerre_basis.sh                  # default: 2000 epochs
# optional: bash scripts/v3_run_stage_a_laguerre_basis.sh 500 v1   # 500 epochs, tag=v1
```

The script prints the absolute log path; tail it with:

```bash
tail -f logs/v3_stage_a_laguerre_basis/train_<timestamp>.log
```

**Expected log signature every 25 epochs**:
```
epoch XXXX  loss=Y.yyyy  pde=...  ortho=...  norm=...  cd=...  λ=...  q=...  T.Ts (S.Ss/step)
```
Convergence expectations (from prompt §7.1):
- `cd` (`L_coeff_decay`) → trending toward 0
- `λ` (`L_lambda_prior`) → ≈ 0 (λ_a stays near λ_init = Z_eff/n)
- `q` (`L_q_residual`) → ≤ 5% perturbation; stays small
- `ortho` → < 1e-3 across the manifest

---

## Step 4 — resume from checkpoint

If training is interrupted (OOM, restart, etc.):

```bash
cd /home/chaos/workspace2/DiracNet_V3/rc_pinn_art_project
python scripts/v3_train_stage_a_laguerre_basis.py \
    --resume checkpoints/v3_stage_a_laguerre_basis/stage_a_last.msgpack \
    --epochs 2000
```

The trainer re-initializes the optimizer state and continues from the saved
`msgpack`.  Make sure `--config` points at the **same** config file as the
original run — `cfg.model.use_laguerre_basis` etc. must match.

---

## Step 5 — evaluate a checkpoint

```bash
cd /home/chaos/workspace2/DiracNet_V3/rc_pinn_art_project
python scripts/v3_evaluate_laguerre_basis.py \
    --ckpt checkpoints/v3_stage_a_laguerre_basis/stage_a_last.msgpack \
    --max-rows 30 \
    --z-min 1 --z-max 8 \
    --out results/laguerre_basis_eval/stage_a_last_$(date +%Y%m%d).json
```

**Expected log**:
```
Loaded checkpoint: ...
Evaluating 30 manifest rows
node-gate pass rate: <PASS>/<TOTAL> (<PCT>%)
mean cos(P_model, P_hydrogenic) over active orbitals: 0.95xx
λ_a drift vs λ_init: mean=0.0X max=0.0X
Wrote evaluation report to results/laguerre_basis_eval/...
```

**Layer-0 acceptance thresholds** (per prompt §7.1):
- `node_gate_pass_rate ≥ 0.95` (≥ 95 % of active orbitals have the right number of radial nodes)
- `mean |cos(P_model, P_hydrogenic)| ≥ 0.95` for hydrogenic-only systems
- `|λ_a − λ_init| / λ_init ≤ 0.30` (softplus drift bounded)
- `L_q_residual ≤ 5 % × ‖Q_skel‖` (encoded as q ≤ 0.05 in `cosine_signed`)

---

## Step 6 — cross-check against cFAC (morphology plot)

The existing `v3_compare_fac_li_pqv.py` plot script overlays PINN-ART P/Q
against cFAC for Li.  After Step 5, re-run it pointing at the Laguerre
checkpoint:

```bash
cd /home/chaos/workspace2/DiracNet_V3/rc_pinn_art_project
python scripts/v3_compare_fac_li_pqv.py \
    --ckpt checkpoints/v3_stage_a_laguerre_basis/stage_a_last.msgpack \
    --out logs/fac_vs_pinn_li_laguerre/
```

(If `--ckpt` flag is not supported by the script, edit the default inside
the script — it currently points at the canonical Stage-A checkpoint.)

---

## Step 7 — promote to Stage B (if Layer-0 passes)

When the Laguerre Stage-A gate passes (`node_gate_pass_rate ≥ 0.95` and
`mean cos ≥ 0.95` on hydrogenic), freeze the Laguerre heads and continue to
Stage B (CI + Slater):

```bash
cd /home/chaos/workspace2/DiracNet_V3/rc_pinn_art_project
python scripts/v3_train_stage_b.py \
    --config configs/v3_stage_a_laguerre_basis.yaml \
    --resume checkpoints/v3_stage_a_laguerre_basis/stage_a_last.msgpack
```

(Equivalent to "freeze Phase-3 ansatz, switch to Round 2's CI head".)

---

## Quick reference

| Step | Script | Time (GPU) | Output |
|------|--------|-----------|--------|
| 0 | env check | < 1 s | `JAX backend=gpu` |
| 1 | pytest | ~30 s | `42 passed` |
| 2 | smoke train | ~30 s | ckpt + history.csv |
| 3 | full train | ~1 h | ckpt + history.csv + log |
| 4 | resume | varies | continues from ckpt |
| 5 | evaluate | ~1 min | results/laguerre_basis_eval/*.json |
| 6 | cFAC compare | ~10 s | logs/fac_vs_pinn_li_laguerre/*.png |
| 7 | Stage B | varies | b-stage ckpt |

---

## Common pitfalls

1. **`JAX backend=cpu`**: install `jax[cuda12]` before Step 3.
2. **Manifest missing**: Step 2/3 will rebuild via
   `scripts/v3_prepare_hydrogenic.py --z-min 1 --z-max 8 --n-levels 6`.
3. **`ModuleNotFoundError: pinn_art`**: the scripts prepend `sys.path.insert(0, ROOT)`
   themselves; do not `cd` elsewhere before launching.
4. **`KeyError: laguerre_init`**: the `build_model_and_params` patch did not
   find the deeponet module; check `pinn_art.models.pinn_art_model` for
   `flax.core.unfreeze(params)` / `freeze(...)` symmetry (R2a patch).
5. **`ConcretizationTypeError`**: `float()` on a traced value inside JIT;
   wrap as `jnp.asarray(x, dtype=jnp.float32)` (already done in
   `LaguerreLambdaHead`).

---

## File locations (after a successful Step 3 run)

```
rc_pinn_art_project/
├── checkpoints/
│   └── v3_stage_a_laguerre_basis/
│       ├── epoch_0250.msgpack   (if cfg.training.ckpt_every_epochs > 0)
│       ├── epoch_0500.msgpack
│       └── stage_a_last.msgpack
├── logs/
│   └── v3_stage_a_laguerre_basis/
│       ├── history.csv          (per-epoch metrics incl. cd/λ/q)
│       └── train_YYYYMMDD_HHMMSS.log
└── results/
    └── laguerre_basis_eval/
        └── stage_a_last_YYYYMMDD.json
```