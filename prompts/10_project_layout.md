# 10 — 项目目录布局

## 1. 顶层

```text
DiracNet_V3/
├── README.md
├── prompts/                         # 本套提示词
│   ├── Overall_design.md
│   ├── README.md
│   └── 00_overview.md … 15_code_structure_generation.md
├── docs/
│   └── design_rationale.md          # 可选，从 Overall 扩展
└── rc_pinn_art_project/             # ★ 代码根（待生成）
    ├── README.md
    ├── pyproject.toml
    ├── requirements.txt
    ├── configs/
    ├── data_cache/                  # gitignore
    ├── data_raw/                    # gitignore
    ├── checkpoints/                 # gitignore
    ├── logs/                        # gitignore
    ├── results/                     # gitignore
    ├── pinn_art/                    # JAX 主包
    ├── scripts/
    └── tests/
```

## 2. `pinn_art/` 包结构

```text
pinn_art/
├── __init__.py
├── constants.py
├── coords/                 # 模块 1
├── nets/                   # DeepONet, SIREN
├── physics/                # Dirac, 正交, 氢解析
├── ci/                     # Slater, H, eigh, NIST
├── observables/            # A_ki, 截面
├── losses/
├── data/                   # dataset, parser, nist
├── training/               # train_state, gates, schedules
└── utils/                  # grid, checkpoint, logging, numeric
```

## 3. `configs/`

```text
configs/
├── default.yaml
├── v3_smoke.yaml
├── v3_phase1_stage_a.yaml
├── v3_phase1_stage_a_extended.yaml
├── v3_phase2_ci_hydrogenic.yaml
├── v3_phase3_nist_infer.yaml
├── v3_loo_z8.yaml
└── ablation_bspline.yaml      # 可选，非主路径
```

## 4. `scripts/`

```text
scripts/
├── v3_prepare_hydrogenic.py
├── v3_prepare_nist_subset.py
├── v3_build_racah_cache.py
├── v3_train_stage_a.py
├── v3_train_stage_b.py
├── v3_infer.py
├── v3_evaluate.py
├── v3_gate_analytic.py
├── v3_profile_latency.py
├── v3_compare_fac.py
└── v3_diagnose_physics_chain.py
```

## 5. `pyproject.toml` 依赖

```toml
[project]
name = "rc-pinn-art"
version = "3.0.0"
requires-python = ">=3.10"
dependencies = [
  "jax[cuda12]>=0.4.28",
  "flax>=0.8",
  "optax",
  "numpy",
  "scipy",
  "pandas",
  "pyarrow",
  "pyyaml",
  "matplotlib",
  "tqdm",
  "orbax-checkpoint",
]
```

CPU-only 开发可用 `jax[cpu]`。

## 6. `.gitignore`

```text
data_cache/
data_raw/
checkpoints/
logs/
results/
__pycache__/
.pytest_cache/
*.egg-info/
```

## 7. `default.yaml` 骨架

见 `15_code_structure_generation.md` §5（完整字段）。
