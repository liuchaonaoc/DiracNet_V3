# 15 — 代码结构生成提示词（Agent 主入口）

> 本文件给 **代码生成 Agent** 使用：严格按文档搭 `rc_pinn_art_project/`，再分模块实现。  
> **不要重新设计物理**；冲突时见 `00_overview.md` §7。

## 0. 必读列表（按序）

1. `Overall_design.md`
2. `00_overview.md` … `14_degenerate_gradient_safety.md`
3. 本文件

## 1. 不可违反的硬约束

### 1.1 框架

- 训练 / 推断 / `grad`：**JAX + Flax（或 Equinox）+ Optax**
- 禁止在 `pinn_art/` 内 `import torch`（预处理脚本除外）

### 1.2 训练纪律

- **Stage A**：backward 仅含 `L_pde, L_ortho, L_asym, L_norm, L_V_*`（见 `06`）
- **禁止** Stage A 使用 NIST MSE 梯度
- **禁止** `level_residual_head`、per-row bias table
- 对角填充：**`fill_h_diagonal_hybrid`** — `nist_mask=True` 用 NIST，否则 **Fall-back** 到 $H_{ii}^{\mathrm{theory}}$（禁止零填充）

### 1.3 主结果定义

- 主能量：**`E_csf`**（CI 本征值），不是 `E_orb_sum`
- 主波函数：**`(P,Q)` in 统一 $V(r)$**
- 报告必须含 Layer-0–4（`08`）

### 1.4 数值

- `eigh` 前 `symmetrize` + `eps_degen`
- 推荐 `safe_eigh` Custom VJP（`14`）
- PDE 残差用预计算 `dPdr`，勿对整个网络高阶 `grad`

## 2. 目标目录树

```text
DiracNet_V3/rc_pinn_art_project/
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── configs/
│   ├── default.yaml
│   ├── v3_smoke.yaml
│   ├── v3_phase1_stage_a.yaml
│   ├── v3_phase2_ci_hydrogenic.yaml
│   └── v3_phase3_nist_infer.yaml
├── pinn_art/
│   ├── __init__.py
│   ├── constants.py
│   ├── coords/
│   │   ├── __init__.py
│   │   ├── mixed_map.py
│   │   └── shell_features.py
│   ├── nets/
│   │   ├── __init__.py
│   │   ├── siren.py
│   │   ├── deeponet.py
│   │   ├── potential_head.py
│   │   └── phase_amplitude.py      # stub 即可
│   ├── physics/
│   │   ├── __init__.py
│   │   ├── dirac_operator.py
│   │   ├── orthogonalizer.py
│   │   └── hydrogenic.py
│   ├── ci/
│   │   ├── __init__.py
│   │   ├── slater_radial.py
│   │   ├── hamiltonian.py
│   │   ├── nist_inject.py          # fill_h_diagonal_hybrid
│   │   └── eigen_solver.py
│   ├── observables/
│   │   ├── __init__.py
│   │   ├── multipole.py
│   │   ├── transition_rates.py
│   │   └── collision.py
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── pde_loss.py
│   │   ├── ortho_loss.py
│   │   ├── asymptotic_loss.py
│   │   ├── potential_prior.py
│   │   └── loss_schedule.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── config_parser.py
│   │   ├── dataset.py
│   │   └── collate.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── pinn_art_model.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_state.py
│   │   ├── stage_a_trainer.py
│   │   ├── gate_a.py
│   │   └── checkpoint.py
│   └── utils/
│       ├── __init__.py
│       ├── grid.py
│       ├── config.py
│       ├── logging.py
│       └── numeric.py
├── scripts/
│   ├── v3_prepare_hydrogenic.py
│   ├── v3_train_stage_a.py
│   ├── v3_gate_analytic.py
│   ├── v3_infer.py
│   ├── v3_evaluate.py
│   ├── v3_profile_latency.py
│   └── v3_diagnose_physics_chain.py
└── tests/
    └── (见 12_test_plan.md)
```

## 3. 核心类与函数签名

### 3.1 `pinn_art/constants.py`

```python
ALPHA: float
C_LIGHT: float
HARTREE_TO_EV: float
FINE_STRUCTURE: float
```

### 3.2 `utils/grid.py`

```python
@dataclass
class RadialGrid:
    r: jnp.ndarray          # [N_grid]
    w: jnp.ndarray          # 积分权重
    def integrate(self, f, axis=-1) -> jnp.ndarray: ...
```

### 3.3 `models/pinn_art_model.py`

```python
class PinnArtModel(nn.Module):
    map_config: ...
    deeponet_config: ...
    n_orb_max: int = 16

    @nn.compact
    def __call__(self, batch, *, train: bool = False, return_ci: bool = False):
        """
        最小实现（Sprint 1）: 仅 V, P, Q, E_orb
        Sprint 3: return_ci=True 时附加 E_csf, V_csf, H
        """
```

### 3.4 `training/train_state.py`

```python
@dataclass
class TrainState:
    step: int
    params: Any
    opt_state: Any
    apply_fn: Callable
    tx: optax.GradientTransformation

    def apply_gradients(self, *, grads, **kw): ...
```

### 3.5 Forward 输出（强制字段）

```python
{
    "V": Array,                    # [B, N_grid]
    "wavefunctions": {
        "P": Array, "Q": Array,
        "dPdr": Array, "dQdr": Array,
    },
    "E_orb": Array,                # [B, N_orb]
    "E_csf": Array | None,
    "V_csf": Array | None,
    "transitions": dict | None,
}
```

## 4. `configs/default.yaml`（完整骨架）

```yaml
seed: 42

grid:
  r_min: 1.0e-4
  r_max: 50.0
  n_grid: 256
  scheme: loglinear

coords:
  c1: 1.0
  c2: 1.0
  r_eps: 1.0e-6
  learn_map: false

model:
  n_orb_max: 16
  n_csf_max: 32
  d_branch: 128
  d_trunk: 128
  n_siren_layers: 4
  omega_0: 30.0
  v_form: nuclear_plus_correction   # V = -Z/r + tanh(corr)

ci:
  enabled: false                    # Sprint 3 起 true
  k_list: [0, 1, 2, 3, 4]
  eps_degen_ev: 1.0e-6
  racah_cache: data_cache/racah_cache_v3.npz
  diagonal_fill: hybrid             # hybrid | nist_only | theory_only

stage_c:
  diagonal_fill: hybrid

continuum:
  enabled: false

stage_a:
  n_epochs: 2000
  batch_size: 8
  weights:
    pde: 1.0
    ortho: 100.0
    asym: 0.01
    norm: 10.0
    v_prior: 0.1
    v_smooth: 1.0e-3
  gate:
    cos_threshold: 0.99
    pde_threshold: 1.0e-3
    e_orb_meV_threshold: 1.0

optimizer:
  lr_trunk: 3.0e-4
  lr_branch: 1.0e-4
  weight_decay: 1.0e-4
  grad_clip: 1.0

training:
  ckpt_dir: ./checkpoints/v3_default
  log_dir: ./logs/v3_default
  val_every_epochs: 10

dataset:
  manifest: data_cache/manifest_hydrogenic_v3.parquet
```

## 5. 实现顺序（严格）

| 序号 | 任务 | Sprint |
|------|------|--------|
| 1 | `pyproject.toml`, `.gitignore`, `constants`, `utils/grid`, `config` | 0 |
| 2 | `coords/*`, `nets/siren`, `nets/deeponet`, `potential_head` | 1 |
| 3 | `physics/*`, `models/pinn_art_model` (no CI) | 1 |
| 4 | `losses/*`, `training/stage_a_trainer`, `gate_a` | 2 |
| 5 | `data/*`, `scripts/v3_prepare_hydrogenic`, `v3_train_stage_a` | 2 |
| 6 | `ci/*`, model `return_ci=True` | 3 |
| 7 | `observables/*`, `v3_infer`, `v3_profile_latency` | 4 |
| 8 | `v3_evaluate`, NIST manifest | 5 |
| 9 | `phase_amplitude`, `v3_compare_fac` | 6 |

**不得在 Gate A 通过前实现完整 CI 训练。**

## 6. 脚本最小行为

### `v3_train_stage_a.py`

```text
--config PATH
加载 yaml → 构建 dataset → TrainState → 循环 train_step → 定期 val → gate → 保存 stage_a_passed
```

### `v3_diagnose_physics_chain.py`

```text
单样本 forward → 打印 ||PDE||, |cos|, E_orb, max|V+Z/r|
```

### `v3_infer.py`

```text
加载 ckpt → jax.jit(infer_observables) → 写 parquet
```

## 7. 验收命令

### Sprint 0

```bash
cd /home/chaos/workspace2/DiracNet_V3/rc_pinn_art_project
pip install -e ".[dev]"
python -c "import pinn_art; print(pinn_art.__version__)"
pytest tests/test_import_smoke.py -v
```

### Sprint 1

```bash
pytest tests/test_coords_mixed_map.py tests/test_deeponet_forward.py \
       tests/test_dirac_operator_h1s.py -v
python scripts/v3_diagnose_physics_chain.py --config configs/v3_smoke.yaml
```

### Sprint 2

```bash
pytest tests/test_pde_loss.py tests/test_train_step_smoke.py -v
python scripts/v3_train_stage_a.py --config configs/v3_smoke.yaml
python scripts/v3_gate_analytic.py --ckpt checkpoints/v3_smoke/stage_a_passed
```

### Sprint 3

```bash
pytest tests/test_eigh_gradient_finite.py tests/test_nist_inject.py -v
```

## 8. Agent 输出格式

每批文件完成后汇报：

```text
Created/updated:
- path — 一行说明

Validation:
- 命令
- 结果（PASS/FAIL）

Next:
- 下一批文件
```

**不得**在无 pytest 的情况下声明 Sprint 完成。

## 9. `pinn_art/__init__.py`

```python
__version__ = "3.0.0"
```

## 10. README.md（项目根）必含

- 安装：`pip install -e .`
- 快速 smoke：`python scripts/v3_train_stage_a.py --config configs/v3_smoke.yaml`
- 指向 `../prompts/README.md`

## 11. 最终提醒

PINN-ART V3 的成功标准是：

```text
物理门禁（波函数 + PDE）→ CI + 对角混合填充（NIST / 理论回退）→ 可观测量与延迟
```

而不是单纯拟合 NIST 表格。任何绕过 Layer-0 的捷径实现都应拒绝合并。
