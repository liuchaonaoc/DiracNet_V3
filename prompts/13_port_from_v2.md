# 13 — 从 V2/V1 移植指南

> V3 **不** `import rc_diracnet_v2`。复制源码后改为 JAX / Flax，并更新 import 路径为 `pinn_art.*`。

## 1. 直接移植（改类型与框架）

| 源 (V2) | 目标 (V3) | 改动要点 |
|---------|-----------|----------|
| `utils/grid.py` | `pinn_art/utils/grid.py` | `torch`→`jax.numpy`；`integrate` 用 `jnp.trapz` |
| `physics/dirac_operator.py` | `physics/dirac_operator.py` | 同上 |
| `physics/hydrogenic_analytic.py` | `physics/hydrogenic.py` | 合并 spinor |
| `readout/orthogonalizer.py` | `physics/orthogonalizer.py` | Löwdin 线性代数 |
| `losses/pde_loss.py` | `losses/pde_loss.py` | API 一致 |
| `losses/asymptotic_loss.py` | `losses/asymptotic_loss.py` | 同上 |
| `data/config_parser.py` | `data/config_parser.py` | 去 torch |
| `data/nist_loader.py` | `scripts/` 预处理用 | 训练读 parquet |

## 2. 参考逻辑、不复制结构

| V2 模块 | V3 替代 |
|---------|---------|
| `basis/bspline_basis.py` | `nets/deeponet.py` + SIREN |
| `kan/*` | branch MLP（非 KAN 主路径） |
| `readout/envelope.py` | 可选 `envelope_bound`；非主自由度 |
| `models/dirac_net_v2.py` | `models/pinn_art_model.py` |
| `training/two_stage_trainer.py` | `training/stage_a_trainer.py` |
| `models/level_residual_head.py` | **弃用**；用 `fill_h_diagonal_hybrid`（NIST + 理论回退） |

## 3. 从 V1 启用（V2 未用）

| V1 模块 | V3 用途 |
|---------|---------|
| `physics/hamiltonian_assembler.py` | 参考装配索引；用 JAX 重写 |
| `physics/eigen_solver.py` | 参考 `eps` 正则；见 `14` |

路径：`DiracNet_V1/rc_diracnet_project/rc_diracnet/`

## 4. 数据 manifest

可复制 V2 生成的 parquet，运行：

```bash
python scripts/v3_migrate_manifest_v2.py --in ../DiracNet_V2/.../manifest_hydrogenic_v2.parquet --out data_cache/manifest_hydrogenic_v3.parquet
```

添加列 `parent_config_id`, `csf_id`（hash）。

## 5. 配置迁移

V2 `grid`, `encoder` 部分字段映射：

```yaml
# V2 → V3
grid.*          → grid.*
encoder.max_z   → branch.max_z
readout.n_orb_max → model.n_orb_max
stage1.weights  → stage_a.weights
```

## 6. 测试移植

| V2 测试 | V3 测试 |
|---------|---------|
| `test_v2_dirac_operator_h1s.py` | `test_dirac_operator_h1s.py` |
| `test_v2_pde_loss.py` | `test_pde_loss.py` |
| `test_v2_gate_pass_on_analytic.py` | `test_gate_pass_on_analytic.py` |

断言阈值保持一致。

## 7. 禁止带入 V2 的"pathology"

- `level_residual_head` 无界校准
- Stage 1 NIST backward
- `zn_bias_table`
- 仅报告 calibrated MAE 而不报 `E_orb`/波函数门禁
