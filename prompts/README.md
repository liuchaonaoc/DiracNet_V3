# PINN-ART (DiracNet V3) 提示词索引

> **上游必读**：[`Overall_design.md`](./Overall_design.md) — 物理目标与四大模块总览。  
> **代码生成入口**：[`15_code_structure_generation.md`](./15_code_structure_generation.md) — Agent 按此文件搭骨架并实现。

## 阅读顺序（给代码生成 Agent）

| 优先级 | 文件 | 内容 |
|--------|------|------|
| ★★★ | [00_overview.md](./00_overview.md) | V3 设计哲学、与 V1/V2 关系、不可违反纪律 |
| ★★★ | [15_code_structure_generation.md](./15_code_structure_generation.md) | 目录树、接口清单、实现顺序、验收命令 |
| ★★ | [01_architecture.md](./01_architecture.md) | 端到端张量流、四模块衔接、JAX `jit`/`vmap` 边界 |
| ★★ | [10_project_layout.md](./10_project_layout.md) | 包结构、`configs/`、`scripts/`、`tests/` |
| ★ | [02_coordinate_mapping.md](./02_coordinate_mapping.md) | 模块 1：混合坐标 $t(r)$ |
| ★ | [03_neural_dirac_solver.md](./03_neural_dirac_solver.md) | 模块 2：DeepONet + SIREN + 相幅分离 |
| ★ | [04_differentiable_ci.md](./04_differentiable_ci.md) | 模块 3：Slater、Racah、NIST 注入 + 理论 Fall-back、`eigh` |
| ★ | [05_observables_inference.md](./05_observables_inference.md) | 模块 4：$A_{ki}$、$gf$、碰撞截面 |
| ★ | [06_physics_losses.md](./06_physics_losses.md) | PDE、正交、渐近、CI 残差等 |
| ★ | [07_training_pipeline.md](./07_training_pipeline.md) | 三阶段训练、权重调度、checkpoint |
| ★ | [08_evaluation.md](./08_evaluation.md) | meV 门禁、截面基准、延迟 profiling |
| ★ | [09_data_pipeline.md](./09_data_pipeline.md) | manifest、NIST、角向系数离线表 |
| ★ | [11_sprint_plan.md](./11_sprint_plan.md) | Sprint 0–6 里程碑 |
| ★ | [12_test_plan.md](./12_test_plan.md) | L0–L4 测试矩阵 |
| ★ | [13_port_from_v2.md](./13_port_from_v2.md) | 从 V2 移植/改写的模块清单 |
| ★ | [14_degenerate_gradient_safety.md](./14_degenerate_gradient_safety.md) | `eigh` 简并梯度、Custom VJP |
| ★★★ | [16_stage_a_selfconsistent_dfs.md](./16_stage_a_selfconsistent_dfs.md) | **Round 2 主入口**：回到 Stage A，自洽 DFS 全组态预训练（Z≤26, n≤10）+ 双组激发能 vs NIST |
| ★★★ | [17_generalized_laguerre_basis.md](./17_generalized_laguerre_basis.md) | **下一轮架构升级**：广义拉盖尔有限基展开（Coulomb-Sturmian 形态 + 网络输出系数），节点数 n-l-1 由多项式阶数硬保证，残差对系数线性可微。**实施进度：R0–R3 完成**（见 [实现报告](#实施进度-2026-06-20)） |

## 与历史版本关系

```text
V1 (PyTorch, RC+MLP)  →  谱学任务定义、NIST、Dirac 算子
V2 (PyTorch, B-spline+KAN)  →  单粒子 PDE 门禁、E_orb-only 纪律
V3 (JAX, DeepONet+SIREN+CI)  →  全栈 PINN-ART：势场 + CI + 截面 + ms 推断
```

## 实施进度（2026-06-20）

[`17_generalized_laguerre_basis.md`](./17_generalized_laguerre_basis.md) 第 10 节（实施清单 R0–R5）当前完成情况：

| 步骤 | 状态 | 落地位置 |
|------|------|----------|
| **R0** — 测试先行 | ✅ | `tests/test_laguerre_basis.py` (16 个测试：氢样极限、节点数、正交性边界、Q 残差、λ 先验) |
| **R1a** — 氢样 Laguerre 系数 | ✅ | `pinn_art/physics/hydrogenic.py::hydrogenic_laguerre_coeffs` |
| **R1b** — Laguerre 基底算子 | ✅ | `pinn_art/nets/laguerre_basis.py`（`LaguerreCoeffHead` / `LaguerreLambdaHead` / `LaguerreQCorrHead` / `laguerre_p_sum` / `laguerre_p_sum_with_r` / `kinetic_balance_q_with_corr` / `q_residual`） |
| **R1c** — 系数正则 | ✅ | `pinn_art/losses/coeff_loss.py`（`L_coeff_decay` + `L_lambda_prior`） |
| **R1d** — Q 残差 | ✅ | `pinn_art/losses/q_corr_loss.py`（`q_residual`） |
| **R2a** — DeepONet 集成 | ✅ | `pinn_art/nets/deeponet.py` 加 `use_laguerre_basis / learn_lambda / K_max / perturb_scale_P / perturb_scale_Q` 五个开关；端到端 forward 验证在 `tests/test_deeponet_forward.py::test_deeponet_laguerre_forward_at_init_matches_hydrogenic` |
| **R2b** — Gram-Schmidt 硬约束 | ✅ | `pinn_art/physics/orthogonalizer.py::gram_schmidt_ortho_pq` + 测试 |
| **R3** — config + 续训脚本 | ✅ | `configs/v3_stage_a_laguerre_basis.yaml` + `scripts/v3_train_stage_a_laguerre_basis.py` |
| **R4** — 形态学验证 | ⏳ | `v3_compare_fac_li_pqv.py` 仍可复用；新一轮形态门禁待加 |
| **R5** — 文档同步 | ✅ | `README.md` 本节；PROGRESS_REPORT 待补 |

**当前验证**：
- 43/43 个相关测试通过（含原 `test_deeponet_forward / test_full_forward_smoke / test_train_step_smoke / test_hydrogenic_P_jax / test_orthogonalizer` + 16 个新 Laguerre 测试 + DeepONet Laguerre 前向 + Gram-Schmidt）
- DeepONet 在 `use_laguerre_basis=True, learn_lambda=False` 时，P 在 epoch 0 即与 `hydrogenic_P_jax` 完全一致（"出场即巅峰" 验证）
- λ_a 初始化 = Z_eff/n（Z=3 时 λ=3.0）

**待办**：
1. **真实 SGD 训练**：当前 `v3_train_stage_a_laguerre_basis.py` 是 smoke 脚本，尚未接通 `ManifestDataset` + `train_step`；下一轮接入 `v3_train_stage_a.py`。
2. **冻结 / 解冻调度**：按 §6.3 表实现 epoch 分段逻辑。
3. **形态门禁**（R4）：把 §7.1 的"节点数硬保证"作为 Stage-A gate。

## 代码根目录（待生成）

```text
DiracNet_V3/rc_pinn_art_project/    # 见 10_project_layout.md
```
