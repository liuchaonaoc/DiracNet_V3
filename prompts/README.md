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

## 与历史版本关系

```text
V1 (PyTorch, RC+MLP)  →  谱学任务定义、NIST、Dirac 算子
V2 (PyTorch, B-spline+KAN)  →  单粒子 PDE 门禁、E_orb-only 纪律
V3 (JAX, DeepONet+SIREN+CI)  →  全栈 PINN-ART：势场 + CI + 截面 + ms 推断
```

## 代码根目录（待生成）

```text
DiracNet_V3/rc_pinn_art_project/    # 见 10_project_layout.md
```
