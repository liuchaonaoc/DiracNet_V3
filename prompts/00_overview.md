# 00 — Overview / V3 (PINN-ART) 设计哲学

> 阅读优先级：**最高**。动手写代码前读完本文 + [`Overall_design.md`](./Overall_design.md)。

## 1. 项目定位

**PINN-ART**（Physics-Informed Neural Network for Atomic Radiative Transfer）= DiracNet V3。

在 V2 已验证的「单粒子 Dirac PDE + 波函数正交」基础上，完成 Overall_design 中的完整管线：

```text
坐标映射 t(r)  →  神经 Dirac 求解器 (V, P, Q)  →  可微 CI（NIST 注入 / 理论回退）  →  辐射/碰撞可观测量
```

| 维度 | V2 | V3 |
|------|----|----|
| 框架 | PyTorch | **JAX**（`jit` / `vmap` / `grad`） |
| 波函数表示 | envelope × B-spline | **DeepONet trunk(SIREN)** + 全局 $V(r)$ |
| 多体能 | Koopmans 求和 $E_\mathrm{orb}$ | **Slater $R^k$ + Racah + `eigh`** |
| 能量精度目标 | PDE 轨道能（~meV 类氢） | **NIST 有则注入、无则理论回退 → 有数据区 meV 级** |
| 输出 | 标量 $E_\mathrm{pred}$ | **$A_{ki}$, $gf$, CE/CI/PI/RR 截面** |
| 推断延迟 | 训练后 forward ~10 ms 级 | **目标 ms–μs（`jit` 热身后）** |

## 2. 四条不可违反纪律

### 2.1 物理主链优先于数据拟合

训练分阶段（见 `07_training_pipeline.md`）：

1. **Stage A**：只优化 Dirac PDE + 正交 + 渐近；**NIST 不进 backward**。
2. **Stage B**：冻结 trunk/branch 主体，只训 CI 径向积分头或低秩修正（若有）。
3. **Stage C**：对角元 **混合填充**——`nist_mask=True` 时用 NIST 实验能级硬替换 $H_{ii}$；`nist_mask=False` 时 **Fall-back** 到 PINN/CI 理论对角元 $H_{ii}^{\mathrm{theory}}$（极高激发态、极高电荷态等 NIST 未收录情形）。不对角元做 lookup bias。

禁止：per-row learnable bias、Stage A 使用 NIST 梯度、用 residual MLP 替代 CI 本征分解、在 mask=False 处用零或任意常数占位（必须走理论回退）。

### 2.1.1 NIST 覆盖与理论回退（Fall-back）

| 数据 | `nist_mask` | $H_{ii}$ 来源 |
|------|-------------|----------------|
| NIST 有收录能级 | `True` | $E_\mathrm{NIST}$（硬注入） |
| NIST 无收录（高 $n$、高电离度等） | `False` | $H_{ii}^{\mathrm{theory}}$（装配后的理论对角元，可微） |

跃迁层：无实验 $A_{ki}$ / $gf$ 时，仅用 CI 本征向量 + 多极矩阵元的 **理论预测**，不插值或臆造实验值。

### 2.2 角向代数离线、径向在线可微

- Racah / Wigner-Eckart 系数 → **预计算 `.npz`，`stop_gradient`**
- Slater 径向积分 $R^k_{ab,cd}$ → **由 $(P,Q)$ 在网格上积分，全程可微**

### 2.3 `eigh` 必须带简并保护

哈密顿量对角化出现在 Stage B/C；梯度含 $(E_i-E_j)^{-1}$。必须：

- 对角元加 $\epsilon \sim 10^{-6}$ eV 破缺，和/或
- Custom VJP 截断（见 `14_degenerate_gradient_safety.md`）

### 2.4 评估分物理层报告

任何报告必须分层：

```text
Layer-0: |cos(P,Q)|, L_PDE, λ/V 合理性
Layer-1: E_orb / Slater 积分自洽
Layer-2: CI 本征能 vs NIST (meV)
Layer-3: A_ki, gf, 截面 vs FAC/R-matrix 基准
Layer-4: 单样本 jit 延迟 (ms)
```

## 3. 神经网络选型（相对 V2 的变更）

| 组件 | 结构 | 理由 |
|------|------|------|
| 算子骨架 | **DeepONet** (branch + trunk) | 离散 $(Z,\omega_{n\kappa})$ → 连续 $(V,P,Q)(r)$ |
| Trunk 激活 | **SIREN** $\sin(\omega x+b)$ | 克服 ReLU 谱偏差，适合 Dirac 节点 |
| 连续谱 | **双头**：振幅 $A(r)$ + 相移 $\phi(r)$ | 相幅分离，避免拟合高频散射振荡 |
| 弃用 | V2 的 B-spline+KAN 作为主假设类 | V3 用算子学习统一势场与波函数 |

KAN/B-spline 可作为 **debug baseline**（`configs/ablation_bspline.yaml`），不是主路径。

## 4. 计算框架：JAX

```python
# 纪律示例
@jax.jit
def forward(params, batch): ...

# 能量网格 / 碰撞能量轴
vmap_over_E = jax.vmap(partial(cross_section, params), in_axes=(None, 0))
```

- 数组默认 `float32`；积分权重、`eigh` 临时用 `float64` 可选。
- 随机性：`jax.random` + 显式 `key` 传入。
- Checkpoint：`orbax` 或 `pickle`+`flax` `train_state`（见 `10_project_layout.md`）。

PyTorch 仅用于 **数据预处理脚本**（复用 V2 `nist_loader`）可接受；**训练与推断核心必须在 JAX**。

## 5. 不在 V3 第一版范围

- 全组态相互作用（完整 MRCI）
- Breit / QED 完整修正
- $Z > 26$ 或 $n > 15$ 的生产数据
- 分子 / 晶体
- 分布式多 GPU 训练（单卡 `jit` 优先）

## 6. 成功判据（Overall_design §七）

| 指标 | 目标 |
|------|------|
| 能级 MAE（仅 `nist_mask=True` 子集） | **< 5 meV**（闭壳层）；复杂组态 < 50 meV；Fall-back 区不评实验 MAE |
| $A_{ki}$ / $gf$ | 与 FAC  Leading % 一致；相对误差中位数 < 20% |
| 碰撞截面 | 10–20% vs 基准；高能 Bethe 标度正确 |
| 单次 `jit` forward（Fe XVII 量级） | **< 10 ms**（A100/4090 级）；热身后 **< 1 ms** 为 stretch goal |

## 7. 文档冲突裁决

```text
14_degenerate_gradient_safety.md  >  04_differentiable_ci.md
07_training_pipeline.md           >  06_physics_losses.md（权重调度）
15_code_structure_generation.md   >  10_project_layout.md（文件清单以 15 为准）
```
