# 毫秒级高精度原子辐射转移计算引擎 (PINN-ART) 核心逻辑开发文档

## 1. 架构总览 (Architecture Overview)

本系统旨在通过深度学习的极速前向推断，替代传统全相对论原子结构代码（如 cFAC、GRASP）中极度耗时的多体自洽场（SCF）迭代计算，以实现 ms 级别的气体辐射转移实时响应。

系统架构分为四个可微模块：

1. **输入与映射层**：对径向坐标与原子特征进行非线性降维映射。
2. **PINN 径向发生器 (Neural Solver)**：求解局域中心势下的单电子 Dirac 径向方程。
3. **可微 CI 层 (Differentiable CI & NIST Injection)**：结合预计算的角向张量和 NIST 实验数据，完成哈密顿矩阵拼装与对角化。
4. **辐射可观测量推断层**：输出高精度的跃迁概率、振子强度与碰撞截面。

---

## 2. 模块一：输入映射与特征预处理 (Feature Preprocessing)

### 2.1 物理特征输入

- **全局参数**：原子序数 $Z$，核电荷分布。
- **组态参数**：采用 $jj$ 耦合；对于给定的平均组态，输入各相对论子壳层 $(n,\kappa)$ 的分数占据数（Fractional occupation numbers）$\omega_{n\kappa}$。相对论角量子数定义为
  $$\kappa = (l-j)(2j+1)\,.$$

### 2.2 径向坐标非线性变换 (Coordinate Transformation)

为克服高激发态和连续态波函数的高频振荡造成的「谱偏差（Spectral Bias）」，神经网络不直接在均匀 $r$ 坐标上学习。

- **变换公式**：采用 cFAC 核心的改进混合形式
  $$t(r) = c_1 r + c_2 \ln r\,.$$
- **逻辑要求**：所有网络输入 $r$ 必须先经此非线性层转化为 $t$，使高里德堡态的波函数振荡周期在 $t$ 空间内包含几乎相同数量的网格点。实现中，损失函数对 $r$ 的导数 $\dfrac{d}{dr}$ 须通过链式法则
  $$\frac{d}{dr} = \frac{dt}{dr}\,\frac{d}{dt}$$
  转化为对网络输入 $t$ 的导数。

---

## 3. 模块二：PINN 径向发生器 (Neural Dirac Solver)

本模块是一个多层感知机（MLP）或隐式神经表示网络，负责在平均势场下生成严格正交的单粒子波函数。

### 3.1 网络输出

给定输入 $(Z,\, t,\, \omega_{n\kappa})$，网络输出：

| 量 | 含义 |
|---|---|
| $V(r)$ | 全局中心有效势（全闭壳层时含核势与电子云屏蔽势） |
| $\epsilon_{n\kappa}$ | 单电子轨道能量（可学习参数） |
| $P_{n\kappa}(r),\, Q_{n\kappa}(r)$ | Dirac 旋量大、小分量 |

### 3.2 物理损失函数 (Physics-Informed Loss)

#### Dirac 方程残差 ($\mathcal{L}_{\mathrm{PDE}}$)

基于一维径向耦合 Dirac 方程（自然原子单位，$\alpha$ 为精细结构常数）构建惩罚项：

$$\mathcal{L}_{\mathrm{PDE1}} = \left\|\left(\frac{d}{dr} + \frac{\kappa}{r}\right) P_{n\kappa} - \alpha\left(\epsilon_{n\kappa} - V(r) + \frac{\alpha^2}{2}\right) Q_{n\kappa}\right\|_2^2$$

$$\mathcal{L}_{\mathrm{PDE2}} = \left\|\left(\frac{d}{dr} - \frac{\kappa}{r}\right) Q_{n\kappa} - \alpha\left(-\epsilon_{n\kappa} + V(r)\right) P_{n\kappa}\right\|_2^2$$

总 PDE 损失可取 $\mathcal{L}_{\mathrm{PDE}} = \mathcal{L}_{\mathrm{PDE1}} + \mathcal{L}_{\mathrm{PDE2}}$（或按实现加权求和）。

#### 正交归一化边界 ($\mathcal{L}_{\mathrm{Norm}}$)

$$\mathcal{L}_{\mathrm{Norm}} = \left(\int_0^\infty \left(P_{n\kappa}^2 + Q_{n\kappa}^2\right)\, dr - 1\right)^2$$

#### 渐近行为硬约束 (Asymptotic Hard-constraint)

在 $r \to \infty$ 区间，可强制网络输出乘以 Whittaker 函数的渐近衰减项，以固化边界条件。

### 3.3 连续谱相幅分离 (Phase-Amplitude Method)

若需计算碰撞截面等涉及自由电子的问题，极高能量的振荡态不宜由 PINN 直接拟合：

- **网络修改**：对连续谱，网络不直接输出 $P(r)$，而输出振幅函数 $A(r)$ 与相位函数 $\phi(r)$。
- **重建逻辑**：
  $$P(r) \approx \eta^{-1/2}(r)\,\sin\phi(r)\,,$$
  再根据相移 $\delta_l$ 进行分波求和（Partial wave summation）。

---

## 4. 模块三：可微组态相互作用层 (Differentiable CI)

利用深度学习框架（PyTorch / JAX）的张量运算，实现「纯代数」毫秒级组装。

### 4.1 角向积分与算符预计算 (Angular Constants)

原子的多体相互作用可因式化为角向积分与径向积分的乘积。角向积分（广义 Racah 代数、$6j/9j$ 符号及 Wigner–Eckart 约化矩阵元）与径向波函数无关。

**代码实现**：在 `init` 阶段预计算所有多体基底间角向耦合的常量张量（`requires_grad=False`），推断时仅做矩阵乘法。

### 4.2 径向张量积分 (Differentiable Radial Integration)

利用 PINN 输出的大、小分量张量，用 `torch.trapz` 等可微数值积分计算广义 Slater 积分 $R^k(\alpha\beta;\gamma\delta)$：

$$R^k = \iint r_>^{\,k+1}\, r_<^{\,k}\,
\bigl[P_\alpha(r_1)P_\gamma(r_1) + Q_\alpha(r_1)Q_\gamma(r_1)\bigr]
\bigl[P_\beta(r_2)P_\delta(r_2) + Q_\beta(r_2)Q_\delta(r_2)\bigr]
\, dr_1\, dr_2\,,$$

其中 $r_< = \min(r_1,r_2)$，$r_> = \max(r_1,r_2)$。

### 4.3 NIST 半经验硬注入 (Semi-empirical NIST Injection)

- **哈密顿矩阵组装**：由预计算角向系数与 Slater 径向积分，可微组装全相对论 CI 哈密顿矩阵 $\mathbf{H}$。
- **对角元残差替换（核心创新）**：将 $\mathbf{H}$ 划分为主组（Main group）与微扰组（Perturbing group）。对主组对角元 $H_{ii}$ 所对应的物理组态，强制用 NIST 高精度能级 $E_{\mathrm{NIST}}$ 替换（等效于 cFAC 的 `CorrectEnergy`）。
- **可微对角化**：
  ```python
  eigvals, eigvecs = torch.linalg.eigh(H_corrected)
  ```
  所得本征值 `eigvals` 具 meV 级绝对精度；本征向量 `eigvecs`（多电子混合系数 $b_\nu$）捕获高精度中间耦合效应。

---

## 5. 模块四：宏观辐射截面推断 (Macroscopic Observables)

借助本征向量与径向轨道，实时输出电磁场响应，供气体流体力学与辐射代码调用。

### 5.1 辐射跃迁率 (Radiative Transitions)

电偶极 (E1) 或更高阶磁多极 (M1, E2 等) 的跃迁振子强度 $gf$：结合混合系数 $b_\nu$ 与跃迁算符径向积分（如偶极积分 $\int P_f\, r\, P_i\, dr$）在自动微分框架下计算。因矩阵对角元已注入 NIST 数据，跃迁波长（能量差）可达极高精度。

### 5.2 碰撞与自由态截面 (Collisional Cross-sections)

根据因式化–插值定理（Factorization–Interpolation Method），碰撞截面分离为纯角向部分与随能量平滑变化的径向积分 $Q^k$。

**推断逻辑**：在少数能量点利用 PINN 给出的散射相移（库仑–玻恩近似或精确 DW 模式）计算 $Q^k$，对碰撞电子能量 $E_{\mathrm{col}}$ 在对数空间做平滑样条插值，在 ms 级输出等离子体碰撞电离（CI）、碰撞激发（CE）与双电子复合（DR）系数。

---

## 6. 开发与训练管线总结 (Training & Execution Pipeline)

| 阶段 | 内容 |
|---|---|
| **预计算** (Python / 符号代数) | 生成前 26 号元素所需的 $jj$ 耦合代数系数（Wigner $3j$、$6j$ 表），存为张量字典 |
| **PINN 预训练** (Offline) | 残差网络（或 Delta PINN），以自动微分约束 $V(r)$ 与 $P,Q$ 正交关系，学习一维相对论 Dirac 波函数生成律 |
| **实时推断** (Online, ms 级) | 输入瞬态等离子体参数、$Z$、目标组态 → PINN → 径向轨道 → 预计算张量相乘 → $\mathbf{H}$ → NIST 覆盖主对角元 → `eigh()` → 跃迁率与碰撞截面 |

**实现注记**：对对角化 `eigh()` 做反向传播或批量处理时，须对简并能级添加极小人工微扰 $\epsilon \mathbf{I}$，以防简并梯度爆炸（Degenerate Gradient Explosion）。
