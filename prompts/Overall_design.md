这份核心逻辑设计文档（Markdown格式）详细梳理了我们讨论的**“基于局部中心势与可微半经验组态相互作用的深度物理网络 (Local Central Potential PINN + Differentiable CI)”**架构。

该文档可直接作为后续编写 PyTorch 或 JAX 代码的底层物理逻辑与管线蓝图（Blueprint）。

***

# 毫秒级高精度原子辐射转移计算引擎 (PINN-ART) 核心逻辑开发文档

## 1. 架构总览 (Architecture Overview)
本系统旨在通过深度学习的极速前向推断，替代传统全相对论原子结构代码（如 cFAC, GRASP）中极度耗时的多体自洽场（SCF）迭代计算，以实现 ms 级别的气体辐射转移实时响应。
系统架构分为四个可微模块：
1. **输入与映射层**：对径向坐标与原子特征进行非线性降维映射。
2. **PINN 径向发生器 (Neural Solver)**：求解局域中心势下的单电子 Dirac 径向方程。
3. **可微 CI 层 (Differentiable CI & NIST Injection)**：结合预计算的角向张量和 NIST 实验数据，完成哈密顿矩阵拼装与对角化。
4. **辐射可观测量推断层**：输出高精度的跃迁概率、振子强度与碰撞截面。

---

## 2. 模块一：输入映射与特征预处理 (Feature Preprocessing)

### 2.1 物理特征输入
*   **全局参数**：原子序数 $Z$，核电荷分布。
*   **组态参数**：采用 $jj$ 耦合，对于给定的平均组态，输入各相对论子壳层 $(n, \kappa)$ 的分数占据数（Fractional occupation numbers） $\omega_{n\kappa}$。其中相对论角量子数 $\kappa = (l - j)(2j + 1)$。

### 2.2 径向坐标非线性变换 (Coordinate Transformation)
为克服高激发态和连续态波函数的高频振荡造成的“谱偏差（Spectral Bias）”，神经网络不直接在均匀 $r$ 坐标上学习。
*   **变换公式**：采用 cFAC 核心的改进混合形式 $t(r) = c_1 \sqrt{r} + c_2 \ln(r)$。
*   **逻辑要求**：所有的网络输入 $r$ 必须先经过此非线性层转化为 $t$，使得高里德堡态的波函数振荡周期在 $t$ 空间内包含几乎相同数量的网格点。在代码实现中，损失函数的导数 $\frac{d}{dr}$ 需要通过链式法则 $\frac{dt}{dr} \frac{d}{dt}$ 转化为对网络输入的导数。

---

## 3. 模块二：PINN 径向发生器 (Neural Dirac Solver)

本模块是一个多层感知机（MLP）或隐式神经表示网络，负责在平均势场下生成严格正交的单粒子波函数。

### 3.1 网络输出
给定输入 $(Z, t, \omega_{n\kappa})$，网络输出：
1.  **全局中心有效势**：$V(r)$（对于全闭壳层，势场本身包含核势和电子云屏蔽势）。
2.  **本征能量预测**：单电子轨道能量参数 $\epsilon_{n\kappa}$（作为可学习变量）。
3.  **单电子旋量分量**：Dirac 旋量的大分量 $P_{n\kappa}(r)$ 和小分量 $Q_{n\kappa}(r)$。

### 3.2 物理损失函数 (Physics-Informed Loss)
*   **Dirac 方程残差 ($Loss_{PDE}$)**：基于一维径向耦合 Dirac 方程 构建惩罚项：
    $$ Loss_{PDE1} = \left\| \left(\frac{d}{dr} + \frac{\kappa}{r}\right)P_{n\kappa} - \alpha\left(\epsilon_{n\kappa} - V(r) + \frac{2}{\alpha^2}\right)Q_{n\kappa} \right\|^2 $$
    $$ Loss_{PDE2} = \left\| \left(\frac{d}{dr} - \frac{\kappa}{r}\right)Q_{n\kappa} - \alpha(-\epsilon_{n\kappa} + V(r))P_{n\kappa} \right\|^2 $$
    *(注：公式中采用自然原子单位，$\alpha$ 为精细结构常数。)*
*   **正交归一化边界 ($Loss_{Norm}$)**：
    $$ Loss_{Norm} = \left( \int_0^\infty (P_{n\kappa}^2 + Q_{n\kappa}^2) dr - 1 \right)^2 $$
*   **渐近行为硬约束 (Asymptotic Hard-constraint)**：在 $r \to \infty$ 区间，可强制网络输出乘以 Whittaker 函数的渐近衰减项以固化边界条件。

### 3.3 连续谱相幅分离 (Phase-Amplitude Method)
若需要计算碰撞截面等涉及自由电子的问题，极高能量的振荡态禁止由 PINN 直接拟合：
*   **网络修改**：对连续谱，网络不输出 $P(r)$，而是输出振幅函数 $A(r)$ 和相位函数 $\phi(r)$。
*   **重建逻辑**：$P(r) \approx \eta^{-1/2}(r) \sin\phi(r)$，根据相移 $\delta_l$ 进行后续的分波法（Partial wave summation）计算。

---

## 4. 模块三：可微组态相互作用层 (Differentiable CI)

这是利用深度学习框架（如 PyTorch/JAX）张量运算特性的“纯代数”毫秒级组装模块。

### 4.1 角向积分与算符预计算 (Angular Constants)
原子的多体相互作用可被因式化为角向积分与径向积分的乘积。
*   **逻辑要求**：角向积分（如广义 Racah 代数、6j/9j 符号 以及 Wigner-Eckart 定理的约化矩阵元）**完全独立于径向波函数**。
*   **代码实现**：在初始化（__init__）阶段，预先计算所有多体基底之间角向耦合的常量张量（Constant Tensors），并在计算图中固化（`requires_grad=False`），推断时仅执行矩阵乘法。

### 4.2 径向张量积分 (Differentiable Radial Integration)
利用 PINN 输出的单电子大分量和小分量张量，使用 `torch.trapz` 等可微数值积分函数计算广义 Slater 积分 $R^k(\alpha\beta; \gamma\delta)$：
$$ R^k = \int \int \frac{r_<^k}{r_{>}^{k+1}} [P_\alpha(r_1)P_\gamma(r_1) + Q_\alpha(r_1)Q_\gamma(r_1)][P_\beta(r_2)P_\delta(r_2) + Q_\beta(r_2)Q_\delta(r_2)] dr_1 dr_2 $$

### 4.3 NIST 半经验硬注入 (Semi-empirical NIST Injection)
*   **哈密顿矩阵组装**：通过预计算的角向系数与 Slater 径向积分，可微地组合出相互作用组态（CI）的全相对论哈密顿矩阵 $H$。
*   **对角元残差替换 (核心创新点)**：
    将 $H$ 矩阵划分出主组（Main group）和微扰组（Perturbing group）。
    寻找对角线元素 $H_{ii}$ 对应的物理组态，**强制用外部提供的 NIST 数据库高精度能级 $E_{NIST}$ 替换该元素的值（等效于 cFAC 的 CorrectEnergy 思想）。**
*   **可微对角化**：
    执行 `eigvals, eigvecs = torch.linalg.eigh(H_corrected)`。
    得到的特征值 `eigvals` 将直接具有 meV 级的绝对精度；特征向量 `eigvecs` (即多电子混合系数 $b_\nu$) 捕获了极高精度的中间耦合效应。

---

## 5. 模块四：宏观辐射截面推断 (Macroscopic Observables)

借助特征向量与径向轨道，实时输出电磁场作用下的各类响应，直接供气体流体力学与辐射代码调用。

### 5.1 辐射跃迁率 (Radiative Transitions)
电偶极 (E1) 或更高阶磁多极 (M1, E2 等) 的跃迁振子强度 $gf$：
直接结合特征向量 $b_\nu$ 与跃迁算符径向积分（如偶极积分 $\int P_f r P_i dr$），调用自动微分框架极速计算，因矩阵的对角元已注入 NIST 数据，辐射跃迁波长（能量差）将达到极高精度。

### 5.2 碰撞与自由态截面 (Collisional Cross-sections)
根据因式化-插值定理（Factorization-Interpolation Method）：
*   碰撞截面分离为纯角向部分与随能量平滑变化的径向积分 $Q_k$。
*   推断逻辑：在少数能量点利用 PINN 给出的散射相移（基于库仑-玻恩近似或精确 DW 模式）计算 $Q_k$，对碰撞电子连续能量 $E_{col}$ 采取对数空间的平滑样条插值，在 ms 级别输出等离子体碰撞电离（CI）、碰撞激发（CE）和双电子复合（DR）系数。

---

## 6. 开发与训练管线总结 (Training & Execution Pipeline)

1.  **预计算阶段 (Python / 符号代数)**：生成前 26 号元素所需的 $jj$ 耦合代数系数（通过 Wigner 3j, 6j 系数表计算），存储为张量字典。
2.  **PINN 预训练阶段 (Offline Training)**：利用残差网络（或 Delta PINN），采用自动微分约束网络权重，使其完美学习到满足 $V(r)$ 与 $P, Q$ 正交关系的一维相对论 Dirac 波函数生成律。
3.  **实时模拟推断阶段 (Online Inference - ms 级)**：
    *   **输入**：瞬态等离子体参数、$Z$、目标组态。
    *   **前向传播**：PINN $\to$ 径向轨道 $\to$ 预计算张量相乘 $\to$ 形成 $H$ 矩阵。
    *   **半经验注入**：读取 NIST 残差词典，覆盖 $H$ 主对角元。
    *   **对角化与截面输出**：计算 `eigh()` 获取高精度本征波函数，进而以解析公式瞬间完成跃迁率、碰撞截面的可微计算并返回模拟主程序。

*实现注记：在实现对角化 `eigh()` 层的反向传播或批量处理时，需对简并能级添加极小的人工微扰项 $\epsilon I$，以防止简并梯度爆炸（Degenerate Gradient Explosion）。*