# 17 — 广义拉盖尔有限基架构（Differentiable Generalized-Laguerre Basis）

> 阅读优先级：**★★★**（Stage A Round 2 的下一轮架构升级）。  
> 上游必读：[`00_overview.md`](./00_overview.md) §2.1、[`03_neural_dirac_solver.md`](./03_neural_dirac_solver.md)、[`16_stage_a_selfconsistent_dfs.md`](./16_stage_a_selfconsistent_dfs.md)。  
> 与 [`04_differentiable_ci.md`](./04_differentiable_ci.md) 配套——本架构下 \(R^k(a,b)\) 积分可对 Laguerre 系数解析或半解析计算。

## 0. 动机：为什么必须放弃「SIREN 直接输出形状」

`v3_compare_fac_li_pqv.py`（cFAC 对比）与 `PROGRESS_REPORT_2026_06_15_FINAL.md` 同时指向同一现象：

| 现象 | 当前架构根因 |
|------|------------|
| Li 2s 的 \(P(r)\) 整体相位右移 ~0.5–1 Bohr | 节点位置只能通过 SIREN 的乘性扰动 `P_H·(1+ε·tanh)` 间接调节，而节点附近 \(P_H=0\)，扰动失效 |
| 节点数偶然正确，但无法保证 | SIREN 输出连续形状，节点数依赖训练隐式涌现 |
| 节点位置梯度被 ∂(tanh)/∂r 的数值噪声淹没 | 256 点网格上 SIREN 的形状抖动全部进入 Dirac PDE 残差 |
| 不同轨道之间需要复杂的 ortho_loss 软约束 | SIREN 形状相互独立，无内在正交性 |
| Z_eff 由 Slater 规则粗估，骨架节点位置先天偏移 | 网络需重新学整条径向曲线 |

**核心诊断**：当前 Trunk 把"形状"当成自由函数去拟合，把本该由**物理量子数 \(n,l\) 锁定**的结构（节点数 = \(n-l-1\)、节点相对顺序、积分权重）让网络重新学一遍——**自由度没限定到物理量**。

## 1. 设计目标

把 \(P_{n\kappa}(r)\) 从「神经网络拟合形状」升级为「**正交有限基展开**，系数由网络预测」：

1. **节点数等于 \(n-l-1\)** —— 由 Laguerre 多项式阶数天然保证（架构保证，硬约束）
2. **节点位置由多项式零点解析决定**，不依赖网络搜索（消除"根位置寻优"的病态梯度）
3. **近核 / 渐近行为自动满足** —— \(P \sim r^{|\kappa|}\) 与 \(\exp(-\lambda r)\) 由显式因子保证
4. **残差对系数 \(c_k\) 是线性叠加关系 → 凸优化** —— 梯度平滑稳定，无 NaN 风险
5. **λ 由网络预测（softplus 正约束）**，初始化 = \(Z_{\mathrm{eff}}/n\)，允许空间展衍度作为可调自由度
6. **Q 由动能平衡骨架生成**，网络只学相对论高阶小修正；**禁止独立拉盖尔展开 Q**（避免变分坍塌）
7. **与现有氢样骨架、动能平衡 \(Q\)、自洽 DFS 势完全兼容**

## 2. 核心数学：广义拉盖尔有限基展开

$$\boxed{
P_{n\kappa}(r) = r^{|\kappa|}\, e^{-\lambda_a r}\, \sum_{k=0}^{K} c_k^{(a)}\, L_k^{\alpha}(2\lambda_a r), \qquad \alpha = 2|\kappa|
}$$

其中：
- \(L_k^\alpha\) 是**广义（缔合）拉盖尔多项式**，阶数 \(k\)，参数 \(\alpha = 2|\kappa|\)
- \(K\) 为截断阶数（见 §3.4 选择规则）
- **\(\lambda_a\) 为可学衰减因子**（per-orbital，softplus 正约束，初始化 = \(Z_{\mathrm{eff}}/n\)）—— 见 §2.5
- \(\{c_k^{(a)}\}_{k=0}^K\) 为**网络要预测的系数**

### 2.1 为什么 Laguerre 基合适

| 属性 | 说明 |
|------|------|
| **包含正确渐近** | 因子 \(e^{-\lambda r}\) 保证 \(r \to \infty\) 衰减 |
| **包含正确近核** | 因子 \(r^{|\kappa|}\) 保证 \(r \to 0\) 行为 |
| **节点数硬保证** | \(L_k^\alpha\) 含 \(k\) 个正实数零点 ⇒ \(\sum c_k L_k\) 含**最多** \(K\) 个节点（若 \(c_K \neq 0\)） |
| **正交性** | \(\int_0^\infty \rho^\alpha e^{-\rho} L_k^\alpha(\rho) L_j^\alpha(\rho) d\rho = \frac{\Gamma(k+\alpha+1)}{k!}\delta_{kj}\)，\(\rho=2\lambda r\) |
| **梯度稳定** | 残差 \(\mathcal{L}\) 对 \(c_k\) 的梯度 \(\partial \mathcal{L} / \partial c_k\) 仅依赖基函数值，**无根位置寻优的病态** |
| **物理对应** | 等价于 MCHF / STO（Slater 型轨道）的有限基展开，与 cFAC / GRASP2k 基组方法同构 |

### 2.2 节点数与 \(K\) 的关系

- 单个 Laguerre 多项式 \(L_k^\alpha\) 含 **\(k\) 个**正根
- 截断展开 \(\sum_{k=0}^K c_k L_k^\alpha\) 在 \(c_K \neq 0\) 时含**最多** \(K\) 个节点
- 物理 \(P_{n\kappa}\) 应含 \(n-l-1\) 个节点 ⇒ **推荐 \(K = n-l\)**（留 1 阶自由度给形状调整）
- 训练初期可对 `c_K` 加大正则，保证初始节点数 = \(K\) 而非 \(K-1\)

### 2.3 与氢样骨架的兼容性

氢样 \(P_H(r; Z, n, l)\) 本身就是 \(K=n-l\) 的 Laguerre 展开（`pinn_art/physics/hydrogenic.py` 已有 `genlaguerre(n-l-1, 2l+1)`），系数可解析算出 \(c_k^H\)。

**初始化策略**：令网络初始输出 \(c_k = c_k^H + \epsilon_k\)，其中 \(\epsilon_k\) 小扰动。**网络初值即精确等于氢样波函数**，无需"先冻结再放开"的多阶段训练。

### 2.4 与 cFAC 有限基方法的对应

| cFAC | PINN-ART（本方案） |
|------|-------------------|
| Slater 型轨道 \(\chi_k = r^{n_k} e^{-\zeta_k r}\) | 拉盖尔多项式 \(r^{|\kappa|} e^{-\lambda r} L_k^{2|\kappa|}(2\lambda r)\) |
| 基组系数（SCF 求解） | 网络输出（梯度下降） |
| 数值 SCF 迭代 | JAX 反向传播（一次 forward + backward） |
| 基函数正交性 | **同 \(|\kappa|\) 同 \(\lambda\) 时**正交；不同轨道间需保留 ortho_loss（见 §2.6） |
| 基组尺寸 Nbasis ≈ 6–10 | 截断 K = n-l（自适应） |

### 2.5 为什么 λ 必须可学

**物理动机**：在多电子屏蔽势中，**空间展衍度（Spatial Extent）** 偏离裸 Slater \(Z_{\mathrm{eff}}/n\)：

- Li 2s 的 cFAC 对比：节点数正确，但 P 的整体空间分布仍有偏差（参见 `logs/fac_vs_pinn_li/PQ_fac_vs_pinn.png` 中 2s 峰位偏移）
- 1s 与 2s 的有效电荷 \(Z_{\mathrm{eff}}\) 截然不同；2p 与 2s 的角动量 \(|\kappa|\) 不同 → 每轨道应有**自己的 \(\lambda_a\)**

**架构**：

```python
λ_a = softplus(lambda_head(branch_feat_a)) + λ_a_init
λ_a_init = Z_eff_a / n_a                              # 氢样初始化
```

- \(\lambda_a\) 由 **per-orbital** 网络头输出（标量），**正约束**通过 softplus 实现
- 初始化 = 氢样值 → 训练初期形态完全等于氢样骨架
- 训练中可微调 → 用最小 \(K\) 即可拟合多体屏蔽引起的展衍

**为什么不能省**：固定 \(\lambda = Z_{\mathrm{eff}}/n\) 时，要拟合大尺度弥散必须把高阶 \(c_k\) 推得很高，违背低截断 \(K\) 的设计目标。

### 2.6 基底正交性的精确边界与 ortho_loss

**数学上的严格正交**仅在**同 \(\alpha\)、同 \(\lambda\)** 下成立：

$$\int_0^\infty \rho^\alpha e^{-\rho} L_k^\alpha(\rho) L_j^\alpha(\rho)\, d\rho = \frac{\Gamma(k+\alpha+1)}{k!}\,\delta_{kj} \quad \Longleftarrow \alpha, \lambda \text{ 相同}$$

**多电子体系中不满足**：

| 轨道对 | 是否同 \(|\kappa|\)？ | 是否同 \(\lambda\)？ | 是否天然正交？ |
|--------|--------------------|--------------------|--------------|
| 1s vs 2s | ✓（都是 l=0） | ✗（\(Z_{\mathrm{eff}}\) 不同） | **否** |
| 2s vs 2p⁻ | ✗（l=0 vs l=1） | ✓ | **否** |
| 2p⁻ vs 2p⁺ | ✓ | ✓ | ✓ |

**结论**：跨轨道正交**不能依赖基底内禀**，必须靠两种方式之一：

1. **保留 ortho_loss**（推荐默认）：权重降为 1.0–10.0（远低于当前的 100.0），足以惩罚偏差
2. **显式 Gram-Schmidt 正交化**：在 forward 最后一步，对 (P, Q) 做轻量正交化（与现有 `orthogonalizer.py` 类似）

**推荐组合**：保留 ortho_loss（软约束）+ 在 forward 中显式正交化（硬约束）= 双重保障。

### 2.7 Q(r) 的生成：动能平衡 + 小修正

**严禁**：把 \(Q(r)\) 当成独立拉盖尔展开去拟合 —— 会导致**变分坍塌（Variational Collapse）**：
- \(Q\) 的物理意义依赖 Dirac 算子：\(Q = \frac{c}{2c^2-V}\left(\frac{dP}{dr} + \frac{\kappa}{r}P\right)\)
- 自由展开会让网络在 \((P, Q)\) 空间中找到数值上残差小但**物理无意义**的解

**正确架构**（动能平衡 + 微扰）：

```python
# 1. 骨架：由 P 的解析梯度与 V 构造（非自由参数）
Q_skel_a(r) = c · (dP_a/dr + κ_a/r · P_a) / (2c² - V(r))

# 2. 修正：网络仅学 Q_skel 附近的小扰动
δQ_a(r) = small_scale_Q · tanh(SIREN_Q_a(r))    # small_scale_Q ≤ 0.05

# 3. 组合
Q_a(r) = Q_skel_a(r) + δQ_a(r)
```

**关键性质**：
- \(\delta Q_a\) 由 SIREN 输出，但**振幅极小**（\(\le 5\%\)），无法破坏相对论极限行为
- 强相对论区（核区 \(r \to 0\)）的 \(Q\) 行为完全由动能平衡决定
- 训练目标：让 \(\delta Q\) 拟合 Breit、QED、真空极化等**高阶相对论修正**

> 注：与 cFAC / MCHF / STO 的对应见 **附录 B**。

## 3. 架构总览

```text
Branch (per orbital a):
   branch_feat ─→ MLP ─→ b [B, D_branch]
                          ├→ coeff_head_a  → coeffs_a [B, K_max+1]    # 全部 K_max+1 个系数
                          │                                            # 无效轨道用 orb_mask 屏蔽
                          ├→ lambda_head_a → λ_a [B]                  # ★ per-orbital 可学衰减因子
                          │                                            #   softplus 正约束，init = Z_eff_a/n_a
                          └→ q_corr_head_a → δQ_a(r) [B, N_grid]    # ★ Q 的高阶修正（振幅 ≤ 5%）

Trunk (per orbital a):
   ρ_a(r)   = r^|κ_a| · exp(-λ_a · r)                                  # λ_a per-orbital，可学
   base_a   = SIREN(t, b)                                              # P 的小幅 SIREN 修正（可选）
   L_stack  = laguerre_stack(2λ_a r, α=2|κ_a|)                          # [B, N_grid, K_max+1]
   perturb_P = tanh(base_a) · perturb_scale_P                          # perturb_scale_P ≤ 0.05

   P_a(r) = ρ_a · Σ_{k=0}^{K_a} coeffs_a[k] · L_stack[..., k] · (1 + perturb_P)

   dP_a/dr = analytic_grad(P_a)                                        # 解析梯度（关键）

   Q_skel_a(r) = c · (dP_a/dr + κ_a/r · P_a) / (2c² - V(r))           # ★ 动能平衡骨架（非自由）
   Q_a(r)     = Q_skel_a(r) + q_corr_a(r)                              # ★ δQ 由网络输出但振幅极小
   dQ_a/dr   = analytic_grad(Q_a)
```

### 3.1 关键算子

- `r^|κ|`：近核因子（与 FAC/Grant 约定一致）
- **`e^{-λ_a r}`**：渐近衰减因子；**λ_a per-orbital 可学**（softplus 正约束，init = Z_eff/n）—— 见 §2.5
- `L_k^{2|κ|}(2λr)`：广义拉盖尔多项式，**JAX 实现见 §3.2**
- 系数 `coeffs_a[k]`：**网络唯一要学的径向自由度**
- 修正项 `1 + perturb_scale · tanh(base)`：可选，幅度 `perturb_scale ≤ 0.05`（远小于 Laguerre 系数贡献）
- **`Q_skel_a(r)`**：动能平衡骨架（**非自由参数**）；`δQ_a` 为 SIREN 输出但振幅 ≤ 5%

### 3.2 Laguerre 多项式的 JAX 实现

项目 `pinn_art/physics/hydrogenic.py` 已有 `_laguerre_generalized_stack`，直接复用：

```python
# 已存在（pinn_art/physics/hydrogenic.py:42-65）
def _laguerre_generalized_stack(
    rho: jnp.ndarray, alpha: jnp.ndarray, max_k: int
) -> jnp.ndarray:
    """L_k^alpha(rho) for k = 0..max_k, return stack [max_k+1, ..., N_g]."""
    # 用三阶递推：L_{k+1} = ((2k+1+α-ρ) L_k - (k+α) L_{k-1}) / (k+1)
```

**无需新写**。直接调用 `laguerre_generalized_stack(2λr, α, K_max)` 即可。

### 3.3 系数参数化

```python
# 每轨道 a 输出 (K_max + 1) 个系数
# K_max = max_{a in active} (n_a - l_a)   ← 取所有活跃轨道的最大截断
coeffs_a = coeff_head(branch_feat_a, kappa_a, n_a, l_a)   # [B, K_max+1]

# orb_mask=False 的位置：coeffs 任意（不影响 forward）
# 无效 k 位置（k > n_a - l_a）：直接置零或 mask 屏蔽
```

**推荐**：让 `coeff_head` 输出所有 `K_max + 1` 个系数，在 forward 中用 `(k <= n_a - l_a)` mask 屏蔽高阶项，使不同轨道的有效阶数自适应。

### 3.4 截断阶数 \(K\) 的选择

| Z 范围 | n 范围 | 推荐 K |
|--------|--------|--------|
| Z ≤ 8 | n ≤ 6 | `n - l`（最少覆盖 1 个节点 + 1 阶形状自由度） |
| Z ≤ 26 | n ≤ 10 | 同上；`K_max = 9` 即可覆盖所有 ns/np/nd 轨道 |

**固定 `K_max = 9`**：所有活跃轨道都用同一组系数头，前 `K_a` 阶有效；不影响参数总量（每轨道最多 10 个系数）。

### 3.5 与氢样骨架的关系

```python
# 氢样展开系数（解析）
def hydrogenic_coefficients(Z, n, l, K_max):
    """P_H(r) = r^l exp(-Z r/n) Σ_k c_k^H L_k^{2l}(2Zr/n)
    c_k^H 是解析常数（标准氢原子）。"""
    alpha = 2 * l
    lam = Z / n
    norm_const = sqrt((2Z/n)^3 · (n-l-1)! / (2n · (n+l)!))
    c_H = {(n-l-1): norm_const · (-1)^(n-l-1)}   # 仅一项非零
    return c_H

# PINN 初始系数：c_init = c_H + 0
coeffs_init = c_H  (对每个活跃轨道)
```

**好处**：网络初始化即精确等于氢样骨架，`perturb_eps · tanh(SIREN)` 修正项从 0 开始 → **无需预热阶段**，训练立即进入物理收敛区。

## 4. 接口与文件改动清单

### 4.1 新增 / 修改文件

| 路径 | 改动 |
|------|------|
| `pinn_art/nets/laguerre_basis.py`（新） | `LaguerreCoeffHead`（系数）、`LaguerreLambdaHead`（**可学 λ**）、`LaguerreQCorrHead`（**Q 修正**）、`laguerre_p_sum`、`kinetic_balance_q_with_corr`、`normalize_laguerre_coeffs` |
| `pinn_art/nets/deeponet.py` | `DeepONetDirac` 增加 `use_laguerre_basis: bool = False`、`K_max: int = 9`、`learn_lambda: bool = True`、`perturb_scale_P: float = 0.05`、`perturb_scale_Q: float = 0.05`、`init_from_hydrogenic: bool = True`；删除 `shape_a` SIREN 头（替换为系数头） |
| `pinn_art/physics/hydrogenic.py` | 新增 `hydrogenic_laguerre_coeffs(Z, n, l, K) -> ndarray`（解析系数），返回归一化 Laguerre 展开系数 |
| `pinn_art/losses/pde_loss.py` | **不变**（Dirac 残差与 P 表达解耦） |
| `pinn_art/losses/ortho_loss.py` | **保留 1.0–10.0 权重**（基底正交仅在同 λ 同 α 下成立；见 §2.6） |
| `pinn_art/losses/coeff_loss.py`（新） | `L_coeff_decay`（防级数发散）、`L_lambda_prior`（λ_a 偏离 Z_eff/n 的轻量正则） |
| `pinn_art/losses/q_corr_loss.py`（新） | `L_q_residual = ∫ ρ · (Q − Q_skel)² dr` —— 强约束 δQ 仅学高阶修正 |
| `pinn_art/ci/slater_radial.py` | 利用 `R^k = ∫ r^k P_a P_b dr` 在 Laguerre 系数上的**解析或半解析公式**（Gauss-Laguerre 积分），加速 CI 装配 |
| `configs/v3_stage_a_laguerre_basis.yaml`（新） | 启用此架构的最小 config |
| `tests/test_laguerre_basis.py`（新） | 单元测试：(a) 与氢样在 Z_eff=Z 极限下一致、(b) 节点数 = n-l-1、(c) 重整化保持 ∫P²dr、(d) Q 的变分残差、(e) 不同 λ 下正交性失效 |

### 4.2 关键接口

```python
class LaguerreCoeffHead(nn.Module):
    """Per-orbital head producing Laguerre expansion coefficients."""
    K_max: int = 9               # 截断阶数
    d_hidden: int = 128
    
    @nn.compact
    def __call__(self, branch_feat: jnp.ndarray, kappa: jnp.ndarray,
                 n_principal: jnp.ndarray, l_orbital: jnp.ndarray) -> jnp.ndarray:
        """
        Returns:
          coeffs: [B, N_orb, K_max+1]
                  无效 k 位置（k > n-l）由 mask 屏蔽
                  初始化 = hydrogenic_laguerre_coeffs(Z, n, l, K_max)
        """
        ...


class LaguerreLambdaHead(nn.Module):
    """Per-orbital learnable decay factor λ_a (softplus, init = Z_eff/n)."""
    d_hidden: int = 64
    
    @nn.compact
    def __call__(self, branch_feat: jnp.ndarray, lambda_init: jnp.ndarray) -> jnp.ndarray:
        """
        Returns:
          lambda_a: [B, N_orb]，softplus(net(branch)) + lambda_init
                    init 偏差接近 0 → 训练初期 = Z_eff/n
        """
        ...


class LaguerreQCorrHead(nn.Module):
    """Per-orbital SIREN correction to kinetic-balance Q."""
    d_hidden: int = 64
    
    @nn.compact
    def __call__(self, t_grid: jnp.ndarray, branch_feat: jnp.ndarray,
                 kappa: jnp.ndarray) -> jnp.ndarray:
        """
        Returns:
          deltaQ: [B, N_orb, N_grid]
                  输出 tanh(...)；forward 中乘 perturb_scale_Q ≤ 0.05
        """
        ...
```

```python
def laguerre_p_sum(
    coeffs: jnp.ndarray,        # [B, N_orb, K_max+1]
    laguerre_stack: jnp.ndarray,# [B, N_orb, K_max+1, N_grid]
    envelope: jnp.ndarray,      # [B, N_orb, N_grid]   = r^|κ| · exp(-λ_a·r)
    perturb_corr: jnp.ndarray | None = None,  # [B, N_orb, N_grid]
    perturb_scale: float = 0.05,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """P_a(r) = envelope_a · Σ_k coeffs_a[k] · L_stack_a[k] · (1 + perturb_scale·perturb_corr_a)
    
    Returns:
      P_a:  [B, N_orb, N_grid]
      dP_a: [B, N_orb, N_grid]   (解析梯度)
    """
    ...


def kinetic_balance_q_with_corr(
    P: jnp.ndarray, dPdr: jnp.ndarray, V: jnp.ndarray,
    kappa: jnp.ndarray, r: jnp.ndarray,
    deltaQ: jnp.ndarray | None = None,    # [B, N_orb, N_grid]
    perturb_scale_Q: float = 0.05,
    c: float = C_LIGHT,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Q = c·(dP + κ/r·P) / (2c² - V) + perturb_scale_Q · deltaQ
    
    骨架由 (P, dP, V) 解析决定；deltaQ 仅学高阶相对论修正。
    
    Returns:
      Q:  [B, N_orb, N_grid]
      dQ: [B, N_orb, N_grid]   (解析梯度)
    """
    inv_r = 1.0 / jnp.clip(r, 1e-8)
    kap_r = kappa[:, None].astype(P.dtype) * inv_r[None, :]
    denom = jnp.clip(2.0 * c * c - V, 1e-6, 1e10)
    Q_skel = c * (dPdr + kap_r * P) / denom
    Q = Q_skel + (perturb_scale_Q * deltaQ if deltaQ is not None else 0.0)
    return Q, ...


def hydrogenic_laguerre_coeffs(
    Z: float | jnp.ndarray,
    n: int | jnp.ndarray,
    l: int | jnp.ndarray,
    K_max: int = 9,
) -> jnp.ndarray:
    """解析氢样 Laguerre 展开系数（用于初始化与基准）。
    
    Returns:
      coeffs: [K_max+1]，c[k] = 0 当 k ≠ n-l-1，
              c[n-l-1] = sqrt((2Z/n)^3 · (n-l-1)! / (2n·(n+l)!))
    """
    ...
```

### 4.3 替换 `DeepONetDirac.__call__` 中的 P / Q 计算

```python
# 旧（基于节点位置的方案，§17 前身）：
nodes_a = nodes[:, a, :]
P_a, dP_a = radial_node_coulomb_sturmian(r_grid, z_skel[:, a], n_a, l_a, nodes_a, ...)
Q_a = kinetic_balance_q(P_a, dP_a, V, kap_a, r_grid)              # 旧：纯骨架

# 新（基于 Laguerre 系数 + 可学 λ + Q 修正）：
# (1) 拉盖尔系数 + 可学 λ 重建 P
coeffs_a   = coeffs[:, a, :]                                       # [B, K_max+1]
L_stack    = laguerre_stack[..., a, :, :]                          # [B, K_max+1, N_grid]
lam_a      = lambdas[:, a]                                         # [B]，可学
envelope   = r_grid[None, :] ** abs_kappa * jnp.exp(-lam_a[:, None] * r_grid[None, :])
P_a, dP_a  = laguerre_p_sum(
    coeffs_a[:, :, None], L_stack, envelope[:, None, :],
    perturb_corr=tanh(base_P_a)[..., None] if base_P_a is not None else None,
    perturb_scale=perturb_scale_P,
)

# (2) Q 由动能平衡骨架 + 网络 δQ 构成（避免变分坍塌）
deltaQ_a   = q_corr[..., a, :]                                     # [B, N_grid]
Q_a, dQ_a  = kinetic_balance_q_with_corr(
    P_a, dP_a, V, kap_a, r_grid,
    deltaQ=deltaQ_a, perturb_scale_Q=perturb_scale_Q,
)
```

**关键变化**：
- **加入可学 λ**（per-orbital softplus 输出）
- **Q 不再是纯骨架**，而是骨架 + 受限 δQ 网络（SIREN 输出振幅 ≤ 5%）
- **不再需要** `node_head`、`softplus_sorted`、`radial_node_coulomb_sturmian`

## 5. 损失函数改动

### 5.1 现有损失（**保留，ortho 权重需谨慎**）

| 损失项 | 公式 | 原权重 | 建议权重 | 备注 |
|------|------|------|--------|------|
| PDE | `∫((H-E)ψ)² dr` | 1.0 | 1.0 | 不变 |
| **ortho** | `Σ_{a<b} ⟨P_a,P_b⟩²` | 100.0 | **10.0**（**不能降到 1.0**） | **基底正交仅在同 λ 同 α 下成立**（§2.6）；降权会导致 1s/2s 不正交 |
| norm | `Σ_a (∫P²+Q² - 1)²` | 10.0 | 10.0 | 不变 |
| scf | `⟨(V_net - sg[V_dfs])²⟩_ρ` | 5.0 | 5.0–15.0 | 不变 |
| asym | `V(r→∞) → -(Z-N+1)/r` | 0.1 | 0.1 | 不变 |
| v_prior | `‖V - V_DFS‖²`（可选） | 1.0 | 1.0 | 不变 |

> **重要修正（Gemini 评审 §II.2）**：拉盖尔基底**仅在同 λ 同 α 下正交**。多电子体系中 1s/2s 的 λ 不同、2s/2p 的 α 不同，**基底不天然正交**。ortho_loss 权重保留 ≥ 10.0。**可选额外硬约束**：在 forward 末尾对 (P, Q) 做一次轻量 Gram-Schmidt 正交化（参考 `pinn_art/physics/orthogonalizer.py`）。

### 5.2 新增损失

```python
# 1. 系数衰减正则（防级数发散）
L_coeff_decay = mean( coeffs^2 / factorial(k_range) )            # k=0..K_max

# 2. 最高阶系数惩罚（保证初始节点数 = n-l 而非 < n-l）
L_top_coeff = mean( softplus(-coeffs[..., -1]) )                   # 鼓励 c_{K_max} > 0

# 3. 系数平滑正则（防相邻 r 处 P 抖动 — 替代 jnp.gradient 噪声）
L_coeff_smooth = mean( (dP/dr - analytic_dP_dr)^2 )               # 仅当用 SIREN 修正时

# 4. λ 偏离先验的轻量正则（防止 λ 飞离合理范围）
L_lambda_prior = mean( (lambda_a - Z_eff_a/n_a)^2 / (Z_eff_a/n_a)^2 )   # 权重 1e-3
# 注：这是软锚点，训练中允许 λ 偏离氢样，但限制发散

# 5. Q 残差约束（防止 δQ 偏离骨架过大 → 防变分坍塌）
L_q_residual = mean( integrate( rho * (Q - Q_skel)^2 ) )            # 权重 1.0（默认）
# 注：用 ρ 加权是关键 —— 高电子密度区 Q 必须贴近骨架
```

- `L_coeff_decay`：默认权重 `1e-3`，与 Stage A 默认一致
- `L_top_coeff`：默认权重 `1e-2`（鼓励 c_K 显著非零 → 节点数 = K）
- `L_coeff_smooth`：仅当 `perturb_scale > 0` 时启用，默认权重 `1e-3`
- `L_lambda_prior`：默认权重 `1e-3`（**软锚点**，允许偏离但防发散；详见 §2.5）
- **`L_q_residual`：默认权重 `1.0`**（**关键约束**，防止 δQ 偏离动能平衡骨架；详见 §2.7）

### 5.3 **不**新增的项（重要）

- ❌ **不**加节点位置单调性正则（系数本身无序约束）
- ❌ **不**加节点数残差（多项式阶数已硬保证）
- ❌ **不**改 PDE 公式
- ❌ **不**加显式 `‖coeffs - coeffs_hydrogenic‖²`（系数应由能量梯度驱动，不强行锚定）

## 6. 训练策略

### 6.1 阶段划分

| 阶段 | epoch 范围 | 配置 | 目标 |
|------|----------|------|------|
| **A — 自洽 DFS 全量预训练** | 0–50 | `use_laguerre_basis=True`，`init_from_hydrogenic=True`，`perturb_scale_P=0.0`，`perturb_scale_Q=0.0` | 仅训 V、SCF loss；P 完全由 Laguerre 系数初始化=氢样；Q 完全等于动能平衡骨架 |
| **B — 放开系数与 λ** | 50+ | `perturb_scale_P=0.0`，`perturb_scale_Q=0.0`，coeffs / λ 全部可学 | 让能量梯度驱动系数偏离氢样，λ 偏离 Z_eff/n |
| **C — 加 P 修正项** | 100+ | `perturb_scale_P=0.05`，`perturb_scale_Q=0.0` | P 加 SIREN 微调（捕捉电子-电子交换、Breit）；Q 仍为纯骨架 |
| **D — 加 Q 修正项** | 150+ | `perturb_scale_P=0.05`，`perturb_scale_Q=0.05` | Q 加受控修正（**强 L_q_residual 约束**）；形态 + 势 + CI 联合 |
| **E — 联合精修** | 200+ | 同上 + `scf` 权重提升到 10–15 | 终态精修 |

### 6.2 初始化

- `coeff_head` 权重 `W = 0, b = coeffs_hydrogenic`（输出初始为氢样系数）
- **`lambda_head`**：softplus 输出，bias 初始化为 `log(expm1(Z_eff/n))`（使初始 λ = Z_eff/n）
- **`q_corr_head`**：SIREN 权重 zero-init（初始 δQ = 0，即 Q 完全等于动能平衡骨架）
- `laguerre_stack` 在 forward 中按当前 `λ` 现算（无需缓存；不同轨道 λ 不同）

### 6.3 冻结 / 解冻调度

| Epoch | `coeff_head` | **`lambda_head`** | `SIREN_P_corr` | **`SIREN_Q_corr`** | `V_head` | ortho/norm |
|-------|-------------|-------------------|----------------|---------------------|----------|------------|
| 0–50 | **冻结**（输出氢样） | **冻结**（= Z_eff/n） | 冻结 | 冻结 | 训练 | 训练 |
| 50–100 | 解冻 | 解冻 | 冻结 | 冻结 | 训练 | 训练 |
| 100–150 | 解冻 | 解冻 | 解冻（small_P） | 冻结 | 训练 | 训练 |
| 150+ | 解冻 | 解冻 | 解冻 | **解冻（small_Q）** | 训练 | 训练 |

> 与原方案（节点位置）相比，**新增 λ 与 Q 修正两个解冻节点**——但因为初始化即等于物理骨架（系数 = 氢样、λ = Z_eff/n、Q = 动能平衡），每个阶段开始时网络已在物理收敛区附近，无预热需求。

## 7. 验收标准

### 7.1 形态层（Layer-0，硬门槛）

| 检查项 | 通过条件 | 测试 |
|--------|---------|------|
| **节点数恒等** | `count_sign_changes(P_a) == n - l - 1`，所有 step 恒成立 | `test_laguerre_basis.py::test_node_count` |
| **与氢样骨架极限一致** | `coeffs = hydrogenic_laguerre_coeffs(Z, n, l)` 且 `λ_a = Z_eff_a/n_a` 时，P 与 `hydrogenic_P_analytic` 数值一致（误差 < 1e-6） | `test_laguerre_basis.py::test_hydrogenic_limit` |
| **近核行为** | `P_a[0:5] / r[0:5]^|κ| → const` | 同上 |
| **渐近衰减** | `P_a[r>20] < 1e-10` | 同上 |
| **归一化** | `∫(P² + Q²) dr = 1`（重建后） | `test_laguerre_basis.py::test_normalization` |
| **基函数正交（同 λ 同 α）** | `⟨ρ^a L_k^a L_j^a⟩ = δ_kj · norm_k`（验证 §2.6 的理论边界） | `test_laguerre_basis.py::test_basis_orthogonality` |
| **不同 λ 下不天然正交** | `⟨P_1s \| P_2s⟩ ≠ 0`（用 λ_1s ≠ λ_2s 验证 §2.6 的失效条件） | `test_laguerre_basis.py::test_orthogonality_breaks` |
| **Q 残差（防变分坍塌）** | `∫ ρ·(Q − Q_skel)² dr / ∫ ρ·Q_skel² dr < 0.05`（即 δQ 振幅 < 5%） | `test_laguerre_basis.py::test_q_residual` |
| **λ 偏离合理性** | `|λ_a − Z_eff_a/n_a| / (Z_eff_a/n_a) < 0.5`（允许 50% 偏离，验证 §2.5 软锚点生效） | `test_laguerre_basis.py::test_lambda_range` |
| **与 cFAC Li 1s/2s 对比** | `cos(P, P_FAC) ≥ 0.95`（1s）/ `≥ 0.95`（2s） | `v3_compare_fac_li_pqv.py` |
| **V(r) 价层偏差** | `max_{r∈[1,8]} |V_net - V_FAC| < 0.2 Ha` | 同上 |

### 7.2 能量层（Layer-2，参考线）

| 体系 | 当前 E-prime | 目标 |
|------|-------------|------|
| Li 1s²2s¹（基态 2s） | ~−12 eV | ~−5.9 eV（cFAC） |
| Li 1s²2s¹ 1s | −95 eV | −65 eV（cFAC） |
| Z=3–4 l=0 训练集 MAE | 16.5 eV | < 5 eV |

> **重要**：能量不是本次架构升级的直接验收指标。**架构验收只看形态（节点数、节点位置、与氢样极限一致、cos(P_FAC)）**。

### 7.3 对比测试矩阵

| 测试 | 命令 |
|------|------|
| 算子测试 | `pytest tests/test_laguerre_basis.py -v` |
| 形态对比 | `python scripts/v3_compare_fac_li_pqv.py --ckpt <new_ckpt> --level "1s2 2s1"` |
| 门禁 | `python scripts/v3_gate_analytic.py --config configs/v3_stage_a_laguerre_basis.yaml` |
| 径向 PDE 分区 | `python scripts/v3_diagnose_pde_radial.py --ckpt <new_ckpt>` |

## 8. 风险与缓解

| 风险 | 缓解 |
|------|------|
| 截断阶数 K 太小 → 形态偏差大 | 默认 K_max = 9（覆盖 n≤10 全部）；K 不足时可通过增加 `perturb_scale` 缓解 |
| 高阶系数发散 → 数值溢出 | `L_coeff_decay` 正则；初始化 = 氢样系数；zero-init SIREN |
| Laguerre 多项式在 r→0 数值不稳 | `_laguerre_generalized_stack` 已用 `jnp.where` 处理边界 |
| V 仍可能过浅（沿用 §16 SCF 路径） | 同时执行 `v3_compare_fac_li_pqv.py` 验证 V，若 V_rmse > 0.2 Ha 优先解决 V |
| 不同 Z 的 K_max 不同 → JAX tracing 重编 | 固定 `K_max = 9`；mask 不同轨道用不同有效 K |
| **不同 λ 下正交性失效**（**§2.6 警示**） | **保留 ortho_loss ≥ 10.0**；可选在 forward 末尾对 (P, Q) 做轻量 Gram-Schmidt 正交化（双重硬约束） |
| **λ 偏离氢样导致 P 退化** | `L_lambda_prior` 软锚点（权重 1e-3，允许 50% 偏离）；初始化 = Z_eff/n 起点 |
| **Q 变分坍塌**（**§2.7 警示**） | **禁止 Q 独立拉盖尔展开**；Q 必须 = 动能平衡骨架 + δQ（δQ 振幅 ≤ 5%）；`L_q_residual` 权重 1.0 强约束 |
| `coeff_head` 维度 = `K_max+1`，参数量增加 | 每轨道仅 10 个数，远小于 SIREN 256 点；总参数反而减少 |
| 阶段 D（Q 解冻）引入新自由度但消解 δQ = 0 | `L_q_residual` 在阶段 D 强制训练 δQ 拟合高阶修正而非核心形态 |

## 9. 与已有 docs 的关系

- **替代** [`03_neural_dirac_solver.md`](./03_neural_dirac_solver.md) §2.1 的 P 输出层（Trunk 大幅简化）
- **兼容** [`16_stage_a_selfconsistent_dfs.md`](./16_stage_a_selfconsistent_dfs.md)：DFS 自洽势、V 解析、SCF 损失完全沿用
- **加速** [`04_differentiable_ci.md`](./04_differentiable_ci.md)：\(R^k\) 积分可用 Gauss-Laguerre 解析公式直接对系数计算，无需重积分网格
- **统一** [`14_degenerate_gradient_safety.md`](./14_degenerate_gradient_safety.md)：系数空间无 `softplus_sorted` 类的退化问题

## 10. 实施清单（按代码生成顺序）

1. **R0 — 测试先行**
   - [ ] `tests/test_laguerre_basis.py::test_hydrogenic_limit`（与氢样极限一致）
   - [ ] `tests/test_laguerre_basis.py::test_node_count`
   - [ ] `tests/test_laguerre_basis.py::test_basis_orthogonality`（同 λ 同 α）
   - [ ] `tests/test_laguerre_basis.py::test_orthogonality_breaks`（**不同 λ 下验证失效**——§2.6）
   - [ ] `tests/test_laguerre_basis.py::test_normalization`
   - [ ] `tests/test_laguerre_basis.py::test_q_residual`（**Q 防变分坍塌**——§2.7）
   - [ ] `tests/test_laguerre_basis.py::test_lambda_range`（λ 偏离合理性）
2. **R1 — 算子实现**
   - [ ] `pinn_art/physics/hydrogenic.py::hydrogenic_laguerre_coeffs`
   - [ ] `pinn_art/nets/laguerre_basis.py::LaguerreCoeffHead`
   - [ ] `pinn_art/nets/laguerre_basis.py::LaguerreLambdaHead`（**可学 λ**）
   - [ ] `pinn_art/nets/laguerre_basis.py::LaguerreQCorrHead`（**Q 受控修正**）
   - [ ] `pinn_art/nets/laguerre_basis.py::laguerre_p_sum`
   - [ ] `pinn_art/nets/laguerre_basis.py::kinetic_balance_q_with_corr`（**Q 骨架+δQ**）
   - [ ] `pinn_art/losses/coeff_loss.py`（含 `L_coeff_decay` + `L_lambda_prior`）
   - [ ] `pinn_art/losses/q_corr_loss.py`（**`L_q_residual`**）
3. **R2 — 模型集成**
   - [ ] `pinn_art/nets/deeponet.py::DeepONetDirac` 增加 `use_laguerre_basis` / `learn_lambda` 分支
   - [ ] `pinn_art/ci/slater_radial.py` 利用 Laguerre 系数的 \(R^k\) 加速
   - [ ] `pinn_art/physics/orthogonalizer.py` 加 `gram_schmidt_ortho_pq_laguerre`（**双硬约束**）
4. **R3 — 训练 config**
   - [ ] `configs/v3_stage_a_laguerre_basis.yaml`
   - [ ] 续训脚本：`scripts/v3_train_stage_a_laguerre_basis.py`
5. **R4 — 验证**
   - [ ] `v3_compare_fac_li_pqv.py` 报告节点数 / cos(P_FAC) / λ_a / δQ_residual
   - [ ] `v3_diagnose_pde_radial.py` 看 core/mid/tail 分区
   - [ ] 与原 eprime 模型对比 Li 1s²2s¹ 能量误差
6. **R5 — 文档**
   - [ ] 更新 `PROGRESS_REPORT` 加 §"形态学进展（拉盖尔基 + 可学 λ + Q 骨架）"
   - [ ] 在 `08_evaluation.md` 加入 Layer-0「节点数硬约束」+ 「Q 残差」门禁
   - [ ] 删除或归档 `17_explicit_radial_node_basis.md` 旧版本（节点位置方案）

## 11. 一句话总结

> 用**拉盖尔多项式系数**代替**SIREN 拟合形状**——节点数由多项式阶数硬保证，节点位置由多项式零点解析决定，系数对残差是线性关系（凸优化）；λ per-orbital 可学以拟合空间展衍；Q 由动能平衡骨架 + 受控 δQ 构成以避免变分坍塌；ortho_loss 保留 ≥ 10 以应对多电子 λ 不同时的正交性失效——**让 DeepONet 把容量花在 \(K+1\) 个系数、1 个 λ、1 个 δQ 上，而不是 256 点的形状细节**。

## 附录 A：与「显式节点位置」方案对比

| 方案 | 节点数 | 节点位置 | 残差对参数的梯度 | ortho | λ | Q | 实现 |
|------|-------|---------|------------------|-------|---|---|------|
| 当前（SIREN 乘性扰动） | 靠 P_H | 间接 | 数值不稳定 | 软约束 100 | 固定 | 动能平衡 | 0 |
| 显式节点位置（前身） | 架构保证 | 可微有序参数 | `softplus_sorted` 退化 | 软约束 | 固定 | 动能平衡 | 中 |
| **拉盖尔系数（本方案）** | **架构保证** | **多项式零点（解析）** | **线性，凸优化** | **软约束 ≥10 + 可选 Gram-Schmidt 硬约束** | **per-orbital 可学** | **骨架+δQ** | 中 |
| 变分有限基（自由选择基） | 架构保证 | 间接（系数） | 取决于基函数 | 取决于基函数 | 取决于基函数 | 取决于基函数 | 大 |

本方案相对前身的**三大增量**：
1. **λ 可学**：拟合多体屏蔽导致的空间展衍
2. **Q 受控修正**：δQ 振幅 ≤ 5%，由 `L_q_residual` 强约束，防变分坍塌
3. **双正交保障**：ortho_loss 软约束 + forward 末尾 Gram-Schmidt 硬约束

附录 B 已涵盖与 cFAC / MCHF / STO 的对应。

## 附录 B：与 cFAC / MCHF / STO 的对应

| 物理方法 | 基函数 | 系数来源 | 数值方法 |
|---------|--------|---------|---------|
| **cFAC** | Slater 型 / 数值径向 | SCF 求解 | 迭代 |
| **MCHF**（多组态 Hartree-Fock） | Slater 型 / 数值 | SCF 求解 | 迭代 |
| **STO-NG** | Slater 型轨道 | 拟合 | 最小二乘 |
| **PINN-ART（本方案）** | 拉盖尔多项式 | 网络输出 | 梯度下降 |

能量上界（cFAC）与精确（GRASP2k）的差距主要来自基组尺寸，本方案用 SIREN 残差 + 可调 λ 补足。**能量上 PINN-ART ≤ cFAC（基组 K_max 相同），但推断速度提升 10²–10³ 量级**（无 SCF 迭代）。

## 附录 C：残差对系数 \(c_k\) 的梯度解析

```python
# 残差定义（Dirac PDE）
def dirac_residual(params, batch):
    coeffs = params['coeffs']                                  # [B, N_orb, K+1]
    L_stack = laguerre_stack(...)                              # [B, N_orb, K+1, N_grid]
    envelope = compute_envelope(...)                            # [B, N_orb, N_grid]
    
    P = jnp.einsum('bok,bokg,bog->bog', coeffs, L_stack, envelope)  # 线性 in coeffs
    dP = compute_dP(coeffs, L_stack, dL_stack, envelope, denvelope)
    Q = kinetic_balance(P, dP, V, kappa, r)
    
    LP, LQ = dirac_apply(P, Q, dP, dQ, V, kappa, r)
    E = orbital_energy(P, Q, LP, LQ, grid)
    residual = (LP - E*P)^2 + (LQ - E*Q)^2
    
    return jnp.sum(residual)

# ∂residual/∂coeffs[k] = 2 · (LP - EP) · ∂LP/∂coeffs[k] + 2 · (LQ - EQ) · ∂LQ/∂coeffs[k]
# ∂P/∂coeffs[k] = envelope · L_stack[..., k]                    (闭式，无链式法则)
# ∂LP/∂coeffs[k] = (V - E) · ∂P/∂coeffs[k] + c · (-∂(dQ) + κ/r · ∂Q)
# ...
```

**关键性质**：∂P/∂coeffs[k] 不依赖其他系数 → 梯度计算图局部、无 NaN、无根位置寻优。

## 附录 D：与氢样骨架（已存在的 `hydrogenic_P_jax`）的代码复用

```python
# 现有函数（pinn_art/physics/hydrogenic.py:42-65）
def _laguerre_generalized_stack(rho, alpha, max_k):
    """L_k^alpha(rho) for k = 0..max_k, return stack [max_k+1, ..., N_g]."""
    ...

# 现有函数（pinn_art/physics/hydrogenic.py:69-121）
def hydrogenic_P_jax(r, Z, n, l, *, max_k=6):
    """P_H(r; Z, n, l) = r · R(r) = r^l · e^{-Zr/n} · Σ_k c_k · L_k^{2l+1}(2Zr/n)"""
    ...
```

**本方案的 `laguerre_p_sum` 与 `hydrogenic_P_jax` 是同一族函数**，只是把"固定系数"改为"网络输出系数"。**绝大部分代码可直接复用**，新增的 `laguerre_basis.py` 仅包含：
- `LaguerreCoeffHead`（替换 SIREN 头）
- `laguerre_p_sum`（组合系数 + 基函数）
- `hydrogenic_laguerre_coeffs`（提取解析系数用于初始化）