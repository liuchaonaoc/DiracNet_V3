# 03 — 模块 2：局部中心势 PINN（DeepONet + SIREN）

## 1. 物理目标

学习全局中心势 $V(r)$ 及在其下正交的单电子 Dirac 旋量 $(P_{n\kappa}, Q_{n\kappa})$，满足：

$$H_D \psi = E \psi, \quad \langle \psi_a | \psi_b \rangle = \delta_{ab}$$

$V(r)$ 由网络输出，**非** 简单 $-Z/r$；多电子屏蔽由 branch 条件化。

## 2. 网络结构

```text
pinn_art/nets/
├── __init__.py
├── siren.py              # SirenLayer, init_siren
├── deeponet.py           # DeepONetDirac
├── potential_head.py     # V(r) 正值/负值约束
└── phase_amplitude.py    # 连续谱双头（可选）
```

### 2.1 DeepONet

```python
class DeepONetDirac(nn.Module):
    """Flax Linen 或 Equinox Module。"""

    def __call__(self, branch_feat, t_grid, kappa, orb_mask):
        """
        branch_feat: [B, D_in]
        t_grid:      [N_grid]
        kappa:       [B, N_orb]
        orb_mask:    [B, N_orb]

        Returns:
            V:   [B, N_grid]
            P,Q: [B, N_orb, N_grid]
        """
```

**Branch**（MLP，3–4 层，宽度 128–256）：

```text
branch_feat → Dense → ... → b [B, D_p]
```

**Trunk**（SIREN，每层 $\sin(\omega_0 W x + b)$，$\omega_0=30$）：

```text
对每个轨道 a：
  x_a = concat(t_grid, embed(kappa_a), broadcast(b))     # [N_grid, D_t]
  hidden = SirenStack(x_a)                                  # [N_grid, D_h]
  P_a = head_P(hidden) * envelope_bound(r)                  # 可选包络
  Q_a = head_Q(hidden) 或 kinetic_balance(P_a, dP, V, kappa)
```

**势场头**（与轨道无关）：

```text
V_raw = SirenTrunk_shared(t)                               # [N_grid]
V = V_nuc + softplus(V_raw)   或   V = -Z/r + tanh(V_corr)
```

推荐：**$V = -Z/r + \Delta V_\mathrm{nn}(t)$**，$\Delta V$ 末层 zero-init。

### 2.2 SIREN 初始化

```python
def siren_init(key, layer_idx, fan_in, omega_0=30.0):
    # 第一层: W ~ U(-1/fan_in, 1/fan_in)
    # 深层:   W ~ U(-sqrt(6/fan_in)/omega_0, ...)
```

### 2.3 导数

在 $r$ 网格上用 **解析链式法则**：

```text
dP/dr = (dP/dt) * (dt/dr)
```

禁止对 PDE 残差使用高阶 `jax.grad` 穿过整个 trunk（慢且噪）；trunk 输出 $P(t)$ 后映射到 $r$。

### 2.4 正交化 `pinn_art/physics/orthogonalizer.py`

移植 V2 `lowdin_orthonormalize` 到 JAX：

```python
def lowdin_orthonormalize(P, Q, grid, orb_mask, dPdr, dQdr) -> dict:
    """返回正交后的 P,Q,dPdr,dQdr；度量 ∫(P²+Q²)dr。"""
```

### 2.5 Dirac 算子 `pinn_art/physics/dirac_operator.py`

与 V1/V2 公式一致（$c=1/\alpha$）：

```python
def dirac_apply(P, Q, dPdr, dQdr, V, kappa, r) -> tuple[Array, Array]:
    """返回 LP, LQ。"""
```

### 2.6 相幅分离（连续谱，Phase 2+）

```python
class PhaseAmplitudeHead(nn.Module):
    def __call__(self, trunk_hidden, t):
        A = softplus(head_A(...))      # 平滑振幅
        phi = head_phi(...)            # 相移，用 sin/cos 包装
        # P_cont = A * cos(kr - phi + ...)
```

束缚态训练 **不启用**；config `continuum.enabled: false` 默认。

## 3. 轨道能量（PDE 监督）

```python
def orbital_energy_from_dirac(P, Q, LP, LQ, grid) -> Array:
    """E_orb = <ψ|H_D|ψ>/<ψ|ψ>  [B, N_orb]"""
```

## 4. 接口形状汇总

| 张量 | 形状 |
|------|------|
| `V` | `[B, N_grid]` |
| `P`, `Q` | `[B, N_orb, N_grid]` |
| `E_orb` | `[B, N_orb]` |
| `branch_feat` | `[B, D_in]` |

## 5. 测试

| 文件 | 断言 |
|------|------|
| `test_siren_init_finite` | 参数有限 |
| `test_deeponet_forward_shapes` | 输出 shape |
| `test_deeponet_zero_init_near_nuclear` | $\Delta V=0$ 时 $V \approx -Z/r$ |
| `test_dirac_h1s_analytic` | 代入解析 $P,Q$ 则 $L_\mathrm{PDE}<10^{-6}$ |
| `test_ortho_identity` | 正交后 $\langle ab\rangle=\delta$ |

## 6. 与 V2 的差异（实现时注意）

- **无** B-spline 系数 $c_k$；**无** 每轨道独立 $\lambda$ envelope 为主自由度。
- 可选：`envelope_bound(r) = r^|κ| exp(-(Z_eff/n)r)` 仅作 **正定性约束**，系数由 branch 预测 $Z_\mathrm{eff}/n$。
