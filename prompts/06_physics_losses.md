# 06 — 物理损失函数

## 1. 总损失（分阶段）

### Stage A — Dirac 预训练（主）

```text
L_A = w_pde·L_pde + w_ortho·L_ortho + w_asym·L_asym + w_norm·L_norm
    + w_V·L_V_prior + w_smooth·L_V_smooth
```

| 项 | 定义 | 默认权重 |
|----|------|----------|
| `L_pde` | $\|LP-E P\|^2 + \|LQ-E Q\|^2$，按轨道、按网格均值，除以 $E_\mathrm{char}^2$ | 1.0 |
| `L_ortho` | $\sum_{a\ne b}(\langle ab\rangle)^2$ | 100.0 |
| `L_asym` | 远区 $P,Q$ 与 Whittaker 渐近形式 | 0.01 |
| `L_norm` | $\|\int(P^2+Q^2)-1\|$ per orb | 10.0 |
| `L_V_prior` | $\|\Delta V\|^2$ 或 $\|V+V_\mathrm{nuc}\|^2$ 正则 | 0.1 |
| `L_V_smooth` | $\|d^2V/dr^2\|^2$ | 1e-3 |

**禁止**：`L_nist` 进入 `jax.grad`。

### Stage B — CI 径向校准（可选）

```text
L_B = w_slat·L_slater_consistency + w_off·L_offdiag
```

- `L_slater_consistency`：与 FAC 参考 $R^k$（若有）或氢极限
- `L_offdiag`：理论 off-diagonal 与参考（弱权重）

### Stage C — 评估态（非训练或弱监督）

NIST 通过 **对角混合填充**（有则注入、无则理论 Fall-back，见 `04` §6），不通过 loss。若启用弱监督：

```text
L_C = w_lead·L_leading_percent  # 可选，权重 ≤ 0.01
```

## 2. 文件

```text
pinn_art/losses/
├── __init__.py
├── pde_loss.py
├── ortho_loss.py
├── asymptotic_loss.py
├── potential_prior.py
└── loss_schedule.py      # 按 epoch 调度 w_*
```

## 3. `pde_loss.py`

```python
def dirac_pde_loss(LP, LQ, P, Q, E_orb, orb_mask, grid, e_char: float = 1.0):
    res_P = LP - E_orb[..., None] * P
    res_Q = LQ - E_orb[..., None] * Q
    integrand = res_P**2 + res_Q**2
    per_orb = grid.integrate(integrand, axis=-1)   # [B, N_orb]
    per_orb = per_orb / (e_char**2 + 1e-12)
    return masked_mean(per_orb, orb_mask)
```

## 4. `asymptotic_loss.py`

远区 $r > r_\mathrm{cut}$（如 $0.5\, r_\mathrm{max}$）：

```python
# 束缚态：P ~ r^γ exp(-ξr), 拟合 ξ 与 Z_eff/n
L_asym = mean( (log|P| - log|P_asym|)^2 )  on tail
```

## 5. Warmup 调度（`loss_schedule.py`）

```yaml
stage_a:
  warmup:
    ortho_start_epoch: 0
    pde_ramp_epochs: [0, 50]      # 0 → full
    asym_start_epoch: 20
```

## 6. 监控量（不进 loss）

- `cos_sim(P, P_analytic)` per (Z,n)
- `E_orb` vs $-Z^2/(2n^2)$
- `cond(H)` before eigh

## 7. 测试

复用 V2 测试逻辑改 JAX：`test_pde_loss_zero_on_analytic`, `test_ortho_loss_zero_when_orthonormal`
