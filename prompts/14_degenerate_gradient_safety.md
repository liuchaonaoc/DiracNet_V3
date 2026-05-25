# 14 — 简并梯度安全（`eigh` Custom VJP）

## 1. 问题

对对称 $H$ 做 `eigh`，损失依赖本征值/向量时，反向传播含：

$$\frac{\partial L}{\partial H_{ij}} \propto \sum_{a\ne b} \frac{(\partial L/\partial V)_{ab}}{E_a - E_b}$$

当 $E_a \approx E_b$（量子简并或数值近简并）时梯度爆炸 → NaN。

## 2. 前向破缺（必做）

```python
def regularize_H(H, eps_degen_ev: float = 1e-6):
    """eps 以 eV 传入，内部转 Hartree。"""
    eps_ha = eps_degen_ev / HARTREE_TO_EV
    M = H.shape[-1]
    eye = jnp.eye(M, dtype=H.dtype)
    H_sym = 0.5 * (H + jnp.swapaxes(H, -1, -2))
    return H_sym + eps_ha * eye
```

Config：

```yaml
ci:
  eps_degen_ev: 1.0e-6
  max_abs_h: 1.0e6
```

## 3. Custom VJP（推荐）

```python
@jax.custom_vjp
def safe_eigh(H):
    H_reg = regularize_H(H)
    E, V = jnp.linalg.eigh(H_reg)
    return E, V

def safe_eigh_fwd(H):
    E, V = safe_eigh(H)
    return (E, V), (H, E, V)

def safe_eigh_bwd(res, g):
    H, E, V = res
    gE, gV = g
    # 截断分母 |Ea-Eb| < delta → 不传播或置零
    H_grad = eigh_vjp_with_gap_cutoff(H, E, V, gE, gV, gap_ev=1e-4)
    return (H_grad,)

safe_eigh.defvjp(safe_eigh_fwd, safe_eigh_bwd)
```

**`gap_ev`**：小于此能量差的配对，**不对该对传播** `gV` 贡献（或 clamp 分母）。

实现可放在 `pinn_art/ci/eigen_solver.py`，参考 V1 `DifferentiableEigenSolver` 语义。

## 4. 训练时监控

每 N step 记录：

```python
metrics["h_cond"] = jnp.linalg.cond(H_reg)
metrics["min_gap_ev"] = jnp.min(jnp.diff(E_sorted)) * HARTREE_TO_EV
```

若 `h_cond > 1e8` 或 `min_gap_ev < 0`，跳过该 batch（`jax.lax.cond`）。

## 5. 测试

- `test_eigh_gradient_finite`：随机 4×4，100 次 seed
- `test_eigh_degenerate_pair_zero_grad`：构造 $H=\mathrm{diag}(1,1,2)$，扰动 loss 仅依赖简并子空间，梯度有界

## 6. 与对角混合填充（NIST + Fall-back）的交互

- `nist_mask=True`：$H_{ii}$ 为实验常数（`stop_gradient`），简并可能被人为消除。
- `nist_mask=False`：$H_{ii}=H_{ii}^{\mathrm{theory}}$ 仍随 $(P,Q)$ 可微，梯度经理论对角与 off-diagonal 传播。
- 无论哪种对角来源，前向均需 `eps_degen` 因 off-diagonal 耦合。

## 7. 优先级

若 Custom VJP 工期不足：**仅** `regularize_H` + `nan_to_num(grad)` 可作为 Sprint 3 临时方案，Sprint 4 前必须补 VJP。
