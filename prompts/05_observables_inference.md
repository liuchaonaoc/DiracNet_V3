# 05 — 模块 4：辐射与碰撞可观测量推断

## 1. 输出清单

| 物理量 | 符号 | 形状 | 单位 |
|--------|------|------|------|
| 能级能量 | $E_i$ | `[B, N_csf]` | Hartree |
| 波长 | $\lambda$ | `[B, N_trans]` | nm |
| 跃迁概率 | $A_{ki}$ | `[B, N_trans]` | s$^{-1}$ |
| 振子强度 | $gf$ | `[B, N_trans]` | 无量纲 |
| CE 截面 | $\sigma_\mathrm{CE}$ | `[B, N_trans, N_E]` | cm$^2$ |
| CI / PI / RR | 同上 | 可选 | |

## 2. 目录

```text
pinn_art/observables/
├── __init__.py
├── multipole.py           # E1, M1, E2 径向矩阵元
├── transition_rates.py    # A_ki, gf
├── collision.py           # CE/CI 因子化 + 样条
└── opacity.py             # 线吸收 opacity（流体力学接口）
```

## 3. 实验数据缺失时的理论回退（Fall-back）

与模块 3 对角元策略一致：推断管线 **始终** 先算理论量，再按 mask 选择性替换实验值。

| 量 | 有实验参考 | 无实验参考（`transition_mask=False`） |
|----|------------|--------------------------------------|
| 能级 $E_i$ | `nist_mask=True` → $E_\mathrm{NIST}$ 已进入 $H_{ii}$ | 使用 `eigh` 后 $E_i^{\mathrm{theory}}$ |
| $A_{ki}$, $gf$ | 可与 FAC/NIST 对比评估 | **仅输出** $A_{ki}^{\mathrm{theory}}$，标注 `source=theory` |
| 碰撞截面 $\sigma(E)$ | 可与 FAC 对比 | **仅输出** 因子化理论 $\sigma^{\mathrm{theory}}(E)$ |

输出字典建议：

```python
"transitions": {
    "A_ki": ...,
    "gf": ...,
    "source": int32[B, N_trans],  # 0=theory, 1=nist/fac_matched
}
```

极高激发态 / 极高电荷态离子的 **未收录跃迁** 不得从 NIST 插值；只报告 PINN-ART 理论预测及不确定度（可选：与 $L_\mathrm{PDE}$ 关联的启发式误差条）。

## 4. 电偶极跃迁（E1）

```python
def radial_dipole_matrix_element(
    P_i, Q_i, P_f, Q_f, grid, multipole: str = "E1"
) -> Array:
    """R_if 径向部分；角向部分由 CSF 系数收缩。"""
```

```python
def transition_rates(
    E_csf, V_csf,
    P, Q, orb_map, grid,
    selection_rules: dict,
) -> dict:
    """
    对允许的 (i→f)：
      ΔE = E_f - E_i
      A_ki = (64 π⁴ ν³ / (3 h c³)) · |D_if|² · g_k/g_i   (具体常数见 constants)
      gf   = (m_e c² / (h ν)) · (g_i/g_f) · A_ki
    返回稀疏 transition_index + 稠密张量。
    """
```

**选择定则**：$\Delta J$, parity, $\Delta n$ 由 `selection_rules` 配置；默认 E1。

## 5. 碰撞截面（因子化 + 对数能量样条）

避免分波求和；在 $N_\mathrm{anchor}\le 8$ 个对数能量点计算 **锚点截面**，再用可微样条：

```python
def collision_cross_section_ce(
    V_csf, P, Q,
    E_grid: Array,           # [N_E]
    n_anchors: int = 6,
) -> Array:
    """
    σ(E) = σ_0 · (E/E_0)^α · spline(log E; anchor_values)
    anchor_values 由库仑-玻恩近似闭式 + 神经网络校正头（可选）
    """
```

高能检查：$\sigma \propto E^{-1}$ Bethe 行为（测试用）。

## 6. 不透明度接口（应用层）

```python
def line_opacity(
    wavelengths_nm, A_ki, populations, temperature, line_profile="voigt"
) -> Array:
    """供宏观流体耦合；V3 第一版可用 Gaussian 近似。"""
```

## 7. 推断 API

```python
@jax.jit
def infer_observables(params, batch, E_grid):
    out = forward_neural_dirac(params, batch)
    out = forward_ci(params, out, batch)
    out = forward_observables(params, out, batch, E_grid)
    return out
```

## 8. 基准对比

| 基准 | 用途 |
|------|------|
| FAC 导出 CSV | $A_{ki}$, $\sigma$ 形状 |
| NIST | 能级、波长 |
| 氢解析 | E1 强度归一化 |

`scripts/v3_compare_fac.py`：读 FAC 表，算相对误差中位数。

## 9. 测试

- `test_einstein_A_hydrogen_lya`：H Lyman-α 与解析比
- `test_gf_sum_rule`：Thomas-Reiche-Kuhn 近似检查（可选）
- `test_collision_bethe_tail`：高能 $\sigma E \to$ const
- `test_infer_jit_shapes`：`jit` 后 shape 不变
