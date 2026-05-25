# 04 — 模块 3：可微 CI、NIST 注入与理论回退（Fall-back）

## 1. 物理分解

相对论 $jj$ 耦合哈密顿量：

$$H_{ij} = \sum_a h_a \delta_{ij} + \sum_{k} R^k_{ab,cd} \cdot C^k_{\mathrm{ang}}(i,j,a,b,c,d)$$

- **角向** $C_{\mathrm{ang}}$：Racah / Wigner-Eckart → **离线预计算**
- **径向** $R^k$：由 PINN 的 $(P,Q)$ 可微积分
- **单粒子对角** $h_a$：$E_\mathrm{orb}$ 或 $ \langle a|H_D|a\rangle$

## 2. 目录

```text
pinn_art/ci/
├── __init__.py
├── slater_radial.py       # R^k 积分
├── hamiltonian.py         # assemble_hamiltonian
├── nist_inject.py         # fill_h_diagonal_hybrid (+ 别名 inject_nist_diagonal)
└── eigen_solver.py        # differentiable_eigh

pinn_art/data/racah/       # 预生成 .npz
scripts/v3_precompute_racah.py
```

## 3. Slater 径向积分

```python
def slater_radial_integral(
    P_a, Q_a, P_b, Q_b,
    k: int,
    grid: RadialGrid,
) -> float:
    """
    R^k(a,b) = ∫ dr r^k · [P_a P_b + Q_a Q_b]  (具体归一化与 FAC 一致，见 constants)
    """
```

对活跃轨道对批量计算：

```python
def compute_all_Rk(
    P, Q, orb_mask, k_max: int, grid
) -> Array:
    """返回 [B, n_k, N_orb, N_orb] 或压缩上三角。"""
```

**$k$ 列表**：config `ci.k_list: [0, 1, 2, 3, 4]`；与角向缓存键一致。

## 4. 角向系数缓存

离线脚本扫描 manifest 中 `parent_config_id`，调用 Python（可用 `sympy` / 专用 Racah 库）生成：

```text
racah_cache.npz:
  keys: (parent_hash, i, j, k) → float64
  metadata: version, coupling scheme (jj)
```

加载：

```python
@functools.partial(jax.jit, static_argnums=(0,))
def lookup_angular(parent_ids, csf_i, csf_j, k_index, cache) -> Array:
    # stop_gradient
    return jax.lax.stop_gradient(C_ang)
```

## 5. 哈密顿量装配

```python
def assemble_hamiltonian(
    E_orb: Array,           # [B, N_orb]
    Rk: Array,              # [B, n_k, ...]
    C_ang: Array,           # [B, N_csf, N_csf] 或稀疏索引
    csf_to_orb: Array,      # [B, N_csf, N_orb] 映射
    csf_mask: Array,
) -> Array:
    """H [B, N_csf, N_csf]，对称。"""
```

**Phase 1 简化**：仅对角 + 一阶 off-diagonal（同 parent 下 2 CSF）用于冒烟；完整 $M\times M$ 为 Phase 2。

## 6. 对角元混合填充：NIST 注入 + 理论回退（Fall-back）

装配完成后的理论哈密顿量记为 $H^{\mathrm{theory}}$，其对角元 $H_{ii}^{\mathrm{theory}}$ 来自 $E_\mathrm{orb}$、Slater $R^k$ 与 Racah 系数（见 §5）。**不在 NIST 中的能级**（极高激发态、极高电荷态、未测量谱项等）必须保留 $H_{ii}^{\mathrm{theory}}$，不得留空、置零或随意常数。

```python
def fill_h_diagonal_hybrid(
    H_theory: Array,           # [B, N_csf, N_csf] 已装配，含 off-diagonal
    E_nist: Array,             # [B, N_csf] Hartree；无数据处可为 NaN
    nist_mask: Array,          # [B, N_csf] bool
) -> tuple[Array, Array]:
    """
    H_diag_theory = diag(H_theory)   # 或显式传入 H_diag_theory [B, N_csf]

    H_ii = where(nist_mask, E_nist, H_diag_theory)

    写回：H_out = H_theory - diag(H_theory) + diag(H_ii)
    返回 (H_out, diag_source)  # diag_source: 0=theory, 1=nist（监控用）
    """
    ...

# 向后兼容别名（实现中调用 fill_h_diagonal_hybrid）
inject_nist_diagonal = fill_h_diagonal_hybrid
```

### 6.1 `nist_mask` 生成规则（数据管线）

在 `collate` / `nist_loader` 中：

```python
nist_mask = jnp.isfinite(E_nist) & (E_nist_uncertainty >= 0)  # 或显式列 has_nist_level
```

典型 `nist_mask=False` 场景：

- 极高主量子数 $n$、Rydberg 系列未入 NIST；
- 高电离度离子（如 He-like Fe XXVI 部分精细结构）；
- 新组态 / 预测谱线，manifest 中无对应行。

### 6.2 可微性

- `where(mask, E_nist, H_diag_theory)` 对 **off-diagonal** 与 **mask=False 的对角** 保持可微（梯度走 $H^{\mathrm{theory}}$ / $(P,Q)$）。
- `E_nist` 在 mask=True 处 **`stop_gradient`**（实验常数，硬注入）。
- 禁止对 `E_nist` 做 MSE backward 训练。

### 6.3 跃迁数据的回退（与能级一致）

模块 4 中：若某 $(i\to f)$ 无 NIST/FAC 跃迁强度参考，`transition_mask=False`，输出 **仅理论** $A_{ki}^{\mathrm{theory}}, gf^{\mathrm{theory}}$（由 $V_\mathrm{csf}$ 与 $(P,Q)$ 计算）。不得用相邻能级实验值外推冒充。

**纪律**：混合填充发生在 **推断** 或 **Stage C 前向**；Stage A 仍不读 NIST。训练 Stage C 若需对齐，只监督 **有 mask 的 off-diagonal** 或 **本征向量**（可选）。

## 7. 可微对角化

```python
def differentiable_eigh(H: Array, eps: float = 1e-6) -> tuple[Array, Array]:
    """
    H_reg = sym(H) + eps * I
    E, V = jnp.linalg.eigh(H_reg)
  见 14_degenerate_gradient_safety.md
    """
```

## 8. CSF 本征向量用法

```python
# 混合系数 b_nu 即 V_csf 列向量
level_energy = E_csf                                    # [B, N_csf]
leading_percent = max_over_nu |V_nu|^2 per CSF
```

## 9. Phase 分阶段

| Phase | CI 内容 | 验收 |
|-------|---------|------|
| P0 | 仅 `diag(E_orb)`，$M=1$ | 与 H 1s 能量一致 |
| P1 | $M\le 5$ 氢化离子，无 Racah | 对角 MAE |
| P2 | 完整 Racah + 对角混合填充（NIST + Fall-back） | 有 NIST 区 meV 门禁 |
| P3 | 复杂铁族组态 | Leading % |

## 10. 测试

- `test_slater_integral_hydrogenic`：与解析比对
- `test_hamiltonian_symmetric`
- `test_nist_inject_only_changes_diag`
- `test_fallback_uses_theory_when_mask_false`：`mask=False` 时 $H_{ii}=H_{ii}^{\mathrm{theory}}$
- `test_fallback_nist_when_mask_true`：`mask=True` 时 $H_{ii}=E_\mathrm{NIST}$，且与理论差可非零
- `test_eigh_recovers_known_spectrum`：构造 2×2 解析 H
- `test_eigh_gradient_finite`：随机 H，检查 grad 无 NaN
