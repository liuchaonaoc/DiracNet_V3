# 12 — 测试计划

> **纪律**：每个 Sprint 结束前，对应测试 ALL GREEN。不允许"下 Sprint 再补"。

## 1. 测试分层

| 层级 | 范围 | 工具 |
|------|------|------|
| L0 | 单函数 | pytest |
| L1 | 模块协作 | pytest |
| L2 | forward + loss + train_step | pytest |
| L3 | 短训练 smoke | pytest + 脚本 |
| L4 | 物理门禁 | gate 脚本 |

## 2. 文件清单

```text
tests/
├── conftest.py
├── test_import_smoke.py
├── test_coords_mixed_map.py
├── test_coords_branch_features.py
├── test_siren_layers.py
├── test_deeponet_forward.py
├── test_dirac_operator_h1s.py
├── test_orthogonalizer.py
├── test_pde_loss.py
├── test_asymptotic_loss.py
├── test_slater_integral_hydrogenic.py
├── test_hamiltonian_assemble.py
├── test_nist_inject.py
├── test_diagonal_fallback.py
├── test_eigh_known_spectrum.py
├── test_eigh_gradient_finite.py
├── test_transition_hydrogen_lya.py
├── test_infer_jit_shapes.py
├── test_collate_batch.py
├── test_gate_pass_on_analytic.py
├── test_gate_fail_on_random.py
├── test_train_step_smoke.py
└── test_full_forward_smoke.py
```

## 3. 关键用例摘要

### 3.1 `test_dirac_operator_h1s`

代入 `physics/hydrogenic.py` 解析 $(P,Q)$，$V=-Z/r$，断言 $L_\mathrm{PDE}<10^{-8}$。

### 3.2 `test_deeponet_forward`

随机 params，`jax.jit(model.apply)`，检查 `V,P,Q` shape，无 NaN。

### 3.3 `test_orthogonalizer`

两轨道人为重叠，正交后 $|\langle 12\rangle|<10^{-6}$。

### 3.4 `test_diagonal_fallback`

```python
def test_fallback_uses_theory_when_mask_false():
    H_th = build_known_H()
    E_nist = jnp.array([0.5, jnp.nan])
    mask = jnp.array([True, False])
    H_out, src = fill_h_diagonal_hybrid(H_th, E_nist, mask)
    assert H_out[1, 1] == H_th[1, 1]
    assert H_out[0, 0] == 0.5

def test_nist_stops_gradient_on_masked_diag():
    ...
```

### 3.5 `test_eigh_gradient_finite`

```python
def test_eigh_gradient_finite():
    H = random_symmetric(4)
    def loss(H):
        E, _ = differentiable_eigh(H, eps=1e-6)
        return E.sum()
    g = jax.grad(loss)(H)
    assert jnp.isfinite(g).all()
```

### 3.6 `test_gate_pass_on_analytic`

将解析波函数 **注入** forward 旁路，Gate A 必须 PASS。

### 3.7 `test_train_step_smoke`

2 step 训练，`loss` 有限且第二步 $\le$ 第一步 × 1.5（宽松）。

## 4. CI 命令

```bash
# Sprint 0
pytest tests/test_import_smoke.py -v

# Sprint 1
pytest tests/test_coords_*.py tests/test_siren_*.py tests/test_deeponet_*.py \
       tests/test_dirac_*.py tests/test_orthogonalizer.py -v

# Sprint 2
pytest tests/test_pde_loss.py tests/test_asymptotic_loss.py \
       tests/test_train_step_smoke.py tests/test_gate_*.py -v

# Sprint 3
pytest tests/test_slater_*.py tests/test_hamiltonian_*.py \
       tests/test_nist_inject.py tests/test_eigh_*.py -v

# Sprint 4+
pytest tests/test_transition_*.py tests/test_infer_jit_*.py \
       tests/test_full_forward_smoke.py -v
```

## 5. 覆盖率目标

第一版：**不强制** coverage 百分比；核心 `physics/`, `ci/`, `nets/` 每个公开函数至少 1 个测试。

## 6. `conftest.py` fixture

```python
@pytest.fixture
def r_grid():
    return make_radial_grid(1e-4, 50.0, 128, "loglinear")

@pytest.fixture
def h1s_batch():
    return {...}  # Z=1, 1s, 见 01_architecture

@pytest.fixture
def rng_key():
    return jax.random.PRNGKey(0)
```
