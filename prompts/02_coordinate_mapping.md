# 02 — 模块 1：非线性坐标映射层

## 1. 物理动机

束缚态径向波函数在高 $n$ 区呈类 Whittaker / 库仑振荡；均匀 $r$ 网格 + ReLU/SIREN 仍会在核区与远区同时浪费自由度。混合坐标：

$$t(r) = c_1 \sqrt{r} + c_2 \ln(r + r_\epsilon)$$

- $\sqrt{r}$：压缩核区，放大外区采样密度。
- $\ln r$：匹配长尾衰减区的缓慢变化。
- $r_\epsilon \sim 10^{-6}$ a.u.，避免 $\ln 0$。

网络 **仅在 $t$ 上** 作为 trunk 输入；$r$ 仅用于 PDE 与积分测度 $dr$。

## 2. 文件与类

```text
pinn_art/coords/
├── __init__.py
├── mixed_map.py          # MixedRadialMap
└── shell_features.py     # build_branch_features
```

### 2.1 `MixedRadialMap`

```python
@dataclass(frozen=True)
class MixedRadialMap:
    c1: float = 1.0
    c2: float = 1.0
    r_eps: float = 1e-6

    def t(self, r: Array) -> Array: ...
    def dt_dr(self, r: Array) -> Array: ...   # 链式法则给 d/dr
```

**可学习性**：Stage A 默认 $c_1,c_2$ **固定**（config 给定）；消融 `learn_map: true` 时用 `softplus` 参数化并加先验 $c_1,c_2 \in [0.5, 2]$。

### 2.2 `build_branch_features`

输入 `batch`，输出 `[B, D_branch_in]`：

| 特征 | 维 | 说明 |
|------|-----|------|
| `embed(Z)` | 32 | 正弦嵌入或 lookup |
| `omega` masked | $N_\mathrm{orb}$ | 分数占据 |
| `n, l, kappa` per orb | $3 N_\mathrm{orb}$ | 归一化到 $[0,1]$ |
| 标量统计 | 4 | $N_\mathrm{ele}$, ion_charge, $\sum \omega$, 价壳 $n_\mathrm{max}$ |

```python
def build_branch_features(batch: dict, *, n_orb_max: int) -> Array:
    """返回 [B, D_in]，padding 轨道用 0。"""
```

## 3. 网格

与 V2 一致，复用逻辑（移植到 `pinn_art/utils/grid.py`）：

```yaml
grid:
  r_min: 1.0e-4
  r_max: 50.0
  n_grid: 256
  scheme: loglinear   # 或 mixed_map 后的均匀 t 网格再反解 r（二选一，默认 r 网格）
```

**默认策略**：在 **$r$ 上 loglinear 采样**，再映射 $t(r)$；PDE 残差在 $r$ 上计算，trunk 输入 $t$。

## 4. 单元测试（`tests/test_coords_mixed_map.py`）

```python
def test_t_monotone_increasing(): ...
def test_dt_dr_positive(): ...
def test_t_at_r_min_finite(): ...
def test_branch_features_shape(batch_fixture): ...
```

## 5. 实现检查清单

- [ ] $t(r)$ 全程有限、单调
- [ ] `dt_dr` 与数值微分一致（atol=1e-4）
- [ ] branch 特征对 padding `orb_mask=False` 不泄漏（乘 0）
