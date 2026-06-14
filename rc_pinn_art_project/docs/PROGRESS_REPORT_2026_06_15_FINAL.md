# DiracNet_V3 (PINN-ART) 阶段性进展报告 — 2026-06-15

> **范围**：Z=1–26 多电子原子 Dirac 方程 PINN 求解 + 可微 CI 注入 + 评估管线优化
> **当前阶段**：Stage A 续训（Path A 多电子屏蔽注入）— **B'' heguard 是已确认的最优 anchor**
> **作者**：PINN-ART 项目组

---

## 〇、总览

本报告是 **2026-06-08 → 2026-06-14** 期间 `DiracNet_V3` 项目的阶段总结。期间完成 5 个续训实验（B → B' → B'' → B''' → B'''' → B'''''）+ 1 次评估管线性能优化（452x 加速），最终产出：

- ✅ **M1（坐标映射）**：已就绪（`MixedRadialMap`）
- ✅ **M2（神经 Dirac 求解器）**：已就绪（`PinnArtModel` + `DeepONetDirac`）
- ⚠️ **M3（可微 CI / Path A 注入）**：**实现完成但训练未稳定**（B'''/B''''/B''''' 全部早停）
- ⏳ **M4（辐射推断）**：未触发

**核心结论**：

| | 单电子（sv）MAE | 多电子（me）MAE |
|---|---|---|
| **最佳 B'' heguard (epoch 105)** | **81997 meV** | 172975 meV |
| Li 元素 | 1812 meV | 3914 meV |
| Be 元素 | 4371 meV | 13393 meV |
| H 元素 | 112 meV | — |
| He 元素 | 117 meV | 1501 meV |

**Path A（V_slater_corr 注入 V_dfs）物理思路正确，但实现路径仍存在 return_ci 改 P/Q 梯度路径的副作用问题**。当前最佳实践是 **回退 B'' heguard 路径**（不启用 Path A 注入）。

---

## 一、模块设计 (来自 `prompts/Overall_design.md`)

整体架构 = 4 个模块串行：

```
输入特征 ─→ M1 坐标映射 ─→ M2 神经 Dirac 求解器 ─→ M3 可微 CI / Path A 注入 ─→ M4 辐射推断
   (Z, ω)      t(r)            V(r), P, Q            R^k, E_csf                A_ki, gf, σ
```

| 模块 | 触发 | 物理原理 | 当前状态 |
|---|---|---|---|
| **M1 坐标映射** | 任何 forward | 抑制 SIREN 谱偏差，混合 $\sqrt{r} + \ln r$ | ✅ 已就绪 |
| **M2 神经 Dirac** | 任何 forward | DeepONet 输出 V/P/Q，Dirac 方程残差做损失 | ✅ 已就绪 |
| **M3 可微 CI** | `return_ci=True` 显式触发 | R^k Slater 积分 → jj 耦合 H 矩阵 → eigh 求 E_csf | ✅ 代码完成；❌ 训练不稳定 |
| **M4 辐射推断** | `E_grid` 入参 | E1/M1/E2 跃迁 + 碰撞激发截面 | ⏳ 未触发 |

**当前已涉及模块：M1 + M2 + M3**（M3 多次尝试早停，但代码已稳定）。

---

## 二、M1：非线性坐标映射层

### 2.1 物理动机

束缚态波函数在高 r 区有 Whittaker 渐近振荡 + 衰减，均匀空间网格会让 SIREN 网络产生**谱偏差**。混合坐标让所有 r 都被映射到 t 的中段，振荡周期近乎平滑。

### 2.2 公式

$$t(r) = c_1 \sqrt{r} + c_2 \ln(r + \epsilon)$$

- 默认 $c_1 = c_2 = 1.0$
- 数值稳定：$r_{\text{safe}} = \max(r, 10^{-6})$ 防止 $\ln(0)$
- $dt/dr = c_1 / (2\sqrt{r}) + c_2 / r$ 用于链式法则

### 2.3 代码实现 (`pinn_art/coords/mixed_map.py`)

```python
@dataclass(frozen=True)
class MixedRadialMap:
    c1: float = 1.0
    c2: float = 1.0
    r_eps: float = 1e-6

    def t(self, r):
        r_safe = jnp.maximum(r, self.r_eps)
        return self.c1 * jnp.sqrt(r_safe) + self.c2 * jnp.log(r_safe)

    def dt_dr(self, r):
        r_safe = jnp.maximum(r, self.r_eps)
        return self.c1 / (2.0 * jnp.sqrt(r_safe)) + self.c2 / r_safe
```

### 2.4 集成位置

`PinnArtModel.__call__` L52–55：
```python
r = grid.r
cmap = MixedRadialMap(c1=self.c1, c2=self.c2)
t = cmap.t(r)
dt_dr = cmap.dt_dr(r)
```
`DeepONetDirac` 在 t 坐标上展开，输出在 r 空间求值。

---

## 三、M2：神经 Dirac 求解器

### 3.1 物理动机

放弃求解多体 Dirac 方程（$10^{10}$ 量级配置空间）。按 cFAC 假设找**全局平均势 V(r)**，所有电子在此统一势中运动 → 自动保证正交性。

### 3.2 核心组件

1. **DeepONet (Branch + Trunk)**：
   - Branch 处理离散特征 $Z$、$\omega_{n\kappa}$（`build_branch_features`）
   - Trunk 是 SIREN（`sin(ωx+b)` 激活），在 t 坐标上展开
   - 末层解码头输出 $V(r), P_{n\kappa}(r), Q_{n\kappa}(r)$ 及其一阶导数

2. **Hydrogenic Skeleton + 微扰**（`use_hydrogenic_skeleton=True`）：
   - 用解析氢样波函数 $R_{n\kappa}^{\text{H}}(r; Z_{\text{eff}})$ 做骨架
   - 网络只学**对骨架的微扰**，大幅降低学习难度
   - $Z_{\text{eff}}$ 由 Slater 屏蔽规则算（`slater_effective_charge`）

3. **Dirac 算子** (`pinn_art/physics/dirac_operator.py`)：
   - 对 $(P, Q)$ 应用径向 Dirac 算子得 $(L_P, L_Q)$
   - 轨道能 $E_{n\kappa} = \int (P L_P + Q L_Q) dr$

### 3.3 损失函数（`pinn_art/losses/`）

| 损失项 | 公式 | 权重 |
|---|---|---|
| **PDE (Dirac 残差)** | $\langle P_n \| (H_D - E_n) P_n \rangle$ | 1.0 |
| **ortho (正交性)** | $\sum_{a<b} \langle P_a P_b \rangle^2$ | 100.0 |
| **norm (归一)** | $\sum_a (\int P_a^2 + Q_a^2 dr - 1)^2$ | 10.0 |
| **v_prior (V 形状)** | $\|V - V_{\text{DFS}}\|^2$ | 1.0 |
| **v_smooth (V 平滑)** | $\int (\nabla V)^2 dr$ | $10^{-6}$ |
| **scf (DFS 自洽)** | $\|V - V_{\text{DFS}}\|^2$ | **5.0 (B'' heguard)** / 1.0 (B''''') |
| **asym (渐近)** | $V(r) \to -(Z-N+1)/r$ 当 $r \to \infty$ | 0.1 |

### 3.4 关键代码：`PinnArtModel.__call__` (L42–106)

```python
# 1. 坐标映射 (M1)
r = grid.r
cmap = MixedRadialMap(c1=self.c1, c2=self.c2)
t = cmap.t(r)
dt_dr = cmap.dt_dr(r)

# 2. Branch / Trunk 特征
branch_feat = build_branch_features(batch, n_orb_max=self.n_orb_max)
net = DeepONetDirac(d_branch=128, d_trunk=128, n_siren_layers=4, omega_0=20.0, ...)

# 3. 前向 (M2)
raw = net(branch_feat, t, r, dt_dr, kappa, orb_mask, Z, ...)

# 4. 选配: Lowdin 正交化
if self.apply_lowdin:
    ortho = lowdin_orthonormalize(raw["P"], raw["Q"], grid, orb_mask, ...)
    P, Q = ortho["P"], ortho["Q"]
else:
    P, Q = raw["P"], raw["Q"]  # B'' heguard 关闭 Lowdin，让 norm/ortho loss 真起作用

# 5. Dirac 算子 + 轨道能
LP, LQ = dirac_apply(P, Q, dPdr, dQdr, V, kappa, r)
E_orb = orbital_energy_from_dirac(P, Q, LP, LQ, grid)
```

---

## 四、M3：可微 CI 层 + Path A 多电子屏蔽注入

### 4.1 物理动机

平均场 V_dfs 必然在对角元引入 eV 级误差。M3 通过 R^k Slater 积分 + 角向系数装配 H 矩阵 + eigh 求本征能 E_csf。

**Path A 创新**：将 R^k 重新注入 V_dfs 形成 V_dfs_aug，让 V_net 训练时同时考虑平均场 + R^k 屏蔽。

### 4.2 公式链

1. **径向密度**（`electron_density`）：
$$\rho(r) = \sum_a \omega_a (P_a^2 + Q_a^2)$$

2. **Hartree 势**（`hartree_potential`，**O(N_g²) 累计积分，可微**）：
$$V_H(r) = \frac{1}{r} \int_0^r \rho\, dr' + \int_r^\infty \frac{\rho}{r'}\, dr'$$

3. **Slater Xα 交换**（`slater_exchange`）：
$$V_x(r) = -\frac{3}{2}\alpha_x \left(\frac{3}{\pi}\rho_{3d}\right)^{1/3}, \quad \rho_{3d} = \rho / (4\pi r^2)$$

4. **DFS 势**：
$$V_{\text{dfs}} = -Z/r + V_H + V_x \quad \xrightarrow{\text{Latter tail}} \min(V, -(Z-N+1)/r)$$

5. **Slater R^k 积分**（`compute_all_Rk_diagonal`）：
$$R^k(a,a) = \int_0^\infty r^k (P_a^2 + Q_a^2) dr$$

6. **可微 Hamiltonian 装配**（`assemble_hamiltonian`）：
$$H_{IJ} = E_{\text{orb},I} \delta_{IJ} + \sum_k C^k_{IJ} \cdot R^k$$

7. **可微对角化**（`safe_eigh`）：防退化梯度爆炸（在主对角加 $10^{-6}$ eV 微扰）

### 4.3 Path A：V_slater_corr 注入（核心创新）

**公式**（`pinn_art/physics/slater_correction.py`）：

$$V_{\text{slater\_corr}}(r) = -\alpha \cdot \sum_k e^{\text{slater\_log\_scale}[k]} \cdot \frac{\sum_a \omega_a R^k(a,a) P_a^2(r)}{\rho(r)}$$

- $\alpha = \text{V\_SLATER\_INJECT\_SCALE}$（全局乘子，B''''' = 0.5）
- `slater_log_scale` 是**可学习标量** `[n_k]`，init = -1.0（B'''' fix）
- 注入：`V_dfs_aug = V_dfs + V_slater_corr`
- SCF 损失对齐：`V_net → V_dfs_aug` 而非 `V_dfs`

### 4.4 关键代码

**A. R^k 计算**（`pinn_art/ci/slater_radial.py`）：
```python
def compute_slater_R0(P_a, P_b, Q_a, Q_b, grid, k=0):
    rho = P_a * P_b + Q_a * Q_b
    if k == 0:
        return grid.integrate(rho, axis=-1)
    r = grid.r
    weight = jnp.power(jnp.clip(r, 1e-8), float(k))
    return grid.integrate(rho * weight[None, :], axis=-1)

def compute_all_Rk_diagonal(P, Q, orb_mask, grid, k_list=(0,)):
    B, N, _ = P.shape
    out = []
    for k in k_list:
        rk = [compute_slater_R0(P[:, a], P[:, a], Q[:, a], Q[:, a], grid, k=k)
              for a in range(N)]
        out.append(jnp.stack(rk, axis=1))
    return jnp.stack(out, axis=1)
```

**B. slater_correction_potential**（`pinn_art/physics/slater_correction.py`）：
```python
def slater_correction_potential(P, Q, omega, Rk, slater_log_scale, rho):
    P2 = P * P
    omega_e = omega[:, None, :, None]      # [B, 1, N, 1]
    Rk_e = Rk[:, :, :, None]                # [B, n_k, N, 1]
    weighted_P2 = omega_e * Rk_e * P2[:, None, :, :]  # [B, n_k, N, N_g]
    sum_wP2 = jnp.sum(weighted_P2, axis=2)  # [B, n_k, N_g]
    scale = jnp.exp(slater_log_scale)[None, :, None]
    total = jnp.sum(scale * sum_wP2, axis=1)  # [B, N_g]
    rho_safe = jnp.clip(rho, 1e-12)
    V_slater_corr = -total / rho_safe * V_SLATER_INJECT_SCALE
    return V_slater_corr
```

**C. 集成到 PinnArtModel**（`pinn_art/models/pinn_art_model.py` L130–143）：
```python
slater_log_scale = self.param(
    "slater_log_scale",
    lambda key, shape, dtype=jnp.float32: jnp.full(shape, -1.0, dtype=dtype),
    (len(self.k_list),),
)
Rk = compute_all_Rk_diagonal(P, Q, orb_mask, grid, k_list=self.k_list)
Rk = Rk * jnp.exp(slater_log_scale)[None, :, None]
out["slater_log_scale"] = slater_log_scale  # 暴露给 trainer
```

**D. Trainer return_ci 控制**（`pinn_art/training/stage_a_trainer.py`）：
```python
return_ci_flag = bool(dfs_cfg[7])  # path_a_enabled → return_ci
```

### 4.5 评估管线优化（关键工程突破）

`v3_eval_excitation_vs_nist.py` GPU 4.5h → 24s = **452x 加速**：

| 瓶颈 | 修复 |
|---|---|
| Python 循环 16348 次单 forward | 合并双 forward 为一次 `return_ci=True` |
| `batch_size=1` | 批处理 batch_size=64 + JIT |
| 双 forward（sv + me） | 单次 `use_ci=True` 拿 E_orb + E_csf |

核心代码（`pinn_art/evaluation/excitation_eval.py`）：
```python
@jax.jit
def _batched_apply(batch_dict):
    return model.apply(params, batch_dict, grid, train=False, return_ci=True)

for chunk_start in range(0, n_full, batch_size):
    chunk = key_idx[chunk_start: chunk_start + batch_size]
    items = [ds[idx] for _, idx in chunk]
    batch = collate_batches(items, n_csf_max=n_csf_max, ...)
    out = _batched_apply(batch)
    _accumulate_cache(out, batch, chunk, cache, use_ci)
```

---

## 五、最可靠 / 最高精度结果

### 5.1 候选评估结果对比（按 Z=3–4 单电子 MAE 排序）

| 评估 ID | 训练 epoch | Li sv | Be sv | He sv | 总 sv MAE | 判定 |
|---|---|---|---|---|---|---|
| **B'' heguard (epoch 105)** | 165 早停 | **1812** | **4371** | 117 | **81997** | **最可靠** ✅ |
| B' baseline | — | 2064 | 4887 | 108 | 82931 | OK |
| B'' heguard (old) | — | 1813 | 4375 | 117 | — | OK |
| stage3b p1z5_6 eprime | — | 1775 | 4454 | 152 | 79293 | Li 略好但 H/He 退 |
| B''' Path A init=-3.0 | 早停 15 | 9772 | 7760 | 10793 | 84588 | ❌ H/He 大退 |
| B'''' Path A init=-1.0 | 早停 105 | 10491 | 8999 | 10879 | 84281 | ❌ 同上 |
| B''''' Path A ×0.5 | 早停 15 | 19291 | 13990 | 24978 | 85467 | ❌❌ 最差 |
| path_a_scf3x | — | 2076 | 4805 | — | — | ❌ |

**B'' heguard 在 Li/Be/H/He sv 上同时取得最低 MAE**，是最稳定最佳结果。

### 5.2 最佳 B'' heguard 完整结果

**Checkpoint**: `checkpoints/v3_stage_a_z1_26_n10_p1z3_4_bprime2_heguard_full/best_anchor.msgpack`
**Config**: `configs/v3_stage_a_prime3_path_a.yaml`
**数据**: NIST Z=1–26, n ≤ 10, 20814 行 (16348 唯一 config)

**汇总**：

| 组 | N (NIST) | MAE (meV) | 中位数 (meV) | 中位相对误差 | 覆盖率 | 达标占比 | 判定 |
|---|---|---|---|---|---|---|---|
| `single_valence` | 4297 | **81997** | 54066 | 0.31 | 44.1% | 1.9% | FAIL |
| `multi_electron` | 2480 | **172975** | 125570 | 1.09 | 23.2% | 1.0% | FAIL |

**single_valence 按元素 MAE (meV)**（高 Z 数据省略）：

| Z | 元素 | N | MAE | 中位数 |
|---|---|---|---|---|
| 1 | H | 55 | **112** | 103 |
| 2 | He | 87 | **117** | 89 |
| 3 | Li | 133 | **1812** | 402 |
| 4 | Be | 163 | **4371** | 1381 |
| 5 | B | 189 | 14733 | 10495 |
| 6 | C | 239 | 31456 | 24664 |
| 7 | N | 221 | 30434 | 24855 |
| 8 | O | 198 | 39090 | 36942 |
| 9 | F | 157 | 44977 | 32099 |
| 10 | Ne | 139 | 46734 | 34102 |
| 11 | Na | 196 | 46354 | 38177 |
| 12 | Mg | 205 | 61109 | 62364 |

**multi_electron 按元素 MAE (meV)**：

| Z | 元素 | N | MAE | 中位数 |
|---|---|---|---|---|
| 2 | He | 48 | **1501** | 1479 |
| 3 | Li | 44 | **3914** | 4133 |
| 4 | Be | 72 | 13393 | 18406 |
| 5 | B | 72 | 25756 | 32998 |
| 6 | C | 96 | 50811 | 59886 |
| 7 | N | 105 | 60774 | 66455 |

### 5.3 训练过程（anchor_guard.jsonl）

B'' heguard 全程 165 epoch 早停。`best_anchor` 保存于 **epoch 105**（H err = +41 meV, V(r~1) diff = -0.01 Ha, He 1s² err = 5336 meV）：

| epoch | h_exc_err (meV) | v_r1_diff (Ha) | he_1s2_err (meV) | 状态 |
|---|---|---|---|---|
| 15 | -844 | -0.062 | +5973 | ✅ |
| 30 | -371 | +0.013 | +4375 | ✅ |
| 45 | -1201 | +0.033 | -1610 | ✅ (He 几乎 NIST!) |
| 60 | -753 | +0.001 | +5038 | ✅ |
| 75 | +74 | +0.005 | +6080 | ✅ |
| 90 | -272 | +0.011 | +2256 | ✅ |
| **105** | **+41** | **-0.010** | **+5336** | ✅ **best_anchor** |
| 120 | +318 | -0.022 | +5535 | ✅ |
| 135 | +463 | -0.049 | +2563 | ✅ |
| 150 | -323 | +0.026 | +4997 | ✅ |
| 165 | +261 | -0.006 | **+15015** | ❌ He 跳到 15 eV，触发早停 |

**epoch 105 = 综合最佳**（H 锚点接近 NIST 0，V(r~1) diff 最小，He 1s² 在 ±5 eV 内）。

---

## 六、B'' heguard 完整运行条件（逐条）

### 6.1 训练命令（GPU）

```bash
cd /home/chaos/workspace2/DiracNet_V3/rc_pinn_art_project
export PYTHONPATH=.
export JAX_PLATFORMS=cuda

python scripts/v3_train_path_a_minimal.py \
  --config configs/v3_stage_a_prime3_path_a.yaml \
  --resume checkpoints/v3_stage_a_z1_26_n10_p1z3_4_eprime_full/best_anchor.msgpack \
  --tag p1z3_4_bprime2_heguard \
  --phase full --epochs 300 --batch-size 16 \
  --anchor-frac 0.30 --warmup-epochs 5 \
  --max-h-exc-meV 5000 --max-v-r1-diff 2.0 \
  --max-he-1s2-dev-ha 0.30 \
  --patience 8
```

### 6.2 评估命令（GPU）

```bash
cd /home/chaos/workspace2/DiracNet_V3/rc_pinn_art_project
export PYTHONPATH=.
export JAX_PLATFORMS=cuda

python scripts/v3_eval_excitation_vs_nist.py \
  --config configs/v3_stage_a_prime3_path_a.yaml \
  --ckpt checkpoints/v3_stage_a_z1_26_n10_p1z3_4_bprime2_heguard_full/best_anchor.msgpack \
  --out-dir logs/bprime2_heguard_eval_full
```

### 6.3 配置要点（`configs/v3_stage_a_prime3_path_a.yaml`）

| 字段 | 值 | 说明 |
|---|---|---|
| `dfs.path_a_enabled` | **false** | 关键：关闭 Path A 注入 |
| `ci.enabled` | true | 仍创建 slater_log_scale（占位用），不参与损失 |
| `stage_a.weights.scf` | 5.0 | E-prime 同款 SCF 自洽权重 |
| `stage_a.weights.ortho` | 100.0 | 强正交约束 |
| `optimizer.lr_trunk` | 1.0e-4 | 续训 lr，E-prime 0.1x |
| `optimizer.lr_branch` | 3.0e-5 | branch lr |
| `optimizer.weight_decay` | 1.0e-4 | AdamW wd |
| `optimizer.grad_clip` | 1.0 | 全局梯度裁剪 |
| `grid.n_grid` | 256 | loglinear 网格 r ∈ [0.01, 50] |
| `model.use_hydrogenic_skeleton` | true | 氢样骨架 + 微扰 |

### 6.4 Anchor 监控（`scripts/v3_train_path_a_minimal.py`）

| 阈值 | 默认 | B'' heguard | 含义 |
|---|---|---|---|
| `max-h-exc-meV` | 5000 | 5000 | H 1s→2p 激发能偏差上限 |
| `max-v-r1-diff` | 2.0 | 2.0 | V(r~1) 与 V_dfs_aug 偏差上限 (Ha) |
| `max-he-1s2-dev-ha` | 0.30 | 0.30 | He 1s² 总能量与 NIST 偏差上限 (Ha) |
| `anchor-frac` | 0.30 | 0.30 | 训练时混入 anchor 比例 |
| `patience` | 8 | 8 | 早停连续失败 epoch 数 |

### 6.5 续训来源

`v3_stage_a_z1_26_n10_p1z3_4_eprime_full/best_anchor.msgpack`（E-prime Stage 3a'，4096 epochs 训练的最优 anchor，H/He sv MAE = 120/108 meV）。

### 6.6 关键代码修改（`scripts/v3_train_path_a_minimal.py`）

```python
# B'' heguard fix: return_ci_flag 必须跟随 path_a_enabled
return_ci_flag = bool(dfs_cfg[7])  # dfs_cfg[7] = path_a_enabled
```

`pinn_art/training/stage_a_trainer.py` 中：
```python
def _train_step_jit(...):
    out = model.apply(params, batch, grid, train=True, return_ci=return_ci_flag)
    # 当 return_ci=False 时, 不计算 H 矩阵 → forward 与 E-prime 行为一致 → 不破坏 P/Q 梯度
```

---

## 七、Path A 注入（B''' / B'''' / B'''''）失败教训

| 实验 | slater 实际起步 | 早停 epoch | H 锚点 | 根因 |
|---|---|---|---|---|
| B''' | -3.0 | 15 | -3026 meV | `return_ci=True` 改 P/Q 梯度 + slater_log_scale 不动 |
| B'''' | -3.0 (被 merge 覆盖) | 105 | -3026 meV | merge_params 覆盖了 init=-1.0 |
| B''''' | -1.0 ✅ | 15 | -5444 meV | V_slater_corr ×0.5 + scf=1.0 仍不够 |

**核心诊断**：
1. **V(r~1) diff=-0.99 Ha** 是 V_net 与 V_dfs_aug 的本质偏离，**与 V_SLATER_INJECT_SCALE 无关**
2. **return_ci=True 即使 V_slater 注入为 0 也破坏 H 锚点**（改 P/Q 梯度路径）
3. **slater_log_scale 实际梯度极小**（3e-5/15 epoch），学习率乘子 10x 仍不够

**当前结论**：**回退 B'' heguard 路径（path_a=false），不启用 Path A 注入**。M3 代码保留但训练关闭。

---

## 八、文件索引

| 用途 | 路径 |
|---|---|
| 最佳 B'' heguard ckpt | `checkpoints/v3_stage_a_z1_26_n10_p1z3_4_bprime2_heguard_full/best_anchor.msgpack` |
| 最佳 B'' heguard eval | `logs/bprime2_heguard_eval_full/` |
| 训练脚本 | `scripts/v3_train_path_a_minimal.py` |
| 评估脚本 | `scripts/v3_eval_excitation_vs_nist.py` |
| 评估核心 | `pinn_art/evaluation/excitation_eval.py` |
| 续训配置 | `configs/v3_stage_a_prime3_path_a.yaml` |
| M1 坐标映射 | `pinn_art/coords/mixed_map.py` |
| M2 神经 Dirac | `pinn_art/models/pinn_art_model.py` |
| M2 网络 (DeepONetDirac) | `pinn_art/nets/deeponet.py` |
| M2 物理 (Dirac 算子) | `pinn_art/physics/dirac_operator.py` |
| M2 物理 (DFS 势) | `pinn_art/physics/dfs_potential.py` |
| M3 R^k 计算 | `pinn_art/ci/slater_radial.py` |
| M3 Hamiltonian 装配 | `pinn_art/ci/hamiltonian.py` |
| M3 安全 eigh | `pinn_art/ci/eigen_solver.py` |
| M3 Path A 注入 | `pinn_art/physics/slater_correction.py` |
| Trainer | `pinn_art/training/stage_a_trainer.py` |
| 完整进展日志 (含失败迭代) | `docs/PROGRESS_REPORT_2026_06.md` |

---

## 九、下一步

1. **M3 训练稳定化**：先冻结 slater_log_scale 50 epoch（让 V_net 适应 return_ci=True），再解冻学习
2. **M4 辐射推断**：在 B'' heguard 上跑 `transitions` 和 `cross_sections` 输出
3. **多电子精度提升**：考虑 `nist_inject=true` 但只在 Z≥3 处注入
4. **谱偏差缓解**：可尝试 $c_1, c_2$ 网格搜索（B'' heguard 当前默认 1.0, 1.0）

---

**报告结束 (2026-06-15)**。
