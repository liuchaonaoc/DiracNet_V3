# Stage A Phase 3：Laguerre 解析骨架 ansatz

> 承接 Phase 2（见 `docs/STAGE_A_PHASE2.md` 和 `logs/gate_analysis/GATE_ANALYSIS_ZH.md`）。
> 路线 C — **类氢解析径向波函数作为 ansatz 骨架，SIREN 只学微扰**。
> **架构变化，不可从 Phase 1/2 ckpt resume。从头训。**

## 1. 动机

Phase 2 验收结果（放宽 Gate）：

- 通过 5/48（全部是 Z≥4 基态 n=1）
- pde 是首要瓶颈：38/48 行 pde>1.0
- **所有激发态（n≥2）cos<0.25**：网络从随机初始化长不出 `n−l−1` 个节点

Phase 3 把这个根因直接消去：用解析类氢径向波函数 `P_H_{n,l}(r;Z)` 作为骨架，节点结构与远场衰减天生正确。

## 2. ansatz 改造

**Phase 1/2**：

```
P_a(r) = env(r;|κ|,Z,n) * shape(r,branch,κ,n)
       = r^{|κ|} · exp(-Zr/n) · [1 + tanh(SIREN)]
```

`env` 没有 Laguerre 多项式 ⇒ 无节点 ⇒ 激发态学不到。

**Phase 3**：

```
P_a(r) = P_H_{n_a, l_a}(r; Z) · (1 + ε · tanh(SIREN(r, branch, n, l, κ)))
                                       ↑ final Dense init = 0  ⇒  P_a init = P_H
```

其中：

- `P_H` 是 JAX-native 实现（`pinn_art/physics/hydrogenic.py: hydrogenic_P_jax`）
- `ε = 0.2`（yaml 可调 `model.perturb_eps`）
- 解析 `dP_H/dr` 由 `hydrogenic_dP_dr_jax` 给出（用恒等式 `dL_k^α/dρ = −L_{k−1}^{α+1}`）

**关键性质（已有测试）**：

| 测试 | 期望 | 实际 |
|------|------|------|
| `init cos(H1s,H2s,H3p,He+1s,Be1s,Be2s)` vs 解析 | > 0.999 | ✅ 全过 |
| `init E_orb` vs `-Z²/(2n²)` | 相对误差 < 0.2 | ✅ 全过 |
| `hydrogenic_P_jax` vs scipy（13 组 n,l,Z） | rel err < 5e-3 | ✅ |
| `hydrogenic_P_jax` 归一化 ∫P²dr | ≈ 1 (±5%) | ✅ |
| `dP/dr` 解析 vs 有限差分 | rel err < 5% | ✅ |
| pytest 全套 | 64 项 | ✅ 64/64 |

## 3. 配置与超参变化

| 项 | Phase 2 | Phase 3 | 备注 |
|----|---------|---------|------|
| `model.use_hydrogenic_skeleton` | — | `true` | 新开关 |
| `model.perturb_eps` | — | `0.2` | 微扰幅度上界 |
| `weights.pde` | 1.0 | **5.0** | Phase 2 主瓶颈 |
| `weights.ortho` | 100 | 50 | 骨架已经接近正交 |
| `weights.asym` | 0.01 | 0.05 | 骨架尾巴已正确，小权重维稳 |
| `weights.norm` | 10 | 5 | 骨架已归一化 |
| `lr_trunk` | 1e-4 | **2e-4** | init 已好，主要微调 V_corr |
| `lr_branch` | 5e-5 | 1e-4 | |
| `gate` 命名 | relaxed/mid/strict（mid 缺） | **relaxed/mid/strict 全 3 档** | 与 phase2 兼容 |
| `resume_from` | Phase 1 ckpt | `null` | 架构变了 |

## 4. 训练命令

```bash
cd ~/workspace2/DiracNet_V3/rc_pinn_art_project
export PYTHONPATH=.

# 从头训 1000 epoch (~6 min on GPU)
python scripts/v3_train_stage_a.py \
  --config configs/v3_phase1_stage_a_z1_8_phase3.yaml
```

预计：JIT 首次编译 ~2 min（Laguerre 展开后图稍大）+ 1000×12 step @ ~0.01s/step。

## 5. 实测结果

### 5.1 Phase 3 @ init（0 epoch 训练）— 已验证

| Gate 档 | Phase 3 init | Phase 2 @1000 ep | 提升 |
|---------|--------------|------------------|------|
| **relaxed** (0.95/1.0/50) | **44/48** | 5/48 | **+39** |
| **mid** (0.98/0.5/50) | **41/48** | 5/48 | +36 |
| **strict** (0.99/0.01/50) | **23/48** | 0/48 | **+23** |

- 全部 48 行 cos = **1.0000**
- 全部 48 行 dE ≤ **5 meV**
- relaxed 失败 4 行：C/N/O 1s + O 2s（全是 pde 大于 1.0）
- strict 失败 25 行：全部因 pde > 0.01（C/N/O 1s 数值微分误差大）

### 5.2 Phase 3 @ 1000 epoch — 待跑

| 指标 | 预期 |
|------|------|
| relaxed | ≥ 46/48 |
| mid | ≥ 44/48 |
| strict | ≥ 30/48 |

理由：init 已极佳，训练只需把 pde 进一步压低。Phase 2 训练时 pde 从 0 epoch
1.x 降到 0.5x，等量级降幅就够把大多数 strict FAIL 转 PASS。

## 6. 验收命令

```bash
# 默认 mid Gate
python scripts/v3_gate_analytic.py \
  --config configs/v3_phase1_stage_a_z1_8_phase3.yaml \
  --ckpt checkpoints/v3_phase1_stage_a_z1_8_phase3/stage_a_last.msgpack

# 三档对照
for mode in relaxed mid strict; do
  python scripts/v3_gate_analytic.py \
    --config configs/v3_phase1_stage_a_z1_8_phase3.yaml \
    --ckpt checkpoints/v3_phase1_stage_a_z1_8_phase3/stage_a_last.msgpack \
    --gate-mode $mode
done
```

## 7. 失败回滚

若 1000 ep 后 mid Gate 仍 < 15/48：

1. 降 `perturb_eps` 至 0.1（让 SIREN 改变更小）
2. 把 `lr_trunk` 降到 5e-5 做 fine-tune
3. 关 `use_hydrogenic_skeleton`（回到 Phase 2 ansatz）—— 这相当于完全放弃路线 C

若效果很好（≥ 40/48 mid pass）：

1. 接 Stage B + NIST manifest（路线 E）
2. 把 `Z` 扩到 1..18，准备真实多电子训练
