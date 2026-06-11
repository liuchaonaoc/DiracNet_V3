# 16 — Stage A Round 2：自洽 DFS 全组态预训练（Z≤26, n≤10）

> 阅读优先级：**最高（本轮主入口）**。动手前先读 [`00_overview.md`](./00_overview.md) §2.1、
> [`06_physics_losses.md`](./06_physics_losses.md)、[`07_training_pipeline.md`](./07_training_pipeline.md)、
> [`09_data_pipeline.md`](./09_data_pipeline.md)、[`08_evaluation.md`](./08_evaluation.md)。
>
> 本文是「回到 Stage A、把 Z≤26 / n≤10 的所有组态跑通」这一轮工作的执行清单（Round 2）。

## 0. 背景与动机

Round 1（Phase 1–3）的 Stage A 现状：

- 训练数据只有 `manifest_hydrogenic_z1_8.parquet`：**Z=1…8、每个仅 `ns1`（单电子、单 s 轨道、n=1…6）**。
- ansatz 是解析氢骨架 `P_H(r;Z)·(1+ε·SIREN)`，势 `V=-Z/r+tanh(V_corr)`，`v_prior` 锚定 **裸核** `-Z/r`。
- 后果：波函数形状好（`cos≈1`），但 `E_orb` 激发能 vs NIST **MAE≈1.56 eV（~0.5%）**，且随 Z 单调放大（见 `logs/v3_phase1_stage_a_z1_8_phase3/ENERGY_VS_NIST.md`）。这是**裸类氢能量标度**导致，不是形状崩坏。

**本轮目标**：把 Stage A 从「裸类氢单电子 + 微扰」升级为「**自洽 DFS 屏蔽局域中心势求解器**」，覆盖 **Z=1…26、n≤10 的基态 + 单电子激发组态**，并在**不注入 NIST** 的前提下，使理论激发能逼近 NIST。完成后再回 Stage C 复检。

**不可违反纪律（沿用 `00_overview.md` §2.1）**：Stage A **NIST 不进 backward**；不得用 per-row learnable bias；不得用 NIST 梯度。屏蔽势必须来自物理（DFS），而非拟合实验值。

## 1. 物理路线：自洽 Dirac-Fock-Slater（DHFS）局域中心势

闭环（全可微，替代传统 SCF 迭代）：

1. 电子密度：$\rho(r)=\sum_a \omega_a\,(P_a^2+Q_a^2)$（按占据数 $\omega_a$ 求和，`stop_gradient` 不施加在此——密度可微）。
2. Hartree 直接屏蔽：$V_H(r)=\int_0^\infty \frac{\rho(r')}{r_>}\,dr' = \frac{1}{r}\!\int_0^r\!\rho\,dr' + \int_r^\infty\!\frac{\rho}{r'}\,dr'$（网格累积积分，可微）。
3. Slater 局域交换：$V_x(r)=-\frac{3}{2}\,\alpha_x\Big[\frac{3}{\pi}\,\rho(r)\Big]^{1/3}$（Xα，`dfs.alpha_x` 默认 1.0；可调）。
4. 总势：$V_\mathrm{DFS}(r)=-\dfrac{Z}{r}+V_H(r)+V_x(r)$，并做**尾部修正**使离子渐近为 $-\dfrac{Z-N+1}{r}$（$N$=电子数；Latter tail 截断）。
5. **自洽一致性**：网络势头输出 $V_\mathrm{net}$，加 loss 让它收敛到 $V_\mathrm{DFS}[\rho]$ 的不动点：
   $$L_\mathrm{scf}=\big\langle\,(V_\mathrm{net}-\mathrm{sg}[V_\mathrm{DFS}[\rho]])^2\,\big\rangle$$
   （`sg`=`stop_gradient`，先稳；后续可去 sg 做完全联立。）

> 骨架仍可用 $Z_\mathrm{eff}$ 暖启动以稳住 `n−l−1` 节点结构，但**能量标度由 $V_\mathrm{DFS}$ 决定**，不再是裸 Z。

## 2. 执行清单（R0 → R1 → R2 → R3）

### Phase R0 — 数据 / 组态枚举 / 双组划分

| # | 产物 | 说明 |
|---|------|------|
| R0.1 | `scripts/v3_enumerate_configs.py` | Z=1…26、`n_max=10`、基态 + **单电子激发** → 组态清单（含 `parent_config`/`level_config`/`J`/`parity`） |
| R0.2 | 扩 `scripts/v3_prepare_nist_manifest.py` | 支持任意 $l,J$、任意 `level_config`（不再只解析 `s`）；输出 `level_meV`（激发能）、`level_abs_meV`、`has_nist_level` |
| R0.3 | 分组标签 `group` | `single_valence`（闭壳芯外仅一个活动价电子：H/Li/Na-like 等序列）/ `multi_electron`（其余）。写入 manifest 一列 |
| R0.4 | `data_cache/manifest_nist_z1_26_n10.parquet` | 两组各自的 NIST 覆盖率统计打印 |

**验收**：

```bash
cd DiracNet_V3/rc_pinn_art_project && export PYTHONPATH=.
python scripts/v3_enumerate_configs.py --z-max 26 --n-max 10
python scripts/v3_prepare_nist_manifest.py --z-max 26 --n-levels 10 \
  --out data_cache/manifest_nist_z1_26_n10.parquet
# 打印：两组行数、各自 has_nist_level 覆盖率
```

### Phase R1 — Stage A 自洽 DFS 物理（核心）

| # | 文件 | 改动 |
|---|------|------|
| R1.1 | `pinn_art/physics/dfs_potential.py`（新） | `build_dfs_potential(P, Q, omega, Z, nele, grid) -> V_dfs`；含 $V_H$、$V_x$、Latter 尾部修正；全可微 |
| R1.2 | `pinn_art/losses/scf_consistency.py`（新） | `scf_consistency_loss(V_net, V_dfs)`；默认对 `V_dfs` 加 `stop_gradient` |
| R1.3 | `pinn_art/nets/deeponet.py` | 势头锚点从 `-Z/r` 改为 `V_dfs`（或 `-Z_eff/r` 暖启动）；骨架可用 $Z_\mathrm{eff}$ |
| R1.4 | `pinn_art/losses/potential_prior.py` / `training/stage_a_trainer.py` | `v_prior` 锚点换为 DFS；总损失加 `w_scf·L_scf`；重平衡 `pde/ortho/asym/norm/scf` |
| R1.5 | `configs/v3_stage_a_z1_26_n10.yaml`（新） | 扩 `Z=1..26`、扩 `n_orb_max`（按枚举上限）；新增 `weights.scf`、`dfs.alpha_x`、`dfs.latter_tail` |

**验收**：

```bash
python scripts/v3_train_stage_a.py --config configs/v3_stage_a_z1_26_n10.yaml  # 50 step 不报错
# 物理自检：V(r) 渐近 → -(Z-N+1)/r；闭壳离子 E_orb 系统差较裸类氢显著下降
pytest tests/test_dfs_potential.py tests/test_scf_consistency.py -q
```

新增测试：

- `test_dfs_potential.py`：He（闭壳 1s²）$V_\mathrm{DFS}$ 大 r 渐近 → $-1/r$；归一化密度积分 = $N$。
- `test_scf_consistency.py`：解析 $V_\mathrm{DFS}[\rho]$ 与有限差分一致；loss 在自洽点为 0。

### Phase R2 — 全组态训练 + 双组评估

| # | 任务 | 验收 |
|---|------|------|
| R2.1 | 训练 + Gate A（可课程式：先闭壳、后开壳；先低 Z、后高 Z） | Gate A 报告 |
| R2.2 | `scripts/v3_eval_excitation_vs_nist.py`（新）：**分两组**输出激发能 vs NIST | 两组各自 MAE/中位数/相对误差/覆盖率/达标占比 |

评估口径（两组都与 NIST 比，**分组分别报告**）：

| 组 | 能量来源（不注入 NIST） | 比较对象 |
|----|--------------------------|----------|
| `single_valence` | 直接用 `E_orb` 之差 | NIST 激发能 |
| `multi_electron` | 理论 CI（`E_orb`+Slater $R^k$+Racah，`nist_inject:false`） | NIST 激发能 |

**达标判据（建议，可在 yaml 调）**：

- `single_valence`：多数组态 < 50 meV 或 < 0.2% 相对误差；
- `multi_electron`：多数组态 < 数百 meV（受 CI 截断与 DFS 近似限制）。

```bash
python scripts/v3_gate_analytic.py --config configs/v3_stage_a_z1_26_n10.yaml \
  --ckpt checkpoints/v3_stage_a_z1_26_n10/stage_a_last.msgpack
python scripts/v3_eval_excitation_vs_nist.py \
  --config configs/v3_stage_a_z1_26_n10.yaml \
  --ckpt checkpoints/v3_stage_a_z1_26_n10/stage_a_last.msgpack \
  --out-dir logs/v3_stage_a_z1_26_n10
# 输出：EXCITATION_VS_NIST.md（single_valence / multi_electron 两节）
```

### Phase R3 — 回 Stage C 复检

| # | 任务 | 验收 |
|---|------|------|
| R3.1 | 用新 Stage A 权重重跑多 CSF Stage C | `layer2_orb`（理论 Fall-back）MAE 下降 |
| R3.2 | 更新 `ENERGY_VS_NIST.md` / `STAGE_C_*_REPORT.md`（双组口径） | 报告刷新 |

```bash
python scripts/v3_evaluate_multicsf_stage_c.py \
  --config configs/v3_phase1_stage_c_multicsf_z1_8.yaml   # 换用新 Stage A ckpt
```

## 3. 新增 / 改动文件清单

```text
新增
  scripts/v3_enumerate_configs.py
  scripts/v3_eval_excitation_vs_nist.py
  pinn_art/physics/dfs_potential.py
  pinn_art/losses/scf_consistency.py
  configs/v3_stage_a_z1_26_n10.yaml
  tests/test_dfs_potential.py
  tests/test_scf_consistency.py

改动
  scripts/v3_prepare_nist_manifest.py    # 任意 l,J,config + group 标签
  pinn_art/nets/deeponet.py              # 势头锚点 → V_dfs
  pinn_art/losses/potential_prior.py     # 锚点 → DFS（或保留为 fallback）
  pinn_art/training/stage_a_trainer.py   # 加 w_scf·L_scf，权重重平衡
```

## 4. 风险与降级

| 风险 | 降级 |
|------|------|
| 全组态枚举爆炸 | 仅基态 + 单激发；高 n 仅保留 NIST 收录组态 |
| DFS 自洽不收敛 | `L_scf` 用 `stop_gradient` + warmup；先固定 $V_\mathrm{DFS}$ 几百 epoch 再联立 |
| 高 Z 开壳组态难训 | 课程式：闭壳 → 开壳；低 Z → 高 Z |
| `n_orb_max` 不足 | 按枚举上限提高；或按组态裁剪占据轨道集 |
| multi_electron CI 精度天花板 | 报告中明确受 DFS + CI 截断限制，作为后续双激发/Breit 工作 |

## 5. 与既有文档的关系

- 纪律与口径：`00_overview.md` §2.1 / §2.1.1（已更新指向本文）。
- 损失项 `L_scf`：`06_physics_losses.md` §1 Stage A。
- 训练数据与状态机：`07_training_pipeline.md` §1。
- 双组评估：`08_evaluation.md` §1 Layer-2。
- 组态枚举与分组：`09_data_pipeline.md` §2、§7。
- 里程碑：`11_sprint_plan.md` Round 2。
