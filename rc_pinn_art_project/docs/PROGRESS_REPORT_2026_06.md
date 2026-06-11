# rc_pinn_art_project 进展报告（2026-05-25 ~ 2026-06-11）

> **报告日期**：2026-06-11
> **覆盖范围**：近 18 天（5 月 25 日 Stage A 首次 1000 epoch 训练 → 6 月 11 日路径 B 评估）
> **目的**：梳理全局推进路线、过去 18 天测试覆盖的步骤、当前问题、优先级与后续工作

---

## 0. 项目一句话

**PINN-ART (PINN-based Atomic Radiation Transfer) v3**：基于 JAX/Flax 的多电子原子 Dirac 方程求解网络，融合 Laguerre 类氢解析骨架 ansatz、自洽 DFS 屏蔽势、Configuration Interaction 矩阵组装与 NIST 实验能级注入。**目标**：毫秒级多电子原子辐射转移计算（`README.md`）。

---

## 1. 全局开发路线（项目分阶段）

| 阶段 | 主题 | 输入 | 关键产物 | 纪律 | 文档 |
|------|------|------|---------|------|------|
| **Stage A Round 1** | 类氢（Z≤8）单粒子 Dirac 训练 | 48 行类氢 manifest（Z=1..8, n=1..6） | 训练好的 DeepONet | NIST 不入梯度 | `STAGE_A_ROUND1.md` |
| **Stage A Phase 2** | 放宽 Gate + 1000 ep 续训 | Round 1 ckpt | 收敛改进 | 同上 | `STAGE_A_PHASE2.md` |
| **Stage A Phase 3** | **Laguerre 解析骨架 ansatz** | 从零训（架构变化） | `p1z8_phase3`（init 即可 41/48 strict） | 同上 | `STAGE_A_PHASE3.md` |
| **Stage A Round 2** | 扩到 Z=1..26、DFS 自洽 | NIST manifest (20814 行) | `v3_stage_a_z1_26_n10`（**全面 FAIL**） | 同上 | — |
| **Stage 2 (p1z8)** | 课程式 Z≤8 全组态 | p1lowz → p1z8_ground → p1z8_full | `p1z8_full`（**条件 PASS**）| 同上 | `STAGE2_EVALUATION_PLAN.md` |
| **Stage 3a (旧)** | Z=3-4 闭壳 + 全组态 | p1z8_full | 灾难性遗忘（H-177 meV）| 同上 | `STAGE3A_EVALUATION_PLAN.md` |
| **Stage 3a'** | 重训：低 Z 混合 + anchor guard | 多轮迭代 (prime_v1→prime_v2→B→E→K) | `E-prime/best_anchor`（H=-3.4 meV，**H 锚点冠军**）| 同上 | 同上 |
| **Stage B** | CI 径向校准（slater_log_scale） | Stage A ckpt + class-H manifest | 冻结 DeepONet，只训 3 个 slater 标量 | NIST 不入梯度 | `STAGE_B.md` |
| **Stage C** | 推断 + NIST 混合对角填充 | Stage B ckpt + NIST manifest | CI 能量 vs NIST | 不训练 | `STAGE_C.md` |
| **Stage C Phase 2c** | 多 CSF 与真实 CI | 2p 双重态 NIST | 8/8 谱组 off-diag | 不训练 | `STAGE_C.md §7` |
| **Phase 3 (后续)** | Layer-3/4 + LOO + FAC | — | 跃迁/截面/延迟性能 | — | `STAGE_C.md §7.3` |

**项目架构层次（Layer-0 → Layer-4）**：
- **Layer-0**：波函数 / PDE / 势场（cos, pde, V_net vs V_dfs, 训练收敛）
- **Layer-1**：轨道能自洽（E_orb, dE vs 参考）
- **Layer-2**：激发能 vs NIST（single_valence / multi_electron 双组）
- **Layer-3**：E1 跃迁速率
- **Layer-4**：延迟（性能 p50_ms）

---

## 2. 过去 18 天的测试覆盖矩阵

| 阶段 | 日期 | 测试名 | 主要结果 | 报告路径 |
|------|------|--------|---------|----------|
| **Stage A P1** | 05-25 | `TRAINING_REPORT_1000ep.md` | 1000 ep 训练完成，pde 2.62→1.14（震荡），0/48 strict | `logs/v3_phase1_stage_a_z1_8/TRAINING_REPORT_1000ep.md` |
| **Stage A P3** | 06-02 | `GATE_ANALYSIS_ZH.md` | Laguerre 骨架 init：**relaxed 48/48 · mid 47/48 · strict 41/48**（首次达到 strict 通过）| `logs/gate_analysis/GATE_ANALYSIS_ZH.md` |
| **Stage A P3** | 06-02 | `ENERGY_VS_NIST.md` | 单粒子能量 vs NIST（基态全 PASS）| `logs/v3_phase1_stage_a_z1_8_phase3/ENERGY_VS_NIST.md` |
| **Stage A P3** | 06-02 | `PDE_DIAGNOSTIC_REPORT.md` | 标量 PDE 几乎全在 core 区域（r<2） | `logs/v3_phase1_stage_a_z1_8_phase3/pde_diagnostic/PDE_DIAGNOSTIC_REPORT.md` |
| **Stage A R2** | 06-04 | `EXCITATION_VS_NIST.md` | **全面 FAIL**：MAE 80.6 eV (single) / 163.5 eV (multi)；H 2s 误差 7960 meV（崩坏 78%）；cos 中位 0.354 | `logs/v3_stage_a_z1_26_n10/EXCITATION_VS_NIST.md` |
| **Stage B** | 06-02 | `STAGE_B_EVALUATION_REPORT.md` | **PASS**：合成 H 2×2/3×3 偏差 <0.01 meV；48/48 类氢 manifest 能量自洽；可进入 Stage C | `logs/v3_phase1_stage_b_z1_8/STAGE_B_EVALUATION_REPORT.md` |
| **Stage C P1** | 06-02 | `STAGE_C_EVALUATION_REPORT.md` | NIST manifest (48 行) 端到端跑通；inject 自检 ~0.001 meV | `logs/v3_phase1_stage_c_z1_8/STAGE_C_EVALUATION_REPORT.md` |
| **Stage C P2a** | 06-02 | Gate C 报告 | `layer2_inject` MAE ≈ 0.001 meV（PASS）；`layer2_orb` MAE 1563 meV（监控）| `logs/v3_phase1_stage_c_z1_8/` |
| **Stage C P2c** | 06-02 | `STAGE_C_MULTICSF_REPORT.md` | 2p 双重态多 CSF：8/8 谱组 `\|H_off\|>0`；管线 PASS | `logs/v3_phase1_stage_c_multicsf_z1_8/STAGE_C_MULTICSF_REPORT.md` |
| **Stage 2** | 06-05 | `STAGE2_EVALUATION_REPORT.md` | p1z8_full：**条件 PASS**；H 锚点 +38 meV；Z≤8 single MAE 15.5 eV；l≥1 形状好，s 轨道崩坏 | `logs/v3_stage_a_z1_26_n10_p1z8_full_eval/stage2_suite/STAGE2_EVALUATION_REPORT.md` |
| **Stage 3a 旧** | 06-07 | `STAGE3A_EVALUATION_REPORT.md` | **未达标**：H 误差 -177 meV（灾难性遗忘）；Li MAE 3989 meV (vs 1558)；He multi 24 eV | `logs/stage3a_p1z3_4_full_eval/suite/STAGE3A_EVALUATION_REPORT.md` |
| **Stage 3a' v1** | 06-08 | `STAGE3A_PRIME_DIAGNOSIS.md` | guard 太严，1 epoch 早停；H 飘到 +293 meV → FAIL | `logs/stage3a_p1z3_4_prime_eval/STAGE3A_PRIME_DIAGNOSIS.md` |
| **Stage 3a' v2** | 06-08 | `STAGE3A_PRIME_V2_EVAL.md` | 放宽 guard；H 恢复 -29 meV；但 Z=3-4 退步（l=0 27 eV）| `logs/stage3a_p1z3_4_prime_v2_eval/STAGE3A_PRIME_V2_EVAL.md` |
| **Stage 3a' B** | 06-08 | `STAGE3A_RESUME_B_EVAL.md` | **scf_weight_mode 切换** (r² → density) 致配置不一致；H 飘到 -465 meV；Z≤2 退化 17× | `logs/stage3a_p1z3_4_resume_b_eval/STAGE3A_RESUME_B_EVAL.md` |
| **Stage 3a' E** | 06-09 | `STAGE3A_EPRIME_EVAL.md` | **H 锚点冠军**：H=-3.4 meV；Z≤2 20.4%；l≥1 dE 59-113 meV；**l=0 27 eV 仍未解决** | `logs/stage3a_p1z3_4_eprime_eval/STAGE3A_EPRIME_EVAL.md` |
| **Stage 3a' K** | 06-09 | `STAGE3A_EPRIME_K_EVAL.md` | K-prime：从 prime_v2 续训恢复 H 到 -18 meV（次好） | `logs/stage3a_p1z3_4_eprime_k_eval/STAGE3A_EPRIME_K_EVAL.md` |
| **Stage 3a' Lowdin** | 06-09 | `STAGE3A_LOWDIN_EVAL.md` | **apply_lowdin=true 是负优化**（只做归一化，破坏 norm_loss）；H 飘到 +61 meV，Z=3-4 0.4% | `logs/stage3a_lowdin_eval/STAGE3A_LOWDIN_EVAL.md` |
| **Stage B (B1)** | 06-09 | `STAGE_B_B1_EVAL.md` | **slater_log_scale 校准不影响单粒子 dE**；Gate A 与 E-prime 完全一致（55/496, l=0 27258 meV）| `logs/stage_b_z1_26_n10_eval/STAGE_B_B1_EVAL.md` |
| **路径 B (zeff)** | 06-10 | `STAGE_A_ZEFF_B_EVAL.md` | **完全失败**：`V_zeff_anchor` 替换 `V_dfs` anchor 导致 V_net 符号反转；H 1s E_orb -0.503 → +0.500 Ha；Z≤2 通过 20.4% → 0% | `logs/stage_a_z1_26_n10_zeff_eval/STAGE_A_ZEFF_B_EVAL.md` |

**覆盖的全局步骤**：
- ✓ Stage A P1 (R1 baseline) → P2 → P3（架构创新：Laguerre 骨架）
- ✓ Stage A R2（Z=1-26 大尺度训练）→ 失败但定位到"DFS 屏蔽势+大数据量"组合问题
- ✓ Stage B（P1 范围）→ **PASS**（CI 管线自洽）
- ✓ Stage C（P1, P2a, P2c）→ **PASS**（多 CSF 2p 精细结构冒烟）
- ✓ Stage 2（p1z8）→ 条件 PASS（s 轨道失败被定位）
- ✓ Stage 3a（4 次迭代）→ 找到 E-prime（H 锚点冠军）
- ✓ 多种 l=0 修复尝试（Lowdin、Stage B Slater、Zeff anchor）→ **均失败**

**未覆盖/未尝试**：
- ✗ 路径 A（R^k → V_dfs 修正项；定义在路径 B 失败后尚未实施）
- ✗ MCSCF 联合优化（重大重构，未启动）
- ✗ Stage C Phase 3（Layer-3/4 + LOO + FAC）
- ✗ NIST manifest 扩到 Z>8（当前 Z=1-26 NIST manifest 已生成，但训练未做）

---

## 3. 当前最佳资产清单（按优先级）

| 资产 | 路径 | 用途 | 状态 |
|------|------|------|------|
| **E-prime (H 锚点冠军)** | `checkpoints/v3_stage_a_z1_26_n10_p1z3_4_eprime_full/best_anchor.msgpack` | **Stage 3a 最佳**：H=-3.4 meV，l≥1 dE 59-113 meV，Z=3-4 11.1% | 2026-06-09 记录 |
| **p1z8 Phase 3 (Laguerre 骨架 init)** | `checkpoints/v3_phase1_stage_a_z1_8_phase3/stage_a_last.msgpack` | **P1 最佳**：init 48/48 relaxed，41/48 strict | 2026-06-02 |
| **Stage C 管线（Z≤8）** | `v3_phase1_stage_c_z1_8/metrics.json` | 端到端 NIST 注入冒烟 | 2026-06-02 |
| **NIST manifest Z=1-26** | `data_cache/manifest_nist_z1_26_n10.parquet` | 20814 行评估/训练 | 已生成 |
| **class-H manifest Z=1-26** | `data_cache/manifest_hydrogenic_z1_26_n10.parquet` | 260 行 Stage B 用 | 06-09 生成 |
| **Racah cache Z=1-26** | `data_cache/racah_cache_z1_26_n10.npz` | 26 Z × 3 k × 8 CSF | 已存在 |

---

## 4. 关键问题的"诊断史"（核心经验）

### 4.1 l=0 Z=3-4 瓶颈

**表现**：E-prime 训练后，l=0 dE_med = **27258 meV (27 eV)**，340 行 l=0 仅 5 行通过；l≥1 59-113 meV（良好）。

**诊断路径**：
1. **Stage 3a 旧** 训练 1200 epoch 后 Li/Be MAE 退步 → 灾难性遗忘
2. **prime_v1/v2/B/K** 系列 → 缓解 H 锚点但 l=0 仍在
3. **Lowdin 尝试** → `apply_lowdin=true` 是负优化（只做归一化）
4. **Stage B (B1)** → slater_log_scale 校准不影响单粒子 dE
5. **路径 B (zeff anchor)** → V_net 符号反转

**根因（当前共识）**：**单粒子 DFS 缺乏多电子屏蔽**——l=0 是 s 轨道，最内层，受内层电子相互屏蔽影响大；DFS 自洽场是平均场近似，**对 2s/1s 屏蔽比不准确**。slater_log_scale 是 CI 矩阵修正，不进 DFS 方程。

### 4.2 灾难性遗忘

**表现**：训练 Z=3-4 时 H 锚点从 -34 meV 漂到 -177 meV 甚至 -465 meV。

**根因**：
- 共享网络参数（DeepONet 全部权重）被 Z=3-4 训练冲掉
- anchor guard 不严格 / 配置切换（scf_weight_mode r² → density）导致 loss landscape 跳变
- prime_v1 的 do_check 逻辑在 epoch 0-1 强制检查 + 阈值 100 meV → 1 epoch 早停

**修复（E-prime 成功）**：
- 50% anchor mix（30%→50%）
- lr_mult 0.1（从 0.3）
- 宽容 guard：H 阈值 400 meV（从 100/200）
- warmup 3 epoch + patience 6
- **配置全程一致**（scf=density 不切换）

### 4.3 配置不一致陷阱

**表现**：B-prime 用 r² 加权 SCF 训的 ckpt 改用 density 加权续训，10 epoch 后 V 反向、H 误差爆炸到 -465 meV。

**教训**：续训必须保证 `scf_weight_mode`、`weights.*`、`model.*` 等所有相关字段与新 config 一致。

### 4.4 apply_lowdin 实现陷阱

**表现**：开 `apply_lowdin: true` 后所有指标退化（Z=3-4 0.4%）。

**根因**：`lowdin_orthonormalize` 的 `use_full_lowdin` 默认 `False`——只做单轨道归一化，**不做 S⁻¹/² 多轨道混合**。强制归一化破坏 `norm_loss` 监督。

**修复建议**：要么不开启，要么修改 `pinn_art_model.py` 传 `use_full_lowdin=True, normalize=False`。

### 4.5 路径 B（V_zeff anchor）失败

**表现**：`anchor_vprior_zeff=true` 让 V_net 整体加 +1 Ha（符号反转）。

**根因**：
- V_zeff_anchor = `-Z_eff/r` 在 r 大时 ≈ -1/r（纯库仑）
- V_dfs 在 r 大时 ≈ Latter tail `-(Z-N+1)/r`
- `v_prior=1.0` + `scf=5.0` 两路 anchor 冲突，V_net 被拉到非物理的中间状态
- 训练时看上去 loss 下降，但 dE 退化在 epoch 100+ 才显现

---

## 5. 当前最重要的 5 个问题（按优先级）

### 优先级 1（**最高**）：**l=0 Z=3-4 瓶颈**

- **症状**：E-prime 后 l=0 dE_med 27 eV（340 行仅 5 通过）
- **影响范围**：决定 Z=3-26 整个体系的精度
- **已尝试方案**：Lowdin、Stage B Slater、Zeff anchor → 均失败
- **下一步候选**：
  - 路径 A（R^k → V_dfs 修正项，~150-200 行新代码，物理推导复杂）
  - MCSCF 联合优化（重大重构）
  - Stage C (nist_inject) 在 CI 矩阵上引入 NIST 差值，迫使单粒子学习屏蔽
  - Stage A 内重写 ansatz（slater_log_scale 直接进入单粒子）

### 优先级 2：**H 锚点保持**

- **症状**：Z=3-4 训练易让 H 误差爆炸到 ±500 meV
- **当前最佳**：E-prime (-3.4 meV)
- **风险**：任何新实验必须从 E-prime 冷启动 + 全程配置一致

### 优先级 3：**Stage C 大 Z 扩展未做**

- **现状**：Z=1-8 NIST manifest 已 PASS（48 行）
- **缺**：Z=1-26 NIST manifest 端到端评估（`manifest_nist_z1_26_n10.parquet` 20814 行已生成但未在 Stage C 跑过）
- **下一步**：写 `v3_evaluate_stage_c.py` 的 Z=1-26 适配，跑 layer-2_orb 报告

### 优先级 4：**PDE 残差偶发尖峰**

- **症状**：Phase 1 1000ep 训练时 norm 在 ep749 突然从 0.5 涨到 5.9
- **当前状态**：未深入分析；可能与 grid 边界 / SIREN 初始化相关
- **风险**：尖峰 → 模型权重跳变 → 早停或评估失真

### 优先级 5：**Racah cache 与 CSF 选择对 Z=1-26 是否充分**

- **现状**：`racah_cache_z1_26_n10.npz` 26 Z × 3 k × 8 CSF（class-H manifest 是 1 电子体系）
- **多电子扩展**：当 Z≥2 (Li, Be, B...) 有 multi-electron 行，需要更多 CSF 描述电子相关
- **风险**：当前 cache 的 CSF 数 (8) 可能不足以表达 Li 1s² 2s¹ 的电子结构

---

## 6. 后续工作步骤（按优先级）

### 第 1 步（**1-2 天**）：**Stage C Z=1-26 端到端评估**

**目标**：把 E-prime 跑全 NIST manifest（20814 行），看 layer-2_orb / layer-2_inject 在 Z=1-26 范围的真实精度。

**命令**：
```bash
# 适配 v3_evaluate_stage_c.py 接受 Z=1-26 manifest
python scripts/v3_evaluate_stage_c.py --config configs/v3_stage_c_z1_26.yaml
# 缺 configs/v3_stage_c_z1_26.yaml，需创建
```

**产物**：
- `logs/v3_stage_c_z1_26/metrics.json`（layer-2_orb / layer-2_inject / multi_electron）
- `STAGE_C_EVALUATION_REPORT.md`（Z=1-26 完整 NIST 对齐报告）

**预期**：与 Phase 2c 一致，注入部分 ~0 meV；E_orb 部分 MAE ~ 1000-3000 meV（H/He 部分好，高 Z 差）。

### 第 2 步（**2-3 天**）：**路径 A（R^k → V_dfs 修正项）实施**

**目标**：把 `slater_log_scale` 真正注入到 V_dfs，让校准 Slater 积分影响单粒子能量。

**范围**：
- 新建 `pinn_art/physics/slater_correction.py`：从 P, Q 计算 k 阶约化密度 ρ_k(r)
- 改 `pinn_art/physics/dfs_potential.py`：在 `build_dfs_potential` 中加 `V_slater_corr = -Σ_k α_k · R^k · ρ_k(r)`
- 改 `pinn_art/models/pinn_art_model.py`：解冻 `slater_log_scale`（Stage A 训练时也更新）
- 新 config `v3_stage_a_z1_26_n10_path_a.yaml`（开启 slater 修正）

**风险**：
- R^k 从 CI 矩阵元转 r 函数是非平凡推导（涉及 Slater 积分密度近似）
- 解冻 slater_log_scale → Stage A 训练不稳定（多 1 自由度）

**期望**：l=0 dE_med 27 eV → 5-10 eV（5× 改善）

### 第 3 步（**1-2 周**）：**MCSCF 联合优化或 Stage A 架构升级**

如果路径 A 仍不解决 l=0，需要根本性变化：
- 轨道旋转 + 变分优化（MCSCF 风格）
- 或用 explicit `Z_eff` 参数化（让网络直接学习"对每个 n,l 屏蔽多少"）

### 第 4 步（**1 周**）：**Phase 3 Layer-3/4 + LOO**

按 `STAGE_C.md §7.3` 路线图：
- E1 跃迁速率评估
- 延迟性能（Fe XVII 规模，p50_ms 目标 <10 ms）
- LOO 验证（leave-one-Z / leave-one-n 表格）

### 第 5 步（**持续**）：**修复 PDE 残差尖峰 + Racah cache 扩展**

并行任务，不阻塞主路径。

---

## 7. 决策树（短期：未来 1-2 周）

```text
是否进入 Stage C Z=1-26 评估？
  ├─ 是（推荐）→ 第 1 步（1-2 天）
  └─ 否 → 第 2 步（路径 A）

路径 A 是否解决 l=0（dE_med < 10 eV）？
  ├─ 是 → 进入 Stage C NIST 注入 + Layer-3 跃迁
  ├─ 否（dE_med 10-20 eV，部分改善）→ 加 Stage A 重训（提高 l=0 数据权重）
  └─ 否（无改善）→ 进入第 3 步（MCSCF / 架构升级）
```

---

## 8. 关键经验教训（避免重蹈覆辙）

1. **配置一致性**：续训必须保证 `scf_weight_mode` / `weights.*` / `model.*` 与新 config 完全一致
2. **每 epoch 监控 dE**：不能只看 loss；路径 B 失败就是 200 epoch 内 dE 退化但 loss 仍在下降
3. **anchor mix 是双刃剑**：太弱（30%）防不住遗忘，太强（>50%）让 Li/Be 学不到
4. **apply_lowdin 当前实现是 normalize-only**：开启前必须读 `orthogonalizer.py` 确认 `use_full_lowdin` 参数
5. **l=0 27 eV 是 DFS 单粒子天花板**：CI 矩阵（slater_log_scale）无法解决
6. **H 锚点 +500 meV = 灾难性遗忘**：H 误差 ±400 meV 内是安全区

---

## 9. 关键文件速查

| 用途 | 路径 |
|------|------|
| 当前 Stage 3a 最佳 ckpt | `checkpoints/v3_stage_a_z1_26_n10_p1z3_4_eprime_full/best_anchor.msgpack` |
| Stage 3a 评估套件 | `scripts/v3_eval_stage3a_suite.sh` |
| Stage B 评估套件 | `scripts/v3_eval_stage_b_suite.sh` |
| 路径 B 评估套件 | `scripts/v3_eval_zeff_suite.sh` |
| Laguerre 骨架 init 配置 | `configs/v3_phase1_stage_a_z1_8_phase3.yaml` |
| E-prime 配置 | `configs/v3_stage_a_z1_26_n10.yaml` |
| 路径 B 配置（失败留档）| `configs/v3_stage_a_z1_26_n10_zeff.yaml` |
| helper（路径 A 起点）| `pinn_art/physics/dfs_potential.py:185-225` `zeff_anchor_potential()` |
| Stage A 训练核心 | `pinn_art/training/stage_a_trainer.py` |
| Stage B 训练核心 | `pinn_art/training/stage_b_trainer.py` |

---

**报告结束**。下一步建议执行 §6 第 1 步（Stage C Z=1-26 评估）——可快速给当前模型在大 Z 上的精度基线。
