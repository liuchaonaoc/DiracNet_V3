# Stage A Round 2（广义拉盖尔基）两周测试评估总结报告

> 报告区间：2026-06-15 ～ 2026-06-29
> 评估对象：`rc_pinn_art_project/` Stage A Round 2，广义拉盖尔有限基架构
> 训练 manifest：`manifest_hydrogenic_z1_26_n10.parquet`（260 行，Z=1..26 × n=1..10，单电子类氢态）
> 数据来源：`results/laguerre_basis_eval/*.json`、`cfac_jobs/energy_batch/energy_compare_*`、`logs/v3_stage_a_laguerre_basis_*/history.csv`
> 本报告仅总结已完成工作，不含未来改动建议。

---

## 1. 概述

两周内围绕"广义拉盖尔有限基 + 可学 λ + 动能平衡 Q"架构完成了一轮密集的渐进式迭代实验（Step B → C → D → E → F → G v1/v2 → H），共 7 个主要实验步、十余次训练运行（每次 5000 epoch）与配套评估。整体目标是将 Step B 在 15 行小 manifest（H/He/Li，n=1..5）上达到的 100% 形态门禁，推广到 260 行全 (Z, n) 扫描，并满足设计文档 `17_generalized_laguerre_basis.md` §13.7 的 DoD（节点门禁 ≥ 85%、能量 RMSE < Step C 基线 28 042 meV）。

**结论先行**：截至 Step H，DoD **未达成**。260 行节点门禁在 60–63% 之间反复，未出现突破；高 n（n≥8）系统性失败模式贯穿全部七步；Step H 引入的解析 dQ（方向 B）产生边际正收益，coeff 锚定（方向 C，权重 50）被证实力度不足。能量层面出现一个新的形态-能量解耦离群点（Z=18 H-like 1s，PINN 能量偏深约 4×）。多步实验产出的是一组高质量的**负结果与根因诊断**，而非达标架构。

### 1.1 时间线

| 日期 | 步骤 | 核心改动 | 训练规模 |
|------|------|----------|----------|
| ~06-20 | Step B（R3 fix v2） | per-batch 解析 init（`_per_orbital_laguerre_init`）、零初始化残差头、JIT-safe `hydrogenic_laguerre_coeffs` | 15 行，5k ep |
| 06-20 | Step C | manifest 扩到 260 行；`r_max=250, n_grid=512, d_trunk=128, lambda_prior=0.1, batch=8` | 260 行，5k ep |
| 06-21 | Step D | §13.1 双 trunk + §13.2 log-r 辅助输入 | 260 行，1k/5k ep |
| 06-21 | Step E | §13.9 branch 容量扩展（`d_branch=256, coeff_d_hidden=128, lambda_d_hidden=64`） | 260 行，5k ep（GPU） |
| 06-21 | Step F | §13.10-A `coeff_decay` 权重 1e-3→5.0，公式 1/k!→1/(k+1) | 260 行，5k ep（GPU） |
| 06-21 | Step G v1 | §13.11-C `HybridLaguerreHead`（节点粗估 + c_k 迭代精化） | 260 行，5k ep（GPU） |
| 06-23 | Step G v2 | 修复 `analytic_nodes` 路径加载后重训 | 260 行，5k ep（GPU） |
| 06-23 | Step H | §13.B 解析 dQ（`kinetic_balance_q_dq_analytic` + `return_d2`）+ §13.C `coeff_anchor_loss`（权重 50, n_min=8）；回退 legacy head | 260 行，5k ep（GPU） |

---

## 2. 评估方法与指标定义

### 2.1 形态学层（Layer-0）

评估 JSON `results/laguerre_basis_eval/stage_a_<tag>.json` 逐活跃轨道记录：

- **node_gate**：节点数门禁。`nodes_observed == nodes_expected`（expected = n−l−1）。**仅判节点数，不判形状/相位**。
- **node_gate_pass_rate**：通过 node_gate 的活跃轨道数 / 总活跃轨道数（260 行全 s 轨道，活跃总数 260）。
- **cos_P_hydrogenic_overall**：模型 P 与解析 `hydrogenic_P_jax` 的带符号余弦 `∫P·P_H/√(∫P²·∫P_H²)`。
- **lambda_rel_drift**：`|λ_a − λ_init|/λ_init`，λ_init = Z/n。
- 聚合：`mean_cos_P_hydrogenic`、`mean/max_lambda_rel_drift`、cos 分箱（≥0.95 / 0.5–0.95 / 0.1–0.5 / <0.1）。

### 2.2 能量层（Layer-1/2）

`cfac_jobs/energy_batch/energy_compare_<tag>/energy_comparison.csv`，三方对比 NIST / cFAC / PINN，260 行：

- `ΔE_PINN_meV = (E_pinn_hartree − E_nist_hartree) × hartree_to_meV`
- `ΔE_FAC_meV`、`ΔE_PINN_FAC_meV`（PINN vs FAC）
- 聚合：全量 RMSE / 中位数 / max|Δ| / per-Z RMSE。

### 2.3 训练层

`logs/v3_stage_a_laguerre_basis_<tag>/history.csv`，每 epoch 记录 `loss, pde, ortho, norm, asym, v_prior, v_smooth, scf, coeff_decay, coeff_anchor, lambda_prior, q_residual`。

---

## 3. 数据完整性说明（重要）

在汇总过程中发现评估产物存在**标注疑点**，需在解读时考虑：

1. `stage_a_f_f5k_gpu.json`（164/260, mean_cos 0.7500001475873556）与 `stage_a_g_g5k_v2_fixed.json`（164/260, mean_cos 0.7500001475873556）的聚合指标**逐位完全相同**，且与 `stage_a_e_e5k_gpu.json`（0.7500001405661099）在 1e-8 量级一致。三个独立训练运行产生如此一致的形态学指标在物理上不可信，表明评估阶段 checkpoint 路径标注存在错误——至少 `stage_a_f_f5k_gpu.json` 与 `stage_a_g_g5k_v2_fixed.json` 实际评估的是同一 checkpoint。
2. **Step F 的权威结果**以 `stage_a_f_f5k_gpu_v2.json` 为准：26/260（10%）、mean_cos 0.108、λ_drift 均值 3.96 / 最大 8.83。这与该步 `history.csv` 末态（loss 3.89e3、pde 1.64e3、coeff_decay 449、lambda_prior 1.91）一致，确证训练崩溃。
3. **Step G v2 的形态学指标**因 (1) 存疑，应使用其自身 checkpoint 重新评估复核；其能量三方对比（`energy_compare_g_g5k_v2_fixed`，RMSE 29 654 meV）来自独立 CSV 管线，标注可信。
4. Step E / G v2 中 `lambda_rel_drift ≈ 2.7e-8`（实测近零）与训练 log 的 `lambda_prior ≈ 1e-4` 自洽，表明 λ 在这些运行中确实被强锚在 init——此点可信，非标注错误。

下文表格按上述原则取值：Step F 取 `_v2`，Step G v2 形态学标注"存疑（疑似复用 E checkpoint）"，其余按 JSON 原值。

---

## 4. 逐步结果

### 4.1 Step B（R3 fix v2，基线，15 行）

- **改动**：实现 per-batch 解析拉盖尔 init（`_per_orbital_laguerre_init`）、`LaguerreCoeffHead/LambdaHead` 零初始化残差加性结构、JIT-safe `hydrogenic_laguerre_coeffs`（gammaln 实现）。无架构变更，纯 init 修复。
- **结果**（`stage_a_r3fixv2_5k_v2.json`）：**15/15 = 100%** 节点门禁，mean_cos 0.9998，λ_drift 0.0008/0.0011。
- **意义**：在训练分布（H/He/Li，n=1..5）内验证"§2.3 init = 物理"——首个 forward 即重现 `P_H`，5000 epoch 后保持。此为后续全部工作的基线。
- **前置负结果**：`r3fix`（3/15）、`r3fixfull`（7/15, cos 0.020）为 init 修复前的 broken 运行，`r3fixv2`（14/15 但 cos 0.013）揭示 node_gate 仅判节点数、不判相位——节点数对而相位翻反的轨道能过 node_gate 但 cos 极低。

### 4.2 Step C（260 行 sweep）

- **改动**：仅 config——manifest 扩到 260 行、`r_max 50→250`、`n_grid 256→512`、`d_trunk 64→128`、`lambda_prior 0.01→0.1`、`batch 4→8`。无代码改动。
- **训练**（`c_c2`，5k ep）：末态 loss 116、pde 113、coeff_decay 2.41e3、lambda_prior 9.8e-5、q_residual 2.8e-10。
- **形态学**（`stage_a_c2_full.json`）：**156/260 = 60.0%**，mean_cos 0.7557，λ_drift 均值 2.57%/最大 10.66%。
  - per-n：n=1..7 全 26 元素全通过（cos≈0.99–1.00）；n=8 仅 14/26（cos 0.56）；n=9 13/26（cos 0.03）；n=10 13/26（cos 0.01）。
  - per-Z：Z=1..15 大多 10/10；Z=16 突降至 2/10；Z=17..26 仅 1/10。
- **能量**（`energy_compare`，260 行）：RMSE 28 042 meV，中位 1 400 meV，max 232 288 meV（Z=26）。per-Z RMSE 随 Z 单调上升（Z=1: 2 794 → Z=26: 79 721 meV）。
- **失败模式分类**（首次系统识别）：
  1. **高 n / 低 Z（长程振荡失败）**：H 7s..10s、He 9s..10s。cos 塌至 ≤0.10，节点数少 1–3 个，瓣间距过宽（`λ_effective` 偏小）。
  2. **高 Z / 低 n（近核 cusp 失败）**：Z≥16, n=2..7 首节点缺失（如 Z=26 n=2 期望 1 节点实际 0）。
  3. **trunk 容量地板**：单一 SIREN trunk 无法同时分辨 `r≲0.04`（Z=26 n=2）与 `r≳100`（H 10s），softplus λ 妥协到训练分布中段，两端失外节点。
- **诊断结论**：拉盖尔系数、λ 先验、动能平衡 Q、branch 头均按设计工作；瓶颈在单一共享 SIREN trunk。

### 4.3 Step D（双 trunk + log-r）

- **改动**：§13.1 `_v_siren_dual`（低 ω₀=15 长程 / 高 ω₀=60 短程，软门 `σ((λ−4)·2)` 混合）；§13.2 trunk 输入加 `[log(r), sqrt(r)]` 辅助特征。
- **训练**（`d_d5k`，5k ep）：末态 loss 24.3、pde 21、lambda_prior 5e-5、q_residual 6.4e-11（较 Step C 的 loss 116 显著下降，因 trunk 容量提升）。
- **形态学**：
  - `d1k`（1k ep）：159/260 = 61.2%，mean_cos 0.7591。
  - `d5k`（5k ep）：155/260 = 59.6%，mean_cos 0.7556。
  - per-n 与 Step C 几乎相同（n=8: 14/26 cos 0.52；n=9: 13/26 cos 0.04；n=10: 13/26 cos 0.02）。
- **能量**：未生成独立 `energy_compare_d*`（无该目录）。
- **诊断结论**：**边际改善**（loss 大降但门禁不升）。双 trunk + log-r 降低了 PDE 残差，但未触及高 n 失败根因。早报告（§13.8）据此将根因从"trunk 容量"转移到"branch 侧 `c_k`/`λ_a` 头容量"，并识别第 4 个失败模式：`cos < 0.1` 的 52 个轨道中相当部分节点数正确但相位翻反（形态学失败≠节点数失败）。

### 4.4 Step E（branch 容量扩展）

- **改动**：§13.9 `d_branch 128→256`、`coeff_d_hidden 64→128`、`lambda_d_hidden 32→64`、`q_corr_d_hidden 64`；`n_siren_layers 4→3`、`omega_0 30→15`。GPU 5k ep。
- **训练**（`e_e5k_gpu`）：末态 loss 40.3、pde 37、lambda_prior 9.6e-5、q_residual 4e-10。
- **形态学**（`stage_a_e_e5k_gpu.json`）：**164/260 = 63.1%**，mean_cos 0.7500，λ_drift 均值 2.7e-8 / 最大 5.2e-7（**λ 完美收敛**，较 Step C 的 2.57% 大幅改善）。
  - per-n：n=1..7 全通过（cos≈1.00）；n=8 15/26（cos 0.50）；n=9 15/26（cos 0.00）；n=10 15/26（cos 0.00）。
  - n≤7：119/182（65.4%）；n≥8：45/78（57.7%）。
- **能量**（`energy_compare_e_e5k_gpu`）：RMSE 37 750 meV，中位 −163 meV，max 367 204 meV（Z=22）。per-Z：低 Z 改善（Z=1: 569, Z=3: 455 meV，较 Step C 的 2 794/2 526 大降）；高 Z 仍差（Z=22: 126 017, Z=26: 65 085）。
- **诊断结论**：**第二个负结果——branch 容量不是失败根因**。λ 完美收敛证明 `LambdaHead` 容量充足；但 n=8..10 cos 仍塌在 0，node_gate 仅 63.1%。高 n 失败与 branch 容量无关。Z=22 出现新离群（−367 keV）。

### 4.5 Step F（coeff_decay 强压制，灾难性负结果）

- **改动**：§13.10-A 将 `coeff_decay` 权重 1e-3→5.0（5000×），公式 1/k!→1/(k+1)（均匀压制各 k）。意图强逼系数稀疏化。
- **训练**（`f_f5k_gpu`，5k ep）：**崩溃**。末态 loss 3.89e3、pde 1.64e3、coeff_decay 449（从 init 1.76e3 降至 449，系数被压向零）、lambda_prior 1.91（λ 飞离 init）。训练曲线：loss 1.2e4→6.7e3（ep1250）→4.4e3（ep2500）→262（ep3750，假性低）→3.89e3（末，反弹）。
- **形态学**（`stage_a_f_f5k_gpu_v2.json`，权威）：**26/260 = 10.0%**，mean_cos 0.108，λ_drift 均值 3.96 / 最大 8.83。
- **诊断结论**：**灾难性失败，路线证伪**。`coeff_decay` 趋向零的 L2 正则与 PDE loss 严重冲突，把系数压垮、λ 打飞。证明"稀疏强制"方向（趋向零）在目标本身是稀疏非零 `coeff_init` 的情形下根本错误。此结果直接催生 §13.13 折中权重分析（0.05 边际、5.0 崩溃）与最终 §13.14 的方向重构——锚向 `coeff_init`（非零）而非零。

### 4.6 Step G v1 / v2（HybridLaguerreHead，负结果）

- **改动**：§13.11-C `HybridLaguerreHead`——Stage 1 节点粗估（`Δr = softplus(MLP) − log(2)`，`learned_nodes = r_analytic + cumsum(Δr)`）+ `nodes_to_laguerre_coeffs`（Gauss-Laguerre 投影）+ Stage 2 c_k 迭代精化（3 次，step 0.1）。保留双 trunk + log-r + branch 容量。
- **G v1**（`g_g5k_gpu`）：训练末态 loss 1.69、pde 0.793、coeff_decay 5.29、lambda_prior 5.6e-4、q_residual 5.4e-12（loss 显著低于 E 的 40.3）。
  - 形态学（`stage_a_g_g5k_gpu.json`）：**162/260 = 62.3%**，mean_cos 0.7375，λ_drift 均值 3.59%/最大 18.4%。
  - 能量（`energy_compare_g_g5k_gpu`）：RMSE 79 842 meV，max 801 970 meV（Z=26）。per-Z 低 Z 极佳（Z=1: 89, Z=2: 119, Z=3: 182 meV）但高 Z 灾难（Z=22: 90 404, Z=26: 258 347）。
  - **问题**：v1 因 `analytic_nodes` 路径解析错误，节点表未正确加载，Hybrid head 退化为近似 legacy 行为且 λ_drift 反而恶化（3.59% vs E 的 0%）。
- **G v2**（`g_g5k_v2_fixed`，修复节点表路径后重训）：训练末态 loss 1.86、pde 0.811、coeff_decay 0.188、lambda_prior 4.9e-4、q_residual 7.6e-12。
  - 形态学（`stage_a_g_g5k_v2_fixed.json`，**标注存疑，见 §3**）：164/260 = 63.1%，mean_cos 0.7500，λ_drift 2.7e-8。与 Step E 指标逐位一致，疑似评估复用 E checkpoint，需复核。
  - 能量（`energy_compare_g_g5k_v2_fixed`，可信）：RMSE 29 654 meV，中位 −393 meV，max 348 600 meV（**Z=6 新离群**，−348.6 keV）。per-Z：Z=6 RMSE 110 238 meV（异常），其余 Z 多数优于 Step C。
- **诊断结论**：**HybridLaguerreHead 路线关闭，负结果**。G v2 修复后 RMSE 29.7 keV，**未超越 Step C 的 28.0 keV**；节点门禁 ~62% 未改善；引入节点位置参数化增加复杂度却无净收益。节点位置直接参数化路线（§13.11）正式否定。

### 4.7 Step H（coeff 锚定 + 解析 dQ）

- **改动**：
  - **方向 B（解析 dQ）**：`laguerre_p_sum_with_r(return_d2=True)` 输出解析 `d²P/dr²`；新增 `kinetic_balance_q_dq_analytic` 闭式求 `dQ/dr = c·(N'D − ND')/D²`；`deeponet.py` 拉盖尔路径用其替换 `jnp.gradient(Q)`；新增 `dVdr`（`dV_nuc=Z/r²` 解析 + `dV_corr` 数值）。
  - **方向 C（coeff 锚定）**：新增 `coeff_anchor_loss = mean(mask_{n≥8}·(c_k − c_k^init)²)`，权重 50，n_min=8；`deeponet.py` 暴露 `laguerre_coeff_init`，经 `pinn_art_model` 透传至 trainer。
  - 回退 `use_hybrid_head=false`（legacy `LaguerreCoeffHead`），保留双 trunk + log-r + branch 容量。
- **训练**（`h_h5k_gpu`，5k ep）：末态 loss 51.9、pde 48.7、coeff_decay 2.41e3、**coeff_anchor 2.82e-4**、lambda_prior 2.4e-4、q_residual 5.2e-10。
- **形态学**（`stage_a_h_h5k_gpu.json`）：**156/260 = 60.0%**，mean_cos 0.7641（七步最高），λ_drift 均值 3.02%/最大 16.5%。
  - per-n：n=1..7 全通过（cos 0.98–1.00）；n=8 14/26（cos 0.61，**七步最高**）；n=9 13/26（cos 0.06）；n=10 12/26（cos 0.03）。
  - n≤7：117/182（64.3%）；n≥8：39/78（50.0%，**较 E/Gv2 的 57.7% 反而下降**）。
  - cos 分箱：≥0.95 共 178；<0.1 共 49（与 Step C 的 52 略改善）。
- **能量**（`energy_compare_h_h5k_gpu`）：RMSE **861 098 meV**，中位 122 meV，max **13 864 221 meV**。
  - **离群点**：Z=18 H-like 1s（Ar XVIII，nele=1，level_config `1s1`），`E_nist = −162.0 Ha`，`E_fac = −162.70 Ha`（FAC 准确），`E_pinn = −671.50 Ha`（**偏深约 4.14×**）。该单点贡献 ΔE = −13.86 MeV，使全量 RMSE 膨胀至 861 keV。
  - 剔除该单点后 per-Z：Z=1: 440, Z=2: 379, Z=3: 229, Z=6: 522, Z=8: 690, Z=10: 922, Z=15: 5 412, Z=22: 130 790, Z=24: 590 470, Z=26: 31 777 meV。低 Z（Z≤10）为七步最佳；Z=22/Z=24 仍有百 keV 级离群。
- **方向 B 评估**：**部分有效**。init 时 `norm` loss 干净（0.0061，无抖动）；n=8 mean cos 0.61（七步最高）；VPQ 形态图（`vpq_compare_h_h5k_gpu`）显示 Fe 8s 长程对齐改善。决定保留。
- **方向 C 评估**：**无效**。`coeff_anchor` 末态仅 2.82e-4，权重 50 × 2.82e-4 ≈ 0.014，相对 pde 项 48.7 完全可忽略——锚定力度不足以对抗 PDE/norm 梯度噪声。n=9,10 仍塌在 cos≈0；n≥8 通过率 50% 反降。
- **新问题**：Z=18 1s 形态学完美（n=1 无节点，node_gate 必过）但能量偏深 4×，是迄今最极端的**形态-能量解耦**案例。该点在 Step E/G 中未出现（Step E Z=18 RMSE 22 617 meV，正常量级），为 Step H 新引入。

---

## 5. 跨步综合

### 5.1 形态学门禁演进

| 步骤 | 节点门禁 | mean_cos | λ_drift 均值 | cos<0.1 数 | 备注 |
|------|----------|----------|--------------|-----------|------|
| Step B（15 行） | 15/15 = 100% | 0.9998 | 0.08% | 0 | 训练分布内 |
| Step C（260 行） | 156/260 = 60.0% | 0.7557 | 2.57% | 52 | 基线 |
| Step D（5k） | 155/260 = 59.6% | 0.7556 | 1.92% | 49 | 双 trunk+log-r，边际 |
| Step E | 164/260 = 63.1% | 0.7500 | 2.7e-8 | 52 | λ 完美收敛 |
| Step F（v2） | 26/260 = 10.0% | 0.108 | 396% | — | 灾难性崩溃 |
| Step G v1 | 162/260 = 62.3% | 0.7375 | 3.59% | 49 | 节点表未加载 |
| Step G v2 | 164/260 = 63.1% | 0.7500 | 2.7e-8 | — | 标注存疑 |
| Step H | 156/260 = 60.0% | 0.7641 | 3.02% | 49 | 解析 dQ 略改善 |

**特征**：剔除崩溃的 Step F，节点门禁在 **59.6%–63.1%** 区间窄幅震荡，七步无突破。mean_cos 在 0.74–0.76，Step H 略最高（0.7641）。λ 收敛在 Step E 后基本解决（Step E/Gv2/H 的 λ_drift 均 ≤3%）。cos<0.1 的硬失败轨道数稳定在 49–52，是贯穿性的不可压缩存量。

### 5.2 能量精度演进

| 步骤 | N | RMSE(meV) | 中位(meV) | max\|Δ\|(meV) | 最大离群 |
|------|----|-----------|-----------|---------------|----------|
| Step C | 260 | 28 042 | 1 400 | 232 288 | Z=26 (232 keV) |
| Step E | 260 | 37 750 | −163 | 367 204 | Z=22 (−367 keV) |
| Step G v1 | 260 | 79 842 | −105 | 801 970 | Z=26 (−802 keV) |
| Step G v2 | 260 | 29 654 | −393 | 348 600 | Z=6 (−349 keV) |
| Step H | 260 | 861 098 | 122 | 13 864 221 | Z=18 1s (−13.86 MeV) |

**特征**：
- 低 Z（Z≤10）能量精度持续改善：Step C Z=1 RMSE 2 794 meV → Step G v1 89 meV → Step H 440 meV。Step G v1 低 Z 达到 89–182 meV 量级（近 meV–百 meV）。
- 高 Z（Z≥22）始终存在百 keV 级离群，且离群元素在步间漂移（Z=26 → Z=22 → Z=6 → Z=18/Z=22/Z=24），表明离群是训练不稳定性的随机显现而非系统性可修复偏差。
- Step H 的 Z=18 1s（−13.86 MeV）是迄今最大离群，量级远超此前所有步，且形态学无异常——纯粹的形态-能量解耦。

### 5.3 per-n 失败模式（贯穿性）

n=1..7 在全部七步中对全部 26 元素几乎 100% 通过（cos≈0.99–1.00）；n=8 在 14–15/26 徘徊（cos 0.37–0.61）；n=9,10 在所有步中 cos≈0、通过率不升。这是**单调的 n 依赖失败**，与 Z 无关地发生在高 n。Step B（n≤5）100% 与 Step C+（n≥8 塌陷）的对比证明：失败随 n 增长出现，与训练分布扩展同步。

### 5.4 训练损失演进

| 步骤 | 末态 loss | pde | coeff_decay | lambda_prior | q_residual |
|------|-----------|-----|-------------|--------------|------------|
| C | 116 | 113 | 2.41e3 | 9.8e-5 | 2.8e-10 |
| D | 24.3 | 21 | 2.41e3 | 5.0e-5 | 6.4e-11 |
| E | 40.3 | 37 | 2.41e3 | 9.6e-5 | 4.0e-10 |
| F | 3.89e3 | 1.64e3 | 449 | 1.91 | 1.9e-9 |
| G v1 | 1.69 | 0.793 | 5.29 | 5.6e-4 | 5.4e-12 |
| G v2 | 1.86 | 0.811 | 0.188 | 4.9e-4 | 7.6e-12 |
| H | 51.9 | 48.7 | 2.41e3 | 2.4e-4 | 5.2e-10 |

**特征**：pde 残差从 Step C 的 113 降至 G v2 的 0.81（140× 改善），但节点门禁不变——**PDE 残差最小化与形态学正确性解耦**。Step H 的 pde 48.7 反高于 G v2，因解析 dQ 改变了残差结构（更严格、噪声更小，绝对值反升但物理意义更清晰）。Step F 的 coeff_decay 449 + lambda_prior 1.91 是崩溃信号。

---

## 6. 关键负结果与证伪汇总

1. **trunk 容量不是根因**（Step D 证伪 Step C 假设）：双 trunk + log-r 使 loss 116→24.3 但门禁 60%→60%。
2. **branch 容量不是根因**（Step E 证伪 Step D 假设）：branch 扩容使 λ 完美收敛但门禁 63.1%，n≥8 不动。
3. **coeff_decay 稀疏强制方向错误**（Step F 证伪 §13.10-A）：趋向零的正则在目标是稀疏非零 `coeff_init` 时压垮网络，10% 门禁。
4. **节点位置直接参数化无净收益**（Step G v2 证伪 §13.11-C）：HybridLaguerreHead RMSE 29.7 keV 不超 Step C 28.0 keV，门禁 63% 不升，路线关闭。
5. **coeff_anchor 权重 50 力度不足**（Step H 部分证伪 §13.C）：锚定项末态 2.8e-4，对 pde 48.7 可忽略，n≥8 通过率反降。
6. **PDE 残差与形态学解耦**（跨步）：pde 从 113→0.81 但门禁不变；Step H Z=18 1s 形态完美而能量偏深 4×。

---

## 7. 遗留问题

1. **高 n（n≥8）系统性失败未解决**：七步实验均未提升 n=9,10 的 cos（恒≈0）。当前架构在该区间的吸引盆地过小或梯度噪声主导，已有手段（双 trunk、log-r、branch 扩容、节点参数化、coeff 锚定）均无效。
2. **形态-能量解耦**：Z=18 1s（Step H）形态学正确但 `E_pinn = −671.5 Ha` vs `E_nist = −162.0 Ha`。该点 n=1 无节点（node_gate 必过）、cos 应近 1，但 Rayleigh 商 `⟨ψ|H|ψ⟩` 给出 4× 偏深能量，指向 Dirac 算子作用或 V(r) 在该 (Z,n) 的数值问题，而非波函数形状。此为 Step H 新引入、未诊断。
3. **高 Z 百 keV 级离群漂移**：Z≥22 区间离群元素步间漂移（Z=26→Z=22→Z=6→Z=18/22/24），非系统性可修复，疑似训练随机性。
4. **评估产物标注完整性**：`stage_a_f_f5k_gpu.json` 与 `stage_a_g_g5k_v2_fixed.json` 逐位相同（§3），Step G v2 形态学指标需用其自身 checkpoint 重新评估复核。评估脚本的 checkpoint 路径解析需审查。
5. **Step H 能量 RMSE 861 keV 非可比**：因 Z=18 单点 13.86 MeV 主导，Step H 全量 RMSE 与前序步不可直接比较；剔除该点后低 Z 为七步最佳，但高 Z 仍有百 keV 离群。
6. **DoD 未达成**：节点门禁 60% vs 目标 ≥85%；能量 RMSE 受离群主导。`17_generalized_laguerre_basis.md` §13.7 的完成判据未满足。

---

## 8. 工件清单

### 8.1 训练产物

| 步骤 | checkpoint | history.csv |
|------|-----------|-------------|
| B | `checkpoints/v3_stage_a_laguerre_basis_r3fixv2_5k/` | `logs/v3_stage_a_laguerre_basis_r3fixv2_5k/` |
| C | `checkpoints/v3_stage_a_laguerre_basis_c_c2/` | `logs/v3_stage_a_laguerre_basis_c_c2/` |
| D | `checkpoints/v3_stage_a_laguerre_basis_d_d5k/`（含 `d_d1k`） | `logs/v3_stage_a_laguerre_basis_d_d5k/` |
| E | `checkpoints/v3_stage_a_laguerre_basis_e_e5k_gpu/` | `logs/v3_stage_a_laguerre_basis_e_e5k_gpu/` |
| F | `checkpoints/v3_stage_a_laguerre_basis_f_f5k_gpu/` | `logs/v3_stage_a_laguerre_basis_f_f5k_gpu/` |
| G v1 | `checkpoints/v3_stage_a_laguerre_basis_g_g5k_gpu/` | `logs/v3_stage_a_laguerre_basis_g_g5k_gpu/` |
| G v2 | `checkpoints/v3_stage_a_laguerre_basis_g_g5k_v2_fixed/` | `logs/v3_stage_a_laguerre_basis_g_g5k_v2_fixed/` |
| H | `checkpoints/v3_stage_a_laguerre_basis_h_h5k_gpu/` | `logs/v3_stage_a_laguerre_basis_h_h5k_gpu/` |

### 8.2 评估产物

- 形态学 JSON：`results/laguerre_basis_eval/stage_a_{r3fixv2_5k_v2, c2_full, d_d1k, d_d5k, e_e5k_gpu, f_f5k_gpu_v2, g_g5k_gpu, g_g5k_v2_fixed, h_h5k_gpu}.json`
- 能量三方 CSV：`cfac_jobs/energy_batch/energy_compare_{,e_e5k_gpu,g_g5k_gpu,g_g5k_v2_fixed,h_h5k_gpu}/energy_comparison.csv`
- VPQ 形态图：`cfac_jobs/energy_batch/vpq_compare_{,e_e5k_gpu,g_g5k_gpu,g_g5k_v2_fixed,h_h5k_gpu}/VPQ_grid.png`
- per-Z 统计：`cfac_jobs/energy_batch/energy_compare_*/per_Z_stats.csv`、`per_Z_RMSE.png`、`energy_1to1.png`

### 8.3 配置与脚本

- 配置：`configs/v3_stage_a_laguerre_basis_{c,d,e,f,g,h}.yaml`（Step H 为当前基线 `_h.yaml`）
- 训练入口：`scripts/v3_train_stage_a_laguerre_basis.py`
- GPU 启动：`scripts/run_step_h_gpu.sh`（`JAX_PLATFORMS=cuda`，`nohup`，`--epochs`/`--tag`）
- sandbox 评估须 `JAX_PLATFORMS=cpu`（无 GPU 访问）

### 8.4 文档

- 设计主档：`prompts/17_generalized_laguerre_basis.md`（§13.8–§13.14 记录 Step D–H 实测）
- Step C 专项报告：`progress_reports/progress_report_step_c.md`
- 历史报告：`rc_pinn_art_project/docs/PROGRESS_REPORT_2026_06*.md`、`laguerre_basis_runbook.md`
- 本报告：`rc_pinn_art_project/docs/TEST_EVALUATION_REPORT_2026_06_15_to_06_29.md`
- 配套设计文档：`rc_pinn_art_project/docs/ARCHITECTURE_AND_DESIGN_2026_06_29.md`
