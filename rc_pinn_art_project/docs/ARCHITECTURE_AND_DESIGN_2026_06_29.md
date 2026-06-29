# PINN-ART (DiracNet V3) 代码框架设计与详细设计文档

> 截止日期：2026-06-29
> 适用代码基线：`rc_pinn_art_project/`（Stage A Round 2，广义拉盖尔有限基架构，Step H 落地后状态）
> 设计依据：`prompts/00_overview.md` … `prompts/17_generalized_laguerre_basis.md`，以及 `pinn_art/` 源码
> 本文档描述**当前代码的实际结构**，而非设计意图的应然状态。代码与文档冲突时以代码为准。

---

## 1. 项目定位与范围

**PINN-ART**（Physics-Informed Neural Network for Atomic Radiative Transfer）= Dirac V3。在 V2（PyTorch、B-spline+KAN、单粒子 Dirac PDE）基础上，迁移到 JAX，采用 DeepONet+SIREN 算子骨架，并加入可微 CI 与辐射/碰撞可观测量推断。

端到端管线：

```
坐标映射 t(r)  →  神经 Dirac 求解器 (V, P, Q)  →  可微 CI（NIST 注入 / 理论回退）  →  辐射/碰撞可观测量
```

**当前实际范围**（Stage A Round 2）：

- 训练数据：`manifest_hydrogenic_z1_26_n10.parquet`，260 行 = 26 个元素（Z=1..26）× 10 个主量子数（n=1..10），全部为单电子类氢态（parent_config `1s1`，level_config `ns1`，nele=1）。
- 网格：`r_min=1e-3, r_max=250, n_grid=512, scheme=loglinear`。
- 范围之外（代码存在但未在当前轮训练）：完整 MRCI、双激发以上 CI、Breit/QED、Z>26 或 n>10、多 GPU。

**四条不可违反纪律**（`00_overview.md` §2）：物理主链优先于数据拟合、角向离线/径向在线可微、`eigh` 简并保护、评估分层报告（Layer-0 形态 / Layer-1 轨道能 / Layer-2 CI vs NIST / Layer-3 截面 / Layer-4 延迟）。

---

## 2. 技术栈与目录布局

**技术栈**：JAX + Flax (linen) + Optax；NumPy/SciPy 用于数据预处理与离线角向系数；pandas 用于 manifest（parquet）；orbax/pickle+flax `train_state` 用于 checkpoint。默认 `float32`，积分权重与 `eigh` 可升 `float64`。随机性由显式 `jax.random` key 传入。

**顶层布局**（`rc_pinn_art_project/`）：

```
configs/        各 Stage 与各实验步的 YAML 配置
data_cache/     manifest parquet、Racah 缓存、Laguerre 节点表
checkpoints/    每 tag 一个子目录，stage_a_last.msgpack
logs/           每 tag 一个子目录，train_<tag>_<stamp>.log + history.csv
results/        评估产物（laguerre_basis_eval/*.json 等）
scripts/        训练/评估/诊断/绘图脚本
tests/          pytest 单元与 smoke 测试（~40 个文件）
pinn_art/       核心包（见 §4）
docs/           本文档与历史报告
```

**包结构**（`pinn_art/`）：

| 子包 | 职责 |
|------|------|
| `constants.py` | 物理常数（`C_LIGHT`、单位换算 `hartree_to_meV` 等） |
| `coords/` | 混合坐标映射 `t(r)`、branch 特征构造 |
| `nets/` | `DeepONetDirac` 算子骨架、SIREN 层、拉盖尔基算子、势场/相位头 |
| `physics/` | Dirac 算子、氢样参考、DFS 势、Slater 修正、正交化器 |
| `losses/` | PDE/正交/归一/渐近/势场先验/SCF/系数正则等损失 |
| `models/` | `PinnArtModel` 顶层模型，编排 net + 物理量 + CI |
| `training/` | Stage A/B trainer、train_state、checkpoint、gate |
| `data/` | manifest 数据集、collate、配置枚举/解析 |
| `ci/` | Hamiltonian 装配、Racah 缓存/计算、Slater 径向积分、`eigh`、NIST 注入 |
| `observables/` | E1 跃迁、多极、碰撞截面 |
| `evaluation/` | Stage C 评估与指标 |
| `utils/` | 网格、配置、日志、数值工具 |

---

## 3. 张量与网格约定

**径向网格** `RadialGrid`（`utils/grid.py`）：

- JAX PyTree（`r, dr, jac` 为 children，`r_min, r_max, n_grid, scheme` 为 aux），可穿过 `jax.jit`。
- `scheme`：`linear` / `log` / `loglinear`（当前用 loglinear，近核 log、远核 linear 混合）。
- `integrate(f, axis)`：梯形权 `dr` 加权求和；`dr = composite_trapezoid_weights(r)`。
- 形状：`r: [N_g]`，`dr: [N_g]`，`jac: [N_g]`。

**张量约定**（贯穿全包）：

- `B` = batch size，`N_orb` = 轨道槽上限（`n_orb_max`，默认 16），`N_g` = 网格点数，`K_max+1` = 拉盖尔系数长度（默认 10）。
- 波函数：`P, Q, dPdr, dQdr : [B, N_orb, N_g]`。
- 势场：`V, dVdr : [B, N_g]`（全行共享，与轨道无关）。
- 拉盖尔系数：`coeffs, coeff_init : [B, N_orb, K_max+1]`；`lambdas, lambda_init : [B, N_orb]`；`deltaQ : [B, N_orb, N_g]`。
- `kappa : [B, N_orb]`（Dirac 角量子数，整数）；`orb_mask : [B, N_orb]`（bool，屏蔽无效槽）；`Z : [B]`（int）。
- `shell_table : [B, N_orb, 4]`（int），`[...,0]=n, [...,1]=l, [...,2:]=占用/其它`。
- 归一化约定：`P = r·R`，`∫ P² dr = 1`（与 `hydrogenic_P_jax` 一致）。
- `λ` 约定：拉盖尔 envelope `r^{|κ|}·exp(-λr)` 的衰减因子，氢样 init = `Z_eff/n`；`α = 2|κ|-1`。

**轨道槽复用**：同一 `N_orb` 槽在不同 manifest 行中可承载不同 (n, l, Z)（行 0=H 1s，行 1=H 2s，…）。因此解析 init 在 JIT 区域内按 per-batch (Z, n, l) 计算（`_per_orbital_laguerre_init`），而非模型构造时固定。

---

## 4. 数据流水线

### 4.1 Manifest 与数据集

- `data/dataset.py::ManifestDataset`：读取 parquet manifest，逐行产出原始样本。
- `data/collate.py::collate_batches`：将样本列表 collate 成 batch dict，并完成 host-side 预处理：
  - 构造 `shell_table`、`kappa`、`orb_mask`、`csf_mask`、`omega`（占据数）、`E_nist`、`nist_mask`。
  - **§13.11-C**：当 `use_hybrid_head=True` 时，按 (Z, n, l) 查 `data_cache/laguerre_nodes_z1_26_n1_10.parquet`，产出 `analytic_nodes : [B, N_orb, K_max]`（JIT 区域外完成，避免 Python 字典查找进入 trace）。
- `data/config_parser.py` / `config_enum.py`：配置枚举与解析（含 degenerate fallback、NIST 注入策略）。

### 4.2 Branch 特征

`coords/shell_features.py::build_branch_features(batch, n_orb_max)` 离散化组态（Z、占据、kappa、n、l 等）为 branch 输入向量；`branch_input_dim(n_orb_max)` 给出维度（`d_in_branch`，默认 84）。branch 特征是 DeepONet 的"离散算子输入"。

### 4.3 坐标映射

`coords/mixed_map.py::MixedRadialMap(c1, c2)`：`t = t(r)`、`dt_dr = dt/dr`。当前 `c1=c2=1.0, learn_map=false`，即近似线性映射，但接口保留可学映射能力。

---

## 5. 顶层模型 `PinnArtModel`

`models/pinn_art_model.py`。Flax Module，`@nn.compact` 单入口 `__call__(batch, grid, *, train, return_ci, E_grid)`。

### 5.1 前向编排

1. 由 `grid.r` 经 `MixedRadialMap` 得 `t, dt_dr`。
2. `build_branch_features(batch)` → `branch_feat`。
3. 解析 `kappa, orb_mask, Z, n_principal, l_orbital`（来自 `shell_table` 或由 kappa 推导：`l = |κ+1/2|-1/2`）。
4. 取 `analytic_nodes`（hybrid head 用，否则零占位）。
5. 可选 `z_eff_orb`（`use_zeff_warmstart` 时由 `slater_effective_charge` 算 Slater 屏蔽有效电荷）。
6. 构造并调用 `DeepONetDirac` → `raw`（含 `V, P, Q, dPdr, dQdr` 及拉盖尔头输出）。
7. 可选 `apply_lowdin`：`lowdin_orthonormalize(P, Q, grid, orb_mask, dPdr, dQdr)`（Stage A 默认 **关闭**，让 norm/ortho loss 直接作用于网络原始输出）。
8. `dirac_apply(P,Q,dPdr,dQdr,V,kappa,r)` → `LP, LQ`（Dirac 算子作用）。
9. `orbital_energy_from_dirac(P,Q,LP,LQ,grid)` → `E_orb : [B, N_orb]`（Rayleigh 商 `⟨ψ|H|ψ⟩/⟨ψ|ψ⟩`），按 `orb_mask` 置零无效槽。
10. 透传拉盖尔头输出至 trainer：`laguerre_coeffs / laguerre_coeff_init / laguerre_lambdas / laguerre_lambda_init / laguerre_deltaQ`。

### 5.2 CI 路径（`return_ci or ci_enabled`）

当前 Stage A 配置 `ci.enabled=false`，CI 路径不激活，但代码完整：

- 取 `csf_mask, csf_to_orb, C_ang`（角向 Racah 系数）。
- 创建可学参数 `slater_log_scale`（init = -1.0，`exp(-1.0)≈0.37`），用于缩放 Slater `R^k` 注入势。
- `compute_all_Rk_diagonal(P,Q,orb_mask,grid,k_list)` → `Rk`，乘 `exp(slater_log_scale)`。
- `assemble_hamiltonian(E_orb, Rk, C_ang, csf_mask, csf_to_orb)`（有角向）或 `assemble_hamiltonian_diagonal`（无角向）→ `H`。
- `nist_inject=True` 时 `fill_h_diagonal_hybrid(H, E_nist, nist_mask)`：`nist_mask=True` 槽硬注入 NIST 实验能级，`False` 槽保留理论对角元（理论回退，禁止零/常数占位）。
- `safe_eigh(H, eps_degen_ev)`（带简并保护，`eps_degen_meV` 默认 1e-3）→ `E_csf, V_csf`。
- `compute_e1_transitions(E_csf, V_csf, P, Q, grid, csf_mask)` → 跃迁。
- `E_grid` 给定时算碰撞截面 `collision_cross_section_ce`。

### 5.3 构造与初始化

`build_model_and_params(cfg, grid, key)`：从 cfg 读取全部模型字段（含拉盖尔/双 trunk/log-r/hybrid 全套开关），构造 `PinnArtModel`，用 `_dummy_batch`（Laguerre 开时给 slot 0 填 `1s` 占位）`model.init` 得 params。

---

## 6. 算子骨架 `DeepONetDirac`

`nets/deeponet.py`。Flax Module，字段即全套架构开关（见 §6.5）。

### 6.1 Branch 主干

```
branch_feat [B, d_in] --Dense(d_branch)--> relu --Dense(d_branch)--> b [B, d_branch]
```

`b` 既喂 trunk（与 `t` 拼接），也喂各拉盖尔头（`LaguerreCoeffHead/LambdaHead/QCorrHead/HybridHead` 都以 `b` 为输入）。

### 6.2 Trunk 与辅助输入

- Trunk 输入：`[t_broadcast, b_broadcast, aux_broadcast]`，形状 `[B, N_g, 1+d_branch+aux]`。
- **§13.2 log-r 辅助**（`use_log_r_input`）：`aux = [log_r_scale·log(r), log_r_scale·sqrt(r)]`，`[B, N_g, 2]`，缓解大动态范围 `r`。
- `t = t_grid[None,:,None]` 广播。

### 6.3 V 头与双 trunk

- `V = V_nuc + V_corr`，`V_nuc = -Z/r`（解析），`V_corr = tanh(V_raw)`（SIREN 输出，有界）。
- **§13.1 双 trunk**（`use_dual_trunk`）：`_v_siren_dual` 跑两个 SIREN——`low`（`ω₀=omega_0`，长程/低 λ）与 `high`（`ω₀=omega_0_high`，短程/高 λ），按 per-row 软门 `α = σ((λ_ref − lambda_split)·gate_temperature)` 混合，`λ_ref = Z/n`（slot 0 代表）。`use_dual=False` 时退化为单 SIREN `_siren_to_scalar`。
- **dV/dr**（§13.B 解析 dQ 需要）：
  - `dV_nuc/dr = +Z/r²`（解析）。
  - `dV_corr/dr = jnp.gradient(V_corr, r)`（V_corr 低频平滑，有限差分非噪声源）。
  - `dVdr = dV_nuc + dV_corr`。

### 6.4 拉盖尔路径（`use_laguerre_basis=True`，当前主路径）

1. `_per_orbital_laguerre_init(Z, n_principal, l_orbital, K_max)` → `coeff_init_per_orb [B,N_orb,K+1]`、`lam_init_per_orb [B,N_orb]`。无效槽（`n≤l`）置零，`λ=1.0` 占位。系数由 `hydrogenic_laguerre_coeffs`（gammaln 实现，JIT 友好）算出——氢样下仅 `k=n-l-1` 槽非零。
2. 逐槽 `a_` 调用头：
   - **legacy 头**（`use_hybrid_head=False`，Step H 当前路径）：`LaguerreCoeffHead(b, init_bias) → coeffs`，输出 = `delta(b) + coeff_init`（`delta` 为零初始化 MLP 残差）。
   - **hybrid 头**（`use_hybrid_head=True`，Step G）：`HybridLaguerreHead(b, init_bias, analytic_nodes[:,a_], λ, α, degree) → coeffs`（节点粗估 + c_k 迭代精化，见 §7.6）。
   - `LaguerreLambdaHead(b, lam_bias) → λ_a`（`softplus(delta + log(expm1(λ_init)))`，零初始化残差，init 严格等于 `λ_init`）。
   - `LaguerreQCorrHead(t_grid, b, kappa) → δQ`（2 层 SIREN，`tanh` 有界，零初始化）。
3. **系数屏蔽**：`coeff_mask = (k_idx ≤ n-l-1)`，`coeffs_all *= coeff_mask`（解析正交边界，硬保证 `k>n-l-1` 槽为零）。
4. 暴露 `laguerre_coeff_init = coeff_init_per_orb * coeff_mask`（供 §13.C anchor loss）。
5. 逐槽重建 `P, dP, d2P`：
   - `laguerre_p_sum_with_r(r, coeffs_a, λ_a, κ_a, α_a, perturb_corr=None, perturb_scale, return_d2=True)` → `P_a, dP_a, d2P_a`（见 §7.2，解析一阶/二阶导）。
   - `δQ' = jnp.gradient(deltaQ_a, r)`。
   - `kinetic_balance_q_dq_analytic(P_a, dP_a, d2P_a, V, dVdr, κ_a, r, δQ, δQ', perturb_scale_Q)` → `Q_a, dQ_a`（§13.B 解析 dQ，替换 `jnp.gradient(Q)`）。
6. `orb_mask` 屏蔽：`P_a *= m` 等。

### 6.5 legacy 骨架路径（`use_laguerre_basis=False`，备份/兼容）

- `use_hydrogenic_skeleton=True`：`P_a = P_H·(1 + perturb_eps·tanh(shape))`，`shape` 为零初始化 SIREN（init 时 `P_a=P_H`）；`dP` 解析 + `jnp.gradient(shape)`；`Q = _kinetic_balance_q(P,dP,V,κ,r)`；`dQ = jnp.gradient(Q)`。
- `use_hydrogenic_skeleton=False`：纯 envelope `r^|κ|·exp(-Zr/n)` × `(1+shape)`。
- 该路径保留以兼容旧 checkpoint，当前训练未用。

### 6.6 输出 dict

```
V, P, Q, dPdr, dQdr,
laguerre_coeffs, laguerre_coeff_init,
laguerre_lambdas, laguerre_lambda_init, laguerre_deltaQ
```

---

## 7. 拉盖尔基算子（`nets/laguerre_basis.py`）

本模块是 Stage A Round 2 的核心，定义全部拉盖尔基头与算子。

### 7.1 头模块

- `LaguerreCoeffHead(K_max, d_hidden)`：`delta = Dense(K+1, zero-init)(relu(relu(Dense(branch))))`，`out = delta + coeff_init`。零初始化保证 init 时 `out == coeff_init`（§2.3 init=物理）。
- `LaguerreLambdaHead(d_hidden)`：`λ = softplus(delta + log(expm1(λ_init)))`，init 严格 `= λ_init`。
- `LaguerreQCorrHead(d_hidden, omega_0)`：2 层 SIREN（`SirenDense`），末层零初始化，`tanh(out)` 有界。`δQ` 幅度由调用方 `perturb_scale_Q`（默认 0.05）封顶，防止破坏相对论极限行为。

### 7.2 `P(r)` 重建与解析导数

`laguerre_p_sum_with_r(r, coeffs, λ, κ, α, perturb_corr=None, perturb_scale=0.05, return_d2=False)`：

- envelope：`env = r^{|κ|}·exp(-λr)`（对数域计算并 clip 防溢出）。
- Laguerre 栈：`L_stack = _laguerre_generalized_stack(ρ=2λr, α, max_k=K_max)`，转置为 `[B,N_orb,K+1,N_g]`。
- `P_core = einsum("bok,bokg->bog", coeffs, L_stack)`；`P = env·P_core`。
- **解析一阶导**（无有限差分）：
  - `env' = env·(|κ|/r − λ)`。
  - `dL_k^α/dρ = −L_{k−1}^{α+1}`（恒等式），`dP_core/dr = (2λ)·Σ c_k·(−L_{k−1}^{α+1})`。
  - `dP/dr = env'·P_core + env·dP_core/dr`。
- **解析二阶导**（`return_d2=True`，§13.B）：
  - `env'' = env·[(|κ|/r − λ)² − |κ|/r²]`。
  - `P_core'' = (2λ)²·Σ c_k·L_{k−2}^{α+2}`（沿 k 轴移位 2）。
  - `d²P/dr² = env''·P_core + 2·env'·dP_core + env·dP_core''`。
- `perturb_corr` 路径：`P_core *= (1+perturb_scale·perturb)`；导数中忽略 `d(perturb)/dr`（≤0.05，亚百分点）。当前前向传 `perturb_corr=None`。

### 7.3 动能平衡 `Q(r)`

`kinetic_balance_q_with_corr(P, dPdr, V, κ, r, δQ, perturb_scale_Q, c)`：

```
Q_skel = c·(dP + (κ/r)·P) / (2c² − V)
Q      = Q_skel + perturb_scale_Q·δQ
```

denom clip 到 `[1e-6, 1e10]`。该函数返回 `dQ=0` 占位（dQ 由模型层解析或数值给出）。

`kinetic_balance_q_dq_analytic(P, dPdr, d2Pdr2, V, dVdr, κ, r, δQ, δQ', perturb_scale_Q, c)`（§13.B，Step H 启用）：

```
N = dP + (κ/r)·P,            D = 2c² − V
N' = d²P + (κ/r)·dP − (κ/r²)·P,   D' = −dV/dr
Q_skel  = c·N/D
Q_skel' = c·(N'·D − N·D') / D²
Q  = Q_skel + s·δQ
Q' = Q_skel' + s·δQ'
```

闭式求 `dQ/dr`，消除 `jnp.gradient(Q)` 在长程对含 `dP/dr` 的量再数值微分（≈数值 d²P）的噪声——高 n 轨道（`r~n²/Z`）在 loglinear 网格长程欠采样时的主要噪声源。

### 7.4 Q 残差约束

`q_residual(P, dPdr, V, κ, grid, δQ, perturb_scale_Q, ρ=None, c)`：`∫ ρ·(Q−Q_skel)² dr`，`ρ` 默认 `|P|²·r²`（电子密度代理）。强约束 `δQ` 贴近骨架，防变分坍塌（Gemini §II.3 警告）。

### 7.5 系数正则

- `coeff_decay_loss(coeffs, K_max)`：`mean(c_k²/(k+1))`（§13.10-A，1/(k+1) 权重，对各 k 均匀压制；替代旧 1/k! 对稀疏 active 槽 `k=n-l-1` 几乎无约束的缺陷）。
- `coeff_anchor_loss(coeffs, coeff_init, n_principal, orb_mask, n_min=8)`（§13.C，Step H 新增）：`mean(mask_{n≥n_min}·(c_k − c_k^init)²)`，**锚向 `coeff_init`（非零）**，仅对高 n（`n≥n_min`）生效，低 n 保持自由。直接对抗高 n 训练退化。
- `lambda_prior_loss(λ_a, λ_init)`：`mean((λ−λ_init)²/λ_init²)`，软锚 `λ` 于 `Z_eff/n`。

### 7.6 节点位置参数化（`HybridLaguerreHead`，Step G，当前已弃用）

`HybridLaguerreHead(K_max, d_hidden_node, d_hidden_ref, n_iter, step_size)`，§13.11-C 联合方案：

1. **Stage 1 节点粗估**：`Δr = softplus(MLP(b)) − log(2)`（零初始化 → init 时 `Δr=0`），`learned_nodes = r_analytic + cumsum(Δr)`，按 `degree=n-l-1` mask。
2. **节点→系数投影** `nodes_to_laguerre_coeffs`：`q(ρ)=Π(ρ−ρ_i)`，按 active 槽匹配 `coeff_init` 幅度后用 64 点 Gauss-Laguerre 正交投影到 `{L_k^α}`。
3. **Stage 2 迭代精化**：`c_k = coeff_init + (q_nodes − q_base_nodes)`，再 `n_iter` 次 `c_k += step·MLP_i(b, c_k, node_feat)`（零初始化 → init 不动）。
4. init 性质保持：`Δr=0, delta=0 → c_k = coeff_init → P = P_H`。

辅助：`load_analytic_nodes_table`（host-side 加载 parquet 节点表）、`_lookup_analytic_nodes`（按 (Z,n,l) 查表）。

> Step G 实测表明该头相对 legacy `LaguerreCoeffHead` 无净收益（RMSE 29.7 keV vs 28.0 keV，节点门禁 62% vs 60%），路线已关闭；Step H 回退 `use_hybrid_head=false`。代码保留但未启用。

---

## 8. 物理层（`physics/`）

### 8.1 Dirac 算子与轨道能（`dirac_operator.py`）

- `dirac_apply(P,Q,dPdr,dQdr,V,κ,r,c=C_LIGHT)`：径向 Dirac `H_D`，
  - `LP = V·P + c·(−dQ/dr + (κ/r)·Q)`
  - `LQ = (V − 2c²)·Q + c·(dP/dr + (κ/r)·P)`
- `orbital_energy_from_dirac(P,Q,LP,LQ,grid)`：Rayleigh 商 `∫(P·LP+Q·LQ)/∫(P²+Q²)`，逐轨道标量。

### 8.2 氢样参考（`hydrogenic.py`）

- `hydrogenic_P_jax(r, Z, n, l, max_k=6)`：JAX 原生批量 `P=r·R_{n,l}(r;Z)`，归一化 `∫P²dr=1`，`n≤l` 返回零。`gammaln` 实现，JIT 友好。
- `hydrogenic_dP_dr_jax`：解析 `dP/dr`（`d/dr[r·R]=R+r·dR/dr`，`dL_k^α/dρ=−L_{k−1}^{α+1}`）。
- `hydrogenic_laguerre_coeffs(Z,n,l,K_max=9)`：氢样拉盖尔展开系数，仅 `k=n-l-1` 非零；`c[n-l-1]=(2Z/n)^l·(Z/n)·sqrt((2Z/n)³·(n-l-1)!/(2n·(n+l)!))`。
- `hydrogenic_P_analytic`（numpy/scipy，单 (n,l,Z)，用于测试/评估参考）、`hydrogenic_energy(Z,n)=−Z²/(2n²)`、`cosine_signed(P_model,P_ref,grid)`。
- `_laguerre_generalized_stack(ρ,α,max_k)`：递推生成 `L_k^α(ρ)`，`k=0..max_k`，返回 `[max_k+1, ..., N_g]`。被拉盖尔基算子与氢样参考共用。

### 8.3 DFS 势与 Slater 修正（`dfs_potential.py`, `slater_correction.py`）

- `build_dfs_potential(P,Q,ω,Z,nele,grid, alpha_x, latter_tail, fermi_amaldi)`：Dirac-Fock-Slater 局域中心势（物理来源，非拟合实验能级）。
- `electron_density(P,Q,ω)`：`ρ=Σ ω_a(P²+Q²)`。
- `slater_effective_charge(...)` / `zeff_anchor_potential(...)`：Slater 屏蔽有效电荷与 `-Z_anchor/r` 锚势（路径 B）。
- `slater_correction_potential(P,Q,ω,Rk,slater_log_scale,ρ)`：`R^k→V` Slater 修正注入（路径 A，`stop_gradient` 防经 SCF loss 反传 P/Q）。
- 当前 Stage A Laguerre 配置 `dfs.enabled=false`、`scf=0`、`path_a_enabled=false`，即 bare-hydrogenic `_NO_DFS` 基线。

### 8.4 正交化器（`orthogonalizer.py`）

- `lowdin_orthonormalize(P,Q,grid,orb_mask,dPdr,dQdr)`：Löwdin 正交归一（`apply_lowdin=false` 时 Stage A 不用）。
- `gram_schmidt_ortho_pq`：Gram-Schmidt 硬约束（R2b）。

---

## 9. 损失层（`losses/`）

| 模块 | 损失 | 作用 |
|------|------|------|
| `pde_loss.py` | `dirac_pde_loss(P,Q,dPdr,dQdr,V,κ,r,grid,orb_mask,E_orb)` | Dirac PDE 残差 `‖H_Dψ − Eψ‖²`，核心项 |
| `norm_loss.py` | `normalization_loss(P,Q,grid,orb_mask)` | `∫(P²+Q²)dr − 1` |
| `ortho_loss.py` | `orthonormality_loss(P,Q,grid,orb_mask)` | 轨道间软正交约束 |
| `asymptotic_loss.py` | `asymptotic_tail_loss(P,Q,r,orb_mask,Z,n_principal)` | 长程渐近行为 |
| `potential_prior.py` | `potential_prior_loss(V,Z,r,V_anchor)`、`potential_smooth_loss(V,r)` | V 锚向物理势 + 平滑正则 |
| `scf_consistency.py` | `scf_consistency_loss(V,V_dfs_aug,weight)` | V 自洽一致（密度加权） |
| `coeff_loss.py` | `coeff_decay_loss`、`coeff_anchor_loss`、`lambda_prior_loss` | 拉盖尔系数/λ 正则（§7.5） |
| `q_corr_loss.py` | `q_residual` | δQ 残差（§7.4） |
| `ci_loss.py` | CI 残差（Stage B/C） | — |
| `loss_schedule.py` | `stage_a_weights(cfg)`、`stage_a_dfs_cfg(cfg)`、`stage_b_weights(cfg)` | 权重 schema 与 DFS 配置解析 |

**Stage A 总损失**（`stage_a_trainer.compute_stage_a_loss`）：

```
total = w_pde·L_pde + w_ortho·L_ortho + w_asym·L_asym + w_norm·L_norm
      + w_v_prior·L_vp + w_v_smooth·L_vs + w_scf·L_scf
      + w_coeff_decay·L_coeff + w_coeff_anchor·L_anchor
      + w_lambda_prior·L_lambda + w_q_residual·L_q
```

`stage_a_weights` 默认值与当前 Step H 配置见 §11.2。`_perturb_scale_Q`、`_coeff_anchor_n_min` 作为伪权重透传给 trainer（非损失项，是算子参数）。

---

## 10. CI 层（`ci/`，Stage B/C，当前未激活）

| 模块 | 职责 |
|------|------|
| `hamiltonian.py` | `assemble_hamiltonian(E_orb,Rk,C_ang,csf_mask,csf_to_orb)`、`assemble_hamiltonian_diagonal` |
| `slater_radial.py` | `compute_all_Rk_diagonal(P,Q,orb_mask,grid,k_list)`：Slater 径向积分 `R^k` 在线可微 |
| `racah_compute.py` / `racah_cache.py` | Racah/Wigner-Eckart 角向系数，预计算 `.npz`，`stop_gradient` |
| `eigen_solver.py` | `safe_eigh(H, eps_degen_ev)`：带简并保护的 `eigh`（Custom VJP / ε 破缺，见 `14_degenerate_gradient_safety.md`） |
| `nist_inject.py` | `fill_h_diagonal_hybrid(H, E_nist, nist_mask)`：对角元混合填充（NIST 硬注入 / 理论回退） |

纪律：禁止 per-row learnable bias、禁止 mask=False 处零/常数占位（必须理论回退）。

---

## 11. 训练流水线（`training/`）

### 11.1 Stage A trainer（`stage_a_trainer.py`）

- `compute_stage_a_loss(out, batch, grid, weights, dfs_cfg)`：聚合 §9 全部损失项，返回 `(total, metrics)`。
- `_train_step_jit`（`@partial(jax.jit, static_argnames=("weights_key","dfs_cfg"))`）：`value_and_grad` → `nan_to_num(grads)` → `apply_gradients`。权重 dict 拆为 hashable keys（static）+ array vals（traced），单次编译跨 epoch 复用。`dfs_cfg` 为 8 元组 static 参数（enabled, alpha_x, latter_tail, anchor_vprior, fermi_amaldi, scf_weight_mode, anchor_vprior_zeff, path_a_enabled）。`return_ci_flag = dfs_cfg[7]`（path_a）。
- `train_step`：外部入口，拆 dict 调 JIT 内核。

### 11.2 权重与配置

`loss_schedule.stage_a_weights` 默认：`pde=1, ortho=100, asym=0.01, norm=10, v_prior=0.1, v_smooth=1e-3, scf=0, coeff_decay=0, coeff_anchor=0, lambda_prior=0, q_residual=0`。

**Step H 配置**（`configs/v3_stage_a_laguerre_basis_h.yaml`，当前基线）：

| 组 | 字段 | 值 |
|----|------|----|
| grid | r_max / n_grid / scheme | 250 / 512 / loglinear |
| model | d_branch / d_trunk / n_siren_layers / omega_0 | 256 / 128 / 3 / 15 |
| model | use_laguerre_basis / K_max / learn_lambda | true / 9 / true |
| model | perturb_scale_P / perturb_scale_Q | 0.05 / 0.05 |
| model | coeff_d_hidden / lambda_d_hidden / q_corr_d_hidden | 128 / 64 / 64（§13.9 branch 容量） |
| model | use_hybrid_head | **false**（回退 legacy head） |
| model | use_log_r_input / log_r_scale | true / 1.0（§13.2） |
| model | use_dual_trunk / omega_0_high / lambda_split | true / 60.0 / 4.0（§13.1） |
| ci | enabled | false |
| stage_a | n_epochs / batch_size / steps_per_epoch | 5000 / 8 / 5 |
| stage_a.weights | pde / ortho / asym / norm | 1.0 / 10.0 / 0.1 / 10.0 |
| stage_a.weights | v_prior / v_smooth | 5.0 / 1e-6 |
| stage_a.weights | coeff_decay | 1e-3 |
| stage_a.weights | **coeff_anchor / coeff_anchor_n_min** | **50.0 / 8**（§13.C，新增） |
| stage_a.weights | lambda_prior / q_residual | 1e-1 / 1.0 |
| optimizer | lr_trunk / lr_branch / weight_decay / grad_clip | 3e-4 / 1e-4 / 1e-4 / 1.0 |

### 11.3 优化器与 train_state

- `train_state.py::create_train_state`：Optax，trunk/branch 差异学习率（`lr_trunk`/`lr_branch`），weight_decay，grad_clip。
- `checkpoint.py`：`save_checkpoint`/`load_params`（msgpack），每 tag 子目录 `checkpoints/v3_stage_a_laguerre_basis_<tag>/stage_a_last.msgpack`。

### 11.4 门禁（`gate_a.py`）

- `check_gate_a(out, batch, grid, *, cos_threshold, pde_threshold, e_orb_meV_threshold, n, l, level_config)`：
  - 逐活跃轨道算 `cos_signed(P_model, P_H)` 与 `E_orb` vs `hydrogenic_energy` 的 meV 误差。
  - `cos_min ≥ cos_threshold` 且 `pde ≤ pde_threshold` 且 `e_orb_mae ≤ e_orb_meV_threshold` → PASS。
  - 当前配置阈值：`cos_threshold=0.5, pde_threshold=1e-1, e_orb_meV_threshold=500`。
- `gate_b.py`：Stage B 门禁。

### 11.5 训练脚本

- `scripts/v3_train_stage_a_laguerre_basis.py`：Stage A Round 2 trainer 入口，`ManifestDataset` + `train_step` + `build_model_and_params`；history.csv 记录 `loss/pde/ortho/norm/asym/v_*/scf/coeff_decay/coeff_anchor/lambda_prior/q_residual`。
- `scripts/run_step_h_gpu.sh`：`JAX_PLATFORMS=cuda`，`nohup` 启动训练，`--epochs`/`--tag` 参数，日志至 `logs/v3_stage_a_laguerre_basis_h/train_<tag>_<stamp>.log`。

---

## 12. 评估与可观测量

### 12.1 评估（`evaluation/` + scripts）

- 形态学评估：`results/laguerre_basis_eval/stage_a_<tag>.json`，逐行记录 `node_gate`、`cos_P_hydrogenic_overall`、`nodes_observed/expected`、`lambda_rel_drift`、`E_orb` 等。
- cFAC 外部对比：`cfac_jobs/energy_batch/vpq_compare_<tag>/`（P/Q/V vs cFAC 网格图）、`energy_compare_<tag>/energy_comparison.csv`（三方能量：NIST / FAC / PINN，列含 `Z, ion_charge, nele, parent_config, level_config, J, term, E_nist_hartree, E_fac_hartree, E_pinn_hartree, ΔE_PINN_meV, ΔE_FAC_meV, ΔE_PINN_FAC_meV`）。
- 评估脚本在 sandbox 内运行时需 `JAX_PLATFORMS=cpu`（无 GPU）。

### 12.2 可观测量（`observables/`，Stage C）

- `transition_rates.py::compute_e1_transitions(E_csf, V_csf, P, Q, grid, csf_mask)`：E1 跃迁 `A_ki`/`gf`。
- `collision.py::collision_cross_section_ce(E_csf, E_grid)`：碰撞激发截面 CE。
- `multipole.py`：多极矩阵元。

---

## 13. 关键不变量与设计约束

### 13.1 §2.3 init = 物理

所有可学头采用零初始化残差 + 解析 init 加性结构：

- `LaguerreCoeffHead`：`delta=0 → out=coeff_init`。
- `LaguerreLambdaHead`：`delta=0 → λ=λ_init=Z/n`。
- `LaguerreQCorrHead`：末层零初始化 → `δQ=0 → Q=Q_skel`。
- `HybridLaguerreHead`：`Δr=0, delta=0 → c_k=coeff_init`。
- `coeff_anchor_loss` 在 init 时为 0（`c_k == coeff_init` 严格）。

→ 首个 forward pass 即精确重现氢样 `P_H`（`|max(P−P_H)|≈1e-7`）。

### 13.2 节点数硬保证

`coeff_mask = (k_idx ≤ n−l−1)` 强制 `k>n−l−1` 系数为零 → 多项式阶 `= n−l−1` → 内部节点数 `= n−l−1`（解析硬约束，不靠 loss 软逼）。

### 13.3 数值稳定性

- envelope 对数域计算 + clip（`-60..30`）。
- `denom = clip(2c²−V, 1e-6, 1e10)`、`r_safe = clip(r, 1e-8/1e-12)`。
- `grads = nan_to_num(grads)`。
- `eigh` 简并保护 `eps_degen_meV`。

### 13.4 单电子训练流形的关键推论

训练 manifest 为纯单电子类氢态 → 解析 `P_H` 即精确解，且已装进 `coeff_init`。因此高 n（n≥8）节点门禁失败的本质是**训练诱导退化**（完美 init 被梯度噪声打偏），而非"学不到"。这是 Step H 引入 `coeff_anchor`（锚向 init）与解析 dQ（消除长程数值噪声）的物理依据。

---

## 14. 文档与代码的对应关系

| 设计文档 | 代码落点 |
|----------|----------|
| `02_coordinate_mapping.md` | `coords/mixed_map.py` |
| `03_neural_dirac_solver.md` | `nets/deeponet.py`, `nets/siren.py` |
| `04_differentiable_ci.md`, `14_degenerate_gradient_safety.md` | `ci/*` |
| `05_observables_inference.md` | `observables/*` |
| `06_physics_losses.md` | `losses/*` |
| `07_training_pipeline.md` | `training/*`, `scripts/v3_train_stage_a*.py` |
| `08_evaluation.md` | `evaluation/*`, `training/gate_a.py` |
| `09_data_pipeline.md` | `data/*` |
| `16_stage_a_selfconsistent_dfs.md` | `physics/dfs_potential.py`, `losses/scf_consistency.py` |
| `17_generalized_laguerre_basis.md` §2–§5 | `nets/laguerre_basis.py`, `nets/deeponet.py` 拉盖尔路径 |
| `17` §13.1 / §13.2 | `use_dual_trunk` / `use_log_r_input` |
| `17` §13.9 | branch 容量字段 `coeff_d_hidden` 等 |
| `17` §13.11-C | `HybridLaguerreHead`, `nodes_to_laguerre_coeffs` |
| `17` §13.13 / §13.10-A | `coeff_decay_loss`（1/(k+1) 权重） |
| `17` §13.14 (Step H) | `coeff_anchor_loss`, `kinetic_balance_q_dq_analytic`, `laguerre_p_sum_with_r(return_d2=True)`, `laguerre_coeff_init` 透传 |

文档冲突裁决（`00_overview.md` §7）：`14 > 04`、`07 > 06`、`15 > 10`。

---

## 15. 当前状态小结

- **架构**：广义拉盖尔有限基（legacy `LaguerreCoeffHead`）+ 可学 λ + 动能平衡 Q + 有界 δQ + 双 trunk + log-r + branch 容量扩展。
- **Step H 新增**：解析 d²P 与解析 dQ（§13.B）、coeff_anchor 锚定损失（§13.C）。
- **训练**：5000 epoch，260 行单电子 manifest，bare-hydrogenic 基线（DFS/SCF/CI 关闭）。
- **未激活**：CI、Slater/DFS 自洽、可观测量推断（Stage B/C 代码就绪，待 Stage A 形态门禁达标后启用）。
- **设计哲学变更**：§13.C 推翻了 `17` §5.3"禁止 `‖c−c_H‖²`"的原始设计——五轮实验证伪了"系数应纯由能量梯度驱动"对单电子数据的适用性。
