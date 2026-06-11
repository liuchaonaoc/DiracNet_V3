# Stage 2（p1z8）详细评估方案

> **Checkpoint**：`checkpoints/v3_stage_a_z1_26_n10_p1z8_full/stage_a_last.msgpack`  
> **训练范围**：课程式 2a（Z≤8 基态 40 行）→ 2b（Z≤8 全组态 2435 行）  
> **纪律**：NIST 不注入；报告必须**分层**（见 `prompts/00_overview.md` §2.4）

---

## 0. 评估前必读：为何不能只看全 manifest 报告

你已跑过的 `logs/v3_stage_a_z1_26_n10_p1z8_full_eval/EXCITATION_VS_NIST.md` 是对 **Z=1…26 全 manifest** 前向，而模型**只在 Z≤8 上训练**。因此：

| 现象 | 解读 |
|------|------|
| 全 manifest MAE ~85 eV、覆盖率 ~44% | **大量 Z>8 行外推失败**，拉低汇总，不能代表 Stage 2 目标 |
| 表中 H MAE ~107 meV、He ~86 meV | **训练区内低 Z 已接近 Round 1**，这才是 Stage 2 主指标 |
| Gate 270/20814 通过、cos 中位 0.31 | 同上：高 Z 行占多数且未训练 |

**本方案核心原则**：所有「是否进入 Stage 3」的判断，以 **in-distribution（Z≤8）** 子集为准；全 manifest 仅作**外推压力测试**（可选）。

---

## 1. 评估分层（Layer-0 → Layer-2）

```text
Layer-0  波函数 / PDE / 势场     cos, pde, V_net vs V_dfs, 训练收敛
Layer-1  轨道能自洽               E_orb, dE vs 参考
Layer-2  激发能 vs NIST           single_valence / multi_electron 双组
```

| 层 | 必做？ | 主要产物 | Stage 2 关注点 |
|----|--------|----------|----------------|
| L0 | **是** | Gate A + 训练 history + H 势诊断 | cos≥0.99？pde 是否下降？价区 V_net |
| L1 | 建议 | Gate 中 `dE_meV` 按 Z/l 分布 | 闭壳 vs 开壳 |
| L2 | **是** | `EXCITATION_VS_NIST.md`（**分 scope**） | Z≤8 内 MAE；H/He 是否保持 <50 meV |
| L2 外推 | 可选 | 全 Z manifest 同左 | 仅记录「未训练区」劣化幅度 |

---

## 2. 评估范围（Scope）定义

| Scope ID | 含义 | manifest 过滤 | 行数（约） | 用途 |
|----------|------|---------------|-----------|------|
| **S0** | 锚点元素 | Z∈{1,2} | ~225 | 与 p1lowz 对比，防回退 |
| **S1** | 训练区全量 | Z≤8 | ~2435 | **Stage 2 主验收** |
| **S2** | 训练区激发（非基态） | Z≤8, `is_ground==False` | ~2395 | 形状+能标难点 |
| **S3** | 外推区 | Z>8 | ~18379 | 不应作为 FAIL 依据 |
| **S4** | 全 manifest | 无 | 20814 | 与 Round 2 原始 run 对比 |

子集 parquet 由评估脚本自动生成（见 §5）。

---

## 3. 对照基线（必须并排）

| 标签 | Checkpoint | 说明 |
|------|------------|------|
| **B0** | `v3_phase1_stage_a_z1_8_phase3/stage_a_last.msgpack` | Round 1，Z≤8 ns1，类氢 |
| **B1** | `v3_stage_a_z1_26_n10/stage_a_last.msgpack` | Round 2 失败 run（全 Z 混训） |
| **B2** | `v3_stage_a_z1_26_n10_p1lowz/stage_a_last.msgpack` | P1-A，Z≤2 only |
| **B3** | `v3_stage_a_z1_26_n10_p1z8_ground/stage_a_last.msgpack` | Stage 2a |
| **T** | `v3_stage_a_z1_26_n10_p1z8_full/stage_a_last.msgpack` | **本次 Stage 2b 待评** |

对比维度（每个 scope）：
- H 1s→2s 激发能（eV）、价区 RMS|V_net+1/r|
- Gate cos 中位、通过率
- Layer-2b single_valence MAE（meV），按 Z 表

---

## 4. 验收门槛（Stage 2 → 是否进 Stage 3）

### 4.1 硬门槛（S1：Z≤8，全部不满足则**不扩 Z**）

| # | 指标 | 门槛 | 测量方式 |
|---|------|------|----------|
| G1 | H 1s→2s 激发能 | \|Δ\| **< 50 meV** vs NIST | `v3_diag_h_potential.py` |
| G2 | H 价区势 | r≈1 bohr \|V_net+1/r\| **< 0.05 Ha** | 同上 |
| G3 | He single_valence MAE | **< 500 meV**（理想 <200） | Layer-2b，Z=2 行 |
| G4 | Gate cos 中位（S1） | **≥ 0.95**（stretch 0.99） | Gate A，Z≤8 子集 |
| G5 | Gate 通过率（S1） | **> 30%**（stretch 50%） | 同上 |
| G6 | single_valence MAE（S1, Z≤8） | **< 5 eV**（stretch 2 eV） | 激发能报告 |
| G7 | 训练收敛 | history 末 100 epoch：pde **中位 < 0.1** 或明确下降 | `history.csv` |

### 4.2 软门槛（指导 Stage 3 课程设计）

| 指标 | 期望 |
|------|------|
| Li–O（Z=3–8）single MAE | 相对 Round 1 同 Z 不恶化超过 2× |
| s 轨道 cos（S1） | 不低于 p 轨道；若 s≪f/g → 优先 ortho/课程 |
| multi_electron（S1） | MAE < 20 eV 或相对误差中位 < 10%（DFS+CI 截断） |

### 4.3 外推区（S3）仅记录、不否决

Z>8 的 FAIL **预期**；记录「相对 B1 是否更差」即可。

---

## 5. 执行步骤（推荐顺序）

### 5.0 环境

```bash
cd /home/chaos/workspace2/DiracNet_V3/rc_pinn_art_project
export PYTHONPATH=.
export JAX_PLATFORMS=cuda   # 或 cpu
```

### 5.1 一键套件（推荐）

```bash
bash scripts/v3_eval_stage2_suite.sh
```

产物目录：`logs/v3_stage_a_z1_26_n10_p1z8_full_eval/stage2_suite/`

### 5.2 分步手动（需单独重跑某层时）

#### Step A — 训练收敛（Layer-0 预备）

套件会生成 `stage2_suite/training_summary.txt`（p1lowz / p1z8_ground / p1z8_full 首/中/末 epoch）。

手动查看：

```bash
awk -F, 'NR==1 || NR%500==0 || NR==NR' logs/v3_stage_a_z1_26_n10_p1z8_full/history.csv
```

关注：`pde`, `scf`, `ortho`, `loss` 末段是否平台/上升。

#### Step B — 势场锚点（Layer-0，H/He）

```bash
CKPT=checkpoints/v3_stage_a_z1_26_n10_p1z8_full/stage_a_last.msgpack
for z in p1lowz p1z8_full; do
  echo "=== $z ==="
  python scripts/v3_diag_h_potential.py --ckpt checkpoints/v3_stage_a_z1_26_n10_${z}/stage_a_last.msgpack
done
```

#### Step C — Gate A（按 scope）

```bash
CKPT=checkpoints/v3_stage_a_z1_26_n10_p1z8_full/stage_a_last.msgpack
CFG=configs/v3_stage_a_z1_26_n10.yaml
MAN=data_cache/manifest_nist_z1_26_n10.parquet

# S1: Z<=8（主验收）
python scripts/v3_gate_analytic.py --config $CFG --ckpt $CKPT \
  --manifest data_cache/manifest_z_le_8.parquet

# S0: 仅 H/He（需先建子集，见 suite 脚本）
python scripts/v3_gate_analytic.py --config $CFG --ckpt $CKPT \
  --manifest data_cache/manifest_z_le_2.parquet
```

Gate 输出默认在 `logs/v3_stage_a_z1_26_n10/`；suite 会复制到 `stage2_suite/gate_*`。

#### Step D — Layer-2b 激发能（按 scope）

```bash
OUT=logs/v3_stage_a_z1_26_n10_p1z8_full_eval/stage2_suite

python scripts/v3_eval_excitation_vs_nist.py \
  --config configs/v3_stage_a_z1_26_n10.yaml \
  --ckpt checkpoints/v3_stage_a_z1_26_n10_p1z8_full/stage_a_last.msgpack \
  --manifest data_cache/manifest_z_le_8.parquet \
  --out-dir $OUT/layer2b_z_le_8

python scripts/v3_eval_excitation_vs_nist.py \
  --config configs/v3_stage_a_z1_26_n10.yaml \
  --ckpt checkpoints/v3_stage_a_z1_26_n10_p1z8_full/stage_a_last.msgpack \
  --manifest data_cache/manifest_z_le_2.parquet \
  --out-dir $OUT/layer2b_z_le_2
```

全 manifest 你已有：`logs/..._p1z8_full_eval/`（= S4）。

#### Step E — 按 Z / 按 l 拆解（从 CSV 后处理）

```bash
python scripts/v3_summarize_excitation_by_Zl.py \
  --csv logs/v3_stage_a_z1_26_n10_p1z8_full_eval/stage2_suite/layer2b_z_le_8/excitation_vs_nist.csv \
  --out logs/v3_stage_a_z1_26_n10_p1z8_full_eval/stage2_suite/summary_z_l.md
```

（由 suite 脚本提供；见 `scripts/v3_summarize_excitation_by_Zl.py`。）

#### Step F — 与基线对比表（汇总）

```bash
python scripts/v3_compare_stage2_baselines.py \
  --target checkpoints/v3_stage_a_z1_26_n10_p1z8_full/stage_a_last.msgpack \
  --baselines p1lowz,p1z8_ground,round2 \
  --manifest data_cache/manifest_z_le_8.parquet \
  --out logs/v3_stage_a_z1_26_n10_p1z8_full_eval/stage2_suite/BASELINE_COMPARISON.md
```

---

## 6. 分析清单（交给我时用）

跑完 §5 后，在对话中说明「Stage 2 评估套件完成」，我将读取：

| 路径 | 用途 |
|------|------|
| `.../stage2_suite/BASELINE_COMPARISON.md` | 基线对比 |
| `.../stage2_suite/layer2b_z_le_8/EXCITATION_VS_NIST.md` | **主验收** |
| `.../stage2_suite/layer2b_z_le_2/` | S0 锚点 |
| `.../stage2_suite/gate_z_le_8/` | cos/pde 分布 |
| `.../stage2_suite/summary_z_l.md` | s/p/d 轨道拆解 |
| `logs/..._p1z8_full/history.csv` | 收敛 |
| `.../stage2_suite/diag_h.txt` | H 势 |

我将输出：
1. **Stage 2 判定**（PASS / 条件 PASS / FAIL）及依据（对照 §4）
2. **相对 p1lowz / Round 1 / Round 2** 的增益与回退点
3. **Stage 3 建议**（Z≤10 闭壳 → 全 Z≤10；或回退 epoch / 调 scf warmup）
4. 更新 `EXCITATION_VS_NIST.md` 的 Stage 2 专节

---

## 7. 决策树（评估完成后）

```text
S1 满足 G1–G3 且 G4≥0.95？
  ├─ 是 → 进入 Stage 3：--resume p1z8_full --z-max 10（或 12），先 ground-only 再 full
  ├─ 否，但 S0 满足、S1 仅 Z=3–8 差 → 加 Stage 2c：Z=3–8 闭壳专项 500 ep，勿扩 Z
  └─ 否，且 S0 回退（H>50meV）→ 检查 2b 是否冲掉 p1lowz：用 p1lowz resume 缩短 2b epoch
```

---

## 8. 已知风险与判读注意

1. **2b 末 epoch loss/pde/scf 偏高**（你当前 history 末行 pde~1.1, scf~4.8）：若 S1 仍达标，可能是高 Z 行在 full eval 中的外推噪声；以 **Z≤8 Gate** 为准。
2. **覆盖率 <100%**：`forward_unique_configs` 未覆盖的 manifest 行不计入 MAE；对比时同时看 `metrics.json` 的 `coverage`。
3. **multi_electron CI**：Stage A 未训 CI；multi 组误差含 DFS+截断 CI 天花板，不与 single 组用同一 meV 门槛。
4. **Gate dE 列**：相对氢参考的轨道能差，与 Layer-2b「激发能」口径不同，勿混读。

---

## 9. 时间预算（GPU）

| 步骤 | 约耗时 |
|------|--------|
| 子集 parquet 生成 | <1 min |
| Gate S0+S1 | 10–30 min |
| Layer-2b S0+S1 | 30–90 min |
| 全 manifest S4 | 已完成（~4 h） |
| 基线对比 + 汇总 | 5 min |

**最小必跑**：Step B + Gate S1 + Layer-2b S1（约 1–2 h GPU）。
