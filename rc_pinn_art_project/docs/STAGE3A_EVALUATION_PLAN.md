# Stage 3a（方案 A：Z=3–4 全组态）评估方案

> **推荐 Checkpoint（2026-06-09）**：`checkpoints/v3_stage_a_z1_26_n10_p1z3_4_eprime_full/best_anchor.msgpack`  
> **训练链**：`p1lowz` → `p1z3_4_eprime_ground` (120 ep) → `p1z3_4_eprime_full` (110 ep)  
> **配置**：`configs/v3_stage_a_z1_26_n10.yaml` (scf=density, apply_lowdin=false)  
> **参数**：anchor 0.5, lr×0.1, H 阈值 400 meV, V 阈值 0.10 Ha, patience 6-10  
> **结果**：H 误差 **-3.4 meV**（历史最佳），Z=3-4 11.1%，l≥1 dE 59-113 meV  
> **遗留**：l=0 27 eV（DFS 单粒子模型硬瓶颈）

> **旧 Checkpoint（已弃）**：`checkpoints/v3_phase1_stage_a_z1_8_p1z3_4_full/stage_a_last.msgpack`  
> **训练链**：`p1lowz` → `p1z3_4_ground` → `p1z3_4_full`（1200 ep）  
> **问题**：H 锚点被破坏到 -177 meV（灾难性遗忘）

---

## 1. 评估原则

1. **双目标**：① Z=3–4 是否比 Stage 2（p1z8）改善；② **H/He 是否回退**（必须从 p1lowz 热启动守住锚点）。
2. **只看 in-distribution**：主验收 scope = **Z=3–4**（496 行）；Z=5–8 / 全 manifest 仅作外推参考。
3. **不要**用全 Z=1–26 报告判定本阶段成败。

---

## 2. Scope

| ID | manifest | 行数 | 用途 |
|----|----------|------|------|
| **A0** | `manifest_z_le_2.parquet` | ~225 | H/He **防回退** |
| **A1** | `manifest_z_3_4.parquet` | ~496 | **本阶段主验收** |
| **A2** | `manifest_z_le_4.parquet` | ~721 | 累计训练区（Z≤4） |

---

## 3. 一键执行

```bash
cd DiracNet_V3/rc_pinn_art_project
export PYTHONPATH=.
export JAX_PLATFORMS=cuda

bash scripts/v3_eval_stage3a_suite.sh
# 若 ckpt 在别处:
# bash scripts/v3_eval_stage3a_suite.sh checkpoints/你的路径/stage_a_last.msgpack
```

产物：`logs/stage3a_p1z3_4_full_eval/suite/`

---

## 4. 验收门槛

### 4.1 硬门槛（不过则不要扩 Z=5–8）

| # | 指标 | 门槛 | 看哪里 |
|---|------|------|--------|
| R1 | H 1s→2s | \|Δ\| **< 50 meV** | `diag_h.txt` |
| R2 | H 价区 V | r≈1 \|ΔV\| **< 0.05 Ha** | `diag_h.txt` |
| R3 | Z≤2 single MAE | **< 200 meV**（stretch 100） | `layer2b_z_le_2/metrics.json` |
| R4 | Z≤2 single pass@50meV | **> 40%** | 同上 |

### 4.2 本阶段目标（Z=3–4）

| # | 指标 | Stage 2 参考 (p1z8, 同 Z) | Stage 3a 目标 |
|---|------|---------------------------|---------------|
| T1 | Li single MAE | **1558 meV** | **< 800 meV** |
| T2 | Be single MAE | **4199 meV** | **< 2 eV** |
| T3 | Z=3–4 Gate cos 中位 | Li~0.59, Be~0.50 | **≥ 0.70** |
| T4 | Z=3–4 single pass@50meV | ~0% | **> 5%**（stretch 10%） |
| T5 | Z=3–4 single 中位 rel err | ~1% 量级 | **< 2%** 且 MAE 下降 |

### 4.3 对照

- `STAGE2_VS_STAGE3A.md`：与 p1z8 在 Z=3–4 上的 MAE/pass 对比。
- `BASELINE_COMPARISON.md`：氢 vs p1lowz / p1z8_full。

---

## 5. 手动最小集（时间紧时）

```bash
CKPT=checkpoints/v3_phase1_stage_a_z1_8_p1z3_4_full/stage_a_last.msgpack

python scripts/v3_diag_h_potential.py --ckpt $CKPT

python scripts/v3_eval_excitation_vs_nist.py \
  --ckpt $CKPT --manifest data_cache/manifest_z_le_2.parquet \
  --out-dir logs/stage3a_quick/z_le_2

python scripts/v3_eval_excitation_vs_nist.py \
  --ckpt $CKPT --manifest data_cache/manifest_z_3_4.parquet \
  --out-dir logs/stage3a_quick/z_3_4
```

---

## 6. 决策

```text
R1–R4 全过 且 T1/T2 至少一项明显改善？
  ├─ 是 → Stage 3b：--z-min 5 --z-max 6（或 Z<=6 全组态），resume 当前 ckpt
  ├─ R 失败 → 从 p1lowz 重训 Z=3–4，缩短 epoch / 降 lr
  └─ R 过但 T 失败 → 加长 Z=3–4 训练或仅 Z=3 再训一轮
```

---

## 7. 交 agent 分析

说「**Stage 3a suite 跑完了**」，并确认 `logs/stage3a_p1z3_4_full_eval/suite/` 存在。

---

## 8. Stage 3a' 重训（Stage 3a 遗忘后）

```bash
cd DiracNet_V3/rc_pinn_art_project && export PYTHONPATH=.
export JAX_PLATFORMS=cuda

# 两阶段：ground(300ep) → full(400ep)，每步 25% Z<=2 混合，H 早停
bash scripts/v3_p1_stage3a_prime.sh

# 仅 full（从 p1lowz 直接训）
bash scripts/v3_p1_stage3a_prime.sh full-only

# 单阶段手动
python scripts/v3_train_stage_a_prime.py --phase full --tag p1z3_4_prime \
  --resume checkpoints/v3_stage_a_z1_26_n10_p1lowz/stage_a_last.msgpack
```

产物：`checkpoints/v3_stage_a_z1_26_n10_p1z3_4_prime_{ground,full}/`  
优先评估 **`best_anchor.msgpack`**（H 误差最小时保存）。  
早停日志：`logs/.../anchor_guard.jsonl`
