# Stage A Phase 2：放宽 Gate + 再训 1000 Epoch

> 承接 Phase 1（1000 ep，见 `logs/v3_phase1_stage_a_z1_8/TRAINING_REPORT_1000ep.md`）。  
> Phase 2 **从 Phase 1 checkpoint 续训**，使用**放宽 Gate** 作为训练期验收标准。

## 1. Gate 分阶段标准

| 名称 | cos | pde | |E−E_H| (meV) | 用途 |
|------|-----|-----|----------------|------|
| **gate**（Phase 2 默认） | ≥ 0.95 | ≤ 1.0 | ≤ 50 | 训练期验收 / `v3_gate_analytic.py` |
| **gate_mid** | ≥ 0.98 | ≤ 0.5 | ≤ 50 | 中期自检 |
| **gate_strict** | ≥ 0.99 | ≤ 0.01 | ≤ 50 | 终态目标（Phase 1 原标准） |

Phase 1 实测（严格 Gate）：0/48 全过，但 **31/48** 满足 pde≤1.0，**34/48** 满足 E≤50 meV。

## 2. 配置与目录

| 项目 | Phase 1 | Phase 2 |
|------|---------|---------|
| 配置 | `configs/v3_phase1_stage_a_z1_8.yaml` | `configs/v3_phase1_stage_a_z1_8_phase2.yaml` |
| Checkpoint | `checkpoints/v3_phase1_stage_a_z1_8/` | `checkpoints/v3_phase1_stage_a_z1_8_phase2/` |
| 日志 | `logs/v3_phase1_stage_a_z1_8/` | `logs/v3_phase1_stage_a_z1_8_phase2/` |
| 学习率 | lr_trunk 3e-4 | **lr_trunk 1e-4**（fine-tune） |
| asym 权重 | 0.1 | **0.01** |

## 3. 训练命令

```bash
cd ~/workspace2/DiracNet_V3/rc_pinn_art_project
export PYTHONPATH=.

# 从 Phase 1 最终权重续训 1000 epoch（推荐）
python scripts/v3_train_stage_a.py \
  --config configs/v3_phase1_stage_a_z1_8_phase2.yaml \
  --resume checkpoints/v3_phase1_stage_a_z1_8/stage_a_last.msgpack
```

或一键脚本：

```bash
bash scripts/v3_round2_train.sh
```

预计墙钟：**~5–6 分钟**（含 JIT 重编译 ~2 min + 1000×12 step @ ~0.01s/step）。

## 4. 训练后验收

### 4.1 放宽 Gate（Phase 2 目标）

```bash
python scripts/v3_gate_analytic.py \
  --config configs/v3_phase1_stage_a_z1_8_phase2.yaml \
  --ckpt checkpoints/v3_phase1_stage_a_z1_8_phase2/stage_a_last.msgpack
```

**期望**：显著多于 Phase 1 的 0/48（基态 n=1 应接近全过）。

### 4.2 严格 Gate（对照，不改动 config 的 gate_strict）

```bash
python scripts/v3_gate_analytic.py \
  --config configs/v3_phase1_stage_a_z1_8_phase2.yaml \
  --ckpt checkpoints/v3_phase1_stage_a_z1_8_phase2/stage_a_last.msgpack \
  --gate-mode strict
```

若脚本尚未支持 `--gate-mode`，可临时改 yaml 中 `stage_a.gate` 为 `gate_strict` 数值后再跑。

### 4.3 中期 Gate（可选）

将 `stage_a.gate` 在 yaml 中替换为 `gate_mid` 块数值，或后续在脚本中加 `--gate-mode mid`。

## 5. 成功判据（Phase 2）

| 级别 | 条件 |
|------|------|
| 最低 | 放宽 Gate **≥ 24/48** 通过（半数） |
| 良好 | 放宽 Gate **≥ 40/48**；严格 Gate pde≤1.0 的样本 **≥ 40/48** |
| 优秀 | 严格 Gate **≥ 8/48** 基态 n=1 全过；H 全系 n=1..6 的 pde ≤ 0.5 |

## 6. 若 Phase 2 仍不足

1. 将 `lr_trunk` 提至 `2e-4` 再训 500 ep  
2. `pde` 权重 ×2 或 `asym` 降至 0  
3. 激发态：Laguerre 基展开或节点数辅助 loss  
4. 终局再跑 1000 ep 并仅收紧 `pde_threshold: 0.1`

## 7. 相关报告

- Phase 1 1000 ep：`logs/v3_phase1_stage_a_z1_8/TRAINING_REPORT_1000ep.md`
- Phase 1 500 ep：`logs/v3_phase1_stage_a_z1_8/TRAINING_REPORT_500ep.md`
- **Gate 失败原因 + Phase1/2 逐行 diff**：`logs/gate_analysis/GATE_ANALYSIS_ZH.md`（`python scripts/v3_analyze_gate_reports.py` 生成）
