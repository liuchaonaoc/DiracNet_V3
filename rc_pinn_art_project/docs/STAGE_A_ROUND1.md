# Stage A 第一轮训练指南（Z = 1..8，H → O）

> **纪律**：Stage A **仅**优化 Dirac PDE + 正交 + 渐近 + 势场正则；**NIST 不进梯度**。

## 1. 数据范围

| 项目 | 值 |
|------|-----|
| 元素 | H, He, Li, Be, B, C, N, **O**（Z = 1..8） |
| 物理模型 | 类氢离子（1 个电子，核电荷 Z，`ion_charge = Z-1`） |
| 能级 | 主量子数 n = 1..6（每 Z 共 6 行） |
| 总行数 | 8 × 6 = **48** |
| Manifest | `data_cache/manifest_hydrogenic_z1_8.parquet` |

## 2. 一键准备数据

```bash
cd /home/chaos/workspace2/DiracNet_V3/rc_pinn_art_project
export PYTHONPATH=.

# 或一条命令（manifest + Racah）
bash scripts/v3_round1_prepare.sh

# 分步等价于：
# 类氢 manifest（Z=1..8）
python scripts/v3_prepare_hydrogenic.py \
  --z-min 1 --z-max 8 --n-levels 6 \
  --out data_cache/manifest_hydrogenic_z1_8.parquet

# Racah 角向缓存（Stage A 不训练 CI，但建议预生成供后续 Stage C）
python scripts/v3_build_racah_cache.py \
  --manifest data_cache/manifest_hydrogenic_z1_8.parquet \
  --out data_cache/racah_cache_z1_8.npz \
  --config configs/v3_phase1_stage_a_z1_8.yaml
```

检查：

```bash
python -c "
import pandas as pd
df = pd.read_parquet('data_cache/manifest_hydrogenic_z1_8.parquet')
print(df.groupby('Z')['element'].first())
print('rows', len(df))
"
```

## 3. Stage A 训练命令

### 3.1 冒烟（约 1 分钟，确认 GPU / 无报错）

```bash
PYTHONPATH=. python scripts/v3_train_stage_a.py --config configs/v3_smoke.yaml
```

### 3.2 第一轮正式训练（推荐）

```bash
PYTHONPATH=. python scripts/v3_train_stage_a.py \
  --config configs/v3_phase1_stage_a_z1_8.yaml
```

训练结束后 checkpoint：

```text
checkpoints/v3_phase1_stage_a_z1_8/stage_a_last.msgpack
logs/v3_phase1_stage_a_z1_8/history.csv
```

### 3.3 断点续训

```bash
PYTHONPATH=. python scripts/v3_train_stage_a.py \
  --config configs/v3_phase1_stage_a_z1_8.yaml \
  --resume checkpoints/v3_phase1_stage_a_z1_8/stage_a_last.msgpack
```

## 4. 训练后验收（Gate A）

```bash
PYTHONPATH=. python scripts/v3_gate_analytic.py \
  --config configs/v3_phase1_stage_a_z1_8.yaml \
  --manifest data_cache/manifest_hydrogenic_z1_8.parquet \
  --ckpt checkpoints/v3_phase1_stage_a_z1_8/stage_a_last.msgpack
```

**通过标准**（见 `configs/v3_phase1_stage_a_z1_8.yaml` → `stage_a.gate`）：

| 指标 | 阈值 |
|------|------|
| \|cos(P, P\_analytic)\| | ≥ 0.99 |
| L_PDE | ≤ 1e-2 |
| \|E_orb - E_H\| | ≤ 50 meV |

单样本诊断：

```bash
PYTHONPATH=. python scripts/v3_diagnose_physics_chain.py \
  --config configs/v3_phase1_stage_a_z1_8.yaml \
  --row 0
```

## 5. 配置要点

| 参数 | 值 | 说明 |
|------|-----|------|
| `grid.n_grid` | 256 | 比 smoke 更密 |
| `model.d_branch/d_trunk` | 128 | 正式容量 |
| `stage_a.n_epochs` | 500 | 48 行小数据集可多轮覆盖 |
| `training.steps_per_epoch` | 12 | 每 epoch 约 96 样本见（含重复采样） |
| `stage_a.batch_size` | 8 | |

## 6. 常见问题

- **GPU**：`python scripts/v3_check_gpu.py` 或 `python -c "import jax; print(jax.devices())"` 应显示 `CudaDevice`。
- **Loss 不降**：跑 `python scripts/v3_diagnose_loss.py`；检查 `E_dirac(H 1s)` ≈ −0.5 Ha、`dQ/dr` 非零、`norm` 非零。
- **快速验证**：先跑 30 epoch 的 `configs/v3_phase1_quick.yaml`，看 `history.csv` 中 `loss` / `pde` / `norm` 是否单调下降。
- **Gate 不过**：增加 `n_epochs` 至 1000，或调整 `omega_0`（15–25）/ `lr_trunk`（1e-4 ~ 5e-4）。

## 7. 关键设计要点

- **Ansatz**：`P_a = r^|κ| · exp(-Z r / n) · shape(t, kappa, n)`；shape 由 SIREN+末层 `Dense(zeros, ones)` 初始化为 1，允许学到任意节点结构。
- **Q（小分量）**：`Q = c (dP/dr + κ/r · P) / (2c² − V)` (正分母，符合标准 Dirac 关系)。
- **dQ/dr**：由 `jnp.gradient` 自动求导（而非硬编码 0）。
- **Stage A 不做 Löwdin**：`model.apply_lowdin=false` → `norm` / `ortho` loss 真正起监督作用。Löwdin 留给 Stage B / 推断时使用。
- **配比**：`v_prior=5.0`（强先验拉 V → −Z/r），`v_smooth=1e-6`（避免 −Z/r 二阶差分主导）。

## 8. Phase 2（放宽 Gate + 再训 1000 epoch）

见 **[STAGE_A_PHASE2.md](STAGE_A_PHASE2.md)** 与 `configs/v3_phase1_stage_a_z1_8_phase2.yaml`。  
1000 ep 结果报告：`logs/v3_phase1_stage_a_z1_8/TRAINING_REPORT_1000ep.md`。

## 9. 下一步（Stage B/C）

- Stage B：CI 径向校准（可选）
- Stage C：`v3_infer.py` + NIST 混合对角填充（`nist_mask` / Fall-back）
