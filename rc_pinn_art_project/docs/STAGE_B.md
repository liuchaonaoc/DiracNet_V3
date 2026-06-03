# Stage B — CI 径向校准（Z = 1..8）

> **纪律**：Stage B 冻结 Stage A 的 DeepONet（branch/trunk/V），仅训练 `slater_log_scale`；**不对 NIST 做梯度**，且本阶段 `ci.nist_inject: false`（类氢 manifest 无可靠实验能级）。

## 1. 前置条件

| 项目 | 路径 |
|------|------|
| Stage A ckpt | `checkpoints/v3_phase1_stage_a_z1_8_phase3/stage_a_last.msgpack` |
| Manifest | `data_cache/manifest_hydrogenic_z1_8.parquet` |
| Racah 缓存 | `data_cache/racah_cache_z1_8.npz` |

生成 Racah（若缺失）：

```bash
cd /home/chaos/workspace2/DiracNet_V3/rc_pinn_art_project
export PYTHONPATH=.

python scripts/v3_build_racah_cache.py \
  --manifest data_cache/manifest_hydrogenic_z1_8.parquet \
  --out data_cache/racah_cache_z1_8.npz \
  --config configs/v3_phase1_stage_b_z1_8.yaml
```

## 2. 训练

```bash
PYTHONPATH=. python scripts/v3_train_stage_b.py \
  --config configs/v3_phase1_stage_b_z1_8.yaml
```

输出：

```text
checkpoints/v3_phase1_stage_b_z1_8/stage_b_last.msgpack
logs/v3_phase1_stage_b_z1_8/history.csv
```

要点：

- `model.apply_lowdin: true`（Löwdin 正交化在 Stage B 启用）
- 可训练参数：仅 `slater_log_scale`（长度 = `len(ci.k_list)`）
- 损失：`L_slat`（R⁰≈1）+ `L_offdiag` + `L_e_csf`（本征能与轨道能一致）+ 弱 `L_pde` 锚定

## 3. Gate B 验证

```bash
PYTHONPATH=. python scripts/v3_gate_ci.py \
  --config configs/v3_phase1_stage_b_z1_8.yaml \
  --ckpt checkpoints/v3_phase1_stage_b_z1_8/stage_b_last.msgpack
```

| 子项 | 判据 |
|------|------|
| B.1 合成 2×2 / 3×3 H | 本征能误差 < 10 meV |
| B.2 `eigh` 梯度 | 有限、无 NaN |
| B.3 Manifest | 48 行 \|E_csf − E_orb\| < 50 meV |

报告：`logs/v3_phase1_stage_b_z1_8/gate_b_report.json`

## 4. 下一步（Stage C）

- `ci.nist_inject: true` + `manifest_nist` 全量推断
- `scripts/v3_infer.py` / `v3_evaluate.py`
