# Stage C — 推断 + NIST 混合对角填充

> **纪律**：不训练网络；加载 Stage B checkpoint；`ci.nist_inject: true` 时对 `nist_mask=True` 的行用 NIST 绝对能标替换 $H_{ii}$，否则回退到理论 $E_\mathrm{orb}$。

## 1. 前置条件

| 产物 | 路径 |
|------|------|
| Stage B 权重 | `checkpoints/v3_phase1_stage_b_z1_8/stage_b_last.msgpack` |
| NIST manifest | `data_cache/manifest_nist_z1_8.parquet` |
| Racah 缓存 | `data_cache/racah_cache_z1_8.npz` |
| NIST 原始 CSV | `DiracNet_V1/rc_diracnet_project/data_raw/nist/` |

## 2. 流程

```bash
cd /home/chaos/workspace2/DiracNet_V3/rc_pinn_art_project
export PYTHONPATH=.

# 1) 生成带 NIST 的 manifest（level_meV=激发能，level_abs_meV=注入用绝对能）
python scripts/v3_prepare_nist_manifest.py

# 2) 单样本 JIT 冒烟
python scripts/v3_infer.py --config configs/v3_phase1_stage_c_z1_8.yaml --row 7

# 3) 全量 Layer-2 评估
python scripts/v3_evaluate_stage_c.py --config configs/v3_phase1_stage_c_z1_8.yaml
```

## 3. 输出

| 文件 | 说明 |
|------|------|
| `logs/v3_phase1_stage_c_z1_8/metrics.json` | Layer-2 指标（meV） |
| `logs/v3_phase1_stage_c_z1_8/STAGE_C_EVALUATION_REPORT.md` | 中文摘要 |
| `logs/v3_phase1_stage_c_z1_8/stage_c_levels.csv` | 逐行能级 |

## 4. 指标解读

- **mae_exc_csf_vs_nist_meV**：注入后 CI 激发能 vs NIST（单 CSF 时应 ≈ 0）
- **mae_exc_orb_vs_nist_meV**：未注入的 Dirac `E_orb` 激发能误差（反映 Stage A 波函数质量）
- **mae_abs_csf_vs_nist_inject_meV**：注入一致性检查

门禁（Phase-1）：全 manifest 激发能 MAE < **50 meV**（`08_evaluation.md`）。

## 5. 与 Stage B 的区别

| | Stage B | Stage C |
|---|---------|---------|
| 训练 | 训 `slater_log_scale` | 无训练 |
| NIST | `nist_inject: false` | `nist_inject: true` |
| Manifest | 类氢理论 | NIST + 绝对能标 |
| 目标 | CI 管线自洽 | 实验能级对齐 |

---

## 6. Phase 1 完成状态（2026-06）

| 项目 | 状态 |
|------|------|
| NIST manifest（48 行，Z=1…8） | ✅ `manifest_nist_z1_8.parquet`，48/48 匹配 NIST |
| `nist_inject` + `fill_h_diagonal_hybrid` + `safe_eigh` | ✅ 端到端跑通 |
| Layer-2 门禁（50 meV） | ✅ **PASS** |
| `diag_source=1` 注入一致性 | ✅ \|E_csf − E_nist_abs\| MAE ≈ 0.001 meV |
| 单 CSF / 无非对角耦合 | ⚠️ 已知局限（见下） |

**Phase 1 已验证什么**

- Stage B 权重 + Löwdin + Slater + Racah + NIST 混合填充 + JIT 推断链路正确。
- `E_csf_exc_meV` 在单 CSF 下 ≈ NIST 激发能（由构造决定，用于检查管线而非模型精度）。

**Phase 1 未解决什么**

- **`E_orb_exc` vs NIST MAE ≈ 1.56 eV**：反映 Stage A Dirac 轨道能，与 `logs/.../ENERGY_VS_NIST.md` 一致；注入不能替代波函数改进。
- 多 CSF、Racah off-diagonal、跃迁/截面（Layer-3）、延迟（Layer-4）均未覆盖。
- `v3_compare_energy_nist.py` 仍默认 Stage A 配置，尚未统一为 Stage C 双轨报告（`E_orb` / `E_csf`）。

---

## 7. 阶段进度

### Phase 2a — 评估与报告 ✅（2026-06）

| 任务 | 状态 |
|------|------|
| `metrics.json` 拆分 `layer2_orb` / `layer2_inject` | ✅ `logs/v3_phase1_stage_c_z1_8/metrics.json` |
| `v3_compare_energy_nist.py --return-ci` | ✅ `logs/.../ENERGY_VS_NIST.md` |
| Fall-back 冒烟 | ✅ `tests/test_stage_c_fallback.py` + `v3_gate_stage_c.py` |
| Gate C | ✅ `gate_c_report.json`（inject + fallback PASS；orb 仅监控） |

**复现（Phase 2a）**

```bash
export PYTHONPATH=.
python scripts/v3_evaluate_stage_c.py --config configs/v3_phase1_stage_c_z1_8.yaml
python scripts/v3_gate_stage_c.py --config configs/v3_phase1_stage_c_z1_8.yaml
python scripts/v3_compare_energy_nist.py \
  --config configs/v3_phase1_stage_c_z1_8.yaml \
  --ckpt checkpoints/v3_phase1_stage_b_z1_8/stage_b_last.msgpack \
  --return-ci --out-dir logs/v3_phase1_stage_c_z1_8
pytest tests/test_stage_c_fallback.py -q
```

**Phase 2a 指标说明**

| 层级 | 典型结果 | 门禁 |
|------|----------|------|
| `layer2_orb` | E_orb 激发能 MAE ≈ **1563 meV** vs NIST | 监控（阈值 500 meV，**不阻塞** Gate C） |
| `layer2_inject` | E_csf 注入自检 ≈ **0.001 meV** | **PASS**（< 1 meV） |

### Phase 2b — Stage A 能量优化 ⏭️ 已跳过

按项目决定暂不实施；Fall-back 精度依赖后续 Stage A 或推断侧标定。

### Phase 2c — 多 CSF 与真实 CI ✅（2026-06，2p 精细结构冒烟）

| 任务 | 状态 |
|------|------|
| `v3_prepare_nist_multicsf.py` | ✅ `manifest_nist_multicsf_z1_8.parquet`（8 谱组 × 2 CSF） |
| `v3_prepare_nist_subset.py` | ✅ Z=1..26 可扩展（调用 `build_nist_manifest`） |
| `collate_spectrum_group` + `csf_to_orb` | ✅ |
| `v3_evaluate_multicsf_stage_c.py` | ✅ `logs/v3_phase1_stage_c_multicsf_z1_8/` |

**复现（Phase 2c）**

```bash
export PYTHONPATH=.
python scripts/v3_prepare_nist_multicsf.py
python scripts/v3_build_racah_cache.py \
  --manifest data_cache/manifest_nist_multicsf_z1_8.parquet \
  --out data_cache/racah_cache_multicsf_z1_8.npz \
  --config configs/v3_phase1_stage_c_multicsf_z1_8.yaml
python scripts/v3_evaluate_multicsf_stage_c.py \
  --config configs/v3_phase1_stage_c_multicsf_z1_8.yaml
```

**Phase 2c 结果（2p 双重态）**

| 项目 | 结果 |
|------|------|
| 谱组 / 非对角耦合 | **8/8** 组 `|H_off| > 0` |
| 管线 VERDICT | **PASS** |
| 精细结构裂距 MAE | 数值仍大（2p 单轨道 + 类氢 E_orb 标度限制）；后续需按 `(Z, parent)` 分 Racah 缓存与 p 轨道训练 |

配置：`configs/v3_phase1_stage_c_multicsf_z1_8.yaml`

### Phase 3 — 可观测量与性能（Sprint 4–6）

| # | 任务 | 验收 |
|---|------|------|
| 1 | Layer-3：`compute_e1_transitions` 全量评估；`v3_compare_fac.py`（用户 FAC CSV） | `median_rel_err(A_ki)` 报告 |
| 2 | Layer-4：`v3_profile_latency.py`，Fe XVII 规模配置 | `p50_ms` 记录，目标 <10 ms |
| 3 | LOO：`configs/v3_loo_z8.yaml` 等 holdout | leave-one-Z / leave-one-n 表格 |
| 4 | 连续谱双头（可选 Sprint 6） | `phase_amplitude.py` + CE 截面 vs FAC |

### 建议执行顺序（路线图）

```text
[已完成] Stage C Phase 1 — 单 CSF + NIST 注入冒烟 (PASS)
[已完成] Phase 2a — 评估/report 拆分 + Gate C
[跳过]   Phase 2b — Stage A E_orb 优化
[已完成] Phase 2c — 2p 多 CSF + off-diagonal CI 冒烟 (PASS)
    ↓
Phase 3  — Layer-3/4 + LOO + FAC + 更大 NIST manifest (Z≤26)
```

### 不在近期范围

- Stage C 上对 NIST 做梯度训练（违反 `00_overview.md` 纪律）。
- 用 `E_csf_exc≈0` 作为模型成功的主要判据（单 CSF + 注入时恒成立）。
- 全元素 Z>26 manifest，直至 Racah 缓存与 Phase 2c 冒烟稳定。
