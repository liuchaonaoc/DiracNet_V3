# 11 — Sprint 计划与里程碑

> 估算：单人全职 **6–8 周**。每 Sprint 结束对应测试 **ALL GREEN**（见 `12_test_plan.md`）。

## Sprint 0 — 项目初始化（1 天）

| 步骤 | 输出 |
|------|------|
| 创建 `rc_pinn_art_project/` 目录 | 空骨架 |
| `pyproject.toml`, `requirements.txt`, `.gitignore` | 可 `pip install -e .` |
| `pinn_art/constants.py`, `utils/grid.py`（JAX 版） | import 通过 |
| `tests/conftest.py` | fixture |
| `configs/default.yaml`, `v3_smoke.yaml` | 配置可读 |

**验收**：

```bash
cd rc_pinn_art_project && pip install -e .
python -c "import pinn_art; print(pinn_art.__version__)"
pytest tests/test_import_smoke.py -v
```

---

## Sprint 1 — 坐标 + DeepONet 前向（3–4 天）

| 步骤 | 提示词 |
|------|--------|
| `coords/mixed_map.py`, `shell_features.py` | `02` |
| `nets/siren.py`, `deeponet.py`, `potential_head.py` | `03` |
| `physics/dirac_operator.py`, `orthogonalizer.py` | `03`, `13` |
| `models/pinn_art_model.py` Stage-A forward | `01` |
| `scripts/v3_diagnose_physics_chain.py` | — |

**验收**：

```bash
pytest tests/test_coords_*.py tests/test_deeponet_*.py tests/test_dirac_*.py -v
python scripts/v3_diagnose_physics_chain.py --config configs/v3_smoke.yaml --row 0
```

---

## Sprint 2 — 物理 Loss + Stage A 训练（4–5 天）

| 步骤 | 提示词 |
|------|--------|
| `losses/*` | `06` |
| `training/train_state.py`, `stage_a_trainer.py` | `07` |
| `scripts/v3_prepare_hydrogenic.py`, `v3_train_stage_a.py` | `09`, `07` |
| `training/gate_a.py`, `scripts/v3_gate_analytic.py` | `08` |

**验收**：

```bash
python scripts/v3_train_stage_a.py --config configs/v3_smoke.yaml   # 50 step 不降报错
python scripts/v3_train_stage_a.py --config configs/v3_phase1_stage_a.yaml
python scripts/v3_gate_analytic.py --ckpt checkpoints/.../stage_a_passed
# 期望 VERDICT=PASS（类氢 18+ 行）
```

---

## Sprint 3 — 可微 CI + 对角混合填充（NIST / Fall-back）（5–7 天）

| 步骤 | 提示词 |
|------|--------|
| `ci/slater_radial.py`, `hamiltonian.py` | `04` |
| `ci/nist_inject.py`, `eigen_solver.py` | `04`, `14` |
| `scripts/v3_build_racah_cache.py`（可先 stub 2×2） | `04` |
| 扩展 forward：CI 分支 | `01` |

**验收**：

```bash
pytest tests/test_slater_*.py tests/test_eigh_*.py tests/test_nist_inject_*.py -v
# 合成 2×2 哈密顿量本征值误差 < 1e-6
```

---

## Sprint 4 — 可观测量 + jit 推断（4–5 天）

| 步骤 | 提示词 |
|------|--------|
| `observables/transition_rates.py`, `collision.py` | `05` |
| `scripts/v3_infer.py`, `v3_profile_latency.py` | `05`, `08` |
| `@jax.jit` 端到端 | `01` |

**验收**：

```bash
pytest tests/test_transition_*.py tests/test_infer_jit_*.py -v
python scripts/v3_profile_latency.py --ckpt ...  # p50 记录
```

---

## Sprint 5 — NIST 全量推断 + 评估（3–4 天）

| 步骤 | 提示词 |
|------|--------|
| `scripts/v3_prepare_nist_subset.py` | `09` |
| `scripts/v3_evaluate.py` | `08` |
| LOO 配置 | `08` |

**验收**：

```bash
python scripts/v3_infer.py --config configs/v3_phase3_nist_infer.yaml
python scripts/v3_evaluate.py --run results/latest
# layer2 mae_meV 报告（Phase2 目标 <50 meV）
```

---

## Sprint 6 — 连续谱 + FAC 对比（可选，1–2 周）

| 步骤 | 提示词 |
|------|--------|
| `nets/phase_amplitude.py` | `03` |
| `observables/collision.py` 完整 | `05` |
| `v3_compare_fac.py` | `08` |

---

## 风险与降级

| 风险 | 降级 |
|------|------|
| Racah 预计算量大 | Phase 2 仅 $M\le 5$ 闭壳层 |
| JAX eigh grad 不稳 | 增大 `eps_degen` + Custom VJP |
| ms 延迟不达标 | 减 `n_grid`、蒸馏小模型 |
| SIREN 不收敛 | 降 $\omega_0$，加 envelope bound |
