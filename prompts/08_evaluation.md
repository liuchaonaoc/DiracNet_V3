# 08 — 评估与性能门禁

## 1. 评估分层

### Layer-0 波函数（Stage A 必须）

```python
@dataclass
class WavefunctionGate:
    cos_min: float          # min |cos(P_pred, P_ref)|
    pde_max: float
    ortho_max: float
    e_orb_mae_meV: float
    verdict: str            # PASS / FAIL
```

阈值（类氢）：

```yaml
gate_a:
  cos_threshold: 0.99
  pde_threshold: 1.0e-3
  ortho_threshold: 1.0e-4
  e_orb_meV_threshold: 1.0
```

### Layer-2 能级（Stage C）

```python
def level_metrics(E_pred, E_ref, nist_mask):
    # 仅对 nist_mask=True 计算 MAE/RMS（实验对比）
    mae_nist_meV = ...
    # 对 mask=False 报告理论自洽性（可选）：E_pred 与 H_diag_theory 一致
    n_fallback = (~nist_mask).sum()
    leading_match_rate = ...
```

目标：

- **有 NIST**（`nist_mask=True`）闭壳层 subset：**MAE < 5 meV**
- **有 NIST** 全 manifest active：**MAE < 50 meV**（Phase 2）
- **无 NIST**（Fall-back 区）：不考核实验 MAE；记录 `n_fallback`、$L_\mathrm{PDE}$、理论 Leading %

`metrics.json` 必须区分：

```json
"layer2": {
  "mae_nist_meV": 3.2,
  "n_nist_matched": 1200,
  "n_theory_fallback": 450,
  "fallback_fraction": 0.27
}
```

### Layer-3 跃迁与截面

```text
median_rel_err(A_ki)  vs FAC
median_rel_err(sigma_CE)  vs FAC  at 3 energies
```

### Layer-4 延迟

```bash
python scripts/v3_profile_latency.py --ckpt ... --n_warmup 10 --n_repeat 100
```

报告：`p50_ms`, `p95_ms`, `compile_s`（首次 `jit`）。

目标：**p50 < 10 ms**（Fe XVII 规模，单卡）。

## 2. 脚本 `v3_evaluate.py`

输出 `results/eval_<run_id>/`：

```text
metrics.json
levels.parquet
transitions.parquet
gate_report.md
latency.json
```

必须字段：

```json
{
  "layer0": {"verdict": "PASS", ...},
  "layer2": {"mae_meV": 3.2, "n_matched": 1200},
  "layer3": {"A_ki_median_rel_err": 0.15},
  "layer4": {"p50_ms": 4.1}
}
```

## 3. 解析对照 `v3_gate_analytic.py`

与 V2 `v2_gate_analytic.py` 等价，读 `manifest_hydrogenic_v3.parquet`，逐行打印表格 + `VERDICT`。

## 4. LOO / OOD

| 测试 | 做法 |
|------|------|
| leave-one-Z | 训练剔除 Z=8，推断 O |
| leave-one-n | 剔除 n=4 壳层 |
| leave-one-ion | 按 ion_charge holdout |

配置：`configs/v3_loo_z8.yaml`。

## 5. 与 FAC 对比流程

1. 用户自备 FAC 输出 → `data_raw/fac/<ion>.csv`
2. `v3_compare_fac.py --fac_dir ... --pred results/...`

## 6. 验收命令（Sprint 末）

```bash
python scripts/v3_train_stage_a.py --config configs/v3_phase1_stage_a.yaml
python scripts/v3_gate_analytic.py --ckpt checkpoints/v3_phase1/stage_a_passed
python scripts/v3_infer.py --ckpt ... --manifest data_cache/manifest_nist_v3.parquet
python scripts/v3_evaluate.py --run results/latest
```
