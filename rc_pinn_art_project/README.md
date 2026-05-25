# PINN-ART (rc-pinn-art) v3.0.0

毫秒级多电子原子辐射转移计算引擎（JAX + Flax）。

## 安装

**GPU（推荐，需 NVIDIA 驱动 + CUDA 12 兼容运行时）：**

```bash
cd rc_pinn_art_project
pip install -U "jax[cuda12]>=0.4.28"
pip install -e ".[dev]"
python -c "import jax; print(jax.devices())"   # 应出现 cuda:0
```

**仅 CPU：**

```bash
pip install -U "jax[cpu]>=0.4.28"
pip install -e ".[dev]"
```

驱动 CUDA 13.x 通常仍可使用 `jax[cuda12]` 轮子（向后兼容）。

## 快速验证

```bash
pytest tests/ -v --tb=short
python scripts/v3_diagnose_physics_chain.py --config configs/v3_smoke.yaml
python scripts/v3_train_stage_a.py --config configs/v3_smoke.yaml
```

设计文档：`../prompts/README.md`
