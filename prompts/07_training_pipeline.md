# 07 — 训练管线

## 1. 三阶段状态机

```text
Stage A (Dirac PINN)  →  gate_A  →  Stage B (CI radial, 可选)  →  gate_B  →  Stage C (infer + 对角混合填充)
```

| 阶段 | 可训练参数 | 数据 |
|------|------------|------|
| A | DeepONet branch+trunk, $\Delta V$ | manifest_hydrogenic + 少量 Z=3..10 |
| B | 仅 Slater 校正头（若有） | 同 A + 小 manifest_ci |
| C | **无训练** 或 极弱 leading% | 全量推断：`nist_mask` 处 NIST，否则理论 Fall-back |

## 2. 优化器

```python
# Optax
optimizer = optax.adamw(learning_rate=schedule, weight_decay=1e-4)
# 参数分组
param_groups = {
    "trunk": trunk_params,   # lr 3e-4
    "branch": branch_params, # lr 1e-4
    "V_head": V_params,      # lr 1e-4
}
```

梯度裁剪：`optax.clip_by_global_norm(1.0)`。

## 3. 训练步（Flax 风格）

```python
@jax.jit
def train_step(state, batch, key):
    def loss_fn(params):
        out = forward(params, batch, train=True)
        loss = compute_stage_a_loss(out, batch)
        return loss, out

    (loss, out), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    grads = jax.tree.map(lambda g: jnp.nan_to_num(g), grads)
    state = state.apply_gradients(grads=grads)
    return state, loss, out
```

## 4. Checkpoint

```text
checkpoints/v3_stage_a/
  step_10000/
    params.msgpack
    opt_state.msgpack
    config.yaml
  stage_a_passed.json    # gate 报告
```

## 5. Gate A（进入 CI）

见 `08_evaluation.md` §2：

- 类氢 18–60 行：$|cos|>0.99$，$|E_\mathrm{orb}+Z^2/(2n^2)|<1$ meV
- $L_\mathrm{PDE}<10^{-3}$
- $\max_{a,b}|\langle ab\rangle|<10^{-4}$

## 6. Gate B（进入全 NIST 推断）

- 2×2、3×3 合成 H 谱：本征能误差 < 10 meV
- `eigh` grad 无 NaN（`test_eigh_gradient_finite`）

## 7. 脚本

```text
scripts/v3_train_stage_a.py
scripts/v3_train_stage_b.py      # 可选
scripts/v3_infer.py              # Stage C jit 推断
scripts/v3_resume.py
```

## 8. 配置片段

```yaml
training:
  batch_size: 8
  n_epochs_a: 2000
  steps_per_epoch: 100
  val_every_epochs: 10
  ckpt_every_epochs: 50
  seed: 42

stage_a:
  enabled: true
  manifest: data_cache/manifest_hydrogenic_v3.parquet

stage_b:
  enabled: false

stage_c:
  diagonal_fill: hybrid          # nist | theory | hybrid（默认 hybrid）
  nist_inject: true              # 启用混合填充（= hybrid）
  manifest: data_cache/manifest_nist_v3.parquet
```

## 9. 失败诊断

| 现象 | 可能原因 | 动作 |
|------|----------|------|
| $L_\mathrm{PDE}$ 不降 | SIREN $\omega_0$ 过大/过小 | 调 15–30 |
| 正交爆炸 | $w_\mathrm{ortho}$ 太小 | 提到 100+ |
| $V$ 发散 | 缺 $L_V$ prior | 开 prior |
| eigh NaN | 未 symmetrize / 无 eps | 见 14 |
