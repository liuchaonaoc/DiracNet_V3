"""Slater R^k correction to V_dfs (Path A: 物理实现 R^k → V_dfs 注入).

定义:
  V_slater_corr(r) = - Σ_k exp(slater_log_scale[k]) Σ_a ω_a R^k(a,a) P_a(r)^2 / ρ(r)

物理直觉 (Slater F^k exchange 的局域简化):
  - 在原始 Dirac-Fock 中, V_exch(a, r) 是非局域的: 需要对所有 b 积分
    V_exch(a, r) P_a(r) = -Σ_b ω_b R^k(a,b) f_k(r; <r_b, >r_b) P_b(r)^2
  - 局域化 (Slater X-α 已做的):  ρ^(1/3) 近似
  - Path A 简化: 只用对角 R^k(a,a) + 局域密度近似
  - 单电子体系 (N=1):  ρ = P^2,  V_slater_corr = - Σ_k exp(slater_log_scale) R^k(a,a)
    = 单电子交换 (Coulomb self-interaction correction)

输入:
  P, Q, omega:  [B, N_orb, N_g]
  Rk:  对角 R^k, 形状 [B, n_k, N_orb]  (来自 compute_all_Rk_diagonal)
  slater_log_scale:  [n_k]  标量
  rho:  [B, N_g]  电子密度 (从同一份 P/Q/omega 算, 用 electron_density)

返回:
  V_slater_corr:  [B, N_g]  局域 Slater 交换修正
"""
from __future__ import annotations

import jax.numpy as jnp

# B''''' fix: 全局注入乘子 (默认 1.0, B''''' 用 0.5)
# 原因 (per §21): B'''' 实跑确认 V_slater_corr 注入 (exp=0.37 倍)
#   让 V_dfs_aug 偏离 V_net 太大 (V(r~1) diff=-0.99 vs B'' -0.10)
#   → He 1s² 30 epoch 退化 6.7 eV
# 修复: 加全局乘子 0.5, 让 V_slater_corr 注入 "半幅" 起步, 缓解 V_dfs_aug 偏移
# 副作用: V_slater_corr 量级小一半 → slater_log_scale 学习信号更弱, 但配合 10x LR 仍可学
# B''''' 调参入口: 在 import 处修改 (后续可放到 config)
V_SLATER_INJECT_SCALE = 0.5


def slater_correction_potential(
    P: jnp.ndarray,
    Q: jnp.ndarray,
    omega: jnp.ndarray,
    Rk: jnp.ndarray,
    slater_log_scale: jnp.ndarray,
    rho: jnp.ndarray,
) -> jnp.ndarray:
    """Path A: 局域 Slater 交换修正 V_slater_corr(r).

    Args:
        P, Q:       [B, N_orb, N_g]   Dirac radial wavefunctions
        omega:      [B, N_orb]         occupation numbers
        Rk:         [B, n_k, N_orb]    对角 R^k(a,a) 积分
        slater_log_scale: [n_k]        可学习标量 (默认 init=0 → 1.0)
        rho:        [B, N_g]           电子密度 (来自 electron_density)

    Returns:
        V_slater_corr: [B, N_g]  注入 V_dfs 的修正项
    """
    B, N, _ = P.shape
    n_k = Rk.shape[1]

    # P_a(r)^2 = P[:,a,:]^2  →  形状 [B, N_orb, N_g]
    P2 = P * P  # [B, N, N_g]

    # 加权:  ω_a R^k(a,a) P_a(r)^2
    # omega: [B, N]  →  [B, 1, N, 1]  (broadcast 在 n_k 维)
    # Rk: [B, n_k, N]  →  [B, n_k, N, 1]
    omega_e = omega[:, None, :, None]             # [B, 1, N, 1]
    Rk_e = Rk[:, :, :, None]                      # [B, n_k, N, 1]
    weighted_P2 = omega_e * Rk_e * P2[:, None, :, :]  # [B, n_k, N, N_g]
    #  Σ_a ω_a R^k(a,a) P_a(r)^2
    sum_wP2 = jnp.sum(weighted_P2, axis=2)        # [B, n_k, N_g]

    # exp(slater_log_scale): [n_k] → [1, n_k, 1]
    scale = jnp.exp(slater_log_scale)[None, :, None]
    # 总和:  Σ_k scale_k * sum_wP2_k
    # 形状 [B, n_k, N_g] → 求和 k → [B, N_g]
    total = jnp.sum(scale * sum_wP2, axis=1)      # [B, N_g]

    # 归一化: 除以 ρ (避免除以 0, clip)
    rho_safe = jnp.clip(rho, 1e-12)              # [B, N_g]
    # Path A 公式:  V_slater_corr = - total / ρ
    V_slater_corr = -total / rho_safe             # [B, N_g]

    # B''''' fix: 全局注入乘子 (默认 1.0, 改成 0.5 缓解 V_dfs_aug 偏移)
    V_slater_corr = V_slater_corr * V_SLATER_INJECT_SCALE

    return V_slater_corr


def augmented_dfs_target(
    V_dfs: jnp.ndarray,
    P: jnp.ndarray,
    Q: jnp.ndarray,
    omega: jnp.ndarray,
    Rk: jnp.ndarray,
    slater_log_scale: jnp.ndarray,
    rho: jnp.ndarray,
) -> jnp.ndarray:
    """V_dfs 增强版: V_dfs + V_slater_corr (Path A 注入).

    用法 (在 stage_a_trainer 的 SCF 损失里):
        V_dfs_aug = augmented_dfs_target(V_dfs, P, Q, omega, Rk, slater_log_scale, rho)
        l_scf = scf_consistency_loss(V_net, V_dfs_aug, weight=scf_weight)
    """
    V_slater = slater_correction_potential(P, Q, omega, Rk, slater_log_scale, rho)
    return V_dfs + V_slater
