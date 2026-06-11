"""Self-consistency loss for the screened DFS potential (R1.2).

Drives the network potential head ``V_net`` to the fixed point of the
density-dependent DFS potential ``V_dfs[rho]``:

    L_scf = < (V_net - sg[V_dfs])^2 >

``sg`` (``stop_gradient``) is applied to ``V_dfs`` by default so the density
side is held fixed within a step (stabilizes early training); set
``stop_grad=False`` to couple both sides fully (later, fully-coupled SCF).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def scf_consistency_loss(
    V_net: jnp.ndarray,
    V_dfs: jnp.ndarray,
    *,
    weight: jnp.ndarray | None = None,
    stop_grad: bool = True,
) -> jnp.ndarray:
    """Mean squared mismatch between the network potential and ``V_dfs[rho]``.

    V_net, V_dfs : [B, N_g]
    weight       : optional [N_g] or [B, N_g] radial weighting (e.g. r^2 to
                   emphasize the valence region over the r->0 cusp).
    """
    target = jax.lax.stop_gradient(V_dfs) if stop_grad else V_dfs
    diff2 = (V_net - target) ** 2
    if weight is None:
        return jnp.mean(diff2)
    w = weight.astype(diff2.dtype)
    if w.ndim == 1:
        w = w[None, :]
    return jnp.sum(w * diff2) / jnp.clip(jnp.sum(jnp.broadcast_to(w, diff2.shape)), 1e-12)
