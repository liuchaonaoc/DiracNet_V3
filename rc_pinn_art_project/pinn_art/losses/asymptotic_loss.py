"""Far-field envelope consistency.

For hydrogenic |P_n l(r)| ~ exp(-Z r / n) at large r, so
    d(log|P|)/dr → -Z / n  (asymptotically).

We compute the slope of log|P| over the outer half of the grid and penalize
deviation from -Z/n.  If n_principal is not supplied we fall back to -Z/2
(reasonable mid-range default).
"""

from __future__ import annotations

import jax.numpy as jnp

from ..utils.numeric import masked_mean


def asymptotic_tail_loss(
    P,
    Q,
    r,
    orb_mask,
    r_cut_frac: float = 0.5,
    Z=None,
    n_principal=None,
):
    n_grid = r.shape[0]
    i_cut = int(n_grid * r_cut_frac)
    tail = slice(i_cut, None)
    P_t = P[..., tail]
    log_P = jnp.log(jnp.clip(jnp.abs(P_t), 1e-30))
    r_t = r[tail]
    dr = r_t[-1] - r_t[0]
    slope = (log_P[..., -1] - log_P[..., 0]) / jnp.clip(dr, 1e-6)
    if Z is not None:
        Zf = Z.astype(jnp.float32)[:, None]
        if n_principal is not None:
            n_f = jnp.clip(n_principal.astype(jnp.float32), 1.0, 30.0)
            target = -Zf / n_f
        else:
            target = -jnp.clip(Zf / 2.0, 0.05, 10.0)
    else:
        target = -0.25
    err = (slope - target) ** 2
    return masked_mean(err, orb_mask)
