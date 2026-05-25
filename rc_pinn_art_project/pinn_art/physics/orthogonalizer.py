"""Per-orbital normalization (+ optional full Lowdin)."""

from __future__ import annotations

import jax.numpy as jnp

from ..utils.grid import RadialGrid


def lowdin_orthonormalize(
    P: jnp.ndarray,
    Q: jnp.ndarray,
    grid: RadialGrid,
    orb_mask: jnp.ndarray,
    dPdr: jnp.ndarray | None = None,
    dQdr: jnp.ndarray | None = None,
    eps: float = 1e-6,
    use_full_lowdin: bool = False,
    normalize: bool = True,
) -> dict[str, jnp.ndarray]:
    """Optionally normalize per-orbital, optionally do full Löwdin S^{-1/2}.

    Caller controls:
        normalize       : per-orbital ‖P‖²+‖Q‖²=1  (default True for backward-compat)
        use_full_lowdin : multi-orbital S^{-1/2} mixing (Stage B/C)

    Stage A should pass normalize=False so that norm_loss is a real supervisor.
    """
    out: dict[str, jnp.ndarray] = {"P": P, "Q": Q}
    if dPdr is not None:
        out["dPdr"] = dPdr
    if dQdr is not None:
        out["dQdr"] = dQdr

    if normalize:
        norm_sq = grid.integrate(P * P + Q * Q, axis=-1)
        norm_sq = jnp.clip(norm_sq, eps)
        inv_norm = jax_rsqrt(norm_sq)[..., None]
        out["P"] = out["P"] * inv_norm
        out["Q"] = out["Q"] * inv_norm
        if dPdr is not None:
            out["dPdr"] = out["dPdr"] * inv_norm
        if dQdr is not None:
            out["dQdr"] = out["dQdr"] * inv_norm
        P = out["P"]
        Q = out["Q"]

    if not use_full_lowdin:
        return out

    integrand = P[:, :, None, :] * P[:, None, :, :] + Q[:, :, None, :] * Q[:, None, :, :]
    S = grid.integrate(integrand, axis=-1)
    B, N_orb, _ = P.shape
    mask_2d = orb_mask[:, :, None] * orb_mask[:, None, :]
    eye = jnp.eye(N_orb, dtype=S.dtype)[None, :, :]
    S = jnp.where(mask_2d, S, eye)
    S = 0.5 * (S + jnp.swapaxes(S, -1, -2)) + eps * eye
    eigvals, eigvecs = jnp.linalg.eigh(S)
    d_inv = 1.0 / jnp.sqrt(jnp.clip(eigvals, eps))
    inv_sqrt = jnp.einsum("...ij,...j,...kj->...ik", eigvecs, d_inv, eigvecs)
    out["P"] = jnp.einsum("bac,bcg->bag", inv_sqrt, P)
    out["Q"] = jnp.einsum("bac,bcg->bag", inv_sqrt, Q)
    if dPdr is not None:
        out["dPdr"] = jnp.einsum("bac,bcg->bag", inv_sqrt, dPdr)
    if dQdr is not None:
        out["dQdr"] = jnp.einsum("bac,bcg->bag", inv_sqrt, dQdr)
    return out


def jax_rsqrt(x: jnp.ndarray) -> jnp.ndarray:
    return 1.0 / jnp.sqrt(x)
