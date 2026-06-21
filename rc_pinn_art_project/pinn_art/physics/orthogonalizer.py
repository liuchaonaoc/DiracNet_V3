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


def gram_schmidt_ortho_pq(
    P: jnp.ndarray,
    Q: jnp.ndarray,
    grid: RadialGrid,
    orb_mask: jnp.ndarray,
    dPdr: jnp.ndarray | None = None,
    dQdr: jnp.ndarray | None = None,
    eps: float = 1e-6,
) -> dict[str, jnp.ndarray]:
    """Lightweight Gram-Schmidt orthonormalization (Stage A Round 2 §2.6).

    Sequentially orthogonalize each orbital against all previous orbitals in
    the inner product ⟨f, g⟩ = ∫ (f_P g_P + f_Q g_Q) dr, then normalize to 1.
    Cheaper than full Löwdin `S^{-1/2}` and avoids the eigh decomposition.

    Returns a dict with keys `P, Q` (and `dPdr, dQdr` if input had them),
    zeroing out masked-out slots.
    """
    B, N_orb, N_g = P.shape
    out = {"P": P, "Q": Q}
    if dPdr is not None:
        out["dPdr"] = dPdr
    if dQdr is not None:
        out["dQdr"] = dQdr

    # Operate per-batch.
    P_out = P
    Q_out = Q
    if dPdr is not None:
        dP_out = dPdr
    if dQdr is not None:
        dQ_out = dQdr
    for b_idx in range(B):
        # Iterate orbital-by-orbital; build list of orthonormalized (P,Q) per
        # batch entry.
        m = orb_mask[b_idx]  # [N_orb]
        active = [a for a in range(N_orb) if bool(m[a])]
        if not active:
            continue
        Ps = []
        Qs = []
        if dPdr is not None:
            dPs = []
        if dQdr is not None:
            dQs = []
        for a in active:
            pa = P_out[b_idx, a]
            qa = Q_out[b_idx, a]
            for pb, qb in zip(Ps, Qs):
                # Subtract projection onto pb: pa ← pa − ⟨pa,pb⟩ pb
                inner = float(grid.integrate(pa * pb + qa * qb, axis=-1))
                pa = pa - inner * pb
                qa = qa - inner * qb
                if dPdr is not None:
                    dpa = dP_out[b_idx, a] - inner * dPs[-1]
                    dPs.append(dpa)
                if dQdr is not None:
                    dqa = dQ_out[b_idx, a] - inner * dQs[-1]
                    dQs.append(dqa)
            # Normalize: ∫(P²+Q²)dr = 1
            norm_sq = float(grid.integrate(pa * pa + qa * qa, axis=-1))
            inv_n = 1.0 / max(norm_sq, eps) ** 0.5
            pa = pa * inv_n
            qa = qa * inv_n
            Ps.append(pa)
            Qs.append(qa)
            if dPdr is not None and not dPs:
                # dPa accumulated inside the inner loop is empty for the
                # first orbital; push current.
                dPs.append(dP_out[b_idx, a] * inv_n)
            elif dPdr is not None:
                dPs[-1] = dPs[-1] * inv_n
            if dQdr is not None and not dQs:
                dQs.append(dQ_out[b_idx, a] * inv_n)
            elif dQdr is not None:
                dQs[-1] = dQs[-1] * inv_n

        # Write back into P_out.
        for k, a in enumerate(active):
            P_out = P_out.at[b_idx, a].set(Ps[k])
            Q_out = Q_out.at[b_idx, a].set(Qs[k])
            if dPdr is not None:
                dP_out = dP_out.at[b_idx, a].set(dPs[k])
            if dQdr is not None:
                dQ_out = dQ_out.at[b_idx, a].set(dQs[k])

    out["P"] = P_out
    out["Q"] = Q_out
    if dPdr is not None:
        out["dPdr"] = dP_out
    if dQdr is not None:
        out["dQdr"] = dQ_out
    return out
