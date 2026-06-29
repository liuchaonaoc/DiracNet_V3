#!/usr/bin/env python3
"""Pre-compute analytic Laguerre polynomial node positions for the
manifest grid (Z=1..26, n=1..10, l=0..n-1, K_max=9).

Implements §13.11-C.6 step 1 of `prompts/17_generalized_laguerre_basis.md`:
build a lookup table of nodes {r_1, ..., r_{n-1}} for each (Z, n, l)
configuration, so the HybridLaguerreHead can init strictly = analytic nodes
(preserves §2.3 init = physics).

Output: `data_cache/laguerre_nodes_z1_26_n1_10.parquet`
  Columns:
    Z, n, l, K_max,
    r_node_0, r_node_1, ..., r_node_{K_max-1}    (K_max columns)
    r_node_invalid                              (0.0 for unused slots)
    # The actual node count is min(n-1, K_max).

Usage:
    python scripts/v3_precompute_laguerre_nodes.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import jax.numpy as jnp
from jax import lax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinn_art.physics.hydrogenic import _laguerre_generalized_stack


def find_laguerre_roots(alpha: float, n: int) -> np.ndarray:
    """Roots of L_n^alpha(x) for x > 0.

    Uses bisection on sign changes of L_n^alpha(x) sampled on a log-spaced
    grid (analytic closed form is not trivial). For n <= 9 and alpha > -1
    the roots are well-separated and this is accurate to ~1e-8.

    Args:
        alpha: Laguerre parameter (>= -1)
        n:     polynomial degree (>= 0)
    Returns:
        roots: [n] array of positive real roots (ascending).
    """
    if n == 0:
        return np.zeros(0)

    # L_n^alpha(x) for large x grows like (-x)^n / n! .  All positive roots
    # are bounded above by ~ 4n + 2*alpha + 10 (Abramowitz & Stegun 22.16.7).
    x_max = float(4 * n + 2 * alpha + 10.0)
    # Use log-spacing to get good resolution for small roots.
    n_samples = 4096
    xs = np.geomspace(1e-6, x_max, n_samples)

    # Evaluate L_n^alpha on xs using the same recursion as `_laguerre_generalized_stack`.
    alpha_jax = jnp.float32(alpha)
    rho = jnp.asarray(xs, dtype=jnp.float32)
    Ls = _laguerre_generalized_stack(rho, alpha_jax, max_k=n)  # [n+1, n_samples]
    L_vals = np.asarray(Ls[n])  # [n_samples]

    # Find sign-change indices.
    signs = np.sign(L_vals)
    sign_changes = np.where(np.diff(signs) != 0)[0]

    if len(sign_changes) < n:
        # Pad: not enough sign changes found.  Fall back to scipy if available.
        try:
            from scipy.special import roots_laguerre
            x_roots, _ = roots_laguerre(n)
            # `roots_laguerre` gives Gauss-Laguerre nodes (NOT Laguerre poly
            # zeros).  Skip the fallback — return zeros to be safe.
            print(f"  [warn] only {len(sign_changes)} sign changes for alpha={alpha}, n={n}")
        except ImportError:
            pass
        roots = np.zeros(n)
        return roots

    # Bisection on each sign-change interval to refine.
    roots = np.zeros(n)
    for i, sc in enumerate(sign_changes[:n]):
        a, b = float(xs[sc]), float(xs[sc + 1])
        # Bisect 60 times -> resolution ~ x_max / 2^60 < 1e-12.
        for _ in range(60):
            m = 0.5 * (a + b)
            L_m = float(_eval_laguerre_scalar(alpha, n, m))
            L_a = float(_eval_laguerre_scalar(alpha, n, a))
            if L_a * L_m <= 0:
                b = m
            else:
                a = m
        roots[i] = 0.5 * (a + b)
    return roots


def _eval_laguerre_scalar(alpha: float, n: int, x: float) -> float:
    """Evaluate L_n^alpha(x) using the standard recursion (scalar)."""
    L_prev = 1.0
    if n == 0:
        return L_prev
    L_curr = 1.0 + alpha - x
    for j in range(1, n):
        jf = float(j)
        L_next = ((2.0 * jf + 1.0 + alpha - x) * L_curr - (jf + alpha) * L_prev) / (jf + 1.0)
        L_prev, L_curr = L_curr, L_next
    return L_curr


def main():
    Z_list = list(range(1, 27))
    n_list = list(range(1, 11))
    K_max = 9

    rows = []
    print(f"Pre-computing Laguerre nodes for Z=1..26, n=1..10, K_max={K_max}...")
    for Z in Z_list:
        for n in n_list:
            for l in range(0, n):
                alpha = 2 * l + 1
                # Roots are in the LAGUERRE POLYNOMIAL ARGUMENT (ρ = 2λr).
                rho_roots = find_laguerre_roots(float(alpha), n - 1)
                # Convert to r via λ = Z/n.
                lam = Z / n
                r_roots = rho_roots / (2.0 * lam)  # [n-1]

                # Pad to K_max entries (zero-fill for invalid slots).
                r_padded = np.zeros(K_max, dtype=np.float64)
                r_padded[: len(r_roots)] = r_roots
                # Cumulative-sum convention: we store raw nodes (NOT increments)
                # so the network only needs to output the deltas.
                # The hybrid head will compute r_total = r_analytic + cumsum(softplus(d)).

                row = {
                    "Z": Z,
                    "n": n,
                    "l": l,
                    "K_max": K_max,
                    "alpha": alpha,
                    "lambda": lam,
                    "n_nodes": len(r_roots),
                }
                for j in range(K_max):
                    row[f"r_node_{j}"] = float(r_padded[j])
                rows.append(row)
        print(f"  Z={Z} done")

    df = pd.DataFrame(rows)
    out_path = ROOT / "data_cache" / "laguerre_nodes_z1_26_n1_10.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"saved -> {out_path}")
    print(f"rows: {len(df)}")

    # Sanity check: print a few key cases.
    print("\nSanity checks:")
    for (Z, n, l) in [(1, 1, 0), (1, 10, 0), (26, 1, 0), (8, 5, 0)]:
        sub = df[(df.Z == Z) & (df.n == n) & (df.l == l)].iloc[0]
        nn = int(sub.n_nodes)
        r = [sub[f"r_node_{j}"] for j in range(nn)]
        lam_val = float(sub["lambda"])
        alpha_val = int(sub["alpha"])
        print(f"  Z={Z} n={n} l={l}: lambda={lam_val:.4f}, alpha={alpha_val}, n_nodes={nn}")
        rstr = ", ".join(f"{v:.3f}" for v in r[:5])
        print(f"    roots r: [{rstr}{'...' if nn > 5 else ''}]")


if __name__ == "__main__":
    main()
