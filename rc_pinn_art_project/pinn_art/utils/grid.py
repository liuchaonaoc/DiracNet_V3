"""Radial grid and integration weights (JAX).

`RadialGrid` is registered as a JAX PyTree so it can be passed through
`jax.jit` (arrays as children, metadata as aux).  This lets the training
step JIT-compile across grids without re-tracing on metadata changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


@jax.tree_util.register_pytree_node_class
@dataclass
class RadialGrid:
    r: jnp.ndarray
    dr: jnp.ndarray
    jac: jnp.ndarray
    r_min: float
    r_max: float
    n_grid: int
    scheme: str

    @property
    def shape(self) -> tuple[int, ...]:
        return self.r.shape

    def integrate(self, f: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
        # Skip eager shape check when tracing under JIT (f.shape is symbolic)
        if not hasattr(f, "aval") and f.shape[axis] != self.n_grid:
            raise ValueError(f"integrate: axis {axis} size {f.shape[axis]} != n_grid {self.n_grid}")
        w = jnp.asarray(self.dr, dtype=f.dtype)
        return jnp.sum(f * w, axis=axis)

    def tree_flatten(self):
        children = (self.r, self.dr, self.jac)
        aux = (self.r_min, self.r_max, self.n_grid, self.scheme)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        r, dr, jac = children
        r_min, r_max, n_grid, scheme = aux
        return cls(r=r, dr=dr, jac=jac, r_min=r_min, r_max=r_max, n_grid=n_grid, scheme=scheme)


def _composite_trapezoid_weights(r: np.ndarray) -> np.ndarray:
    n = r.shape[0]
    w = np.zeros_like(r)
    w[1:-1] = 0.5 * (r[2:] - r[:-2])
    w[0] = 0.5 * (r[1] - r[0])
    w[-1] = 0.5 * (r[-1] - r[-2])
    return w


def make_radial_grid(
    r_min: float = 1.0e-4,
    r_max: float = 50.0,
    n_grid: int = 256,
    scheme: str = "loglinear",
) -> RadialGrid:
    if r_min <= 0.0:
        raise ValueError("r_min must be > 0")
    if n_grid < 4:
        raise ValueError("n_grid must be >= 4")

    x = np.linspace(0.0, 1.0, n_grid, dtype=np.float64)
    if scheme == "linear":
        r = r_min + (r_max - r_min) * x
        jac = np.full_like(r, r_max - r_min)
    elif scheme == "log":
        log_min, log_max = np.log(r_min), np.log(r_max)
        r = np.exp(log_min + (log_max - log_min) * x)
        jac = r * (log_max - log_min)
    elif scheme == "loglinear":
        log_min, log_max = np.log(r_min), np.log(r_max)
        r_log = np.exp(log_min + (log_max - log_min) * x)
        r_lin = r_min + (r_max - r_min) * x
        mix = x
        r = (1.0 - mix) * r_log + mix * r_lin
        dr_dx_log = r_log * (log_max - log_min)
        dr_dx_lin = np.full_like(r_lin, r_max - r_min)
        jac = (1.0 - mix) * dr_dx_log + mix * dr_dx_lin
    else:
        raise ValueError(f"unknown scheme {scheme!r}")

    dr = _composite_trapezoid_weights(r)
    return RadialGrid(
        r=jnp.asarray(r, dtype=jnp.float32),
        dr=jnp.asarray(dr, dtype=jnp.float32),
        jac=jnp.asarray(jac, dtype=jnp.float32),
        r_min=r_min,
        r_max=r_max,
        n_grid=n_grid,
        scheme=scheme,
    )
