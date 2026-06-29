"""Coefficient-regularization losses for the Laguerre basis ansatz.

Exposed as thin re-exports of the operators in
`pinn_art.nets.laguerre_basis`, so loss schedules / training scripts can
import a uniform `pinn_art.losss.coeff_loss.*` namespace.
"""

from __future__ import annotations

from ..nets.laguerre_basis import (
    coeff_anchor_loss,
    coeff_decay_loss,
    lambda_prior_loss,
)


__all__ = ["coeff_anchor_loss", "coeff_decay_loss", "lambda_prior_loss"]