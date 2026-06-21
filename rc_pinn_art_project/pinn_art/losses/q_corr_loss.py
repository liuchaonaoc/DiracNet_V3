"""Q-correction loss: penalize (Q − Q_skel)² weighted by ρ.

The head `LaguerreQCorrHead` emits a `tanh(·)` correction whose amplitude is
multiplied by `perturb_scale_Q` (≤ 5%). To keep δQ truly "small", we add this
loss in the optimizer; without it the network is free to deviate from the
kinetic-balance skeleton and recover variational-collapse solutions.

Re-export of `pinn_art.nets.laguerre_basis.q_residual`.
"""

from __future__ import annotations

from ..nets.laguerre_basis import q_residual


__all__ = ["q_residual"]