"""End-to-end PINN-ART model."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn

from ..ci.eigen_solver import safe_eigh
from ..ci.hamiltonian import assemble_hamiltonian_diagonal
from ..ci.nist_inject import fill_h_diagonal_hybrid
from ..coords.mixed_map import MixedRadialMap
from ..coords.shell_features import branch_input_dim, build_branch_features
from ..nets.deeponet import DeepONetDirac
from ..observables.collision import collision_cross_section_ce
from ..observables.transition_rates import compute_e1_transitions
from ..physics.dirac_operator import dirac_apply, orbital_energy_from_dirac
from ..physics.orthogonalizer import lowdin_orthonormalize
from ..utils.grid import RadialGrid


class PinnArtModel(nn.Module):
    n_orb_max: int = 16
    n_csf_max: int = 32
    d_branch: int = 128
    d_trunk: int = 128
    n_siren_layers: int = 4
    omega_0: float = 30.0
    c1: float = 1.0
    c2: float = 1.0
    ci_enabled: bool = False
    eps_degen_ev: float = 1e-6
    apply_lowdin: bool = False  # Stage A：保持网络原始输出（让 norm/ortho loss 真正起作用）

    @nn.compact
    def __call__(
        self,
        batch: dict,
        grid: RadialGrid,
        *,
        train: bool = False,
        return_ci: bool = False,
        E_grid=None,
    ) -> dict:
        r = grid.r
        cmap = MixedRadialMap(c1=self.c1, c2=self.c2)
        t = cmap.t(r)
        dt_dr = cmap.dt_dr(r)

        branch_feat = build_branch_features(batch, n_orb_max=self.n_orb_max)
        d_in = branch_input_dim(self.n_orb_max)
        net = DeepONetDirac(
            d_branch=self.d_branch,
            d_trunk=self.d_trunk,
            n_siren_layers=self.n_siren_layers,
            omega_0=self.omega_0,
            n_orb_max=self.n_orb_max,
            d_in_branch=d_in,
        )
        kappa = batch["kappa"][:, : self.n_orb_max]
        orb_mask = batch["orb_mask"][:, : self.n_orb_max]
        Z = batch["Z"]

        shell_table = batch.get("shell_table")
        if shell_table is not None:
            n_principal = shell_table[:, : self.n_orb_max, 0]
        else:
            n_principal = jnp.maximum(jnp.abs(kappa), 1)

        raw = net(branch_feat, t, r, dt_dr, kappa, orb_mask, Z, n_principal=n_principal)

        if self.apply_lowdin:
            ortho = lowdin_orthonormalize(
                raw["P"], raw["Q"], grid, orb_mask, raw["dPdr"], raw["dQdr"]
            )
            P, Q = ortho["P"], ortho["Q"]
            dPdr, dQdr = ortho["dPdr"], ortho["dQdr"]
        else:
            P, Q = raw["P"], raw["Q"]
            dPdr, dQdr = raw["dPdr"], raw["dQdr"]
        V = raw["V"]
        LP, LQ = dirac_apply(P, Q, dPdr, dQdr, V, kappa, r)
        E_orb = orbital_energy_from_dirac(P, Q, LP, LQ, grid)
        E_orb = jnp.where(orb_mask, E_orb, 0.0)

        out = {
            "t_grid": t,
            "V": V,
            "wavefunctions": {"P": P, "Q": Q, "dPdr": dPdr, "dQdr": dQdr},
            "LP": LP,
            "LQ": LQ,
            "E_orb": E_orb,
            "E_csf": None,
            "V_csf": None,
            "H": None,
            "transitions": None,
            "cross_sections": None,
        }

        if return_ci or self.ci_enabled:
            csf_mask = batch.get("csf_mask")
            if csf_mask is None:
                csf_mask = orb_mask[:, :1].astype(bool)
                csf_mask = jnp.broadcast_to(csf_mask, (Z.shape[0], self.n_csf_max))
            else:
                csf_mask = csf_mask[:, : self.n_csf_max]

            H = assemble_hamiltonian_diagonal(E_orb, csf_mask)
            E_nist = batch.get("E_nist")
            nist_mask = batch.get("nist_mask")
            if E_nist is not None and nist_mask is not None:
                E_nist = E_nist[:, : self.n_csf_max]
                nist_mask = nist_mask[:, : self.n_csf_max]
                if E_nist.dtype == jnp.float32 or E_nist.dtype == jnp.float64:
                    nist_mask = nist_mask & jnp.isfinite(E_nist)
                H, diag_src = fill_h_diagonal_hybrid(H, E_nist, nist_mask)
                out["diag_source"] = diag_src
            E_csf, V_csf = safe_eigh(H, self.eps_degen_ev)
            out["E_csf"] = E_csf
            out["V_csf"] = V_csf
            out["H"] = H
            out["transitions"] = compute_e1_transitions(E_csf, V_csf, P, Q, grid, csf_mask)
            if E_grid is not None:
                out["cross_sections"] = {"CE": collision_cross_section_ce(E_csf, E_grid)}

        return out


def build_model_and_params(cfg, grid: RadialGrid, key: jax.Array):
    model_cfg = getattr(cfg, "model", cfg)
    ci_cfg = getattr(cfg, "ci", None)
    coords = getattr(cfg, "coords", None)
    model = PinnArtModel(
        n_orb_max=int(getattr(model_cfg, "n_orb_max", 16)),
        n_csf_max=int(getattr(model_cfg, "n_csf_max", 32)),
        d_branch=int(getattr(model_cfg, "d_branch", 128)),
        d_trunk=int(getattr(model_cfg, "d_trunk", 128)),
        n_siren_layers=int(getattr(model_cfg, "n_siren_layers", 4)),
        omega_0=float(getattr(model_cfg, "omega_0", 30.0)),
        c1=float(getattr(coords, "c1", 1.0)) if coords else 1.0,
        c2=float(getattr(coords, "c2", 1.0)) if coords else 1.0,
        ci_enabled=bool(getattr(ci_cfg, "enabled", False)) if ci_cfg else False,
        eps_degen_ev=float(getattr(ci_cfg, "eps_degen_ev", 1e-6)) if ci_cfg else 1e-6,
        apply_lowdin=bool(getattr(model_cfg, "apply_lowdin", False)),
    )
    batch = _dummy_batch(int(getattr(model_cfg, "n_orb_max", 16)), int(getattr(model_cfg, "n_csf_max", 32)))
    params = model.init(key, batch, grid, train=False, return_ci=model.ci_enabled)
    return model, params


def _dummy_batch(n_orb_max, n_csf_max):
    B = 2
    return {
        "Z": jnp.ones(B, dtype=jnp.int32),
        "ion_charge": jnp.zeros(B, dtype=jnp.int32),
        "nele": jnp.ones(B, dtype=jnp.int32),
        "omega": jnp.ones((B, n_orb_max), dtype=jnp.float32) / n_orb_max,
        "kappa": jnp.ones((B, n_orb_max), dtype=jnp.int32) * -1,
        "orb_mask": jnp.array([[True] + [False] * (n_orb_max - 1)] * B),
        "shell_table": jnp.zeros((B, n_orb_max, 4), dtype=jnp.int32),
        "csf_mask": jnp.array([[True] + [False] * (n_csf_max - 1)] * B),
        "E_nist": jnp.zeros((B, n_csf_max)),
        "nist_mask": jnp.array([[True] + [False] * (n_csf_max - 1)] * B),
    }
