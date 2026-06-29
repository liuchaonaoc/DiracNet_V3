"""End-to-end PINN-ART model."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import flax.core
from flax import linen as nn

from ..ci.eigen_solver import safe_eigh
from ..ci.hamiltonian import assemble_hamiltonian, assemble_hamiltonian_diagonal
from ..ci.nist_inject import fill_h_diagonal_hybrid
from ..ci.slater_radial import compute_all_Rk_diagonal
from ..coords.mixed_map import MixedRadialMap
from ..coords.shell_features import branch_input_dim, build_branch_features
from ..nets.deeponet import DeepONetDirac
from ..nets.laguerre_basis import (
    LaguerreCoeffHead,
    LaguerreLambdaHead,
    LaguerreQCorrHead,
    laguerre_p_sum_with_r,
)
from ..physics.hydrogenic import hydrogenic_laguerre_coeffs  # noqa: F401  (used by `DeepONetDirac` via the `_per_orbital_laguerre_init` re-export)
from ..observables.collision import collision_cross_section_ce
from ..observables.transition_rates import compute_e1_transitions
from ..physics.dfs_potential import slater_effective_charge
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
    k_list: tuple[int, ...] = (0,)
    nist_inject: bool = False
    eps_degen_ev: float = 1e-6
    apply_lowdin: bool = False  # Stage A：保持网络原始输出（让 norm/ortho loss 真正起作用）
    use_hydrogenic_skeleton: bool = True
    perturb_eps: float = 0.2
    use_zeff_warmstart: bool = False  # R1.3: Slater Z_eff skeleton warm-start

    # --- Stage A Round 2 (Laguerre basis) toggles ---
    use_laguerre_basis: bool = False
    K_max: int = 9
    learn_lambda: bool = True
    perturb_scale_P: float = 0.05
    perturb_scale_Q: float = 0.05
    # --- §13.1 / §13.2 of 17_generalized_laguerre_basis.md ---
    use_log_r_input: bool = False
    log_r_scale: float = 1.0
    use_dual_trunk: bool = False
    d_trunk_high: int = 128
    omega_0_high: float = 60.0
    lambda_split: float = 4.0
    gate_temperature: float = 2.0
    # --- §13.9 Branch Capacity (Step E) ---
    coeff_d_hidden: int = 64
    lambda_d_hidden: int = 32
    q_corr_d_hidden: int = 64
    # --- §13.11-C Joint scheme (Step G) ---
    use_hybrid_head: bool = False   # when True, replace LaguerreCoeffHead with HybridLaguerreHead
    nodes_table_path: str = "data_cache/laguerre_nodes_z1_26_n1_10.parquet"
    hybrid_d_hidden_node: int = 64   # Stage 1 node MLP hidden width
    hybrid_d_hidden_ref: int = 32    # Stage 2 refinement MLP hidden width
    hybrid_n_iter: int = 3           # refinement iterations
    hybrid_step_size: float = 0.1    # per-iteration c_k update step
    # Analytic Laguerre init is computed per-batch inside
    # `DeepONetDirac.__call__` from the (batched) (Z, n, l) so it follows
    # whatever row the manifest feeds (Round-3 fix v2).  No model-level
    # constants are needed.
    #
    # §13.11-C (round-2 fix): the analytic node table is loaded by
    # `collate_batches` (host-side) and stored in `batch['analytic_nodes']`.
    # The model itself does NOT load the table (no `setup()` is needed).
    # This keeps the JIT region free of Python-side table lookups.

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
        kappa = batch["kappa"][:, : self.n_orb_max]
        orb_mask = batch["orb_mask"][:, : self.n_orb_max]
        Z = batch["Z"]

        shell_table = batch.get("shell_table")
        if shell_table is not None:
            n_principal = shell_table[:, : self.n_orb_max, 0]
            l_orbital = shell_table[:, : self.n_orb_max, 1]
        else:
            n_principal = jnp.maximum(jnp.abs(kappa), 1)
            l_orbital = jnp.where(kappa < 0, -kappa - 1, kappa)

        # §13.11-C: analytic Laguerre node positions [B, N_orb, K_max] are
        # looked up host-side in `collate_batches` and passed in as
        # `batch['analytic_nodes']`.  This keeps the JIT region free of
        # Python-side table lookups.  When `use_hybrid_head` is off, we
        # still need a key for the call signature; pass zeros.
        if "analytic_nodes" in batch:
            analytic_nodes = batch["analytic_nodes"]
        else:
            analytic_nodes = jnp.zeros(
                (Z.shape[0], self.n_orb_max, self.K_max), dtype=jnp.float32
            )

        # Stage A Round 2 (Round-3 fix v2): per-batch analytic Laguerre
        # init is computed inside `DeepONetDirac.__call__` from the
        # (batched) (Z, n, l) — so the very first forward pass
        # reproduces the analytic hydrogenic P for each row.

        net = DeepONetDirac(
            d_branch=self.d_branch,
            d_trunk=self.d_trunk,
            n_siren_layers=self.n_siren_layers,
            omega_0=self.omega_0,
            n_orb_max=self.n_orb_max,
            d_in_branch=d_in,
            use_hydrogenic_skeleton=self.use_hydrogenic_skeleton,
            perturb_eps=self.perturb_eps,
            use_laguerre_basis=self.use_laguerre_basis,
            K_max=self.K_max,
            learn_lambda=self.learn_lambda,
            perturb_scale_P=self.perturb_scale_P,
            perturb_scale_Q=self.perturb_scale_Q,
            use_log_r_input=self.use_log_r_input,
            log_r_scale=self.log_r_scale,
            use_dual_trunk=self.use_dual_trunk,
            d_trunk_high=self.d_trunk_high,
            omega_0_high=self.omega_0_high,
            lambda_split=self.lambda_split,
            gate_temperature=self.gate_temperature,
            coeff_d_hidden=self.coeff_d_hidden,
            lambda_d_hidden=self.lambda_d_hidden,
            q_corr_d_hidden=self.q_corr_d_hidden,
            # §13.11-C
            use_hybrid_head=self.use_hybrid_head,
            hybrid_d_hidden_node=self.hybrid_d_hidden_node,
            hybrid_d_hidden_ref=self.hybrid_d_hidden_ref,
            hybrid_n_iter=self.hybrid_n_iter,
            hybrid_step_size=self.hybrid_step_size,
        )

        z_eff_orb = None
        if self.use_zeff_warmstart:
            omega = batch.get("omega")
            if omega is not None:
                z_eff_orb = slater_effective_charge(
                    Z, n_principal, omega[:, : self.n_orb_max], orb_mask
                )

        raw = net(
            branch_feat, t, r, dt_dr, kappa, orb_mask, Z,
            n_principal=n_principal, l_orbital=l_orbital, z_eff_orb=z_eff_orb,
            analytic_nodes=analytic_nodes,
        )

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
            # Stage A Round 2: forward Laguerre-head outputs through to
            # the trainer (consumed by coeff_loss / q_corr_loss).
            "laguerre_coeffs": raw.get("laguerre_coeffs"),
            "laguerre_coeff_init": raw.get("laguerre_coeff_init"),
            "laguerre_lambdas": raw.get("laguerre_lambdas"),
            "laguerre_lambda_init": raw.get("laguerre_lambda_init"),
            "laguerre_deltaQ": raw.get("laguerre_deltaQ"),
            "cross_sections": None,
        }

        if return_ci or self.ci_enabled:
            csf_mask = batch.get("csf_mask")
            if csf_mask is None:
                csf_mask = orb_mask[:, :1].astype(bool)
                csf_mask = jnp.broadcast_to(csf_mask, (Z.shape[0], self.n_csf_max))
            else:
                csf_mask = csf_mask[:, : self.n_csf_max]

            slater_log_scale = self.param(
                "slater_log_scale",
                # B'''' fix: init 到 -1.0 (exp=0.37, 诊断最优值)
                # 原因 (per §13, §20):
                #  - -3.0 (exp=0.05): 210 epoch 不动, V_slater 实质为 0
                #  - -1.0 (exp=0.37): §13 诊断最优值, 一次性给到
                #  - 配合 multi_transform 10x lr (B' fix v2 移除), 让 slater_log_scale 真正能动
                lambda key, shape, dtype=jnp.float32: jnp.full(shape, -1.0, dtype=dtype),
                (len(self.k_list),),
            )
            Rk = compute_all_Rk_diagonal(P, Q, orb_mask, grid, k_list=self.k_list)
            Rk = Rk * jnp.exp(slater_log_scale)[None, :, None]
            # B' (Path A): 把 slater_log_scale 暴露给 trainer (注入 SCF 损失)
            out["slater_log_scale"] = slater_log_scale

            csf_to_orb = batch.get("csf_to_orb")
            if csf_to_orb is not None:
                csf_to_orb = csf_to_orb[:, : self.n_csf_max, : self.n_orb_max]

            C_ang = batch.get("C_ang")
            if C_ang is not None:
                C_ang = C_ang[:, : len(self.k_list), : self.n_csf_max, : self.n_csf_max]
                H = assemble_hamiltonian(E_orb, Rk, C_ang, csf_mask, csf_to_orb)
            else:
                H = assemble_hamiltonian_diagonal(E_orb, csf_mask, csf_to_orb)

            E_nist = batch.get("E_nist")
            nist_mask = batch.get("nist_mask")
            if self.nist_inject and E_nist is not None and nist_mask is not None:
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
            out["Rk"] = Rk
            out["transitions"] = compute_e1_transitions(E_csf, V_csf, P, Q, grid, csf_mask)
            if E_grid is not None:
                out["cross_sections"] = {"CE": collision_cross_section_ce(E_csf, E_grid)}

        return out


def _resolve_eps_degen_ev(ci_cfg) -> float:
    """Regularization for eigh: config in meV (preferred) or legacy eV → internal eV."""
    if ci_cfg is None:
        return 1e-6
    mev = getattr(ci_cfg, "eps_degen_meV", None)
    if mev is not None:
        return float(mev) / 1000.0
    return float(getattr(ci_cfg, "eps_degen_ev", 1e-6))


def build_model_and_params(cfg, grid: RadialGrid, key: jax.Array):
    model_cfg = getattr(cfg, "model", cfg)
    ci_cfg = getattr(cfg, "ci", None)
    coords = getattr(cfg, "coords", None)

    # Stage A Round 2 (Round-3 fix v2): no model-level Laguerre init
    # constants — `DeepONetDirac.__call__` computes the per-batch
    # analytic init from the (batched) (Z, n, l).

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
        k_list=tuple(int(k) for k in getattr(ci_cfg, "k_list", [0])) if ci_cfg else (0,),
        nist_inject=bool(getattr(ci_cfg, "nist_inject", False)) if ci_cfg else False,
        eps_degen_ev=_resolve_eps_degen_ev(ci_cfg),
        apply_lowdin=bool(getattr(model_cfg, "apply_lowdin", False)),
        use_hydrogenic_skeleton=bool(getattr(model_cfg, "use_hydrogenic_skeleton", True)),
        perturb_eps=float(getattr(model_cfg, "perturb_eps", 0.2)),
        use_zeff_warmstart=bool(getattr(model_cfg, "use_zeff_warmstart", False)),
        use_laguerre_basis=bool(getattr(model_cfg, "use_laguerre_basis", False)),
        K_max=int(getattr(model_cfg, "K_max", 9)),
        learn_lambda=bool(getattr(model_cfg, "learn_lambda", True)),
        perturb_scale_P=float(getattr(model_cfg, "perturb_scale_P", 0.05)),
        perturb_scale_Q=float(getattr(model_cfg, "perturb_scale_Q", 0.05)),
        # §13.1 / §13.2 of 17_generalized_laguerre_basis.md
        use_log_r_input=bool(getattr(model_cfg, "use_log_r_input", False)),
        log_r_scale=float(getattr(model_cfg, "log_r_scale", 1.0)),
        use_dual_trunk=bool(getattr(model_cfg, "use_dual_trunk", False)),
        d_trunk_high=int(getattr(model_cfg, "d_trunk_high", 128)),
        omega_0_high=float(getattr(model_cfg, "omega_0_high", 60.0)),
        lambda_split=float(getattr(model_cfg, "lambda_split", 4.0)),
        gate_temperature=float(getattr(model_cfg, "gate_temperature", 2.0)),
        # §13.9 of 17_generalized_laguerre_basis.md
        coeff_d_hidden=int(getattr(model_cfg, "coeff_d_hidden", 64)),
        lambda_d_hidden=int(getattr(model_cfg, "lambda_d_hidden", 32)),
        q_corr_d_hidden=int(getattr(model_cfg, "q_corr_d_hidden", 64)),
        # §13.11-C of 17_generalized_laguerre_basis.md
        use_hybrid_head=bool(getattr(model_cfg, "use_hybrid_head", False)),
        nodes_table_path=str(getattr(model_cfg, "nodes_table_path",
                                     "data_cache/laguerre_nodes_z1_26_n1_10.parquet")),
        hybrid_d_hidden_node=int(getattr(model_cfg, "hybrid_d_hidden_node", 64)),
        hybrid_d_hidden_ref=int(getattr(model_cfg, "hybrid_d_hidden_ref", 32)),
        hybrid_n_iter=int(getattr(model_cfg, "hybrid_n_iter", 3)),
        hybrid_step_size=float(getattr(model_cfg, "hybrid_step_size", 0.1)),
    )
    batch = _dummy_batch(int(getattr(model_cfg, "n_orb_max", 16)), int(getattr(model_cfg, "n_csf_max", 32)))
    # When the Laguerre basis is on, populate the dummy shell_table with
    # valid (n, l) for slot 0 so the analytic init has somewhere to write
    # a non-zero value (purely a smoke-test safety net; the forward path
    # is robust to all-zero shell_table because invalid slots are masked).
    if getattr(model_cfg, "use_laguerre_basis", False):
        n_orb_max = int(getattr(model_cfg, "n_orb_max", 16))
        shell = np.zeros((batch["shell_table"].shape[0], n_orb_max, 4), dtype=np.int32)
        shell[0, 0] = np.array([1, 0, 1, 1], dtype=np.int32)
        batch["shell_table"] = jnp.asarray(shell, dtype=jnp.int32)
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
