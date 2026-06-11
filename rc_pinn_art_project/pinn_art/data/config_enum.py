"""Configuration enumeration & two-group classification (Round 2, no torch).

Pure-Python helpers shared by ``scripts/v3_enumerate_configs.py`` and
``scripts/v3_prepare_nist_manifest.py``:

- Aufbau (Madelung) ground configurations for an arbitrary electron count.
- Single-electron (valence) excitation enumeration up to ``n_max``.
- The ``single_valence`` / ``multi_electron`` grouping rule (see
  ``prompts/09_data_pipeline.md`` §7 and ``prompts/16_stage_a_selfconsistent_dfs.md`` R0.3).

A configuration is represented as a list of ``(n, l, occ)`` subshell tuples
(``nl`` resolution, *not* j-split), matching the manifest ``level_config``
strings such as ``"1s2 2s2 2p1"``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_L_LETTERS = "spdfghi"  # l = 0..6
_L_LETTER_TO_INT = {ch: i for i, ch in enumerate(_L_LETTERS)}
_L_MAX = len(_L_LETTERS) - 1  # 6 (i orbital): parser/letter ceiling

# Spectroscopic term letters indexed by total L (here L == l of the active e-)
_TERM_LETTERS = "SPDFGHI"

Shell = tuple[int, int, int]  # (n, l, occ)

_SHELL_TOKEN_RE = re.compile(r"(\d+)([spdfghi])(\d+)?", re.IGNORECASE)


def subshell_capacity(l: int) -> int:
    """Full occupancy of an ``nl`` subshell = 2(2l+1)."""
    return 2 * (2 * l + 1)


def madelung_order(n_max: int) -> list[tuple[int, int]]:
    """Orbital filling order (n+l, then n) up to ``n_max``, capped at l<=6."""
    orbs = [
        (n, l)
        for n in range(1, n_max + 1)
        for l in range(0, min(n, _L_MAX + 1))
    ]
    orbs.sort(key=lambda nl: (nl[0] + nl[1], nl[0]))
    return orbs


def ground_config(nele: int, n_max: int = 10) -> list[Shell]:
    """Aufbau ground configuration for ``nele`` electrons."""
    if nele <= 0:
        return []
    shells: list[Shell] = []
    remaining = nele
    for (n, l) in madelung_order(n_max):
        if remaining <= 0:
            break
        occ = min(subshell_capacity(l), remaining)
        shells.append((n, l, occ))
        remaining -= occ
    if remaining > 0:
        raise ValueError(f"cannot place {nele} electrons within n_max={n_max}")
    return shells


def shell_token(n: int, l: int, occ: int) -> str:
    return f"{n}{_L_LETTERS[l]}{occ}"


def config_to_string(shells: list[Shell]) -> str:
    """Canonical ``"1s2 2s2 2p1"`` string (sorted by (n,l), occ>0 only)."""
    toks = [shell_token(n, l, occ) for (n, l, occ) in sorted(shells) if occ > 0]
    return " ".join(toks)


def parse_config_tuples(s: str) -> list[Shell]:
    """Parse a config string into canonical ``(n, l, occ)`` tuples.

    Tolerant of NIST-normalized strings; implicit occ defaults to 1.
    Returns ``[]`` on any unparseable token (caller decides to skip).
    """
    s = (s or "").strip().lower().replace(",", " ")
    if not s:
        return []
    acc: dict[tuple[int, int], int] = {}
    for tok in s.split():
        m = _SHELL_TOKEN_RE.fullmatch(tok)
        if not m:
            return []
        n = int(m.group(1))
        l = _L_LETTER_TO_INT[m.group(2).lower()]
        occ = int(m.group(3)) if m.group(3) else 1
        acc[(n, l)] = acc.get((n, l), 0) + occ
    return [(n, l, occ) for (n, l), occ in sorted(acc.items())]


def complete_core(shells: list[Shell], nele: int, n_max: int = 10) -> list[Shell]:
    """Prepend the omitted Aufbau closed core so the config holds ``nele`` e-.

    NIST exports inconsistently drop the inner closed core (e.g. C I lists
    ``2s2 2p2`` instead of ``1s2 2s2 2p2``). Given the known electron count we
    refill the lowest *closed* subshells that are absent until the count matches.
    A no-op when the config is already complete.
    """
    present = {(n, l): occ for (n, l, occ) in shells}
    missing = nele - sum(present.values())
    if missing <= 0:
        return [(n, l, occ) for (n, l), occ in sorted(present.items())]
    for (n, l) in madelung_order(n_max):
        if missing <= 0:
            break
        if (n, l) in present:
            continue
        cap = subshell_capacity(l)
        if cap <= missing:
            present[(n, l)] = cap
            missing -= cap
    return [(n, l, occ) for (n, l), occ in sorted(present.items())]


def canonical_config(s: str, nele: int | None = None) -> str:
    """Canonicalize an arbitrary config string for cross-source matching.

    When ``nele`` is given, the omitted inner closed core is refilled first so
    that NIST strings (which may drop the core) line up with full enumerations.
    """
    shells = parse_config_tuples(s)
    if nele is not None and shells:
        shells = complete_core(shells, nele)
    return config_to_string(shells)


def parity_of(shells: list[Shell]) -> int:
    """Configuration parity: 0 = even, 1 = odd (= (-1)^Σ l*occ)."""
    return sum(l * occ for (_, l, occ) in shells) % 2


def n_open_electrons(shells: list[Shell]) -> int:
    """Electrons sitting in *unfilled* subshells (closed shells contribute 0)."""
    return sum(occ for (_, l, occ) in shells if 0 < occ < subshell_capacity(l))


def classify_group(shells: list[Shell]) -> str:
    """``single_valence`` iff exactly one electron lives outside closed shells."""
    return "single_valence" if n_open_electrons(shells) == 1 else "multi_electron"


def _open_shell(shells: list[Shell]) -> Shell | None:
    """Return the single open subshell if there is exactly one, else None."""
    open_shells = [s for s in shells if 0 < s[2] < subshell_capacity(s[1])]
    return open_shells[0] if len(open_shells) == 1 else None


def twice_j_values(l: int) -> list[int]:
    """Single-electron 2j values for orbital angular momentum ``l``."""
    if l == 0:
        return [1]
    return [2 * l - 1, 2 * l + 1]


def term_symbol(l: int, multiplicity: int = 2) -> str:
    """Best-effort spectroscopic term, e.g. l=1 -> ``"2P"`` (single active e-)."""
    return f"{multiplicity}{_TERM_LETTERS[l]}" if 0 <= l <= _L_MAX else ""


@dataclass(frozen=True)
class Level:
    """One enumerated atomic level (fine-structure resolved when meaningful)."""

    Z: int
    ion_charge: int
    nele: int
    parent_config: str   # ground configuration string
    level_config: str    # this level's configuration string
    twice_j: int         # 2*J; -1 == unspecified (multi_electron placeholder)
    parity: int
    group: str
    term: str
    n_active: int        # principal n of the active (excited/valence) orbital
    l_active: int        # l of the active orbital (-1 if none)
    is_ground: bool

    @property
    def J(self) -> float:
        return float("nan") if self.twice_j < 0 else self.twice_j / 2.0


def _levels_for_config(
    Z: int,
    ion_charge: int,
    nele: int,
    parent_str: str,
    shells: list[Shell],
    active: Shell | None,
    is_ground: bool,
) -> list[Level]:
    """Expand one configuration into fine-structure (J) resolved Level rows."""
    group = classify_group(shells)
    parity = parity_of(shells)
    cfg_str = config_to_string(shells)

    if group == "single_valence" and active is not None:
        _, l_a, _ = active
        n_a = active[0]
        out = []
        for tj in twice_j_values(l_a):
            out.append(Level(
                Z=Z, ion_charge=ion_charge, nele=nele,
                parent_config=parent_str, level_config=cfg_str,
                twice_j=tj, parity=parity, group=group,
                term=term_symbol(l_a, 2), n_active=n_a, l_active=l_a,
                is_ground=is_ground,
            ))
        return out

    # multi_electron (open core / >1 valence e-) or closed-shell ground:
    # J/term require full angular coupling -> left to downstream CI (R2).
    n_a = active[0] if active is not None else -1
    l_a = active[1] if active is not None else -1
    closed = (active is None) and (n_open_electrons(shells) == 0)
    return [Level(
        Z=Z, ion_charge=ion_charge, nele=nele,
        parent_config=parent_str, level_config=cfg_str,
        twice_j=0 if closed else -1, parity=parity, group=group,
        term="1S" if closed else "", n_active=n_a, l_active=l_a,
        is_ground=is_ground,
    )]


def enumerate_levels(
    Z: int,
    ion_charge: int,
    n_max: int = 10,
) -> list[Level]:
    """Ground + single-(valence)-electron excitations for one (Z, ion).

    The outermost occupied electron is promoted to every orbital ranked above
    it in Madelung order (up to ``n_max``, l<=6). Fine structure (J) is resolved
    for ``single_valence`` levels; ``multi_electron`` levels carry a single
    placeholder row (J unspecified) for downstream CI.
    """
    nele = Z - ion_charge
    if nele <= 0:
        return []
    g_shells = ground_config(nele, n_max=n_max)
    parent_str = config_to_string(g_shells)
    order = madelung_order(n_max)
    rank = {nl: i for i, nl in enumerate(order)}

    # Valence = occupied subshell with the largest Madelung rank.
    valence = max(
        (s for s in g_shells if s[2] > 0),
        key=lambda s: rank.get((s[0], s[1]), -1),
    )
    v_n, v_l, _ = valence

    levels: list[Level] = []
    levels.extend(_levels_for_config(
        Z, ion_charge, nele, parent_str, g_shells,
        active=_open_shell(g_shells), is_ground=True,
    ))

    seen: set[tuple[str, int, int]] = {
        (lv.level_config, lv.twice_j, lv.parity) for lv in levels
    }
    v_rank = rank[(v_n, v_l)]
    occ_map = {(n, l): occ for (n, l, occ) in g_shells}

    for (tn, tl) in order:
        if rank[(tn, tl)] <= v_rank:
            continue  # only promote upward
        if occ_map.get((tn, tl), 0) >= subshell_capacity(tl):
            continue  # target already full
        # Build excited configuration: valence -1, target +1.
        new_occ = dict(occ_map)
        new_occ[(v_n, v_l)] = new_occ.get((v_n, v_l), 0) - 1
        new_occ[(tn, tl)] = new_occ.get((tn, tl), 0) + 1
        ex_shells = [
            (n, l, occ) for (n, l), occ in sorted(new_occ.items()) if occ > 0
        ]
        for lv in _levels_for_config(
            Z, ion_charge, nele, parent_str, ex_shells,
            active=(tn, tl, new_occ[(tn, tl)]), is_ground=False,
        ):
            key = (lv.level_config, lv.twice_j, lv.parity)
            if key in seen:
                continue
            seen.add(key)
            levels.append(lv)
    return levels


def enumerate_all(
    z_min: int = 1,
    z_max: int = 26,
    n_max: int = 10,
    ion_charges: str = "all",
) -> list[Level]:
    """Enumerate levels for a Z range.

    ``ion_charges``:
      - ``"all"``    : every ion stage (electron count 1..Z),
      - ``"neutral"``: only neutral atoms (ion_charge=0),
      - ``"bare"``   : only fully stripped (hydrogen-like, ion_charge=Z-1).
    """
    out: list[Level] = []
    for Z in range(z_min, z_max + 1):
        if ion_charges == "neutral":
            charges = [0]
        elif ion_charges == "bare":
            charges = [Z - 1]
        else:
            charges = list(range(0, Z))  # nele = 1..Z
        for q in charges:
            out.extend(enumerate_levels(Z, q, n_max=n_max))
    return out
