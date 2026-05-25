"""Electron configuration string parsing (no torch)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

import numpy as np

_L_LETTER_TO_INT = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5, "i": 6}
_SHELL_PATTERN = re.compile(r"(?P<n>\d+)(?P<l>[spdfghi])(?P<j>\d+/2)?(?P<occ>\d+)?")


@dataclass(frozen=True)
class Shell:
    n: int
    l: int
    j: float
    occ: int

    @property
    def kappa(self) -> int:
        if self.j == self.l - 0.5:
            return self.l
        return -(self.l + 1)


def parse_config_string(s: str) -> list[Shell]:
    s = s.strip()
    if not s:
        return []
    shells: list[Shell] = []
    for tok in s.split():
        m = _SHELL_PATTERN.fullmatch(tok)
        if not m:
            raise ValueError(f"cannot parse shell token: {tok!r}")
        n = int(m.group("n"))
        l = _L_LETTER_TO_INT[m.group("l")]
        occ = int(m.group("occ") or "0")
        j_str = m.group("j")
        if j_str is not None:
            j_val = int(j_str.split("/")[0]) / 2.0
            shells.append(Shell(n=n, l=l, j=j_val, occ=occ))
        else:
            if l == 0:
                shells.append(Shell(n=n, l=0, j=0.5, occ=occ))
            else:
                cap_low = 2 * l
                cap_high = 2 * l + 2
                if occ <= cap_low:
                    shells.append(Shell(n=n, l=l, j=l - 0.5, occ=occ))
                else:
                    shells.append(Shell(n=n, l=l, j=l - 0.5, occ=cap_low))
                    shells.append(Shell(n=n, l=l, j=l + 0.5, occ=min(occ - cap_low, cap_high)))
    return shells


def encode_config_to_array(shells: Sequence[Shell], max_seq: int) -> np.ndarray:
    out = np.zeros((max_seq, 4), dtype=np.int32)
    for i, sh in enumerate(shells[:max_seq]):
        out[i, 0] = sh.n
        out[i, 1] = sh.l
        out[i, 2] = int(2 * sh.j)
        out[i, 3] = sh.occ
    return out
