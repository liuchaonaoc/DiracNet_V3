#!/usr/bin/env python3
"""Analyze gate_a_detailed.csv: failure reasons + Phase1 vs Phase2 diff."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GATES = {
    "relaxed": {"cos": 0.95, "pde": 1.0, "e_meV": 50.0},
    "mid": {"cos": 0.98, "pde": 0.5, "e_meV": 50.0},
    "strict": {"cos": 0.99, "pde": 0.01, "e_meV": 50.0},
}


def _fail_reasons(row: pd.Series, th: dict) -> list[str]:
    reasons = []
    if row["cos"] < th["cos"]:
        reasons.append(f"cos<{th['cos']}")
    if row["pde"] > th["pde"]:
        reasons.append(f"pde>{th['pde']}")
    if row["dE_meV"] > th["e_meV"]:
        reasons.append(f"dE>{th['e_meV']}meV")
    return reasons if reasons else ["PASS"]


def _verdict(row: pd.Series, th: dict) -> str:
    return "PASS" if _fail_reasons(row, th) == ["PASS"] else "FAIL"


def analyze_failure_stats(df: pd.DataFrame, label: str, gate_name: str) -> pd.DataFrame:
    th = GATES[gate_name]
    rows = []
    for _, r in df.iterrows():
        reasons = _fail_reasons(r, th)
        rows.append({
            "phase": label,
            "gate": gate_name,
            "Z": int(r["Z"]),
            "element": r["element"],
            "n": int(r["n"]),
            "fail_reason": "+".join(reasons),
            "cos": r["cos"],
            "pde": r["pde"],
            "dE_meV": r["dE_meV"],
            "verdict": _verdict(r, th),
        })
    out = pd.DataFrame(rows)
    return out


def summarize_reasons(det: pd.DataFrame) -> pd.DataFrame:
    """Count failure reason combinations."""
    fail = det[det["verdict"] == "FAIL"]
    if fail.empty:
        return pd.DataFrame([{"fail_reason": "PASS", "count": len(det)}])
    g = fail.groupby(["phase", "gate", "fail_reason"]).size().reset_index(name="count")
    g = g.sort_values(["phase", "gate", "count"], ascending=[True, True, False])
    return g


def summarize_by_criterion(det: pd.DataFrame) -> pd.DataFrame:
    """Per criterion fail counts."""
    rows = []
    for (phase, gate), g in det.groupby(["phase", "gate"]):
        n = len(g)
        th = GATES[gate]
        rows.append({
            "phase": phase,
            "gate": gate,
            "n_total": n,
            "pass_all": int((g["verdict"] == "PASS").sum()),
            "fail_cos": int((g["cos"] < th["cos"]).sum()),
            "fail_pde": int((g["pde"] > th["pde"]).sum()),
            "fail_dE": int((g["dE_meV"] > th["e_meV"]).sum()),
            "pass_cos_only": int(((g["cos"] >= th["cos"]) & (g["verdict"] == "FAIL")).sum()),
            "pass_dE_only": int(((g["dE_meV"] <= th["e_meV"]) & (g["verdict"] == "FAIL")).sum()),
        })
    return pd.DataFrame(rows)


def build_diff(p1: pd.DataFrame, p2: pd.DataFrame, gate_name: str) -> pd.DataFrame:
    th = GATES[gate_name]
    key = ["Z", "n"]
    m = p1.merge(p2, on=key, suffixes=("_p1", "_p2"))
    m["gate"] = gate_name
    m["cos_delta"] = m["cos_p2"] - m["cos_p1"]
    m["pde_delta"] = m["pde_p2"] - m["pde_p1"]
    m["dE_delta_meV"] = m["dE_meV_p2"] - m["dE_meV_p1"]
    m["verdict_p1"] = m.apply(
        lambda r: _verdict(pd.Series({"cos": r["cos_p1"], "pde": r["pde_p1"], "dE_meV": r["dE_meV_p1"]}), th),
        axis=1,
    )
    m["verdict_p2"] = m.apply(
        lambda r: _verdict(pd.Series({"cos": r["cos_p2"], "pde": r["pde_p2"], "dE_meV": r["dE_meV_p2"]}), th),
        axis=1,
    )
    m["status"] = m.apply(_diff_status, axis=1)
    cols = [
        "gate", "Z", "element_p1", "n",
        "cos_p1", "cos_p2", "cos_delta",
        "pde_p1", "pde_p2", "pde_delta",
        "dE_meV_p1", "dE_meV_p2", "dE_delta_meV",
        "verdict_p1", "verdict_p2", "status",
    ]
    return m[cols].sort_values(["Z", "n"])


def _diff_status(r) -> str:
    if r["verdict_p1"] == "PASS" and r["verdict_p2"] == "PASS":
        return "PASS→PASS"
    if r["verdict_p1"] == "FAIL" and r["verdict_p2"] == "PASS":
        return "FAIL→PASS ✓"
    if r["verdict_p1"] == "PASS" and r["verdict_p2"] == "FAIL":
        return "PASS→FAIL ✗"
    # both fail — which metric got worse?
    worse = []
    if r["cos_delta"] < -0.01:
        worse.append("cos↓")
    if r["pde_delta"] > 0.05:
        worse.append("pde↑")
    if r["dE_delta_meV"] > 5:
        worse.append("dE↑")
    better = []
    if r["cos_delta"] > 0.01:
        better.append("cos↑")
    if r["pde_delta"] < -0.05:
        better.append("pde↓")
    if r["dE_delta_meV"] < -5:
        better.append("dE↓")
    tag = ",".join(better) if better else ""
    if worse:
        tag = (tag + " " if tag else "") + ",".join(worse)
    return f"FAIL→FAIL ({tag or 'mixed'})"


def write_markdown(
    out_path: Path,
    crit: pd.DataFrame,
    reason_summary: pd.DataFrame,
    diff_relaxed: pd.DataFrame,
    diff_strict: pd.DataFrame,
) -> None:
    lines = [
        "# Gate A 分析报告（自动生成）",
        "",
        "> 由 `scripts/v3_analyze_gate_reports.py` 生成。",
        "",
        "## 1. 分项失败统计（按阈值）",
        "",
        crit.to_markdown(index=False),
        "",
        "## 2. 失败原因组合频次",
        "",
        reason_summary.to_markdown(index=False),
        "",
        "## 3. Phase 1 vs Phase 2 逐行对比（放宽 Gate）",
        "",
        "阈值: cos≥0.95, pde≤1.0, dE≤50 meV",
        "",
        diff_relaxed.to_markdown(index=False),
        "",
        "## 4. Phase 1 vs Phase 2 逐行对比（严格 Gate）",
        "",
        "阈值: cos≥0.99, pde≤0.01, dE≤50 meV",
        "",
        diff_strict.to_markdown(index=False),
        "",
        "## 5. 变化摘要",
        "",
    ]
    for gate_name, diff in [("relaxed", diff_relaxed), ("strict", diff_strict)]:
        n_improve = (diff["status"] == "FAIL→PASS ✓").sum()
        n_regress = (diff["status"] == "PASS→FAIL ✗").sum()
        lines.append(f"### {gate_name}")
        lines.append(f"- FAIL→PASS: **{n_improve}** 行")
        lines.append(f"- PASS→FAIL: **{n_regress}** 行")
        lines.append(f"- 仍 FAIL: **{(diff['status'].str.startswith('FAIL→FAIL')).sum()}** 行")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    p1_path = ROOT / "logs/v3_phase1_stage_a_z1_8/gate_a_detailed.csv"
    p2_path = ROOT / "logs/v3_phase1_stage_a_z1_8_phase2/gate_a_detailed.csv"
    out_dir = ROOT / "logs/gate_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not p1_path.exists():
        print(f"Missing {p1_path}")
        sys.exit(1)
    if not p2_path.exists():
        print(f"Missing {p2_path}")
        sys.exit(1)

    p1 = pd.read_csv(p1_path)
    p2 = pd.read_csv(p2_path)

    all_det = []
    for gate in ("relaxed", "mid", "strict"):
        all_det.append(analyze_failure_stats(p1, "phase1_1000ep", gate))
        all_det.append(analyze_failure_stats(p2, "phase2_1000ep", gate))
    det = pd.concat(all_det, ignore_index=True)
    det.to_csv(out_dir / "failure_reasons_per_row.csv", index=False)

    crit = summarize_by_criterion(det)
    crit.to_csv(out_dir / "failure_by_criterion.csv", index=False)

    reason_summary = summarize_reasons(det)
    reason_summary.to_csv(out_dir / "failure_reason_combos.csv", index=False)

    diff_relaxed = build_diff(p1, p2, "relaxed")
    diff_strict = build_diff(p1, p2, "strict")
    diff_relaxed.to_csv(out_dir / "phase1_vs_phase2_relaxed.csv", index=False)
    diff_strict.to_csv(out_dir / "phase1_vs_phase2_strict.csv", index=False)

    write_markdown(
        out_dir / "GATE_ANALYSIS.md",
        crit,
        reason_summary,
        diff_relaxed,
        diff_strict,
    )

    print(f"Wrote analysis to {out_dir}/")
    print("\n=== 分项失败统计 ===")
    print(crit.to_string(index=False))
    print("\n=== Phase2 放宽 Gate 通过 ===")
    p2r = det[(det["phase"] == "phase2_1000ep") & (det["gate"] == "relaxed") & (det["verdict"] == "PASS")]
    print(p2r[["Z", "element", "n", "cos", "pde", "dE_meV"]].to_string(index=False))
    print("\n=== FAIL→PASS (relaxed) ===")
    print(diff_relaxed[diff_relaxed["status"] == "FAIL→PASS ✓"][["Z", "n", "element_p1", "cos_delta", "pde_delta", "dE_delta_meV"]].to_string(index=False))


if __name__ == "__main__":
    main()
