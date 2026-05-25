#!/usr/bin/env python3
"""Analyze gate_a_detailed.csv: failure reasons + multi-phase diffs.

Supports Phase 1 / 2 / 3 comparison when corresponding CSV files exist.
Outputs under ``logs/gate_analysis/`` by default.
"""

from __future__ import annotations

import argparse
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

PHASE_PATHS = {
    "phase1_1000ep": ROOT / "logs/v3_phase1_stage_a_z1_8/gate_a_detailed.csv",
    "phase2_1000ep": ROOT / "logs/v3_phase1_stage_a_z1_8_phase2/gate_a_detailed.csv",
    "phase3_1000ep": ROOT / "logs/v3_phase1_stage_a_z1_8_phase3/gate_a_detailed.csv",
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
    return pd.DataFrame(rows)


def summarize_reasons(det: pd.DataFrame) -> pd.DataFrame:
    fail = det[det["verdict"] == "FAIL"]
    if fail.empty:
        return pd.DataFrame(columns=["phase", "gate", "fail_reason", "count"])
    g = fail.groupby(["phase", "gate", "fail_reason"]).size().reset_index(name="count")
    return g.sort_values(["phase", "gate", "count"], ascending=[True, True, False])


def summarize_by_criterion(det: pd.DataFrame) -> pd.DataFrame:
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


def build_pass_matrix(phases: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Rows = gate tier; columns = phase labels with pass counts."""
    rows = []
    for gate_name, th in GATES.items():
        row = {"gate": gate_name, "cos_th": th["cos"], "pde_th": th["pde"], "dE_meV_th": th["e_meV"]}
        for label, df in phases.items():
            det = analyze_failure_stats(df, label, gate_name)
            row[label] = int((det["verdict"] == "PASS").sum())
        rows.append(row)
    return pd.DataFrame(rows)


def build_diff(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    gate_name: str,
    *,
    label_a: str = "a",
    label_b: str = "b",
) -> pd.DataFrame:
    th = GATES[gate_name]
    key = ["Z", "n"]
    m = df_a.merge(df_b, on=key, suffixes=("_a", "_b"))
    m["gate"] = gate_name
    m["phase_a"] = label_a
    m["phase_b"] = label_b
    m["cos_delta"] = m["cos_b"] - m["cos_a"]
    m["pde_delta"] = m["pde_b"] - m["pde_a"]
    m["dE_delta_meV"] = m["dE_meV_b"] - m["dE_meV_a"]
    m["verdict_a"] = m.apply(
        lambda r: _verdict(
            pd.Series({"cos": r["cos_a"], "pde": r["pde_a"], "dE_meV": r["dE_meV_a"]}), th
        ),
        axis=1,
    )
    m["verdict_b"] = m.apply(
        lambda r: _verdict(
            pd.Series({"cos": r["cos_b"], "pde": r["pde_b"], "dE_meV": r["dE_meV_b"]}), th
        ),
        axis=1,
    )
    m["status"] = m.apply(_diff_status, axis=1)
    elem_col = "element_a" if "element_a" in m.columns else "element"
    cols = [
        "gate", "phase_a", "phase_b", "Z", elem_col, "n",
        "cos_a", "cos_b", "cos_delta",
        "pde_a", "pde_b", "pde_delta",
        "dE_meV_a", "dE_meV_b", "dE_delta_meV",
        "verdict_a", "verdict_b", "status",
    ]
    return m[cols].sort_values(["Z", "n"])


def _diff_status(r) -> str:
    if r["verdict_a"] == "PASS" and r["verdict_b"] == "PASS":
        return "PASS→PASS"
    if r["verdict_a"] == "FAIL" and r["verdict_b"] == "PASS":
        return "FAIL→PASS ✓"
    if r["verdict_a"] == "PASS" and r["verdict_b"] == "FAIL":
        return "PASS→FAIL ✗"
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


def _diff_summary(diff: pd.DataFrame, label_a: str, label_b: str, gate_name: str) -> list[str]:
    n_improve = int((diff["status"] == "FAIL→PASS ✓").sum())
    n_regress = int((diff["status"] == "PASS→FAIL ✗").sum())
    both_pass = int((diff["status"] == "PASS→PASS").sum())
    both_fail = int(diff["status"].str.startswith("FAIL→FAIL").sum())
    return [
        f"### {label_a} → {label_b}（{gate_name}）",
        f"- FAIL→PASS: **{n_improve}**",
        f"- PASS→FAIL: **{n_regress}**",
        f"- 两阶段都过: **{both_pass}**",
        f"- 仍 FAIL: **{both_fail}**",
        "",
    ]


def summarize_strict_by_Z(df: pd.DataFrame, phase_label: str) -> pd.DataFrame:
    th = GATES["strict"]
    rows = []
    for Z, g in df.groupby("Z"):
        sub = g.copy()
        pass_mask = (
            (sub["cos"] >= th["cos"])
            & (sub["pde"] <= th["pde"])
            & (sub["dE_meV"] <= th["e_meV"])
        )
        fail_n = sorted(sub.loc[~pass_mask, "n"].astype(int).tolist())
        rows.append({
            "phase": phase_label,
            "Z": int(Z),
            "element": str(sub["element"].iloc[0]),
            "strict_pass": int(pass_mask.sum()),
            "strict_total": len(sub),
            "fail_n": ",".join(str(x) for x in fail_n) if fail_n else "",
        })
    return pd.DataFrame(rows)


def write_markdown(
    out_path: Path,
    *,
    crit: pd.DataFrame,
    pass_matrix: pd.DataFrame,
    reason_summary: pd.DataFrame,
    strict_by_z: pd.DataFrame,
    diff_sections: list[str],
    diff_tables: list[tuple[str, pd.DataFrame]],
    phases_present: list[str],
) -> None:
    lines = [
        "# Gate A 分析报告（Phase 1 / 2 / 3）",
        "",
        "> 自动生成：`python scripts/v3_analyze_gate_reports.py`",
        "",
        f"> 包含阶段：{', '.join(phases_present)}",
        "",
        "## 1. 三阶段 Gate 通过数总览",
        "",
        pass_matrix.to_markdown(index=False),
        "",
        "## 2. 分项失败统计（按阶段 × 阈值）",
        "",
        crit.to_markdown(index=False),
        "",
        "## 3. strict Gate 按元素汇总",
        "",
        strict_by_z.to_markdown(index=False),
        "",
        "## 4. 失败原因组合频次",
        "",
    ]
    if reason_summary.empty:
        lines.append("（无 FAIL 行）")
    else:
        lines.append(reason_summary.to_markdown(index=False))
    lines.extend(["", "## 5. 逐行对比摘要", ""])
    lines.extend(diff_sections)
    for title, diff_df in diff_tables:
        lines.extend([
            f"### {title}",
            "",
            f"阈值见 Gate 表；共 {len(diff_df)} 行。完整 CSV 见 `logs/gate_analysis/`。",
            "",
        ])
        # Show only status changes + worst regressions in markdown (full data in CSV)
        changes = diff_df[diff_df["status"] != "PASS→PASS"]
        if len(changes) > 40:
            lines.append(f"*（仅展示 {len(changes)} 行中有变化的前 40 行；完整见 CSV）*")
            lines.append("")
            lines.append(changes.head(40).to_markdown(index=False))
        elif len(changes) > 0:
            lines.append(changes.to_markdown(index=False))
        else:
            lines.append("（无变化：全部 PASS→PASS）")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def load_phases(extra_paths: dict[str, Path] | None = None) -> dict[str, pd.DataFrame]:
    paths = {**PHASE_PATHS}
    if extra_paths:
        paths.update(extra_paths)
    phases: dict[str, pd.DataFrame] = {}
    for label, path in paths.items():
        if path.exists():
            phases[label] = pd.read_csv(path)
        else:
            print(f"Skip missing: {path}")
    if not phases:
        raise FileNotFoundError("No gate_a_detailed.csv found for any phase")
    return phases


def main():
    ap = argparse.ArgumentParser(description="Gate A multi-phase analysis")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "logs/gate_analysis")
    args = ap.parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    phases = load_phases()
    phase_labels = list(phases.keys())

    # Per-row failure detail (all phases × all gates)
    all_det = []
    for label, df in phases.items():
        for gate in GATES:
            all_det.append(analyze_failure_stats(df, label, gate))
    det = pd.concat(all_det, ignore_index=True)
    det.to_csv(out_dir / "failure_reasons_per_row.csv", index=False)

    crit = summarize_by_criterion(det)
    crit.to_csv(out_dir / "failure_by_criterion.csv", index=False)

    reason_summary = summarize_reasons(det)
    reason_summary.to_csv(out_dir / "failure_reason_combos.csv", index=False)

    pass_matrix = build_pass_matrix(phases)
    pass_matrix.to_csv(out_dir / "phase123_pass_matrix.csv", index=False)

    strict_parts = [summarize_strict_by_Z(df, label) for label, df in phases.items()]
    strict_by_z = pd.concat(strict_parts, ignore_index=True)
    strict_by_z.to_csv(out_dir / "strict_pass_by_element.csv", index=False)

    # Pairwise diffs for available phases
    diff_sections: list[str] = []
    diff_tables: list[tuple[str, pd.DataFrame]] = []
    pairs = [
        ("phase1_1000ep", "phase2_1000ep"),
        ("phase1_1000ep", "phase3_1000ep"),
        ("phase2_1000ep", "phase3_1000ep"),
    ]
    for gate in ("relaxed", "mid", "strict"):
        for la, lb in pairs:
            if la not in phases or lb not in phases:
                continue
            diff = build_diff(phases[la], phases[lb], gate, label_a=la, label_b=lb)
            fname = f"{la}_vs_{lb}_{gate}.csv"
            diff.to_csv(out_dir / fname, index=False)
            diff_sections.extend(_diff_summary(diff, la, lb, gate))
            if gate == "relaxed" and (la, lb) in (
                ("phase1_1000ep", "phase2_1000ep"),
                ("phase1_1000ep", "phase3_1000ep"),
                ("phase2_1000ep", "phase3_1000ep"),
            ):
                diff_tables.append((f"{la} vs {lb}（{gate}）", diff))

    write_markdown(
        out_dir / "GATE_ANALYSIS_ZH.md",
        crit=crit,
        pass_matrix=pass_matrix,
        reason_summary=reason_summary,
        strict_by_z=strict_by_z,
        diff_sections=diff_sections,
        diff_tables=diff_tables,
        phases_present=phase_labels,
    )
    # English alias for backward compat
    write_markdown(
        out_dir / "GATE_ANALYSIS.md",
        crit=crit,
        pass_matrix=pass_matrix,
        reason_summary=reason_summary,
        strict_by_z=strict_by_z,
        diff_sections=diff_sections,
        diff_tables=diff_tables,
        phases_present=phase_labels,
    )

    # Legacy filenames (phase1 vs phase2 only)
    if "phase1_1000ep" in phases and "phase2_1000ep" in phases:
        build_diff(phases["phase1_1000ep"], phases["phase2_1000ep"], "relaxed",
                   label_a="phase1_1000ep", label_b="phase2_1000ep").to_csv(
            out_dir / "phase1_vs_phase2_relaxed.csv", index=False
        )
        build_diff(phases["phase1_1000ep"], phases["phase2_1000ep"], "strict",
                   label_a="phase1_1000ep", label_b="phase2_1000ep").to_csv(
            out_dir / "phase1_vs_phase2_strict.csv", index=False
        )

    print(f"Wrote analysis to {out_dir}/")
    print("\n=== 三阶段 Gate 通过数 ===")
    print(pass_matrix.to_string(index=False))
    if "phase3_1000ep" in phases:
        p3 = phases["phase3_1000ep"]
        th = GATES["relaxed"]
        fail = p3[(p3.cos < th["cos"]) | (p3.pde > th["pde"]) | (p3.dE_meV > th["e_meV"])]
        if len(fail):
            print("\n=== Phase3 relaxed FAIL ===")
            print(fail[["Z", "element", "n", "cos", "pde", "dE_meV"]].to_string(index=False))


if __name__ == "__main__":
    main()
