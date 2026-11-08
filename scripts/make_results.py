"""Compute verdicts and RESULTS.md from results/raw_runs.json.

Verdict rules are the frozen ones in PREREGISTRATION.md:
H1: triton/eager bandwidth ratio at the verdict size at least 1.30.
H2: triton/compiled bandwidth ratio at the verdict size inside
    0.85 to 1.18.
H3: triton max abs error within 1e-5 at every size.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

H1_LINE = 1.30
H2_LO = 0.85
H2_HI = 1.18


def compute(raw: dict) -> dict:
    by_size = raw["results_by_size"]
    verdict_cols = str(raw["verdict_cols"])
    v = by_size[verdict_cols]
    ratio_eager = (v["triton"]["bandwidth_gb_s_median"]
                   / v["eager"]["bandwidth_gb_s_median"])
    ratio_compiled = (v["triton"]["bandwidth_gb_s_median"]
                      / v["compiled"]["bandwidth_gb_s_median"])
    h3_sizes = {c: bool(by_size[c]["triton"]["within_tolerance"])
                for c in by_size}
    return {
        "verdict_cols": raw["verdict_cols"],
        "ratio_triton_over_eager_at_verdict": ratio_eager,
        "ratio_triton_over_compiled_at_verdict": ratio_compiled,
        "h3_within_tolerance_by_size": h3_sizes,
        "verdicts": {
            "H1": "CONFIRMED" if ratio_eager >= H1_LINE else "REFUTED",
            "H2": "CONFIRMED" if H2_LO <= ratio_compiled <= H2_HI
                  else "REFUTED",
            "H3": "CONFIRMED" if all(h3_sizes.values()) else "REFUTED",
        },
    }


def fmt(x: float, digits: int = 4) -> str:
    return f"{x:.{digits}f}"


def render(raw: dict, summary: dict) -> str:
    by_size = raw["results_by_size"]
    device = raw["device"]
    lines = []
    lines.append("# RESULTS")
    lines.append("")
    lines.append(f"Device of record: BlackBox, {device} (8 GB VRAM). "
                 f"Environment: torch {raw['torch']}, triton "
                 f"{raw['triton']}, Python on the machine of record. "
                 f"Rows {raw['rows']}, float32, "
                 f"{raw['warmup_calls']} warmup calls then "
                 f"{raw['timed_calls']} timed calls per arm per size, "
                 "CUDA events around the call only.")
    lines.append("")
    lines.append("| Cols | Arm | Median ms (IQR) | GB/s | Max abs err |")
    lines.append("| --- | --- | --- | --- | --- |")
    for cols in raw["cols_ladder"]:
        arms = by_size[str(cols)]
        for arm in ("eager", "compiled", "triton"):
            r = arms[arm]
            lines.append(
                f"| {cols} | {arm} | {fmt(r['time_ms_median'])} "
                f"({fmt(r['time_ms_q1'])} to {fmt(r['time_ms_q3'])}) "
                f"| {fmt(r['bandwidth_gb_s_median'], 1)} "
                f"| {r['max_abs_err_vs_eager']:.3g} |")
    lines.append("")
    vc = summary["verdict_cols"]
    lines.append(f"Ratios at the verdict size ({vc} columns): triton "
                 f"over eager bandwidth "
                 f"{fmt(summary['ratio_triton_over_eager_at_verdict'], 3)}; "
                 f"triton over compiled bandwidth "
                 f"{fmt(summary['ratio_triton_over_compiled_at_verdict'], 3)}.")
    lines.append("")
    lines.append("## Verdicts (frozen rules, PREREGISTRATION.md)")
    lines.append("")
    vd = summary["verdicts"]
    lines.append(f"- H1 (triton beats eager by at least 1.30x at "
                 f"{vc}): {vd['H1']}.")
    lines.append(f"- H2 (triton within 0.85 to 1.18 of compiled at "
                 f"{vc}): {vd['H2']}.")
    lines.append(f"- H3 (triton within 1e-5 of the eager reference at "
                 f"every size): {vd['H3']}.")
    lines.append("")
    lines.append("Context, not scored: the compiled arm's first-call "
                 "wall time per size (compilation included) was "
                 + ", ".join(
                     f"{c} cols: "
                     f"{by_size[str(c)]['compiled']['first_call_wall_s']:.2f} s"
                     for c in raw["cols_ladder"]) + ".")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    raw = json.loads((ROOT / "results" / "raw_runs.json").read_text())
    summary = compute(raw)
    (ROOT / "results" / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    (ROOT / "RESULTS.md").write_text(render(raw, summary))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
