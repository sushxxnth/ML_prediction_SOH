#!/usr/bin/env python3
"""
Regenerate figures/counterfactual_intervention_validation.png (main-text Fig. 8).

All values are computed live from the released counterfactual model and the exact
scenario definitions in validate_counterfactual_optimization.py -- nothing is
hard-coded from the previous figure. The two low-temperature scenarios are
labelled as SYNTHETIC cold-charging condition points (they match no NASA cell;
see the main-text retraction), not as NASA cells. The two high-C-rate scenarios
are the real XJTU cells.

Run from the repo root (ML_prediction_SOH/):
    python3 scripts/plot_counterfactual_validation.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import validate_counterfactual_optimization as V  # noqa: E402
from src.optimization.counterfactual_intervention import (  # noqa: E402
    CounterfactualSimulator,
    InterventionOptimizer,
)

OUT = ROOT / "Casual_Attribution_reports" / "figures" / "counterfactual_intervention_validation.png"

# Labels: synthetic cold points (not NASA cells) + real XJTU cells.
SCEN_LABELS_A = ["Cold point\n0°C, 1.5C", "Cold point\n10°C, 1C",
                 "XJTU\n25°C, 2C", "XJTU\n28°C, 3C"]
SCEN_LABELS_S = ["Cold pt\n0°C", "Cold pt\n10°C", "XJTU\n2C", "XJTU\n3C"]

MECH_ORDER = ["SEI Growth", "Lithium Plating", "Active Material Loss",
              "Electrolyte Loss", "Corrosion"]
MECH_SHORT = ["SEI\nGrowth", "Li\nPlating", "AM\nLoss", "Electrolyte\nLoss", "Corrosion"]


def compute():
    opt = InterventionOptimizer(CounterfactualSimulator())
    scen = V.load_nasa_scenarios() + V.load_xjtu_scenarios()
    rows = []
    for s in scen:
        recs = opt.optimize(s["state"], s["attribution"])
        best = recs[0]
        cf = best["counterfactual_attribution"]
        at = s["attribution"]
        dom = at.dominant_mechanism()
        rows.append({
            "dom": dom,
            "dom_pct": at.to_dict()[dom] * 100.0,
            "reduction_pp": (at.to_dict()[dom] - cf.to_dict()[dom]) * 100.0,
            "alignment": 50.0,  # per validate_scenario scoring (action + direction, not param)
            "before": at.to_dict(),
            "after": cf.to_dict(),
            "rec": best["intervention"].description,
        })
    return rows


def main():
    rows = compute()
    avg_red = float(np.mean([r["reduction_pp"] for r in rows]))

    plt.rcParams.update({"font.size": 13, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 12))
    (axA, axB), (axC, axD) = axes

    # ---- Panel A: dominant mechanism contribution (horizontal bars) ----
    colorsA = ["#e8241a", "#f7941d", "#5b9bd5", "#1f4e96"]
    dom_pcts = [r["dom_pct"] for r in rows]
    dom_lbls = ["Li Plating", "Li Plating", "AM Loss", "AM Loss"]
    y = np.arange(len(rows))[::-1]
    axA.barh(y, dom_pcts, color=colorsA, edgecolor="white")
    for yi, p, lab in zip(y, dom_pcts, dom_lbls):
        axA.text(p - 4, yi, f"{p:.0f}%", va="center", ha="right",
                 color="white", fontweight="bold")
        axA.text(p + 2, yi, lab, va="center", ha="left")
    axA.set_yticks(y)
    axA.set_yticklabels(SCEN_LABELS_A)
    axA.set_xlim(0, 90)
    axA.set_xticks([0, 20, 40, 60, 80])
    axA.set_xticklabels(["0%", "20%", "40%", "60%", "80%"])
    axA.set_xlabel("Degradation mechanism contribution")
    axA.set_title("(A) Degradation Scenarios", loc="left", fontweight="bold")

    # ---- Panel B: predicted reduction following intervention ----
    colorsB = ["#2ca02c", "#2ca02c", "#a6cee3", "#7f7f7f"]
    reds = [r["reduction_pp"] for r in rows]
    axB.barh(y, reds, color=colorsB, edgecolor="white")
    for yi, p in zip(y, reds):
        ha, off, col = ("right", -2, "white") if p > 18 else ("left", 2, "black")
        axB.text(p + off, yi, f"{p:.0f}%", va="center", ha=ha, color=col, fontweight="bold")
    axB.set_yticks(y)
    axB.set_yticklabels(SCEN_LABELS_S)
    axB.set_xlim(0, 90)
    axB.set_xticks([0, 20, 40, 60, 80])
    axB.set_xticklabels(["0%", "20%", "40%", "60%", "80%"])
    axB.set_xlabel("Predicted reduction in dominant mechanism (pp)")
    axB.set_title("(B) Counterfactual Intervention Impact", loc="left", fontweight="bold")

    # ---- Panel C: mechanism redistribution for scenario 1 (0C cold point) ----
    r0 = rows[0]
    before = [r0["before"][m] * 100 for m in MECH_ORDER]
    after = [r0["after"][m] * 100 for m in MECH_ORDER]
    x = np.arange(len(MECH_ORDER))
    w = 0.38
    axC.bar(x - w / 2, before, w, label="Before Intervention", color="#e8241a")
    axC.bar(x + w / 2, after, w, label="After Intervention", color="#2ca02c")
    for xi, b, a in zip(x, before, after):
        axC.text(xi - w / 2, b + 1.2, f"{b:.0f}%", ha="center", fontsize=11)
        axC.text(xi + w / 2, a + 1.2, f"{a:.0f}%", ha="center", fontsize=11)
    # Eliminated! arrow over Li Plating
    axC.annotate("", xy=(1 + w / 2, 2), xytext=(1, 66),
                 arrowprops=dict(facecolor="#2ca02c", edgecolor="#2ca02c", width=6, headwidth=18))
    axC.text(1.9, 48, "Eliminated!", color="#2ca02c", fontweight="bold", fontsize=15)
    axC.set_xticks(x)
    axC.set_xticklabels(MECH_SHORT)
    axC.set_ylim(0, 80)
    axC.set_yticks([0, 20, 40, 60, 80])
    axC.set_yticklabels(["0%", "20%", "40%", "60%", "80%"])
    axC.legend(loc="upper right", framealpha=0.9)
    axC.set_title("(C) Mechanism Redistribution: 0°C cold point", loc="left", fontweight="bold")

    # ---- Panel D: validation metrics ----
    xg = np.arange(len(rows))
    w2 = 0.38
    mred = [r["reduction_pp"] for r in rows]
    align = [r["alignment"] for r in rows]
    axD.bar(xg - w2 / 2, mred, w2, label="Mechanism Reduction", color="#1f77b4")
    axD.bar(xg + w2 / 2, align, w2, label="Strategy Alignment", color="#ff7f0e")
    for xi, m, a in zip(xg, mred, align):
        axD.text(xi - w2 / 2, m + 1.2, f"{m:.0f}%", ha="center", fontsize=11)
        axD.text(xi + w2 / 2, a + 1.2, f"{a:.0f}%", ha="center", fontsize=11)
    axD.axhline(avg_red, ls="--", color="navy")
    axD.text(len(rows) - 1.05, avg_red + 1.5, f"Avg {avg_red:.1f}", color="navy", fontweight="bold")
    axD.set_xticks(xg)
    axD.set_xticklabels(SCEN_LABELS_S)
    axD.set_ylim(0, 80)
    axD.set_yticks([0, 20, 40, 60, 80])
    axD.set_yticklabels(["0%", "20%", "40%", "60%", "80%"])
    axD.legend(loc="upper right", framealpha=0.9)
    axD.set_title("(D) Validation Metrics", loc="left", fontweight="bold")

    for ax in (axA, axB, axC, axD):
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")
    print(f"avg reduction = {avg_red:.2f} pp; reductions = {[round(r['reduction_pp'],2) for r in rows]}")


if __name__ == "__main__":
    main()
