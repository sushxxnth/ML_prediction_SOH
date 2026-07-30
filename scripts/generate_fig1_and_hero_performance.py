"""Regenerate Fig1.png and hero_performance.png with numbers verified against
the revised cce_paper.tex (honest zero-shot rebuild, 2026-07-09).

Every number below is sourced from the revised paper text:
- Zero-shot table (tab:zeroshot), reproducible from run_zeroshot_table_rebuild.py
  (seed 42) and the released HERO checkpoint (run_zeroshot_bank_injection.py):
    RUL MAE:  LSTM 146.4, GRU 154.9, CNN-LSTM 105.7, Transformer 98.2,
              MLP 183.1, RF 100.3, HERO 181.1
    SOH MAE:  13.9, 15.2, 7.9, 12.6, 17.4, 13.1, 9.7
    Adaptation rows: LSTM+CORAL 9.1/107.9, Transformer+CORAL 5.6/110.0,
              MLP+CORAL 6.0/111.1, MLP+MMD 6.4/111.5, Transformer+FT 7.6/110.8
- In-distribution SOH MAE 0.74% (held-out cells, training chemistries; l.570)
- Attribution: 96.0% in-distribution, 100% corroborated subset (abstract)
- Counterfactual: 34.6% avg reduction (abstract)
- PATT: 99.9\% held-out accuracy (l.884); the 99.2\% in the case studies is per-inference confidence
- Early warning: 121 cycles mean lead, F1 74.3%, recall 92.9%
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "/Users/sushanth.c/physics_informed_model/ML_prediction_SOH/Casual_Attribution_reports"

# ---- palette (dataviz reference, light mode) ----
BLUE   = "#2a78d6"   # emphasis / HERO
AQUA   = "#1baf7a"   # adaptation methods
VIOLET = "#4a3aa7"
GRAY   = "#b9b8b3"   # neutral baseline bars
INK    = "#0b0b0b"
INK2   = "#52514e"
SURF   = "#ffffff"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#d8d7d2",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "#ecebe7",
    "grid.linewidth": 0.7,
    "axes.axisbelow": True,
    "text.color": INK,
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "figure.facecolor": SURF,
    "axes.facecolor": SURF,
    "savefig.facecolor": SURF,
})

# ============================================================
# Figure 1: four-stage architecture overview
# ============================================================
fig, ax = plt.subplots(figsize=(11.0, 9.2))
ax.set_xlim(0, 100); ax.set_ylim(0, 100)
ax.axis("off"); ax.grid(False)

def box(x, y, w, h, fc, ec, lw=1.6, r=2.2):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, mutation_aspect=1.0)
    ax.add_patch(p)
    return p

def arrow(x1, y1, x2, y2, lw=3.2):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle="-|>,head_width=3.4,head_length=5.4",
                        color="#8f8e89", lw=lw, shrinkA=0, shrinkB=0)
    ax.add_patch(a)

# Title
ax.text(50, 97.0, "Physics-Informed Battery Health Management",
        ha="center", va="center", fontsize=19, fontweight="bold", color=INK)
ax.text(50, 92.9, "Four Core Processing Stages",
        ha="center", va="center", fontsize=13.5, color=INK2)

# Input box (left)
box(2.5, 55, 17, 26, "#e3f0e3", "#4d7a4d")
ax.text(11, 76.0, "Battery\nMeasurement\nData", ha="center", va="center",
        fontsize=11.5, fontweight="bold", color=INK)
ax.text(11, 63.5, "Voltage\nTemperature\nCurrent\nCycle count", ha="center",
        va="center", fontsize=10, color=INK2, linespacing=1.5)

# Processing container
box(24, 50, 73.5, 38, "#fbfbfa", "#3f3e3b", lw=2.0, r=2.6)
ax.text(60.75, 53.4, "Physics-Informed Processing", ha="center", va="center",
        fontsize=12, color=INK2)

stages = [
    ("HERO\nPredictor",        "0.74% SOH MAE\n(held-out cells,\nin-distrib.)",  "#dcecfb", BLUE),
    ("Causal\nAttribution",    "96.0%\nagreement\n(in-distrib.)",   "#e6e2f4", VIOLET),
    ("Counterfactual\nOptimizer", "34.6% avg\nmechanism\nreduction","#fdeecd", "#a86f00"),
    ("Advisory\nSystem",       "99.9%\nusage-mode\nclassification", "#fadedd", "#b23837"),
]
sx, sw, gap = 26.5, 16.0, 1.8
for i, (name, metric, fc, ec) in enumerate(stages):
    x = sx + i * (sw + gap)
    box(x, 57.5, sw, 26.5, fc, ec, lw=1.8)
    ax.text(x + sw/2, 78.6, name, ha="center", va="center",
            fontsize=11.5, fontweight="bold", color=INK, linespacing=1.25)
    ax.text(x + sw/2, 66.0, metric, ha="center", va="center",
            fontsize=10, color=INK2, linespacing=1.45)

# arrows: input -> container ; container -> outputs
arrow(20.0, 68, 23.6, 68)
arrow(42, 49.6, 25.5, 41.5)
arrow(79, 49.6, 74.5, 41.5)

# Key outputs (bottom-left)
box(2.5, 7, 45, 33, "#e3f0e3", "#4d7a4d")
ax.text(25, 35.6, "Key Outputs", ha="center", va="center",
        fontsize=12.5, fontweight="bold", color=INK)
outputs = ["1.  SOH / RUL prediction",
           "2.  Degradation-mechanism attribution",
           "3.  Counterfactual intervention scenarios",
           "4.  Early warning of accelerated fade",
           "5.  Actionable user guidance"]
for i, t in enumerate(outputs):
    ax.text(5.5, 30.4 - i * 4.6, t, ha="left", va="center",
            fontsize = 10.5, color=INK)

# Validated metrics (bottom-right)
box(52.5, 7, 45, 33, "#e3f0e3", "#4d7a4d")
ax.text(75, 35.6, "Validated Performance", ha="center", va="center",
        fontsize=12.5, fontweight="bold", color=INK)
metrics = [
    "SOH MAE 0.74% on held-out cells (in-distribution)",
    "Reproducible zero-shot protocol + DA baselines",
    "   (cross-chemistry transfer open for all methods)",
    "Attribution: 96.0% in-distrib., 100% corroborated",
    "Counterfactual: 34.6% avg mechanism reduction",
    "Early warning: 121-cycle mean lead  (F1 74.3%)",
    "Usage classification: 99.9% (held-out)",
]
for i, t in enumerate(metrics):
    ax.text(55.0, 31.6 - i * 3.7, t, ha="left", va="center",
            fontsize=9.4, color=INK)

fig.savefig(f"{OUT}/Fig1.png", dpi=220, bbox_inches="tight", pad_inches=0.25)
plt.close(fig)
print("Fig1.png written")

# ============================================================
# hero_performance.png : 2 honest panels (SOH, RUL)
#   values from reports/zeroshot_table_rebuild.json +
#   reports/zeroshot_bank_injection.json (HERO checkpoint)
# ============================================================
methods = ["CNN-\nLSTM", "Trans-\nformer", "RF", "LSTM", "GRU", "MLP",
           "HERO\n(Ours)",
           "Trans.\n+CORAL", "MLP\n+CORAL", "MLP\n+MMD", "Trans.\n+FT", "LSTM\n+CORAL"]
soh = [7.9, 12.6, 13.1, 13.9, 15.2, 17.4, 9.7, 5.6, 6.0, 6.4, 7.6, 9.1]
rul = [105.7, 98.2, 100.3, 146.4, 154.9, 183.1, 181.1, 110.0, 111.1, 111.5, 110.8, 107.9]
colors = [GRAY]*6 + [BLUE] + [AQUA]*5

fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.4))
fig.suptitle("Zero-Shot Chemistry Transfer (Train: LCO $\\rightarrow$ Test: TJU NCM/NCA)\n"
             "gray: zero-shot   ·   blue: HERO (zero-shot)   ·   green: + adaptation on one held-out target cell",
             fontsize=12.5, fontweight="bold", y=1.02)

axA = axes[0]
bars = axA.bar(methods, soh, color=colors, width=0.62, zorder=3)
for b, v in zip(bars, soh):
    axA.text(b.get_x() + b.get_width()/2, v + 0.25, f"{v:g}",
             ha="center", va="bottom", fontsize=8.5, color=INK)
axA.set_ylabel("SOH MAE (%)")
axA.set_title("A.  SOH Error", fontsize=11.5, fontweight="bold", loc="left")
axA.set_ylim(0, 19.5)
axA.tick_params(axis="x", labelsize=8)

axB = axes[1]
bars = axB.bar(methods, rul, color=colors, width=0.62, zorder=3)
for b, v in zip(bars, rul):
    axB.text(b.get_x() + b.get_width()/2, v + 2.4, f"{v:g}",
             ha="center", va="bottom", fontsize=8.5, color=INK)
axB.set_ylabel("RUL MAE (cycles, uncapped)")
axB.set_title("B.  RUL Error", fontsize=11.5, fontweight="bold", loc="left")
axB.set_ylim(0, 205)
axB.tick_params(axis="x", labelsize=8)

fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(f"{OUT}/hero_performance.png", dpi=220, bbox_inches="tight",
            pad_inches=0.2)
plt.close(fig)
print("hero_performance.png written")
