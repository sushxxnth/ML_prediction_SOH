"""Data-flow schematic of the three major models (main-text Fig. 2).

Shows how inputs flow through the framework's three trained model stages
- HERO (retrieval-augmented prediction + early warning)
- Hybrid PINN (causal mechanism attribution)
- Counterfactual optimizer (intervention selection)
with the shared PATT classifier supplying operating-mode context, and the
advisory layer emitting the final recommendation. Each stage is annotated
with the input it consumes and the output it produces.

Publication-grade redesign: restrained palette, soft card shadows for depth,
consistent typography, no brittle hardcoded equation numbers.

Output: Casual_Attribution_reports/figures/dataflow_overview.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "Casual_Attribution_reports/figures/dataflow_overview.png"

# --- restrained, cohesive palette -----------------------------------------
BLUE   = "#3b6ea5"   # HERO
VIOLET = "#5b4b8a"   # Hybrid PINN
AMBER  = "#b07d2b"   # Counterfactual optimizer
TEAL   = "#2a9d8f"   # PATT / advisory
GRAY   = "#9aa0a6"   # raw signals
INK    = "#1a1a2e"
INK2   = "#5b5b66"
SHADOW = "#d7d7dc"


def tint(hex_color, amount=0.90):
    """Light tint of a hex color for card fills."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return f"#{r:02x}{g:02x}{b:02x}"


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": INK,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

fig, ax = plt.subplots(figsize=(15.0, 7.8))
ax.set_xlim(0, 100)
ax.set_ylim(0, 60)
ax.axis("off")

BOX_Y, BOX_H = 25, 15
TOP = BOX_Y + BOX_H
IN_Y = TOP + 3.4
OUT_Y = BOX_Y - 3.6
MID = BOX_Y + BOX_H / 2


def _card(x, y, w, h, edge, face, lw=2.0, pad=0.6, rad=0.6):
    # soft drop shadow, then the card
    ax.add_patch(FancyBboxPatch((x + 0.45, y - 0.55), w, h,
                                boxstyle=f"round,pad={pad},rounding_size={rad}",
                                fc=SHADOW, ec="none", zorder=2))
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad={pad},rounding_size={rad}",
                                fc=face, ec=edge, lw=lw, zorder=3))


def model_box(x, w, tag, title, body, inp, out, edge):
    _card(x, BOX_Y, w, BOX_H, edge, tint(edge, 0.93))
    ax.text(x + w / 2, TOP - 2.1, tag, ha="center", va="top",
            fontsize=9, fontweight="bold", color=edge, zorder=4)
    ax.text(x + w / 2, TOP - 5.3, title, ha="center", va="top",
            fontsize=12, fontweight="bold", color=INK, zorder=4)
    ax.text(x + w / 2, BOX_Y + 4.7, body, ha="center", va="top",
            fontsize=8.4, color=INK2, zorder=4, linespacing=1.4)
    ax.text(x + w / 2, IN_Y, "in: " + inp, ha="center", va="center",
            fontsize=8.4, color=edge, style="italic", zorder=4, linespacing=1.3)
    ax.text(x + w / 2, OUT_Y, "out: " + out, ha="center", va="center",
            fontsize=8.4, color=INK, fontweight="bold", zorder=4, linespacing=1.3)


def small_box(x, w, title, body, edge, y=BOX_Y, h=BOX_H):
    _card(x, y, w, h, edge, tint(edge, 0.94), lw=1.7)
    ax.text(x + w / 2, y + h - 2.7, title, ha="center", va="top",
            fontsize=10.5, fontweight="bold", color=INK, zorder=4)
    ax.text(x + w / 2, y + h - 6.6, body, ha="center", va="top",
            fontsize=8.4, color=INK2, zorder=4, linespacing=1.4)


def arrow(x1, y1, x2, y2, color=INK2, lw=2.2, rad=None):
    cs = "arc3" if rad is None else f"arc3,rad={rad}"
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 connectionstyle=cs, mutation_scale=14,
                                 color=color, lw=lw, shrinkA=2, shrinkB=2, zorder=2))


ax.text(50, 58.0, "Data flow through the three trained model stages",
        ha="center", fontsize=14.5, fontweight="bold", color=INK)

small_box(1.5, 12.5, "Raw signals",
          "voltage, current,\ntemperature,\ncycle count", GRAY)
small_box(87.5, 11.0, "Advisory",
          "ranked user\nrecommendation", TEAL)

model_box(17, 19, "MODEL 1", "HERO",
          "retrieval + cross-attention\nover a memory bank of\nhistorical trajectories",
          "recent trajectory window", "SOH / RUL forecast;\nearly-warning trigger", BLUE)
model_box(40.5, 19, "MODEL 2", "Hybrid PINN",
          "expert priors + NN residual\n+ bounded physics heads\n(softmax attribution)",
          "features $x$ (9-d) +\ncontext $c$ (6-d)", "attribution $\\boldsymbol{\\alpha}$ (5 mech.)", VIOLET)
model_box(64, 21, "MODEL 3", "Counterfactual optimizer",
          "physics-based sensitivity\nmodel on $s'=$ apply$(s,i)$;\nrank by benefit $\\Delta(i)$",
          "attribution $\\boldsymbol{\\alpha}$ + candidates $\\mathcal{I}$", "best intervention $i^*$", AMBER)

for x1, x2 in [(14.0, 17), (36, 40.5), (59.5, 64), (85, 87.5)]:
    arrow(x1, MID, x2, MID)

# early-warning feedback arc (above the boxes)
ax.text(38, 53.6, "early-warning trigger invokes attribution",
        ha="center", fontsize=8.4, color=BLUE, style="italic")
arrow(26.5, TOP + 6.2, 50, TOP + 6.2, color=BLUE, lw=1.7, rad=-0.35)

# PATT shared component (bottom)
small_box(30, 40, "PATT  —  shared physics-aware classifier",
          "storage vs. cycling  $\\rightarrow$  sets the operating-mode entry of context $c$\n"
          "and selects which intervention pool the optimizer searches",
          TEAL, y=4, h=11)
arrow(48, 15, 50, BOX_Y, color=TEAL, lw=1.8)
arrow(60, 15, 73, BOX_Y, color=TEAL, lw=1.8)

fig.savefig(OUT, dpi=220, bbox_inches="tight", pad_inches=0.28)
print("wrote", OUT)
