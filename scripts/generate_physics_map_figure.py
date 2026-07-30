"""'Where the physics lives' consolidated map figure (main-text Fig. 3).

Every governing equation mapped to the architectural component it enters, the
learned parameter it produces (with bounds), and what it guarantees at inference
vs. what it shapes only during training.

Content verified against the implementation:
  src/models/pinn_physics_module.py     (equations, parameter bounds, residuals)
  src/models/pinn_causal_attribution.py (priors, hard plating gate, softmax)
and against cce_paper.tex learned values (Pure-PINN fit beta=1.48, gamma=0.52).

House-style redesign to match the paper's data figures: flat fills, no drop
shadows, the paper's blue/navy/teal/gray palette, straight connectors, bold
sans-serif labels.

Output: Casual_Attribution_reports/figures/physics_map.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "Casual_Attribution_reports/figures/physics_map.png"

# --- palette taken from the paper's other figures -------------------------
BLUE = "#4C9BC0"   # governing equations
NAVY = "#1C3D5F"   # structural rules / model components
TEAL = "#2CA089"   # hard constraint / inference guarantees
GRAY = "#8A8F96"   # neutral I/O and scope
INK  = "#1A1A1A"
INK2 = "#454545"
FILL = "#FFFFFF"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": INK,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

fig, ax = plt.subplots(figsize=(14.2, 9.6))
ax.set_xlim(0, 100)
ax.set_ylim(2, 103)
ax.axis("off")


def box(x, y, w, h, title, body, edge=GRAY, fs_title=10, fs_body=8.4, lw=1.7):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.5,rounding_size=0.28",
                                fc=FILL, ec=edge, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top",
            fontsize=fs_title, fontweight="bold", color=INK, zorder=3)
    ax.text(x + w / 2, y + h - 5.2, body, ha="center", va="top",
            fontsize=fs_body, color=INK2, linespacing=1.5, zorder=3)


def arrow(x1, y1, x2, y2, color=INK2, lw=1.9):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle="-|>", mutation_scale=12,
                                 color=color, lw=lw,
                                 shrinkA=1, shrinkB=1, zorder=1))


# ---- column headers
for xx, label in [(15, "Governing physics"), (50, "Where it enters"),
                  (85, "What it buys")]:
    ax.text(xx, 102, label, ha="center", fontsize=13,
            fontweight="bold", color=INK)

# ---- left column: physics boxes
box(2, 84, 26, 14.5, "Arrhenius + $\\sqrt{t}$ diffusion (SEI)",
    "$Q_{SEI}=k\\,\\sqrt{t}\\,e^{-E_a/2RT}f(SOC)$\n"
    "bound: $E_a\\in[35,60]$ kJ/mol, $k_{SEI}>0$\n"
    "(bounded head, structural)", edge=BLUE)
box(2, 68, 26, 14.5, "Butler–Volmer kinetics (plating)",
    "$Q_{pl}=k\\,\\sigma(\\frac{T_{crit}-T}{\\delta})\\,e^{(T_{ref}-T)/\\tau}\\,C^{\\alpha_p}$\n"
    "bound: $\\alpha_p\\in[0.3,0.7]$, $k_p>0$\n"
    "(bounded head, structural)", edge=BLUE)
box(2, 52, 26, 14.5, "Fatigue power law (AM loss)",
    "$Q_{AM}=k\\,C^{\\beta}N^{\\gamma}$\n"
    "bound: $\\beta\\in[1,2]$, $\\gamma\\in[0.3,1]$\n"
    "(Pure-PINN fit: $\\beta$=1.48, $\\gamma$=0.52)", edge=BLUE)
box(2, 38.5, 26, 12, "Mass conservation",
    "$\\sum_i \\alpha_i = 1$ (softmax)\n"
    "structural: fade split sums to $1-SOH$", edge=NAVY)
box(2, 22.5, 26, 14.5, "Expert condition rules (5 mech.)",
    "frozen logit priors from\n(T, C-rates, SOC, mode);\n"
    "plating < 10 °C, corrosion SOC < 25%", edge=NAVY)
box(2, 8, 26, 13, "No current $\\Rightarrow$ no plating",
    "hard constraint: plating logit\n$\\to -10^4$ whenever mode = storage",
    edge=TEAL)

# ---- middle column: architecture flow
box(38, 90, 24, 8.5, "Inputs", "features (9-d) + context (6-d)", edge=GRAY)
box(38, 78, 24, 9, "Encoders + fusion", "fused representation $h$", edge=GRAY)
box(33.5, 60, 15.5, 14.5, "Physics-param.\nnetwork",
    "\n\nbounded heads:\n$E_a, k, \\alpha, \\beta, \\gamma$", fs_title=9.3, edge=NAVY)
box(51, 60, 15.5, 14.5, "Mechanism\nheads (5)",
    "\n\nNN logits\n$\\mathbf{z}_{NN}$", fs_title=9.3, edge=NAVY)
box(38, 44.5, 24, 10.5, "Prior injection",
    "$\\mathbf{z}_{prior}$ (frozen rules) $+\\;\\mathbf{z}_{NN}$", edge=NAVY)
box(38, 30, 24, 9.5, "Hard plating gate", "mask plating logit in storage", edge=TEAL)
box(38, 13.5, 24, 11.5, "Softmax attribution",
    "$\\boldsymbol{\\alpha}$: SEI, plating, AM loss,\nelectrolyte, corrosion", edge=NAVY)

# flow arrows (middle)
for a in [(50, 90, 50, 87.3), (45, 78, 41.5, 74.8), (55, 78, 58.5, 74.8),
          (58.5, 60, 53, 55.3), (50, 44.5, 50, 39.8), (50, 30, 50, 25.3)]:
    arrow(*a, color=INK2)

# physics -> architecture arrows
arrow(28, 90.5, 37.5, 69, color=BLUE)
arrow(28, 74.5, 35, 68, color=BLUE)
arrow(28, 59, 33.5, 65, color=BLUE)
arrow(28, 44, 42, 19.5, color=NAVY)
arrow(28, 29.5, 38, 48.5, color=NAVY)
arrow(28, 14.5, 38, 34, color=TEAL)

# ---- right column: training objective vs inference guarantees
box(70, 68, 29, 28, "Training objective (Hybrid)",
    "single cross-entropy on the\ndominant-mechanism label\n\n"
    "no physics penalty terms:\nphysics is structural, not a\nsoft loss\n\n"
    "[Pure-PINN ablation adds PDE\nresidual $+$ param-reg losses]", edge=BLUE)
box(70, 31, 29, 34, "Guaranteed at inference",
    "$\\sum_i \\alpha_i = 1$ exactly (softmax)\n\n"
    "plating $\\equiv 0$ in storage\n(37.1% violations $\\to$ 0)\n\n"
    "parameters inside literature bounds\nfor any input (bounded heads)\n\n"
    "frozen, human-readable priors\n$\\to$ every attribution auditable", edge=TEAL)
box(70, 8, 29, 20, "Scope (stated honestly)",
    "electrolyte + corrosion enter only via\nthe frozen rules (no learned params,\n"
    "no residuals); the priors' accuracy\ngain is in-distribution only", edge=GRAY)

arrow(66.5, 67, 70, 80, color=BLUE)
arrow(62, 18, 70, 44, color=TEAL)

fig.savefig(OUT, dpi=300, bbox_inches="tight", pad_inches=0.25)
print("saved", OUT)
