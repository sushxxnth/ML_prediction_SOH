"""
Regenerate pinn_architecture_comparison.png to match the original layout PERFECTLY
and FIX the flaws identified: NO vertical lines in the table, accurate panel D bar chart,
exact colors of the logits, white background, precise gradient in C.
Numbers: 60.0% Pure PINN, 96.0% Hybrid PINN. Stack sizes: 21.7% Base, 71.6% NN, 2.7% Prior.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Global Setup
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 12,
    'axes.linewidth': 1.2,
    'figure.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.dpi': 300,
})

fig = plt.figure(figsize=(15, 13.5))
fig.suptitle(
    'Battery Degradation Mechanism Attribution: Pure PINN vs. Hybrid PINN Architectures',
    fontsize=20, fontweight='bold', y=0.97
)

# Colors matching the original pixel values
C_SEI = '#9ecbeb'
C_PLA = '#f6bd8c'
C_AM  = '#b3e0a6'
C_NN_L = '#a87cb3'   # purple logits
C_PR_L = '#4c7aa1'   # blue logits
C_PHYS_L_LEFT = '#9ecbeb'
C_PHYS_L_RIGHT = '#f6bd8c'
C_BG_PHYS = '#e8ebec'  # light gray outer box
C_BG_BOX = '#fdfdfd'
C_TEXT = 'black'

# ══════════════════════════════════════════════════════════════════════════════
# Panel A – Architecture Diagram
# ══════════════════════════════════════════════════════════════════════════════
ax_a = fig.add_axes([0.02, 0.50, 0.46, 0.40])
ax_a.set_xlim(0, 10)
ax_a.set_ylim(0, 10)
ax_a.axis('off')

ax_a.text(0.0, 1.03, 'a', fontsize=24, fontweight='bold', transform=ax_a.transAxes)
ax_a.text(5.0, 9.8, 'Architecture Diagram', fontsize=16, fontweight='bold', ha='center')

# Pure PINN header
ax_a.text(2.2, 8.8, 'Pure PINN', fontsize=14, fontweight='bold', ha='center')

def draw_box(x, y, w, h, text, bg, ec='black', lw=1.2, fs=12, tc='black', rad=0.2):
    r = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={rad}",
                       facecolor=bg, edgecolor=ec, linewidth=lw)
    ax_a.add_patch(r)
    if text:
        ax_a.text(x + w/2, y + h/2, text, fontsize=fs, color=tc, ha='center', va='center')

# Arrows helper
def draw_arrow(x0, y0, x1, y1):
    ax_a.annotate('', xy=(x1, y1), xytext=(x0, y0),
                  arrowprops=dict(arrowstyle='->,head_width=0.15,head_length=0.2', color='black', lw=1.2))

# Pure PINN side
draw_box(0.3, 7.5, 3.8, 0.6, 'Input: Features + Context', C_BG_BOX, rad=0.1)
draw_arrow(2.2, 7.4, 2.2, 7.0)

# Outer Physics Box
r = FancyBboxPatch((0.3, 3.8), 3.8, 3.1, boxstyle="round,pad=0.1",
                   facecolor=C_BG_PHYS, edgecolor='black', linewidth=1.2)
ax_a.add_patch(r)
ax_a.text(2.2, 6.6, 'Physics Equations', fontsize=13, ha='center', va='center')

draw_box(1.0, 5.6, 2.4, 0.45, 'SEI', C_SEI, rad=0.1)
draw_box(1.0, 4.8, 2.4, 0.45, 'Plating', C_PLA, rad=0.1)
draw_box(1.0, 4.0, 2.4, 0.45, 'AM Loss', C_AM, rad=0.1)

draw_arrow(2.2, 5.5, 2.2, 5.6)
draw_arrow(2.2, 4.7, 2.2, 4.8)

for y in [5.85, 5.05, 4.25]:
    draw_arrow(1.0, y, 0.5, y)
    draw_arrow(3.4, y, 3.9, y)

draw_arrow(2.2, 3.7, 2.2, 3.3)
draw_box(0.7, 2.5, 3.0, 0.8, 'Output\nprobabilities', C_BG_PHYS, rad=0.1)
draw_arrow(2.2, 2.4, 2.2, 1.8)

ax_a.text(2.2, 1.3, '60.0% accuracy', fontsize=14, fontweight='bold', ha='center', color='black')

# Hybrid PINN side
ax_a.text(7.7, 8.8, 'Hybrid PINN', fontsize=14, fontweight='bold', ha='center')
draw_box(5.5, 7.5, 4.4, 0.6, 'Input: Features + Context', C_BG_BOX, rad=0.1)
draw_arrow(7.7, 7.4, 7.7, 7.0)

# Components Connectors
ax_a.plot([5.8, 5.8], [6.95, 4.4], '-', color='black', lw=1.2)
ax_a.plot([5.8, 5.8], [6.95, 6.95], '-', color='black', lw=1.2) # Corner

def draw_hybrid_row(y, text1, text2, out_c, is_split=False):
    draw_arrow(5.8, y+0.35, 6.0, y+0.35)
    draw_box(6.0, y, 1.8, 0.7, text1, C_BG_BOX, rad=0.1)
    draw_arrow(7.8, y+0.35, 8.3, y+0.35)
    if is_split:
        # custom gradient block for Physics Loss box
        r2 = FancyBboxPatch((8.3, y), 1.6, 0.7, boxstyle="round,pad=0.1",
                            facecolor=C_PHYS_L_LEFT, edgecolor='black', linewidth=1.2)
        ax_a.add_patch(r2)
        r_half = mpatches.Rectangle((9.1, y-0.1), 0.9, 0.9, facecolor=C_PHYS_L_RIGHT, zorder=1)
        ax_a.add_patch(r_half)
        # Re-draw the edge
        r2_edge = FancyBboxPatch((8.3, y), 1.6, 0.7, boxstyle="round,pad=0.1",
                                 facecolor='none', edgecolor='black', linewidth=1.2, zorder=2)
        ax_a.add_patch(r2_edge)
        ax_a.text(9.1, y+0.35, text2, fontsize=12, color='black', ha='center', va='center', zorder=3)
    else:
        tc = 'white' if out_c != C_BG_BOX else 'black'
        draw_box(8.3, y, 1.6, 0.7, text2, out_c, tc=tc, rad=0.1)

draw_hybrid_row(6.0, 'Neural\nNetwork', 'NN\nLogits', C_NN_L)
draw_hybrid_row(5.0, 'Physics\nEquations', 'Physics\nLoss', None, is_split=True)
draw_hybrid_row(4.0, 'Expert\nPriors', 'Prior\nLogits', C_PR_L)

ax_a.plot([9.1, 9.1], [3.9, 3.4], '-', color='black', lw=1.2)
ax_a.plot([7.7, 9.1], [3.4, 3.4], '-', color='black', lw=1.2)
draw_arrow(7.7, 3.4, 7.7, 3.0)

draw_box(6.4, 2.5, 2.6, 0.5, 'Output', C_BG_BOX, rad=0.1)
draw_arrow(7.7, 2.4, 7.7, 1.8)
ax_a.text(7.7, 1.3, '96.0% accuracy', fontsize=14, fontweight='bold', ha='center', color='black')

# Center separator
ax_a.plot([4.8, 4.8], [0.5, 9.0], ':', color='#999999', lw=1.5)


# ══════════════════════════════════════════════════════════════════════════════
# Panel B – Physics Parameter Learning Table
# NO VERTICAL LINES, thick horiz lines, green checks
# ══════════════════════════════════════════════════════════════════════════════
ax_b = fig.add_axes([0.52, 0.50, 0.45, 0.40])
ax_b.axis('off')
ax_b.text(0.0, 1.03, 'b', fontsize=24, fontweight='bold', transform=ax_b.transAxes)
ax_b.text(0.5, 0.98, 'Physics Parameter Learning', fontsize=16, fontweight='bold', ha='center', transform=ax_b.transAxes)

cols = ['Parameter', 'Learned\nValue', 'Literature\nRange', 'Status']
rows = [
    ['$E_a$ (SEI\nactivation)', '50.3\nkJ/mol', '35-60\nkJ/mol', ''],
    ['β (C-rate\nexponent)',    '1.48',           '~1.5',           ''],
    ['γ (cycle\nexponent)',     '0.52',           '0.3-1.0',        ''],
    ['α (plating\ncoefficient)','0.51',           '0.3-0.7',        ''],
]

# We will draw the table manually to have perfect line control
y_start = 0.85
row_h = 0.15
x_cols = [0, 0.35, 0.55, 0.8]  # relative X positions
w_cols = [0.35, 0.2, 0.25, 0.2]

# Top thick line
ax_b.plot([0, 1], [y_start, y_start], '-', color='black', lw=2.0)

# Header
for j, col in enumerate(cols):
    ax_b.text(x_cols[j]+w_cols[j]/2, y_start - row_h/2, col, fontweight='bold', ha='center', va='center', fontsize=13)

# Line below header
ax_b.plot([0, 1], [y_start - row_h, y_start - row_h], '-', color='black', lw=1.5)

for i, row in enumerate(rows):
    yst = y_start - row_h - (i)*row_h
    # alternating bg
    if i % 2 == 1:
        rect = mpatches.Rectangle((0, yst-row_h), 1, row_h, facecolor='#f0f0f0', edgecolor='none')
        ax_b.add_patch(rect)
    
    for j, cell in enumerate(row):
        if j == 3: # checkmark
            ax_b.text(x_cols[j]+w_cols[j]/2, yst - row_h/2, cell, color='#4aa758', 
                      fontweight='bold', fontsize=24, fontfamily='DejaVu Sans', ha='center', va='center')
        else:
            ax_b.text(x_cols[j]+w_cols[j]/2, yst - row_h/2, cell, ha='center', va='center', fontsize=13)

# Bottom line
ax_b.plot([0, 1], [y_start - 5*row_h, y_start - 5*row_h], '-', color='black', lw=2.0)


# ══════════════════════════════════════════════════════════════════════════════
# Panel C – The Disambiguation Challenge
# ══════════════════════════════════════════════════════════════════════════════
ax_c = fig.add_axes([0.08, 0.05, 0.38, 0.38])
ax_c.text(-0.15, 1.05, 'c', fontsize=24, fontweight='bold', transform=ax_c.transAxes)
ax_c.set_title('The Disambiguation Challenge', fontsize=16, fontweight='bold', pad=15)

# Gradient background
gradient = np.zeros((100, 100, 4))
for iy in range(100):
    for ix in range(100):
        t = (ix/100.0 + (99-iy)/100.0) / 2
        r = 0.55 + t*(0.94 - 0.55)
        g = 0.73 + t*(0.65 - 0.73)
        b = 0.85 + t*(0.40 - 0.85)
        gradient[iy, ix] = [r, g, b, 0.6]

ax_c.imshow(gradient, extent=[0, 2.1, 0, 50], aspect='auto', zorder=0)

np.random.seed(42)
sei_cr = np.random.uniform(0.1, 1.2, 12);  sei_tp = np.random.uniform(10, 40, 12)
am_cr  = np.random.uniform(1.3, 2.0, 12);  am_tp  = np.random.uniform(15, 45, 12)
amb_cr = np.random.uniform(0.7, 1.4, 15);  amb_tp = np.random.uniform(20, 38, 15)

ax_c.scatter(sei_cr, sei_tp, c='#4882b5', s=60, edgecolors='black', linewidths=0.8, zorder=3)
ax_c.scatter(am_cr,  am_tp,  c='#e17b48', s=60, edgecolors='black', linewidths=0.8, zorder=3)
ax_c.scatter(amb_cr, amb_tp, c='#a690c5', s=60, edgecolors='black', linewidths=0.8, zorder=3)

# Ambiguous rectangle
rect = mpatches.Rectangle(
    (0.6, 19), 0.9, 21, facecolor='#b9b4d8', edgecolor='black', 
    linewidth=1.5, linestyle='--', zorder=2, alpha=0.4
)
ax_c.add_patch(rect)
ax_c.text(1.05, 29.5, 'Ambiguous\nZone\n(Similar Signals)', fontsize=13, fontweight='bold', ha='center', va='center', zorder=4)

ax_c.text(0.3, 44, 'SEI\nDominant', fontsize=14, fontweight='bold', ha='center')
ax_c.text(1.8, 5, 'AM Loss\nDominant', fontsize=14, fontweight='bold', ha='center')

ax_c.set_xlabel('C-rate', fontsize=13)
ax_c.set_ylabel('Temperature (°C)', fontsize=13)
ax_c.set_xlim(0, 2.1)
ax_c.set_ylim(0, 50)
ax_c.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
ax_c.set_xticklabels(['0', '0.5C', '1C', '1.5C', '2C'])

arr = dict(arrowstyle='->', color='black', lw=1.2)
ax_c.annotate('Strong\nSEI\nSignal', xy=(0.3, 16), xytext=(0.2, 6), ha='center', arrowprops=arr)
ax_c.annotate('Strong\nAM Loss\nSignal', xy=(1.85, 39), xytext=(1.8, 47), ha='center', arrowprops=arr)
ax_c.annotate('Overlap &\nUncertainty', xy=(1.05, 19), xytext=(1.05, 9), ha='center', arrowprops=arr)


# ══════════════════════════════════════════════════════════════════════════════
# Panel D – Component Contribution Analysis 
# No axes lines except maybe one at bottom
# ══════════════════════════════════════════════════════════════════════════════
ax_d = fig.add_axes([0.62, 0.05, 0.35, 0.38])
ax_d.axis('off')

ax_d.text(-0.25, 1.05, 'd', fontsize=24, fontweight='bold', transform=ax_d.transAxes)
ax_d.text(0.5, 1.05, 'Component Contribution Analysis', fontsize=16, fontweight='bold', ha='center', transform=ax_d.transAxes)

# Base line for bottom of chart
ax_d.plot([-0.2, 1.2], [0, 0], '-', color='black', lw=1.2)

# Values updated as requested: 21.7% base, 71.6% NN, 2.7% Priors = 96%
base_val = 21.7
nn_val = 71.6
prior_val = 2.7
total_val = 96.0

bar_x = 0.2
bar_width = 0.5

# Draw Stacked bar
# Colors from orig: Base=light gray, NN=purple, Prior=blue
C_B = '#e6e6e6'
C_N = '#81579d'
C_P = '#4c7aa1'

ax_d.bar(bar_x, base_val, width=bar_width, color=C_B, edgecolor='black', lw=1.2)
ax_d.bar(bar_x, nn_val, width=bar_width, bottom=base_val, color=C_N, edgecolor='black', lw=1.2)
ax_d.bar(bar_x, prior_val, width=bar_width, bottom=base_val+nn_val, color=C_P, edgecolor='black', lw=1.2)

ax_d.text(bar_x, total_val + 1.5, f'{total_val:.0f}%', fontsize=18, ha='center', va='bottom')

ax_d.text(bar_x, base_val/2, f'{base_val:.0f}%\ncontribution', fontsize=14, ha='center', va='center', color='black')
ax_d.text(bar_x, base_val + nn_val/2, f'{nn_val:.0f}%\ncontribution', fontsize=14, ha='center', va='center', color='white')
# 2.7% is too small to fit the word contribution inside, so we skip inner text for top slice.

# Right side labels matching original
def draw_line_label(y, text):
    # simple horiz line from right of bar
    x_end = bar_x + bar_width/2 + 0.1
    ax_d.plot([bar_x + bar_width/2, x_end], [y, y], '-', color='black', lw=1.0)
    ax_d.text(x_end + 0.05, y, text, fontsize=14, va='center')

draw_line_label(base_val/2, 'Random Baseline')
draw_line_label(base_val + nn_val/2, 'Neural Network')

# Expert Priors Box
bbox_props = dict(boxstyle="round,pad=0.4", fc="#eaebff", ec="black", lw=1.5)
ax_d.annotate(
    f'Expert Priors\n{prior_val:.1f}% contribution',
    xy=(bar_x + bar_width/2, base_val + nn_val + prior_val/2),
    xytext=(bar_x + bar_width/2 + 0.35, 88),
    fontsize=14, fontweight='bold', va='center', bbox=bbox_props,
    arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=0", color='black', lw=1.5)
)
ax_d.text(bar_x + bar_width/2 + 0.35, 75, 'Knowledge distillation:\nNN internalizes priors\nduring training', fontsize=13, va='top')

ax_d.set_xlim(-0.2, 1.5)
ax_d.set_ylim(0, 105)

out = 'Casual_Attribution_reports/pinn_architecture_comparison.png'
plt.savefig(out, dpi=300, facecolor='white')
print(f"✅  Saved {out}")
plt.close()
