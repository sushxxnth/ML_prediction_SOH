"""
Regenerate pinn_architecture_comparison.png with correct numbers:
- Pure PINN: 60.0% (not 77.3%)
- Hybrid PINN: 96.0% (not 92.0%)
- Ablation: NN-only 93.3%, Priors-only 81.3%, Combined 96.0%
- New narrative: knowledge distillation effect, not "priors contribute >3x"
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Set up figure
fig = plt.figure(figsize=(14, 12))
fig.suptitle('Battery Degradation Mechanism Attribution: Pure PINN vs. Hybrid PINN Architectures', 
             fontsize=14, fontweight='bold', y=0.98)

# ============================================================
# Panel A: Architecture Diagram
# ============================================================
ax_a = fig.add_axes([0.02, 0.52, 0.45, 0.42])
ax_a.set_xlim(0, 10)
ax_a.set_ylim(0, 10)
ax_a.axis('off')
ax_a.text(0.0, 9.8, 'a', fontsize=16, fontweight='bold', transform=ax_a.transAxes, va='top')
ax_a.text(5.0, 9.7, 'Architecture Diagram', fontsize=13, fontweight='bold', ha='center')

# Pure PINN side
ax_a.text(2.2, 9.0, 'Pure PINN', fontsize=11, fontweight='bold', ha='center')

# Input box
rect = FancyBboxPatch((0.5, 8.0), 3.4, 0.7, boxstyle="round,pad=0.1", 
                       facecolor='#E8E8E8', edgecolor='#666666', linewidth=1)
ax_a.add_patch(rect)
ax_a.text(2.2, 8.35, 'Input: Features + Context', fontsize=8, ha='center', va='center')

# Arrow
ax_a.annotate('', xy=(2.2, 7.8), xytext=(2.2, 8.0),
              arrowprops=dict(arrowstyle='->', color='#666666', lw=1.5))

# Physics Equations box
rect = FancyBboxPatch((0.5, 5.5), 3.4, 2.2, boxstyle="round,pad=0.1", 
                       facecolor='#B8D4E8', edgecolor='#4A86B8', linewidth=1)
ax_a.add_patch(rect)
ax_a.text(2.2, 7.3, 'Physics Equations', fontsize=9, fontweight='bold', ha='center', va='center')

# Mechanism boxes inside
for j, (name, color) in enumerate([('SEI', '#9BC4A0'), ('Plating', '#7AB5D4'), ('AM Loss', '#8DB87A')]):
    y_pos = 6.7 - j * 0.5
    rect = FancyBboxPatch((0.8, y_pos-0.15), 1.8, 0.35, boxstyle="round,pad=0.05",
                           facecolor=color, edgecolor='#333333', linewidth=0.5)
    ax_a.add_patch(rect)
    ax_a.text(1.7, y_pos, name, fontsize=7, ha='center', va='center')

# Arrow  
ax_a.annotate('', xy=(2.2, 5.2), xytext=(2.2, 5.5),
              arrowprops=dict(arrowstyle='->', color='#666666', lw=1.5))

# Output box
rect = FancyBboxPatch((0.5, 4.5), 3.4, 0.6, boxstyle="round,pad=0.1",
                       facecolor='#E8E8E8', edgecolor='#666666', linewidth=1)
ax_a.add_patch(rect)
ax_a.text(2.2, 4.8, 'Output probabilities', fontsize=8, ha='center', va='center')

# Arrow
ax_a.annotate('', xy=(2.2, 4.2), xytext=(2.2, 4.5),
              arrowprops=dict(arrowstyle='->', color='#666666', lw=1.5))

# Accuracy
ax_a.text(2.2, 3.9, '60.0% accuracy', fontsize=11, fontweight='bold', ha='center', color='#C44E52')

# Hybrid PINN side
ax_a.text(7.5, 9.0, 'Hybrid PINN', fontsize=11, fontweight='bold', ha='center')

# Input box
rect = FancyBboxPatch((5.5, 8.0), 4.0, 0.7, boxstyle="round,pad=0.1",
                       facecolor='#E8E8E8', edgecolor='#666666', linewidth=1)
ax_a.add_patch(rect)
ax_a.text(7.5, 8.35, 'Input: Features + Context', fontsize=8, ha='center', va='center')

# Arrow
ax_a.annotate('', xy=(7.5, 7.8), xytext=(7.5, 8.0),
              arrowprops=dict(arrowstyle='->', color='#666666', lw=1.5))

# Three components
components = [
    ('Neural\nNetwork', '#D4A5E8', 'NN\nLogits', '#9B59B6'),
    ('Physics\nEquations', '#B8D4E8', 'Physics\nLoss', '#4A86B8'),
    ('Expert\nPriors', '#E8D4A5', 'Prior\nLogits', '#D4A017'),
]

for j, (name, bg_color, output_name, out_color) in enumerate(components):
    y_pos = 7.2 - j * 0.9
    # Component box
    rect = FancyBboxPatch((5.5, y_pos-0.25), 2.0, 0.6, boxstyle="round,pad=0.05",
                           facecolor=bg_color, edgecolor='#333333', linewidth=0.8)
    ax_a.add_patch(rect)
    ax_a.text(6.5, y_pos+0.05, name, fontsize=7, ha='center', va='center')
    
    # Arrow
    ax_a.annotate('', xy=(8.0, y_pos+0.05), xytext=(7.7, y_pos+0.05),
                  arrowprops=dict(arrowstyle='->', color='#666666', lw=1))
    
    # Output box
    rect = FancyBboxPatch((8.0, y_pos-0.2), 1.3, 0.5, boxstyle="round,pad=0.05",
                           facecolor=out_color, edgecolor='#333333', linewidth=0.5, alpha=0.7)
    ax_a.add_patch(rect)
    ax_a.text(8.65, y_pos+0.05, output_name, fontsize=6, ha='center', va='center', color='white', fontweight='bold')

# Output box
rect = FancyBboxPatch((6.0, 4.5), 3.0, 0.6, boxstyle="round,pad=0.1",
                       facecolor='#E8E8E8', edgecolor='#666666', linewidth=1)
ax_a.add_patch(rect)
ax_a.text(7.5, 4.8, 'Output', fontsize=8, ha='center', va='center')

# Arrow
ax_a.annotate('', xy=(7.5, 4.2), xytext=(7.5, 4.5),
              arrowprops=dict(arrowstyle='->', color='#666666', lw=1.5))

# Accuracy
ax_a.text(7.5, 3.9, '96.0% accuracy', fontsize=11, fontweight='bold', ha='center', color='#2E8B57')

# Dividing line
ax_a.plot([4.5, 4.5], [3.5, 9.5], '--', color='#999999', lw=1)

# ============================================================
# Panel B: Physics Parameter Learning (Table)
# ============================================================
ax_b = fig.add_axes([0.52, 0.52, 0.46, 0.42])
ax_b.axis('off')
ax_b.text(0.0, 1.0, 'b', fontsize=16, fontweight='bold', transform=ax_b.transAxes, va='top')
ax_b.text(0.5, 0.95, 'Physics Parameter Learning', fontsize=13, fontweight='bold', 
          ha='center', transform=ax_b.transAxes)

# Table data
cols = ['Parameter', 'Learned\nValue', 'Literature\nRange', 'Status']
rows = [
    ['$E_a$ (SEI\nactivation)', '50.3\nkJ/mol', '35-60\nkJ/mol', '✓'],
    ['β (C-rate\nexponent)', '1.48', '~1.5', '✓'],
    ['γ (cycle\nexponent)', '0.52', '0.3-1.0', '✓'],
    ['α (plating\ncoefficient)', '0.51', '0.3-0.7', '✓'],
]

table = ax_b.table(cellText=rows, colLabels=cols, loc='center',
                   cellLoc='center', colColours=['#4A86B8']*4)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.2)

# Style header
for j in range(4):
    table[0, j].set_text_props(color='white', fontweight='bold')
    table[0, j].set_facecolor('#4A86B8')

# Style check marks
for i in range(1, 5):
    table[i, 3].set_text_props(color='#2E8B57', fontsize=14, fontweight='bold')

# ============================================================
# Panel C: Disambiguation Challenge (Scatter plot)
# ============================================================
ax_c = fig.add_axes([0.05, 0.04, 0.42, 0.42])
ax_c.text(-0.05, 1.05, 'c', fontsize=16, fontweight='bold', transform=ax_c.transAxes)
ax_c.set_title('The Disambiguation Challenge', fontsize=13, fontweight='bold', pad=10)

# Generate scatter data
np.random.seed(42)
# SEI dominant points (low C-rate, various temps)
sei_crates = np.random.uniform(0.1, 1.2, 15)
sei_temps = np.random.uniform(10, 45, 15)

# AM loss dominant points (high C-rate)
am_crates = np.random.uniform(1.3, 2.0, 12)
am_temps = np.random.uniform(20, 45, 12)

# Ambiguous zone
amb_crates = np.random.uniform(0.8, 1.5, 10)
amb_temps = np.random.uniform(20, 40, 10)

ax_c.scatter(sei_crates, sei_temps, c='#4A86B8', s=50, alpha=0.7, label='SEI Dominant', zorder=3)
ax_c.scatter(am_crates, am_temps, c='#C44E52', s=50, alpha=0.7, label='AM Loss Dominant', zorder=3)
ax_c.scatter(amb_crates, amb_temps, c='#999999', s=50, alpha=0.5, label='Ambiguous', zorder=3)

# Ambiguous zone rectangle
rect = mpatches.FancyBboxPatch((0.7, 19), 0.9, 22, boxstyle="round,pad=0.1",
                                facecolor='#FFE0B2', edgecolor='#FF9800', 
                                linewidth=2, linestyle='--', alpha=0.3)
ax_c.add_patch(rect)
ax_c.text(1.15, 30, 'Ambiguous\nZone\n(Similar Signals)', fontsize=8, 
          ha='center', va='center', style='italic', color='#E65100', fontweight='bold')

# Region labels
ax_c.text(0.3, 42, 'SEI\nDominant', fontsize=9, fontweight='bold', color='#4A86B8')
ax_c.text(1.7, 8, 'AM Loss\nDominant', fontsize=9, fontweight='bold', color='#C44E52')

ax_c.set_xlabel('C-rate', fontsize=10)
ax_c.set_ylabel('Temperature (°C)', fontsize=10)
ax_c.set_xlim(0, 2.1)
ax_c.set_ylim(0, 50)
ax_c.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
ax_c.set_xticklabels(['0', '0.5C', '1C', '1.5C', '2C'])

# Annotations
ax_c.annotate('Strong\nSEI Signal', xy=(0.3, 20), fontsize=7, ha='center',
              arrowprops=dict(arrowstyle='->', color='#4A86B8'), xytext=(0.1, 12))
ax_c.annotate('Strong\nAM Loss Signal', xy=(1.8, 38), fontsize=7, ha='center',
              arrowprops=dict(arrowstyle='->', color='#C44E52'), xytext=(1.9, 45))
ax_c.annotate('Overlap &\nUncertainty', xy=(1.0, 22), fontsize=7, ha='center',
              arrowprops=dict(arrowstyle='->', color='#FF9800'), xytext=(0.5, 14))

# ============================================================
# Panel D: Component Contribution Analysis (NEW - Knowledge Distillation)
# ============================================================
ax_d = fig.add_axes([0.55, 0.04, 0.42, 0.42])
ax_d.text(-0.05, 1.05, 'd', fontsize=16, fontweight='bold', transform=ax_d.transAxes)
ax_d.set_title('Component Contribution Analysis', fontsize=13, fontweight='bold', pad=10)

# Stacked bar showing the corrected breakdown
# Full model: 96.0%, NN-only: 93.3%, Priors-only: 81.3%, Random: 21.7%
categories = ['Random\nBaseline', 'Priors\nOnly', 'NN\nOnly', 'Full\nHybrid']
values = [21.7, 81.3, 93.3, 96.0]
colors = ['#B0BEC5', '#E8D4A5', '#D4A5E8', '#2E8B57']
edge_colors = ['#78909C', '#D4A017', '#9B59B6', '#1B5E20']

bars = ax_d.bar(categories, values, color=colors, edgecolor=edge_colors, linewidth=1.5, width=0.6)

# Add value labels on bars
for bar, val in zip(bars, values):
    ax_d.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
              f'{val}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add annotation arrows showing contributions
ax_d.annotate('', xy=(2.7, 93.3), xytext=(2.7, 96.0),
              arrowprops=dict(arrowstyle='<->', color='#C44E52', lw=2))
ax_d.text(3.2, 94.7, 'Priors add\n+2.7 pp', fontsize=8, fontweight='bold', 
          color='#C44E52', va='center')

# Knowledge distillation annotation
ax_d.annotate('Knowledge\nDistillation\nEffect', 
              xy=(2, 93.3), xytext=(0.5, 60),
              fontsize=9, fontweight='bold', color='#7B1FA2',
              ha='center', va='center',
              arrowprops=dict(arrowstyle='->', color='#7B1FA2', lw=1.5,
                            connectionstyle='arc3,rad=-0.2'),
              bbox=dict(boxstyle='round,pad=0.3', facecolor='#F3E5F5', edgecolor='#7B1FA2'))

# Safety net annotation
ax_d.annotate('Safety Net:\n2 cold-weather\nplating cases', 
              xy=(3, 96.0), xytext=(3.5, 75),
              fontsize=7, ha='center', va='center',
              arrowprops=dict(arrowstyle='->', color='#1B5E20', lw=1),
              bbox=dict(boxstyle='round,pad=0.2', facecolor='#E8F5E9', edgecolor='#1B5E20'))

ax_d.set_ylabel('Accuracy (%)', fontsize=10)
ax_d.set_ylim(0, 105)
ax_d.spines['top'].set_visible(False)
ax_d.spines['right'].set_visible(False)

plt.savefig('Casual_Attribution_reports/pinn_architecture_comparison.png', 
            dpi=150, bbox_inches='tight', facecolor='white')
print("✅ Saved pinn_architecture_comparison.png")
plt.close()
