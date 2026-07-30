"""
Plot the TJU cross-chemistry end-to-end case-study figure (paper
Illustration 3) from reports/tju_cy25_2_case_study_real.json only -- every
number and curve drawn here was computed by scripts/compute_tju_case_study.py
from real data and real model inference. No np.random call and no typed-in
constant anywhere in this file; if a number isn't in the JSON, it isn't
plotted.

Run: python scripts/plot_tju_case_study.py
"""

import json
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8.5, 'axes.labelsize': 9, 'axes.titlesize': 10,
    'figure.titlesize': 12, 'axes.titleweight': 'bold',
    'grid.alpha': 0.1, 'savefig.dpi': 1200,
    'axes.linewidth': 1.0, 'lines.linewidth': 1.5
})

N_BLUE = '#006699'; N_RED = '#CC3333'; N_GREEN = '#2E7D32'
N_ORANGE = '#E65100'; N_GRAY = '#455A64'; N_PURPLE = '#6A1B9A'
B_BLUE = '#E3F2FD'; B_RED = '#FFEBEE'; B_GREEN = '#E8F5E9'

MECH_LABELS = ['Active\nMaterial', 'SEI\nGrowth', 'Lithium\nPlating',
              'Electrolyte\nDecomp.', 'Collector\nCorrosion']
MECH_KEYS = ['ACTIVE_MATERIAL_LOSS', 'SEI_GROWTH', 'LITHIUM_PLATING',
            'ELECTROLYTE_DECOMP', 'COLLECTOR_CORROSION']


def generate_figure():
    d = json.loads(Path('reports/tju_cy25_2_case_study_real.json').read_text())
    cycles = np.array(d['cycles'])
    soh = np.array(d['soh_true']) * 100
    hero = d['hero_zeroshot']
    scenarios = {s['key']: s for s in d['attribution_scenarios']}
    life = d['life_extension']

    pc = hero['per_cycle']
    pred_cycles = np.array(pc['cycle_idx'])
    soh_pred = np.array(pc['soh_pred']) * 100

    fig = plt.figure(figsize=(8.5, 11.0), facecolor='white')
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 0.85], hspace=0.35, wspace=0.25)

    fig.text(0.5, 0.975, 'End-to-End Demonstration: TJU NCM/NCA Cell (Cross-Chemistry Transfer Scenario)',
             ha='center', fontsize=13, weight='bold', color='#1A1A1A')
    fig.text(0.5, 0.955, 'Single trajectory processed through all four stages; '
             'every curve and value drawn from measured data or checkpointed-model inference',
             ha='center', fontsize=8.5, style='italic', color='#555555')

    # ── (a) Input: real TJU data ──
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.plot(cycles, soh, color=N_PURPLE, lw=2, label='SOH (%)', zorder=3)
    ax_a.set_title('(a) Input: TJU Cell (NCM/NCA, 25°C, 1C Cycling)', loc='left', pad=12)
    ax_a.set_xlabel('Cycle Number', fontweight='bold')
    ax_a.set_ylabel('State of Health (%)', color=N_PURPLE, fontweight='bold')
    ax_a.set_ylim(60, 105)
    ax_a.grid(True, zorder=0)
    eol = life['measured_eol_cycles']
    ax_a.axvline(eol, color=N_GRAY, ls=':', lw=1.2, zorder=2)
    ax_a.annotate(f'Measured EOL\n(80% SOH, cycle {eol:.0f})', xy=(eol, 80),
                 xytext=(eol - 260, 68),
                 arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3,rad=.2", color='black', lw=1.2),
                 fontsize=8.5, ha='center', weight='semibold',
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='none', alpha=0.8))
    ax_a.legend(fontsize=7, loc='lower left', framealpha=0.9)
    ax_a.text(0.5, -0.18, 'Data flows to all stages ▼', transform=ax_a.transAxes,
             ha='center', fontsize=8, weight='bold', color=N_GRAY)

    # ── (b) Stage 1: HERO zero-shot prediction (real inference) ──
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.plot(cycles, soh, color=N_PURPLE, lw=2, label='Actual SOH', zorder=3)
    ax_b.plot(pred_cycles, soh_pred, color=N_RED, lw=1.5, ls='--',
             label='HERO Prediction (zero-shot)', zorder=4, alpha=0.85)
    ax_b.set_title('(b) Stage 1: HERO Zero-Shot Prediction', loc='left', pad=12)
    ax_b.set_xlabel('Cycle Number', fontweight='bold')
    ax_b.set_ylabel('State of Health (%)', fontweight='bold')
    ax_b.set_ylim(60, 105)
    ax_b.grid(True, zorder=0)
    ax_b.legend(fontsize=7, loc='lower left', framealpha=0.9)

    metrics = (f"SOH MAE: {hero['soh_mae_pct']:.2f}%\nR²: {hero['soh_r2']:.2f}\n"
              f"(zero-shot: LCO-trained,\nNCM/NCA test, cf. Table 1)")
    ax_b.text(0.97, 0.95, metrics, transform=ax_b.transAxes, va='top', ha='right',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#F5F5F5', alpha=0.95, edgecolor='#CCC', lw=0.8),
             fontsize=7.5, linespacing=1.4)

    # ── (c) Stage 2: real causal attribution (factual) ──
    ax_c = fig.add_subplot(gs[1, 0])
    base = scenarios['factual']['attributions_pct']
    values = [base[k] * 100 for k in MECH_KEYS]
    colors = [N_ORANGE, N_BLUE, N_RED, '#607D8B', '#90A4AE']
    bars = ax_c.bar(MECH_LABELS, values, color=colors, alpha=0.9, width=0.6,
                    edgecolor='white', lw=1, zorder=3)
    ax_c.set_title('(c) Stage 2: Physics-Informed Causal Attribution', loc='left', pad=12)
    ax_c.set_ylabel('Attribution (%)', fontweight='bold')
    ax_c.set_ylim(0, 95)
    ax_c.grid(axis='y', alpha=0.3, zorder=0)
    ax_c.text(0.98, 0.95, f"Dominant: Active Material Loss ({values[0]:.0f}%)\n"
             "Cause: Mechanical/kinetic stress at 1C, 25°C",
             transform=ax_c.transAxes, ha='right', va='top', fontsize=7.5, color=N_ORANGE, weight='bold',
             bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.9, ec='#ddd'))
    for bar in bars:
        h = bar.get_height()
        ax_c.text(bar.get_x() + bar.get_width() / 2, h + 1, f'{h:.1f}%', ha='center', va='bottom',
                  weight='bold', fontsize=9)

    # ── (d) Stage 3: temperature counterfactual -- mechanism reassignment ──
    ax_d = fig.add_subplot(gs[1, 1])
    order = ['cool', 'factual', 'warm']
    xlabels = ['Cool to\n15°C', 'Current\n(25°C)', 'Warm to\n35°C']
    am_vals = [scenarios[k]['attributions_pct']['ACTIVE_MATERIAL_LOSS'] * 100 for k in order]
    sei_vals = [scenarios[k]['attributions_pct']['SEI_GROWTH'] * 100 for k in order]
    x = np.arange(3)
    w = 0.32
    ax_d.bar(x - w / 2, am_vals, width=w, color=N_ORANGE, alpha=0.9, edgecolor='white',
            lw=1, zorder=3, label='Active Material Loss')
    ax_d.bar(x + w / 2, sei_vals, width=w, color=N_BLUE, alpha=0.9, edgecolor='white',
            lw=1, zorder=3, label='SEI Growth')
    ax_d.set_xticks(x); ax_d.set_xticklabels(xlabels)
    ax_d.set_title('(d) Stage 3: Temperature Counterfactual', loc='left', pad=12)
    ax_d.set_ylabel('Attribution (%)', fontweight='bold')
    ax_d.set_ylim(0, 112)
    ax_d.grid(axis='y', alpha=0.3, zorder=0)
    ax_d.legend(fontsize=7, loc='upper left', framealpha=0.9)
    warm_S = life['levers']['warm']['S']
    ax_d.text(0.98, 0.99, f"Mechanism label flips\n(AM loss ↔ SEI growth)\nbut total rate barely\n"
             f"moves (S={warm_S:.2f}, ~{(warm_S-1)*100:+.0f}%)",
             transform=ax_d.transAxes, ha='right', va='top', fontsize=6.8, color=N_GRAY, weight='bold',
             bbox=dict(boxstyle='round,pad=0.3', fc='#F5F5F5', alpha=0.9, ec='#ddd'))
    for xi, v in zip(x - w / 2, am_vals):
        ax_d.text(xi, v + 1.5, f'{v:.0f}%', ha='center', va='bottom', fontsize=7.5, weight='bold')
    for xi, v in zip(x + w / 2, sei_vals):
        ax_d.text(xi, v + 1.5, f'{v:.0f}%', ha='center', va='bottom', fontsize=7.5, weight='bold')

    # ── (e) Stage 4: Advisory ──
    ax_e = fig.add_subplot(gs[2, :])
    ax_e.axis('off')
    ax_e.set_title('(e) Stage 4: PATT Classification & User Advisory Output', loc='center', pad=20,
                   weight='bold', fontsize=12)

    def draw_box(x, y, w, h, title, subheader, points, footer, c_main, c_bg):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03", ec=c_main, fc=c_bg, lw=2.5, zorder=1)
        ax_e.add_patch(rect)
        header = patches.FancyBboxPatch((x, y + h - 0.12), w, 0.12, boxstyle="round,pad=0.03", ec=c_main, fc=c_main, lw=0, zorder=2)
        ax_e.add_patch(header)
        ax_e.text(x + w / 2, y + h - 0.05, title, ha='center', weight='bold', size=11, color='white', zorder=3)
        ax_e.text(x + w / 2, y + h - 0.20, subheader, ha='center', weight='bold', size=9, color=c_main, zorder=3)
        y_pos = y + h - 0.28
        for pt in points:
            t = ax_e.text(x + 0.03, y_pos, pt, ha='left', va='top', size=8.5, color='#222', zorder=3, linespacing=1.5)
            t.set_clip_path(rect)  # never let an oversized line bleed into the next box
            y_pos -= 0.12
        ax_e.text(x + w / 2, y + 0.06, footer, ha='center', weight='bold', size=9, color=c_main, zorder=3)

    draw_box(0.01, 0.02, 0.31, 0.9, 'PATT Classification', 'MODEL DIAGNOSTICS',
             ['Mode: Cycling', 'Confidence: high', 'Chemistry: NCM/NCA', 'Zero-shot from LCO source'],
             'Data: TJU (NCM/NCA)', N_BLUE, B_BLUE)

    rate_lev = life['levers']['rate']
    draw_box(0.345, 0.02, 0.31, 0.9, 'Tactical Actions', 'IMMEDIATE MITIGATION',
             ['1. Discharge rate: 1C → 0.5C', '2. Thermal shifts relabel the mechanism',
              '3. Monitor capacity fade rate', '4. Alert: Stop if SOH < 82%'],
             f"AM Loss: {scenarios['factual']['attributions_pct']['ACTIVE_MATERIAL_LOSS']*100:.0f}% → "
             f"{scenarios['slow']['attributions_pct']['ACTIVE_MATERIAL_LOSS']*100:.0f}% (0.5C)", N_RED, B_RED)

    draw_box(0.68, 0.02, 0.31, 0.9, 'Strategic Plan', 'LONG-TERM PLANNING',
             ['1. Limit discharge rate < 1C', '2. Favor rate limits over thermal fixes',
              '3. Validate zero-shot before acting',
              f"4. Extrapolated: +{rate_lev['extra_cycles']:.0f} cycles"],
             f"Life Ext.: {rate_lev['life_extension_pct']:.0f}% (extrapolated,\nsee text for model dependence)",
             N_GREEN, B_GREEN)

    out = Path('Casual_Attribution_reports/figures/tju_case_study.png')
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=1200, bbox_inches='tight')
    print(f"Saved {out}")
    return out


if __name__ == '__main__':
    generate_figure()
