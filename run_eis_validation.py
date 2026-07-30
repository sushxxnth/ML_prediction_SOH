"""
EIS-Based Independent Validation of Mechanism Attribution (Reviewer R3.1)

The PLN storage study provides EIS spectra for 105 cells stored at four
temperatures (-40, -5, 25, 50 C) and three SOCs (0, 50, 100 %), measured
after 3 weeks and 3 months. SEI growth raises the interfacial (charge
transfer + film) resistance, so the RELATIVE growth of R_ct from 3W to 3M
within each storage condition is an independent, capacity-free probe of
SEI-dominated calendar aging.

Validation: across storage conditions, compare
  (a) EIS-measured relative R_ct growth  (independent measurement)
  (b) the Hybrid PINN's SEI attribution probability for that condition
  (c) the PINN's Arrhenius-based physics SEI prediction

Agreement in ranking (Spearman) supports the attribution outputs as
condition-based probability assignments; disagreement bounds their validity.

Outputs: reports/eis_attribution_validation.json (+ .png figure)
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))

from src.data.eis_impedance_loader import EISImpedanceLoader
from src.models.pinn_causal_attribution import PINNCausalAttributionModel
from test_unified_validation import make_context, BASE_FEATURES

MECHS = ['SEI Layer Growth', 'Lithium Plating', 'Active Material Loss',
         'Electrolyte Decomposition', 'Collector Corrosion']

OUT_JSON = Path('reports/eis_attribution_validation.json')
OUT_FIG = Path('reports/eis_attribution_validation.png')


def rct_from_spectrum(s):
    """Interfacial resistance estimate: Re(Z) at 1 Hz minus the
    high-frequency intercept. More robust across measurement temperatures
    than semicircle-apex detection (which fails on distorted spectra)."""
    z_real, z_imag = np.asarray(s.z_real), np.asarray(s.z_imag)
    freq = np.asarray(s.frequency)
    finite = np.isfinite(z_real) & np.isfinite(z_imag) & np.isfinite(freq)
    z_real, freq = z_real[finite], freq[finite]
    if len(z_real) < 5:
        return None
    r0 = float(z_real[np.argmax(freq)])  # high-frequency intercept
    r_1hz = float(z_real[np.argmin(np.abs(freq - 1.0))])
    rct = r_1hz - r0
    if rct <= 0:
        return None
    return r0, rct


def collect_eis():
    loader = EISImpedanceLoader('.')
    cells = loader.load()
    groups = defaultdict(lambda: defaultdict(list))  # (T, SOC) -> period -> [rct]
    for c in cells.values():
        for s in c.spectra:
            est = rct_from_spectrum(s)
            if est is None:
                continue
            groups[(s.temperature, s.soc)][s.storage_period].append(est[1])
    rows = []
    for (temp, soc), periods in sorted(groups.items()):
        if '3W' not in periods or '3M' not in periods:
            continue
        rct_3w = float(np.median(periods['3W']))
        rct_3m = float(np.median(periods['3M']))
        rows.append({
            'temp_C': temp, 'soc_pct': soc,
            'n_3W': len(periods['3W']), 'n_3M': len(periods['3M']),
            'median_rct_3W_ohm': round(rct_3w, 4),
            'median_rct_3M_ohm': round(rct_3m, 4),
            'relative_rct_growth': round((rct_3m - rct_3w) / rct_3w, 4),
        })
    return rows


@torch.no_grad()
def model_outputs(model, temp, soc_pct):
    context = make_context(temp, 0.0, 0.0, soc_pct / 100.0, 'storage')
    features = torch.FloatTensor(BASE_FEATURES).unsqueeze(0)
    context_t = torch.FloatTensor(context).unsqueeze(0)
    out = model(features, context_t)
    probs = torch.softmax(out['logits'], dim=-1).squeeze(0)
    sei_physics = float(out['physics_predictions']['sei'].squeeze())
    return {
        'attr_probs': {m: round(float(probs[i]), 4) for i, m in enumerate(MECHS)},
        'dominant': MECHS[int(probs.argmax())],
        'alpha_sei': round(float(probs[0]), 4),
        'physics_sei_prediction': round(sei_physics, 6),
    }


def main():
    print("[1/3] Extracting EIS R_ct growth per storage condition...")
    rows = collect_eis()

    print("[2/3] Querying Hybrid PINN for matching storage contexts...")
    model = PINNCausalAttributionModel(feature_dim=9, context_dim=6)
    model.load_state_dict(torch.load(
        'reports/pinn_causal/pinn_causal_retrained.pt',
        map_location='cpu', weights_only=True))
    model.eval()

    for r in rows:
        r.update(model_outputs(model, r['temp_C'], r['soc_pct']))

    print(f"\n{'T(C)':>6} {'SOC%':>5} {'Rct 3W':>8} {'Rct 3M':>8} {'growth':>8} "
          f"{'a_SEI':>6} {'dominant':>22}")
    for r in rows:
        print(f"{r['temp_C']:>6.0f} {r['soc_pct']:>5.0f} "
              f"{r['median_rct_3W_ohm']:>8.3f} {r['median_rct_3M_ohm']:>8.3f} "
              f"{r['relative_rct_growth']:>8.3f} {r['alpha_sei']:>6.2f} "
              f"{r['dominant']:>22}")

    print("\n[3/3] Rank correlations across conditions...")
    growth = [r['relative_rct_growth'] for r in rows]
    temps = [r['temp_C'] for r in rows]
    alpha = [r['alpha_sei'] for r in rows]
    physics = [r['physics_sei_prediction'] for r in rows]

    corr = {}
    corr['growth_vs_temp'] = spearmanr(growth, temps)
    corr['growth_vs_alpha_sei'] = spearmanr(growth, alpha)
    corr['growth_vs_physics_sei'] = spearmanr(growth, physics)

    # SEI-labeled subset only (exclude 0% SOC where corrosion is expected)
    sei_rows = [r for r in rows if r['soc_pct'] > 0]
    corr['growth_vs_physics_sei_soc>0'] = spearmanr(
        [r['relative_rct_growth'] for r in sei_rows],
        [r['physics_sei_prediction'] for r in sei_rows])

    # Absolute R_ct level at 3W vs storage temperature. Warm measurement
    # temperature LOWERS R_ct (faster kinetics), so a positive ordering with
    # storage T is a conservative indicator of real interfacial degradation.
    corr['rct_level_3W_vs_temp'] = spearmanr(
        [r['median_rct_3W_ohm'] for r in rows], temps)

    results_corr = {}
    for name, (rho, p) in corr.items():
        results_corr[name] = {'spearman_rho': round(float(rho), 3),
                              'p_value': round(float(p), 4)}
        print(f"  {name:32}: rho={rho:+.3f}  p={p:.4f}")

    out = {'conditions': rows, 'correlations': results_corr}
    OUT_JSON.parent.mkdir(exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(out, f, indent=2)

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    socs = sorted(set(r['soc_pct'] for r in rows))
    markers = {0.0: 's', 50.0: 'o', 100.0: '^'}
    for soc in socs:
        rr = [r for r in rows if r['soc_pct'] == soc]
        axes[0].plot([r['temp_C'] for r in rr],
                     [r['relative_rct_growth'] for r in rr],
                     markers.get(soc, 'o') + '-', label=f'{soc:.0f}% SOC')
    axes[0].set_xlabel('Storage temperature (°C)')
    axes[0].set_ylabel('Relative R$_{ct}$ growth (3W → 3M)')
    axes[0].set_title('(a) EIS-measured interfacial resistance growth')
    axes[0].legend()

    sc = axes[1].scatter(growth, physics,
                         c=temps, cmap='coolwarm', s=60)
    axes[1].set_xlabel('Relative R$_{ct}$ growth (EIS, measured)')
    axes[1].set_ylabel('PINN physics SEI prediction')
    axes[1].set_title('(b) Model SEI output vs. independent EIS probe')
    plt.colorbar(sc, ax=axes[1], label='Storage T (°C)')
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=150)
    print(f"\nSaved {OUT_JSON} and {OUT_FIG}")


if __name__ == '__main__':
    main()
