"""
Early-Warning Slope-Threshold Sensitivity Analysis (Reviewer R2.7)

The paper's early-warning engine fires when the 20-cycle SOH slope exceeds
0.17 %/cycle (src/advisory/warning_engine.py, FAST_DEGRADATION_RATE=0.0017),
a value selected on NASA data. This script sweeps the threshold across
0.10-0.25 %/cycle and reports precision / recall / F1 / lead time on:

  - NASA Ames (34 cells, summary.csv, EOL = SOH <= 80%)  [paper setting]
  - TJU  (NCM, unified cache, EOL = SOH <= 80%)
  - XJTU (NCM, unified cache; tests stop at ~82-85% SOH so no cell reaches
    EOL: used to measure the false-alarm rate on non-failing cells)

Outputs: reports/early_warning_threshold_sensitivity.json (+ .png figure)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SLOPE_WINDOW = 20          # cycles, matches WarningEngine.SLOPE_WINDOW
EOL_SOH = 0.80
MIN_CYCLES = 20
THRESHOLDS = [0.0010, 0.0013, 0.0015, 0.0017, 0.0020, 0.0025]  # frac/cycle

OUT_JSON = Path('reports/early_warning_threshold_sensitivity.json')
OUT_FIG = Path('reports/early_warning_threshold_sensitivity.png')


def clean_trajectory(cycles, sohs):
    """Sort by cycle, drop non-physical SOH values, apply rolling median."""
    df = pd.DataFrame({'cycle': cycles, 'soh': sohs}).dropna()
    df = df[(df.soh > 0.3) & (df.soh < 1.2)].sort_values('cycle')
    if len(df) < MIN_CYCLES:
        return None
    df['soh'] = df['soh'].rolling(5, center=True, min_periods=1).median()
    return df.reset_index(drop=True)


def load_nasa():
    df = pd.read_csv('data/nasa_set5/summary.csv')
    df = df[df.cycle_type == 'discharge']
    out = {}
    for cid, g in df.groupby('cell_id'):
        traj = clean_trajectory(g.cycle_index.values, g.SOH_Q.values)
        if traj is not None:
            out[cid] = traj
    return out


def load_cache(name):
    d = json.load(open(f'data/unified_cache/{name}/{name}_processed.json'))
    out = {}
    for cid, c in d['cells'].items():
        cycles = [cy['cycle_index'] for cy in c['cycles']]
        sohs = [cy.get('soh_capacity') for cy in c['cycles']]
        traj = clean_trajectory(cycles, sohs)
        if traj is not None:
            out[cid] = traj
    return out


def first_eol_cycle(traj):
    hit = traj[traj.soh <= EOL_SOH]
    return None if hit.empty else float(hit.cycle.iloc[0])


def first_warning_cycle(traj, threshold):
    """First cycle where the trailing SLOPE_WINDOW-cycle fade rate >= threshold."""
    cyc = traj.cycle.values.astype(float)
    soh = traj.soh.values.astype(float)
    for i in range(len(cyc)):
        mask = (cyc >= cyc[i] - SLOPE_WINDOW) & (cyc <= cyc[i])
        if mask.sum() < 3:
            continue
        slope = np.polyfit(cyc[mask], soh[mask], 1)[0]
        if -slope >= threshold:
            return cyc[i]
    return None


def evaluate(cells, threshold):
    tp = fp = fn = tn = 0
    lead_times = []
    for cid, traj in cells.items():
        eol = first_eol_cycle(traj)
        warn = first_warning_cycle(traj, threshold)
        if eol is not None:
            if warn is not None and warn < eol:
                tp += 1
                lead_times.append(eol - warn)
            else:
                fn += 1
        else:
            if warn is not None:
                fp += 1
            else:
                tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'precision': round(precision * 100, 1),
        'recall': round(recall * 100, 1),
        'f1': round(f1 * 100, 1),
        'mean_lead_time_cycles': round(float(np.mean(lead_times)), 1) if lead_times else None,
        'false_alarm_rate': round(fp / (fp + tn) * 100, 1) if fp + tn else None,
    }


def main():
    datasets = {
        'NASA': load_nasa(),
        'TJU': load_cache('tju'),
        'XJTU': load_cache('xjtu'),
    }
    for name, cells in datasets.items():
        pos = sum(first_eol_cycle(t) is not None for t in cells.values())
        print(f"{name}: {len(cells)} usable cells, {pos} reach EOL (SOH<={EOL_SOH})")

    results = {'slope_window': SLOPE_WINDOW, 'eol_soh': EOL_SOH, 'datasets': {}}
    for name, cells in datasets.items():
        results['datasets'][name] = {}
        print(f"\n{name}:")
        print(f"  {'thr %/cyc':>10} {'P':>6} {'R':>6} {'F1':>6} {'lead':>7} {'FAR':>6}")
        for thr in THRESHOLDS:
            ev = evaluate(cells, thr)
            results['datasets'][name][f"{thr*100:.2f}"] = ev
            print(f"  {thr*100:>9.2f}% {ev['precision']:>6} {ev['recall']:>6} "
                  f"{ev['f1']:>6} {str(ev['mean_lead_time_cycles']):>7} "
                  f"{str(ev['false_alarm_rate']):>6}")

    OUT_JSON.parent.mkdir(exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)

    # Figure: F1 and lead time vs threshold
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = [t * 100 for t in THRESHOLDS]
    for name in datasets:
        f1s = [results['datasets'][name][f"{t*100:.2f}"]['f1'] for t in THRESHOLDS]
        axes[0].plot(x, f1s, 'o-', label=name)
        leads = [results['datasets'][name][f"{t*100:.2f}"]['mean_lead_time_cycles'] or np.nan
                 for t in THRESHOLDS]
        axes[1].plot(x, leads, 'o-', label=name)
    for ax, ylab in zip(axes, ['F1 (%)', 'Mean lead time (cycles)']):
        ax.axvline(0.17, color='gray', ls='--', lw=1, label='paper threshold' if ylab.startswith('F1') else None)
        ax.set_xlabel('Slope threshold (%/cycle)')
        ax.set_ylabel(ylab)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=150)
    print(f"\nSaved {OUT_JSON} and {OUT_FIG}")


if __name__ == '__main__':
    main()
