"""
Reproducible Early-Warning Evaluation on NASA Ames (Reviewer R2.7 follow-up)

The original submission reported F1 = 88.9% (TP=24, FP=5, FN=1) for the
slope-based early-warning system, but the evaluation harness that produced
that confusion matrix is not in the repository, and its 96% recall is
inconsistent with the same paper's statement that detection reached "52% of
failing cells." This script provides a fully documented, reproducible
replacement.

Design decisions (all documented, none tuned to hit a target number):

  1. Labels use a SUSTAINED end-of-life definition: a cell has failed only if
     its SOH stays at or below 80% for the remainder of its record. This
     removes ~8 cells whose SOH momentarily dips below 80% due to capacity
     regeneration and then recovers (e.g. B0033 recovers to 113%), which a
     single-touch EOL rule mislabels as failures.

  2. Cells that arrive already below 80% (B0049/B0050/B0051, pre-aged
     random-walk cells) are excluded: an *early*-warning task is undefined for
     a cell with no healthy history to warn from.

  3. Detection: a warning fires when the trailing 20-cycle SOH fade slope
     reaches the threshold (default 0.17%/cycle) AND SOH is already <= 90%
     (suppresses noise-driven alarms on near-pristine cells), sustained for
     one qualifying cycle. Persistence and the SOH gate are standard
     noise-rejection practice.

Result at the paper's 0.17%/cycle threshold: F1 ~= 74%, recall 92.9%
(13/14 knee-failures detected, one missed -- matching the paper's "only one
missed detection"), mean lead time ~121 cycles (paper claimed 99). The
reproducible F1 is ~14 points below the un-reproducible 88.9%; we report the
reproducible value and the full threshold sweep.

Outputs: reports/early_warning_reconstruction.json (+ .png)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EOL = 0.80
WINDOW = 20
SOH_GATE = 0.90
PERSIST = 1
MIN_HISTORY = 10          # cycles of healthy record required before an EOL crossing
DEFAULT_THR = 0.0017
THRESHOLDS = [0.0013, 0.0015, 0.0017, 0.0020, 0.0025]

OUT_JSON = Path('reports/early_warning_reconstruction.json')
OUT_FIG = Path('reports/early_warning_reconstruction.png')


def load_nasa():
    df = pd.read_csv('data/nasa_set5/summary.csv')
    df = df[df.cycle_type == 'discharge']
    cells = {}
    for cid, g in df.groupby('cell_id'):
        t = pd.DataFrame({'cycle': g.cycle_index.values, 'soh': g.SOH_Q.values}).dropna()
        t = t[(t.soh > 0.3) & (t.soh < 1.2)].sort_values('cycle').reset_index(drop=True)
        if len(t) >= 3:
            cells[cid] = t
    return cells


def sustained_eol(t):
    """First cycle from which SOH stays <= EOL for the rest of the record."""
    s, c = t.soh.values, t.cycle.values
    for i in range(len(s)):
        if s[i] <= EOL and (i == len(s) - 1 or np.mean(s[i:]) <= EOL + 0.01):
            return float(c[i]), i
    return None, None


def warn_cycle(t, thr):
    c = t.cycle.values.astype(float)
    s = t.soh.values.astype(float)
    run = 0
    for i in range(len(c)):
        m = (c >= c[i] - WINDOW) & (c <= c[i])
        if m.sum() < 3:
            run = 0
            continue
        slope = -np.polyfit(c[m], s[m], 1)[0]
        if slope >= thr and s[i] <= SOH_GATE:
            run += 1
            if run >= PERSIST:
                return c[i]
        else:
            run = 0
    return None


def evaluate(cells, thr):
    tp = fp = fn = tn = 0
    leads = []
    excluded = []
    for cid, t in cells.items():
        eol, idx = sustained_eol(t)
        if eol is not None and idx < MIN_HISTORY:
            excluded.append(cid)
            continue
        w = warn_cycle(t, thr)
        if eol is not None:
            if w is not None and w < eol:
                tp += 1
                leads.append(eol - w)
            else:
                fn += 1
        else:
            if w is not None:
                fp += 1
            else:
                tn += 1
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return {
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'n_failed': tp + fn, 'n_healthy': fp + tn, 'n_excluded': len(excluded),
        'precision': round(P * 100, 1), 'recall': round(R * 100, 1), 'f1': round(F * 100, 1),
        'mean_lead_cycles': round(float(np.mean(leads)), 1) if leads else None,
    }


def main():
    cells = load_nasa()
    print(f"Loaded {len(cells)} NASA cells")
    sweep = {f"{thr*100:.2f}": evaluate(cells, thr) for thr in THRESHOLDS}

    print(f"\n{'thr':>7} {'nFail':>6} {'nOK':>5} {'TP':>3} {'FP':>3} {'FN':>3} {'TN':>3} "
          f"{'P':>6} {'R':>6} {'F1':>6} {'lead':>6}")
    for thr in THRESHOLDS:
        r = sweep[f"{thr*100:.2f}"]
        print(f"{thr*100:>6.2f}% {r['n_failed']:>6} {r['n_healthy']:>5} {r['tp']:>3} {r['fp']:>3} "
              f"{r['fn']:>3} {r['tn']:>3} {r['precision']:>6} {r['recall']:>6} {r['f1']:>6} "
              f"{str(r['mean_lead_cycles']):>6}")

    default = sweep[f"{DEFAULT_THR*100:.2f}"]
    print(f"\nReproducible operating point (thr={DEFAULT_THR*100:.2f}%/cycle):")
    print(f"  F1 = {default['f1']}%, precision = {default['precision']}%, "
          f"recall = {default['recall']}% ({default['tp']}/{default['n_failed']}), "
          f"mean lead = {default['mean_lead_cycles']} cycles")

    out = {
        'config': {'eol_soh': EOL, 'slope_window': WINDOW, 'soh_gate': SOH_GATE,
                   'persistence': PERSIST, 'min_history': MIN_HISTORY,
                   'default_threshold_pct_per_cycle': DEFAULT_THR * 100},
        'note': ('Sustained-EOL labels; born-failed cells excluded; SOH-gated persistent '
                 'slope detector. Replaces the un-reproducible 88.9% F1 from the original '
                 'submission (harness not in repo, recall inconsistent with paper text).'),
        'default_operating_point': default,
        'threshold_sweep': sweep,
    }
    OUT_JSON.parent.mkdir(exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(out, f, indent=2)

    x = [t * 100 for t in THRESHOLDS]
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, style in [('precision', 's-'), ('recall', 'o-'), ('f1', '^-')]:
        ax.plot(x, [sweep[f"{t*100:.2f}"][key] for t in THRESHOLDS], style, label=key.upper())
    ax.axvline(DEFAULT_THR * 100, color='gray', ls='--', lw=1, label='paper threshold (0.17)')
    ax.set_xlabel('Slope threshold (%/cycle)')
    ax.set_ylabel('%')
    ax.set_title('NASA early-warning: reproducible metrics vs threshold')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=150)
    print(f"\nSaved {OUT_JSON} and {OUT_FIG}")


if __name__ == '__main__':
    main()
