"""
Physics-value experiments: quantify what the physics priors buy, honestly.

Addresses the editor's core concern ("physics not clearly presented") with two
measurable claims that do not rest on the modest 2.7-point in-distribution
accuracy margin:

  1. DATA EFFICIENCY: priors act as training-time regularizers, so their value
     should grow as labeled scenarios shrink. Train the hybrid (priors) and the
     purely data-driven variant on stratified subsets of the 75 canonical
     scenarios (20%/50%/80%, 3 seeds each) and evaluate on the HELD-OUT
     scenarios only (never on scenarios seen in training).

  2. PHYSICAL CONSISTENCY: sweep a dense grid of operating contexts and count
     physically impossible dominant attributions. The only rule used is
     textbook-uncontroversial: lithium plating requires charging current, so
     attributing plating as dominant in storage (no charging) is a violation.
     The hybrid with the hard mode-gate is 0% by construction; the data-driven
     variant is measured empirically.

Run:  PYTHONPATH=. arch -arm64 python3 run_physics_value_experiments.py
Output: reports/causal_attribution/unified_validation/physics_value_experiments.json
"""

import sys
import json
import random
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from train_datadriven_baseline import train_once, evaluate, set_seed
from train_pinn_correct import get_all_scenarios

OUT = Path('reports/causal_attribution/unified_validation/physics_value_experiments.json')
FRACTIONS = (0.2, 0.5, 0.8)
SEEDS = (42, 123, 7)
EPOCHS = 300
DATASETS = ['NASA', 'Panasonic', 'Nature', 'Randomized', 'HUST']


def stratified_subset(scenarios, fraction, seed):
    """Sample a fraction of scenarios stratified by dataset group."""
    rng = np.random.default_rng(seed)
    train_idx = []
    for ds in DATASETS:
        idx = [i for i, s in enumerate(scenarios) if s['dataset'] == ds]
        n = max(1, int(round(fraction * len(idx))))
        train_idx.extend(rng.choice(idx, size=n, replace=False).tolist())
    train_set = set(train_idx)
    train = [scenarios[i] for i in sorted(train_set)]
    heldout = [scenarios[i] for i in range(len(scenarios)) if i not in train_set]
    return train, heldout


def data_efficiency():
    scenarios = get_all_scenarios()
    results = {}
    for frac in FRACTIONS:
        for variant, use_priors in (('hybrid', True), ('data_driven', False)):
            accs_held, accs_all = [], []
            for seed in SEEDS:
                train, heldout = stratified_subset(scenarios, frac, seed)
                tag = f"frac{frac}/{variant}/seed{seed}"
                print(f"[data-eff] {tag}: train={len(train)} heldout={len(heldout)}", flush=True)
                model = train_once(train, use_priors, epochs=EPOCHS, seed=seed,
                                   log_prefix=tag)
                accs_held.append(evaluate(model, heldout)['accuracy'])
                accs_all.append(evaluate(model, scenarios)['accuracy'])
                print(f"[data-eff] {tag}: heldout={accs_held[-1]:.3f} all75={accs_all[-1]:.3f}", flush=True)
            results[f"{variant}@{frac}"] = {
                'heldout_mean': float(np.mean(accs_held)),
                'heldout_std': float(np.std(accs_held)),
                'all75_mean': float(np.mean(accs_all)),
                'all75_std': float(np.std(accs_all)),
                'heldout_runs': [float(a) for a in accs_held],
            }
    return results


def context_grid(n=4000, seed=0):
    """Random physically-realizable contexts spanning the deployment envelope.
    Context layout (test_unified_validation.make_context): [temp_norm=(T-25)/20,
    charge_c/3, discharge_c/4, soc, 0.0, mode (cycling=1.0, storage=0.0)].
    Storage rows have zero charge/discharge current by definition."""
    from test_unified_validation import BASE_FEATURES
    rng = np.random.default_rng(seed)
    ctx = np.zeros((n, 6), dtype=np.float32)
    ctx[:, 0] = rng.uniform(-1.5, 1.25, n)          # -5degC .. 50degC
    is_cyc = rng.uniform(0, 1, n) > 0.5
    ctx[:, 5] = is_cyc.astype(np.float32)           # mode: 1 cycling / 0 storage
    ctx[is_cyc, 1] = rng.uniform(0.2, 3.0, is_cyc.sum()) / 3.0   # charge C-rate
    ctx[is_cyc, 2] = rng.uniform(0.2, 3.0, is_cyc.sum()) / 4.0   # discharge C-rate
    ctx[:, 3] = rng.uniform(0.05, 1.0, n)           # SOC
    # index 4 stays 0.0, matching the scenario builder
    feats = (BASE_FEATURES[None, :] +
             rng.normal(0, 0.05, (n, 9))).clip(0, 1).astype(np.float32)
    return torch.tensor(feats), torch.tensor(ctx), torch.tensor(is_cyc)


@torch.no_grad()
def violation_rate(model, feats, ctx, is_cyc):
    """Fraction of STORAGE contexts where plating is the dominant attribution."""
    from train_datadriven_baseline import stacked_logits, MECHS
    model.eval()
    out = model(feats, ctx)
    logits = stacked_logits(out, ctx.shape[0])
    pred = logits.argmax(dim=1)
    plating_idx = MECHS.index('lithium_plating')
    storage = ~is_cyc
    n_storage = int(storage.sum())
    viol = int(((pred == plating_idx) & storage).sum())
    return viol, n_storage


def physical_consistency():
    scenarios = get_all_scenarios()
    feats, ctx, is_cyc = context_grid()
    results = {}
    for variant, use_priors in (('hybrid', True), ('data_driven', False)):
        rates = []
        for seed in SEEDS:
            set_seed(seed)
            model = train_once(scenarios, use_priors, epochs=EPOCHS, seed=seed,
                               log_prefix=f"[consistency] {variant}/seed{seed}")
            viol, n_storage = violation_rate(model, feats, ctx, is_cyc)
            rates.append(viol / n_storage)
            print(f"[consistency] {variant}/seed{seed}: {viol}/{n_storage} "
                  f"storage-plating violations ({100*rates[-1]:.2f}%)", flush=True)
        results[variant] = {
            'violation_rate_mean': float(np.mean(rates)),
            'violation_rate_std': float(np.std(rates)),
            'runs': [float(r) for r in rates],
            'n_storage_contexts': n_storage,
        }
    return results


def main():
    out = {
        'date': datetime.now().isoformat(),
        'epochs': EPOCHS,
        'seeds': list(SEEDS),
        'fractions': list(FRACTIONS),
        'data_efficiency': data_efficiency(),
        'physical_consistency': physical_consistency(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {OUT}")

    print("\n=== DATA EFFICIENCY (held-out scenario accuracy) ===")
    de = out['data_efficiency']
    for frac in FRACTIONS:
        h = de[f"hybrid@{frac}"]; d = de[f"data_driven@{frac}"]
        print(f"  {int(frac*100)}% data: hybrid {h['heldout_mean']:.3f}±{h['heldout_std']:.3f}"
              f"  vs data-driven {d['heldout_mean']:.3f}±{d['heldout_std']:.3f}")
    print("\n=== PHYSICAL CONSISTENCY (storage-plating violation rate) ===")
    for v, r in out['physical_consistency'].items():
        print(f"  {v}: {100*r['violation_rate_mean']:.2f}% ± {100*r['violation_rate_std']:.2f}%")


if __name__ == '__main__':
    main()
