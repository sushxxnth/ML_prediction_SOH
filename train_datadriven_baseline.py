"""
Pure Data-Driven Baseline for Causal Attribution (Reviewer R3.2 / R2.4)

Trains the SAME multi-head architecture (PINNCausalAttributionModel) with
use_physics_priors=False, i.e. no rule-based prior injection, using the
identical training protocol as train_pinn_correct.py (same augmentation,
same loss computation, same optimizer/schedule).

Two evaluation protocols:
  A. Paper protocol: train on all 75 scenarios (augmented), evaluate on the
     same 75 (matches how pinn_causal_retrained.pt was produced).
  B. Leave-one-dataset-out (LODO): train on 4 scenario groups, evaluate on
     the held-out 5th group. This is the honest generalization test and is
     run for BOTH variants (with and without physics priors).

Outputs: reports/causal_attribution/unified_validation/datadriven_baseline_results.json
"""

import sys
import json
import random
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent))

from src.models.pinn_causal_attribution import PINNCausalAttributionModel
from train_pinn_correct import (
    get_all_scenarios, generate_augmented_training_data, BatteryDataset,
)

DATASETS = ['NASA', 'Panasonic', 'Nature', 'Randomized', 'HUST']
MECHS = ['sei_growth', 'lithium_plating', 'am_loss', 'electrolyte', 'corrosion']

OUT_PATH = Path('reports/causal_attribution/unified_validation/datadriven_baseline_results.json')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def stacked_logits(output, batch_size):
    """Same logit extraction as train_pinn_correct.py (CE over attributions)."""
    attr_list = []
    for m in MECHS:
        attr = output['attributions'].get(m, torch.zeros(batch_size))
        if attr.dim() == 0:
            attr = attr.expand(batch_size)
        attr_list.append(attr)
    logits = torch.stack(attr_list, dim=1)
    if logits.dim() > 2:
        logits = logits.reshape(batch_size, 5)
    return logits


def train_once(train_scenarios, use_priors, epochs=300, lr=0.002,
               augment_factor=30, batch_size=32, seed=42, log_prefix=""):
    set_seed(seed)
    train_list = generate_augmented_training_data(train_scenarios, augment_factor)
    dataset = BatteryDataset(train_list)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = PINNCausalAttributionModel(feature_dim=9, context_dim=6,
                                       use_physics_priors=use_priors)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for batch in loader:
            features, context, target = batch['features'], batch['context'], batch['target']
            optimizer.zero_grad()
            output = model(features, context)
            logits = stacked_logits(output, context.shape[0])
            loss = criterion(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        if (epoch + 1) % 50 == 0:
            print(f"  {log_prefix} epoch {epoch+1}/{epochs} loss={loss.item():.4f}", flush=True)

    return model


@torch.no_grad()
def evaluate(model, scenarios):
    model.eval()
    dataset = BatteryDataset(scenarios)
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(scenarios), shuffle=False)
    by_dataset = defaultdict(lambda: {'correct': 0, 'total': 0})
    by_mech = defaultdict(lambda: {'correct': 0, 'total': 0})
    correct = 0
    for batch in loader:
        output = model(batch['features'], batch['context'])
        logits = stacked_logits(output, batch['context'].shape[0])
        pred = logits.argmax(dim=1)
        for i, s in enumerate(scenarios):
            ok = bool(pred[i].item() == s['expected_idx'])
            by_dataset[s['dataset']]['total'] += 1
            by_mech[s['expected']]['total'] += 1
            if ok:
                correct += 1
                by_dataset[s['dataset']]['correct'] += 1
                by_mech[s['expected']]['correct'] += 1
    return {
        'accuracy': correct / len(scenarios),
        'correct': correct,
        'total': len(scenarios),
        'by_dataset': {k: dict(v) for k, v in by_dataset.items()},
        'by_mechanism': {k: dict(v) for k, v in by_mech.items()},
    }


def main(epochs=300, seeds=(42, 123, 7)):
    scenarios = get_all_scenarios()
    print(f"Loaded {len(scenarios)} scenarios", flush=True)

    results = {
        'date': datetime.now().isoformat(),
        'epochs': epochs,
        'protocol_A_train_on_all': {},
        'protocol_B_lodo': {},
    }

    # Protocol A: paper protocol, both variants, multiple seeds
    for use_priors, label in [(False, 'data_driven'), (True, 'hybrid_pinn')]:
        accs = []
        details = []
        for seed in seeds:
            print(f"\n[Protocol A] {label}, seed={seed}", flush=True)
            model = train_once(scenarios, use_priors, epochs=epochs, seed=seed,
                               log_prefix=f"A/{label}/s{seed}")
            ev = evaluate(model, scenarios)
            print(f"  -> accuracy {ev['correct']}/{ev['total']} = {ev['accuracy']*100:.1f}%", flush=True)
            accs.append(ev['accuracy'])
            details.append({'seed': seed, **ev})
        results['protocol_A_train_on_all'][label] = {
            'mean_accuracy': float(np.mean(accs)),
            'std_accuracy': float(np.std(accs)),
            'runs': details,
        }
        _save(results)

    # Protocol B: leave-one-dataset-out, both variants, single seed
    for use_priors, label in [(False, 'data_driven'), (True, 'hybrid_pinn')]:
        folds = {}
        total_correct, total_n = 0, 0
        for held_out in DATASETS:
            train_s = [s for s in scenarios if s['dataset'] != held_out]
            test_s = [s for s in scenarios if s['dataset'] == held_out]
            print(f"\n[Protocol B] {label}, held-out={held_out} "
                  f"(train {len(train_s)}, test {len(test_s)})", flush=True)
            model = train_once(train_s, use_priors, epochs=epochs, seed=42,
                               log_prefix=f"B/{label}/{held_out}")
            ev = evaluate(model, test_s)
            print(f"  -> held-out accuracy {ev['correct']}/{ev['total']} = {ev['accuracy']*100:.1f}%", flush=True)
            folds[held_out] = ev
            total_correct += ev['correct']
            total_n += ev['total']
        results['protocol_B_lodo'][label] = {
            'pooled_accuracy': total_correct / total_n,
            'pooled_correct': total_correct,
            'pooled_total': total_n,
            'folds': folds,
        }
        _save(results)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for proto in ['protocol_A_train_on_all', 'protocol_B_lodo']:
        print(f"\n{proto}:")
        for label, r in results[proto].items():
            if 'mean_accuracy' in r:
                print(f"  {label:15}: {r['mean_accuracy']*100:.1f}% ± {r['std_accuracy']*100:.1f}%")
            else:
                print(f"  {label:15}: {r['pooled_accuracy']*100:.1f}% "
                      f"({r['pooled_correct']}/{r['pooled_total']})")
    print(f"\nSaved to {OUT_PATH}")


def _save(results):
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300)
    args = parser.parse_args()
    main(epochs=args.epochs)
