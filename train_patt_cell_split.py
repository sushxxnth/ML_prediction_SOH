"""
PATT Cell-Level Split Experiment

Addresses the leakage concern in the reported 99.9% PATT cycling/storage
classification accuracy: the original protocol (train_patt_classifier.py)
uses torch random_split over individual WINDOWS, so windows from the same
cell appear in both train and test. This script re-runs the identical model
and training protocol under two split regimes, multiple seeds:

  1. window-level random split (control; reproduces paper protocol)
  2. cell-level grouped split (no cell appears in more than one partition)

Note: the storage/cycling labels remain confounded with source dataset
(storage = Stanford calendar aging only; cycling = NASA/CALCE/Oxford/XJTU).
A cell-level split removes within-cell leakage but cannot remove the
dataset-level confound; that is discussed in the paper text.

Outputs: reports/patt_cell_split/patt_cell_split_results.json
"""

import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent))

from src.models.physics_aware_transformer import PATTDomainClassifier, PATTConfig, PhysicsInformedLoss
from src.data.unified_pipeline import UnifiedDataPipeline
from train_patt_classifier import DomainClassificationDataset


def load_data_with_cells(data_root: str = 'data', seed: int = 42):
    """Same feature extraction as train_patt_classifier.load_data, but keeps cell IDs."""
    rng = np.random.RandomState(seed)

    print("  Loading cycling data (NASA, CALCE, Oxford, XJTU)...")
    cycling_pipeline = UnifiedDataPipeline(data_root, use_lithium_features=False)
    try:
        cycling_pipeline.load_datasets(['nasa', 'calce', 'oxford', 'xjtu'])
    except Exception as e:
        print(f"  Warning: Could not load some datasets: {e}")
        cycling_pipeline.load_datasets(['nasa', 'xjtu'])

    cyc_feats, cyc_temps, cyc_times, cyc_cells = [], [], [], []
    for s in cycling_pipeline.samples:
        if not np.isfinite(s.soh) or s.soh < 0.5 or s.soh > 1.1:
            continue
        feat = np.zeros(5, dtype=np.float32)
        feat[0] = np.clip(s.soh, 0.5, 1.0)
        feat[1] = (getattr(s, 'temperature', 25) + 40) / 100
        if len(s.features) >= 3:
            feat[2] = np.clip(abs(s.features[2]) if np.isfinite(s.features[2]) else 0.015, 0, 0.1)
        else:
            feat[2] = 0.015
        feat[3] = np.clip(s.cycle_idx / 500, 0, 1) if hasattr(s, 'cycle_idx') else 0.5
        feat[4] = np.clip(s.features[4] if len(s.features) > 4 and np.isfinite(s.features[4]) else 0.05, 0, 0.2)
        feat = np.nan_to_num(feat, nan=0.5)
        cyc_feats.append(feat)
        cyc_temps.append(getattr(s, 'temperature', 25) + 273.15)
        cyc_times.append(feat[3])
        cyc_cells.append(f"CYC::{s.cell_id}")
    print(f"    {len(cyc_feats)} cycling windows from {len(set(cyc_cells))} cells")

    print("  Loading storage data (Stanford Calendar Aging)...")
    stanford_csv = Path(data_root) / 'stanford_calendar' / 'stanford_sampled_diagnostic.csv'
    sto_feats, sto_temps, sto_times, sto_cells = [], [], [], []
    df = pd.read_csv(stanford_csv)
    for cell_id, cell_df in df.groupby('cell_id'):
        cell_df = cell_df.sort_values('month')
        if len(cell_df) < 2:
            continue
        initial_capacity = cell_df.iloc[0]['capacity_ah']
        if initial_capacity <= 0:
            continue
        for idx, row in cell_df.iterrows():
            soh = row['capacity_ah'] / initial_capacity
            if not np.isfinite(soh) or soh < 0.5 or soh > 1.1:
                continue
            feat = np.zeros(5, dtype=np.float32)
            feat[0] = np.clip(soh, 0.5, 1.0)
            feat[1] = (25 + 40) / 100
            cap_values = cell_df['capacity_ah'].values
            time_values = cell_df['month'].values
            deg_rate = abs((cap_values[-1] - cap_values[0]) / (time_values[-1] - time_values[0] + 1e-6))
            feat[2] = np.clip(deg_rate / initial_capacity, 0, 0.05)
            feat[3] = np.clip(row['month'] / 70, 0, 1)
            if len(cell_df) >= 3:
                recent_caps = cell_df.iloc[max(0, idx - 2):idx + 1]['capacity_ah'].values
                feat[4] = np.clip(np.std(recent_caps) / initial_capacity, 0, 0.1)
            else:
                feat[4] = 0.02
            feat = np.nan_to_num(feat, nan=0.5)
            sto_feats.append(feat)
            sto_temps.append(25 + 273.15)
            sto_times.append(feat[3])
            sto_cells.append(f"STO::{cell_id}")
    print(f"    {len(sto_feats)} storage windows from {len(set(sto_cells))} cells")

    # Balance classes by window count (same cap as original protocol), seeded
    n_samples = min(len(cyc_feats), len(sto_feats), 10000)
    if len(cyc_feats) > n_samples:
        idx = rng.choice(len(cyc_feats), n_samples, replace=False)
        cyc_feats = [cyc_feats[i] for i in idx]
        cyc_temps = [cyc_temps[i] for i in idx]
        cyc_times = [cyc_times[i] for i in idx]
        cyc_cells = [cyc_cells[i] for i in idx]
    if len(sto_feats) > n_samples:
        idx = rng.choice(len(sto_feats), n_samples, replace=False)
        sto_feats = [sto_feats[i] for i in idx]
        sto_temps = [sto_temps[i] for i in idx]
        sto_times = [sto_times[i] for i in idx]
        sto_cells = [sto_cells[i] for i in idx]

    features = np.vstack([cyc_feats, sto_feats])
    labels = np.array([1] * len(cyc_feats) + [0] * len(sto_feats))
    temps = np.array(cyc_temps + sto_temps)
    times = np.array(cyc_times + sto_times)
    cells = np.array(cyc_cells + sto_cells)

    perm = rng.permutation(len(labels))
    return features[perm], labels[perm], temps[perm], times[perm], cells[perm]


def window_level_indices(n_total, seed):
    """70/15/15 random split over windows (paper protocol)."""
    g = np.random.RandomState(seed)
    perm = g.permutation(n_total)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    return perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]


def cell_level_indices(cells, labels, seed):
    """70/15/15 split over cells, stratified by class; no cell crosses partitions."""
    g = np.random.RandomState(seed)
    train_idx, val_idx, test_idx = [], [], []
    for cls in [0, 1]:
        cls_cells = np.unique(cells[labels == cls])
        g.shuffle(cls_cells)
        n = len(cls_cells)
        n_train = int(0.7 * n)
        n_val = int(0.15 * n)
        splits = {
            'train': set(cls_cells[:n_train]),
            'val': set(cls_cells[n_train:n_train + n_val]),
            'test': set(cls_cells[n_train + n_val:]),
        }
        for i, (c, l) in enumerate(zip(cells, labels)):
            if l != cls:
                continue
            if c in splits['train']:
                train_idx.append(i)
            elif c in splits['val']:
                val_idx.append(i)
            else:
                test_idx.append(i)
    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def run_batch(model, batch, device):
    features = batch['features'].to(device)
    temp = batch.get('temperature')
    time = batch.get('time_fraction')
    if temp is not None:
        temp = temp.to(device)
    if time is not None:
        time = time.to(device)
    return model(features, temp_kelvin=temp, time_fraction=time), features


def train_and_eval(dataset, tr_idx, va_idx, te_idx, seed, device='cpu',
                   epochs=50, batch_size=64, lr=1e-3):
    """Identical model/hyperparameters to train_patt_classifier.train_patt."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = DataLoader(Subset(dataset, tr_idx.tolist()), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, va_idx.tolist()), batch_size=batch_size)
    test_loader = DataLoader(Subset(dataset, te_idx.tolist()), batch_size=batch_size)

    config = PATTConfig(d_model=64, n_heads=4, n_layers=2, d_ff=128, dropout=0.1)
    model = PATTDomainClassifier(input_dim=5, config=config).to(device)
    criterion = PhysicsInformedLoss(lambda_temporal=0.1, lambda_physics=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_val_acc = 0.0
    best_state = None
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            labels_batch = batch['labels'].to(device)
            optimizer.zero_grad()
            outputs, features = run_batch(model, batch, device)
            loss = criterion(outputs, labels_batch, features[:, 2])['total']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                outputs, _ = run_batch(model, batch, device)
                val_preds.extend(outputs['prediction'].cpu().numpy())
                val_labels.extend(batch['labels'].numpy())
        val_acc = accuracy_score(val_labels, val_preds)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            outputs, _ = run_batch(model, batch, device)
            test_preds.extend(outputs['prediction'].cpu().numpy())
            test_labels.extend(batch['labels'].numpy())

    acc = accuracy_score(test_labels, test_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, test_preds, average='binary', pos_label=1, zero_division=0)
    cm = confusion_matrix(test_labels, test_preds)
    return {
        'accuracy': float(acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': cm.tolist(),
        'best_val_accuracy': float(best_val_acc),
        'n_train': len(tr_idx), 'n_val': len(va_idx), 'n_test': len(te_idx),
    }


def main():
    device = 'cpu'
    seeds = [42, 123, 999]
    out_dir = Path('reports/patt_cell_split')
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PATT SPLIT-LEVEL EXPERIMENT: window-level vs cell-level")
    print("=" * 70)

    # Load once with fixed seed so both regimes see the same window pool
    features, labels, temps, times, cells = load_data_with_cells('data', seed=42)
    dataset = DomainClassificationDataset(features, labels, temps, times)
    n_cells = {
        'cycling': int(len(np.unique(cells[labels == 1]))),
        'storage': int(len(np.unique(cells[labels == 0]))),
    }
    print(f"\nTotal windows: {len(labels)} "
          f"(cycling {int((labels == 1).sum())} / storage {int((labels == 0).sum())}), "
          f"cells: {n_cells}")

    results = {'window_level': [], 'cell_level': []}
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        tr, va, te = window_level_indices(len(labels), seed)
        r = train_and_eval(dataset, tr, va, te, seed, device)
        # count test windows whose cell also has windows in train (leakage measure)
        train_cells = set(cells[tr])
        r['test_windows_with_cell_in_train'] = int(sum(c in train_cells for c in cells[te]))
        results['window_level'].append({'seed': seed, **r})
        print(f"  window-level: acc={r['accuracy']:.4f}, f1={r['f1']:.4f} "
              f"(leaked test windows: {r['test_windows_with_cell_in_train']}/{r['n_test']})")

        tr, va, te = cell_level_indices(cells, labels, seed)
        r = train_and_eval(dataset, tr, va, te, seed, device)
        r['test_cells'] = int(len(np.unique(cells[te])))
        assert not (set(cells[tr]) & set(cells[te])), "cell leakage in grouped split"
        assert not (set(cells[va]) & set(cells[te])), "cell leakage in grouped split"
        results['cell_level'].append({'seed': seed, **r})
        print(f"  cell-level:   acc={r['accuracy']:.4f}, f1={r['f1']:.4f} "
              f"({r['test_cells']} held-out cells)")

    summary = {}
    for regime in ['window_level', 'cell_level']:
        accs = [r['accuracy'] for r in results[regime]]
        f1s = [r['f1'] for r in results[regime]]
        summary[regime] = {
            'accuracy_mean': float(np.mean(accs)), 'accuracy_std': float(np.std(accs)),
            'f1_mean': float(np.mean(f1s)), 'f1_std': float(np.std(f1s)),
        }

    output = {
        'experiment': 'PATT window-level vs cell-level split',
        'date': datetime.now().isoformat(),
        'seeds': seeds,
        'protocol': 'identical model/hyperparams to train_patt_classifier.py '
                    '(PATT d64/h4/l2, 50 epochs, AdamW 1e-3, PhysicsInformedLoss)',
        'n_windows': int(len(labels)),
        'n_cells': n_cells,
        'caveat': 'storage/cycling labels remain confounded with source dataset; '
                  'cell-level split removes within-cell leakage only',
        'summary': summary,
        'runs': results,
    }
    with open(out_dir / 'patt_cell_split_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 70)
    print("SUMMARY (mean ± std over seeds)")
    for regime, s in summary.items():
        print(f"  {regime:13s}: acc = {s['accuracy_mean']:.4f} ± {s['accuracy_std']:.4f}, "
              f"f1 = {s['f1_mean']:.4f} ± {s['f1_std']:.4f}")
    print(f"\nResults saved to {out_dir / 'patt_cell_split_results.json'}")


if __name__ == '__main__':
    main()
