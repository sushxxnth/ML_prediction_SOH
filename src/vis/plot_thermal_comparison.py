"""
Plot Thermal Generalization Comparison: Room (25C) vs High (40C)

This script loads a model checkpoint and evaluates on NASA Randomized
(room + 40C) test samples, then plots side-by-side SOH (true vs predicted)
for room-temp and high-temp groups.

Usage:
    python3 src/vis/plot_thermal_comparison.py \
        --model reports/cross_dataset_randomized/model.pt

Outputs:
    reports/thermal_generalization/thermal_comparison.png
    reports/thermal_generalization/thermal_comparison.svg
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.context.extended_context import normalize_temperature, normalize_crate
from src.data.randomwalk_loader import RandomizedBatteryLoader
from src.train.train_multi_dataset import ContextAwareSOHPredictor


def build_samples(loader, cell_ids: List[str], temp_c: float) -> List[Tuple[np.ndarray, float, np.ndarray, int]]:
    samples = []
    cells = loader.cells
    temp_norm = normalize_temperature(temp_c)
    for cid in cell_ids:
        cell = cells[cid]
        cap0 = cell.initial_capacity if cell.initial_capacity > 0 else None
        for cyc in cell.cycles:
            cap = cyc.capacity
            soh = cap / cap0 if cap0 else 1.0
            feat = cyc.to_feature_vector().astype(np.float32)
            current_max = abs(cyc.current_max)
            charge_norm = normalize_crate(current_max / max(cell.nominal_capacity, 0.1))
            discharge_norm = charge_norm
            ctx = np.array([temp_norm, charge_norm, discharge_norm], dtype=np.float32)
            samples.append((feat, float(soh), ctx, 0))  # chem_id=0 (LCO)
    return samples


def to_loader(samples, batch_size=64):
    class _DS(torch.utils.data.Dataset):
        def __len__(self): return len(samples)
        def __getitem__(self, idx):
            f, soh, ctx, chem = samples[idx]
            return {
                'features': torch.tensor(f, dtype=torch.float32),
                'soh': torch.tensor(soh, dtype=torch.float32),
                'context': torch.tensor(ctx, dtype=torch.float32),
                'chem_id': torch.tensor(chem, dtype=torch.long)
            }
    return torch.utils.data.DataLoader(_DS(), batch_size=batch_size, shuffle=False)


def predict(model, samples, device='cpu'):
    loader = to_loader(samples)
    model.eval()
    preds = []
    truths = []
    with torch.no_grad():
        for batch in loader:
            f = batch['features'].to(device)
            c = batch['context'].to(device)
            chem = batch['chem_id'].to(device).unsqueeze(-1)
            soh_true = batch['soh'].cpu().numpy()
            soh_pred, _, _ = model(f, c, chem, return_latent=True)
            p = soh_pred.cpu().numpy()
            p = np.atleast_1d(p.squeeze())
            t = np.atleast_1d(soh_true.squeeze())
            preds.extend(p)
            truths.extend(t)
    return np.array(truths), np.array(preds)


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred))) if len(y_true) else np.nan


def r2(y_true, y_pred):
    if len(y_true) == 0:
        return np.nan
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def main():
    parser = argparse.ArgumentParser(description="Plot Thermal Generalization Comparison")
    parser.add_argument('--model', type=str, default='reports/cross_dataset_randomized/model.pt',
                        help='Path to trained model checkpoint')
    parser.add_argument('--room_path', type=str,
                        default='data/randomized_battery_usage/raw_1/Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/data/Matlab',
                        help='Path to room-temp randomized data')
    parser.add_argument('--high_path', type=str,
                        default='data/randomized_battery_usage/raw_40_high',
                        help='Path to high-temp randomized data')
    args = parser.parse_args()

    out_dir = Path('reports/thermal_generalization')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    room_loader = RandomizedBatteryLoader(args.room_path, use_cache=False)
    room_loader.load()
    high_loader = RandomizedBatteryLoader(args.high_path, use_cache=False)
    high_loader.load()

    # Define cells
    room_cells = ['RW9', 'RW10', 'RW11', 'RW12']
    high_cells = ['RW25', 'RW26', 'RW27', 'RW28']

    room_samples = build_samples(room_loader, room_cells, temp_c=25.0)
    high_samples = build_samples(high_loader, high_cells, temp_c=40.0)

    # Load model
    ckpt = torch.load(args.model, map_location='cpu')
    model = ContextAwareSOHPredictor(
        feature_dim=9,
        context_numeric_dim=3,
        chem_emb_dim=4,
        hidden_dim=192,
        latent_dim=96
    )
    model.load_state_dict(ckpt['model_state_dict'])

    # Predict
    y_room, yhat_room = predict(model, room_samples)
    y_high, yhat_high = predict(model, high_samples)

    # Metrics
    mae_room = mae(y_room, yhat_room)
    mae_high = mae(y_high, yhat_high)
    r2_room = r2(y_room, yhat_room)
    r2_high = r2(y_high, yhat_high)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    # Room
    axes[0].plot(y_room, color='black', linewidth=1.8, label='True SOH')
    axes[0].plot(yhat_room, color='blue', linestyle='--', linewidth=1.6, label='Pred SOH')
    axes[0].axhline(0.8, color='gray', linestyle=':', linewidth=1.2, label='80% EOL')
    axes[0].set_title(f'Room Temp (25°C)\nMAE={mae_room:.3f}, R²={r2_room:.3f}', fontsize=11)
    axes[0].set_xlabel('Cycle (reference benchmarks)')
    axes[0].set_ylabel('SOH')
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    # High
    axes[1].plot(y_high, color='black', linewidth=1.8, label='True SOH')
    axes[1].plot(yhat_high, color='red', linestyle='--', linewidth=1.6, label='Pred SOH')
    axes[1].axhline(0.8, color='gray', linestyle=':', linewidth=1.2, label='80% EOL')
    axes[1].set_title(f'High Temp (40°C)\nMAE={mae_high:.3f}, R²={r2_high:.3f}', fontsize=11)
    axes[1].set_xlabel('Cycle (reference benchmarks)')
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.suptitle('Thermal Comparison: Room vs High Temperature', fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_dir / 'thermal_comparison.png', dpi=150)
    plt.savefig(out_dir / 'thermal_comparison.svg')
    print(f"Saved plots to {out_dir / 'thermal_comparison.png'} and .svg")


if __name__ == '__main__':
    main()

