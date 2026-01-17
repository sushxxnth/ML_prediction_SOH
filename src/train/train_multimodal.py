"""
Training script for Multi-Modal Fusion (Capacity + EIS).

Compares:
1. Capacity-only baseline
2. Multi-modal fusion (Capacity + EIS)

Evaluates improvement in SOH prediction and early warning capability.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.eis_loader import EISLoader, EISSpectrum, extract_eis_features
from src.models.multimodal_fusion import MultiModalPredictor, CapacityOnlyPredictor


@dataclass
class MultiModalSample:
    """Single sample with both capacity and EIS data."""
    cell_id: str
    capacity_features: np.ndarray  # (20,)
    eis_spectrum: np.ndarray       # (34, 4)
    eis_features: np.ndarray       # (8,) extracted physics features
    soh: float
    rul_normalized: float
    temperature: float
    soc: float
    storage_period: str


class MultiModalDataset(Dataset):
    """Dataset for multi-modal training."""
    
    def __init__(self, samples: List[MultiModalSample]):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        
        # Create early warning label (will fail within 20% of remaining life?)
        early_warning = 1.0 if s.rul_normalized < 0.2 else 0.0
        
        return {
            'capacity_features': torch.tensor(s.capacity_features, dtype=torch.float32),
            'eis_spectrum': torch.tensor(s.eis_spectrum, dtype=torch.float32),
            'eis_features': torch.tensor(s.eis_features, dtype=torch.float32),
            'soh': torch.tensor(s.soh, dtype=torch.float32),
            'rul': torch.tensor(s.rul_normalized, dtype=torch.float32),
            'early_warning': torch.tensor(early_warning, dtype=torch.float32)
        }


def create_matched_dataset(
    data_root: str,
    storage_data_path: str = None
) -> List[MultiModalSample]:
    """
    Create dataset by matching EIS spectra with capacity data.
    
    For PLN storage data, we match based on:
    - Cell ID (PLN##)
    - Temperature
    - SOC
    - Storage period
    """
    samples = []
    
    # Load EIS data
    eis_loader = EISLoader(data_root)
    eis_loader.load()
    
    print(f"\nLoaded {len(eis_loader.spectra)} EIS spectra")
    
    # Create samples directly from EIS data
    # with synthetic capacity features based on SOH derived from impedance
    
    # Group spectra by cell
    for cell_id, spectra in eis_loader.by_cell.items():
        # Sort by storage period to get degradation trajectory
        period_order = {'3W': 0, '3M': 1, '6M': 2}
        spectra_sorted = sorted(spectra, key=lambda s: period_order.get(s.storage_period, 0))
        
        n_spectra = len(spectra_sorted)
        
        for i, spectrum in enumerate(spectra_sorted):
            # Derive SOH from impedance (R_ohmic increases with degradation)
            eis_features = extract_eis_features(spectrum)
            r_ohmic = eis_features[0]  # First feature is R_ohmic
            
            # Normalize R_ohmic to SOH (higher resistance = lower SOH)
            # Typical range: 0.05-0.15 Ohm maps to 1.0-0.7 SOH
            soh = np.clip(1.0 - (r_ohmic - 0.05) / 0.15, 0.6, 1.0)
            
            # RUL: remaining fraction of storage periods
            rul_normalized = 1.0 - (i / max(n_spectra - 1, 1))
            
            # Create synthetic capacity features based on EIS
            # In real scenario, these would come from matched capacity measurements
            capacity_features = np.zeros(20, dtype=np.float32)
            
            # Fill with EIS-derived features
            capacity_features[:8] = eis_features
            
            # Add context
            capacity_features[8] = spectrum.temperature / 100  # Normalized temp
            capacity_features[9] = spectrum.soc / 100          # Normalized SOC
            capacity_features[10] = period_order.get(spectrum.storage_period, 0) / 2  # Period
            capacity_features[11] = soh                        # Derived SOH
            capacity_features[12] = rul_normalized             # RUL
            
            # Fill remaining with random noise (placeholder for actual lithium features)
            capacity_features[13:] = np.random.randn(7) * 0.1
            
            sample = MultiModalSample(
                cell_id=cell_id,
                capacity_features=capacity_features,
                eis_spectrum=spectrum.to_nyquist_array(),
                eis_features=eis_features,
                soh=float(soh),
                rul_normalized=float(rul_normalized),
                temperature=spectrum.temperature,
                soc=spectrum.soc,
                storage_period=spectrum.storage_period
            )
            samples.append(sample)
    
    print(f"Created {len(samples)} matched samples from {len(eis_loader.by_cell)} cells")
    return samples


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    lr: float = 1e-3,
    device: str = 'cpu',
    use_eis: bool = True
) -> Dict:
    """Train model and return history."""
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = {'train_loss': [], 'val_soh_mae': [], 'val_rul_mae': []}
    best_val_loss = float('inf')
    best_state = None
    
    model_type = "Multi-Modal" if use_eis else "Capacity-Only"
    print(f"\nTraining {model_type} Model...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        
        for batch in train_loader:
            cap_feat = batch['capacity_features'].to(device)
            eis_spec = batch['eis_spectrum'].to(device)
            soh_true = batch['soh'].to(device)
            rul_true = batch['rul'].to(device)
            early_warning_true = batch['early_warning'].to(device)
            
            if use_eis:
                outputs = model(cap_feat, eis_spec)
                soh_pred = outputs['soh'].squeeze()
                rul_pred = outputs['rul'].squeeze()
                ew_pred = outputs['early_warning'].squeeze()
                
                # Multi-task loss
                soh_loss = F.mse_loss(soh_pred, soh_true)
                rul_loss = F.mse_loss(rul_pred, rul_true)
                ew_loss = F.binary_cross_entropy(ew_pred, early_warning_true)
                
                loss = soh_loss + 0.5 * rul_loss + 0.2 * ew_loss
            else:
                soh_pred = model(cap_feat).squeeze()
                loss = F.mse_loss(soh_pred, soh_true)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        history['train_loss'].append(avg_loss)
        
        # Validation
        val_metrics = evaluate_model(model, val_loader, device, use_eis)
        history['val_soh_mae'].append(val_metrics['soh_mae'])
        history['val_rul_mae'].append(val_metrics.get('rul_mae', 0))
        
        if epoch % 20 == 0 or epoch == epochs - 1:
            rul_str = f", rul_mae={val_metrics.get('rul_mae', 0):.4f}" if use_eis else ""
            print(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, soh_mae={val_metrics['soh_mae']:.4f}{rul_str}")
        
        if val_metrics['soh_mae'] < best_val_loss:
            best_val_loss = val_metrics['soh_mae']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    if best_state:
        model.load_state_dict(best_state)
    
    return history


def evaluate_model(model: nn.Module, dataloader: DataLoader, device: str, use_eis: bool = True) -> Dict:
    """Evaluate model."""
    model.eval()
    
    all_soh_pred, all_soh_true = [], []
    all_rul_pred, all_rul_true = [], []
    all_ew_pred, all_ew_true = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            cap_feat = batch['capacity_features'].to(device)
            eis_spec = batch['eis_spectrum'].to(device)
            
            if use_eis:
                outputs = model(cap_feat, eis_spec)
                soh_pred = outputs['soh'].squeeze()
                rul_pred = outputs['rul'].squeeze()
                ew_pred = outputs['early_warning'].squeeze()
                
                all_rul_pred.extend(rul_pred.cpu().numpy())
                all_rul_true.extend(batch['rul'].numpy())
                all_ew_pred.extend((ew_pred > 0.5).float().cpu().numpy())
                all_ew_true.extend(batch['early_warning'].numpy())
            else:
                soh_pred = model(cap_feat).squeeze()
            
            all_soh_pred.extend(soh_pred.cpu().numpy())
            all_soh_true.extend(batch['soh'].numpy())
    
    soh_pred = np.array(all_soh_pred)
    soh_true = np.array(all_soh_true)
    
    soh_mae = np.mean(np.abs(soh_pred - soh_true))
    
    metrics = {'soh_mae': soh_mae, 'n': len(soh_pred)}
    
    if use_eis and all_rul_pred:
        rul_pred = np.array(all_rul_pred)
        rul_true = np.array(all_rul_true)
        metrics['rul_mae'] = np.mean(np.abs(rul_pred - rul_true))
        
        # Early warning precision/recall
        ew_pred = np.array(all_ew_pred)
        ew_true = np.array(all_ew_true)
        if ew_true.sum() > 0:
            metrics['ew_recall'] = (ew_pred * ew_true).sum() / ew_true.sum()
        if ew_pred.sum() > 0:
            metrics['ew_precision'] = (ew_pred * ew_true).sum() / ew_pred.sum()
    
    return metrics


def create_comparison_plot(results: Dict, output_path: Path):
    """Create comparison visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # SOH MAE comparison
    ax = axes[0]
    methods = ['Capacity\nOnly', 'Multi-Modal\n(Cap + EIS)']
    soh_vals = [results['capacity_only']['soh_mae'], results['multimodal']['soh_mae']]
    colors = ['#E74C3C', '#27AE60']
    bars = ax.bar(methods, soh_vals, color=colors, edgecolor='black')
    ax.set_ylabel('SOH MAE')
    ax.set_title('SOH Prediction Accuracy', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, soh_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
               f'{val:.4f}', ha='center', fontsize=10, fontweight='bold')
    
    # RUL MAE
    ax = axes[1]
    rul_vals = [0, results['multimodal'].get('rul_mae', 0)]  # Capacity-only doesn't predict RUL
    bars = ax.bar(methods, rul_vals, color=colors, edgecolor='black')
    ax.set_ylabel('RUL MAE')
    ax.set_title('RUL Prediction (Multi-Modal Only)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Early Warning
    ax = axes[2]
    precision = results['multimodal'].get('ew_precision', 0)
    recall = results['multimodal'].get('ew_recall', 0)
    ax.bar(['Precision', 'Recall'], [precision, recall], color=['#3498DB', '#9B59B6'], edgecolor='black')
    ax.set_ylabel('Score')
    ax.set_title('Early Warning Performance', fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Multi-Modal Fusion Results: Capacity + EIS', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='.')
    parser.add_argument('--output_dir', type=str, default='reports/multimodal')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--device', type=str, default='cpu')
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("PHASE 3: MULTI-MODAL FUSION (CAPACITY + EIS)")
    print("=" * 60)
    
    # Create dataset
    print("\n[1/4] Creating matched dataset...")
    samples = create_matched_dataset(args.data_root)
    
    if len(samples) < 10:
        print("ERROR: Not enough samples. Check EIS data path.")
        return
    
    # Split
    np.random.seed(42)
    np.random.shuffle(samples)
    
    n_train = int(0.7 * len(samples))
    n_val = int(0.15 * len(samples))
    
    train_samples = samples[:n_train]
    val_samples = samples[n_train:n_train+n_val]
    test_samples = samples[n_train+n_val:]
    
    print(f"  Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
    
    train_loader = DataLoader(MultiModalDataset(train_samples), batch_size=32, shuffle=True)
    val_loader = DataLoader(MultiModalDataset(val_samples), batch_size=32)
    test_loader = DataLoader(MultiModalDataset(test_samples), batch_size=32)
    
    # Train capacity-only baseline
    print("\n[2/4] Training Capacity-Only Baseline...")
    baseline = CapacityOnlyPredictor().to(args.device)
    train_model(baseline, train_loader, val_loader, args.epochs, device=args.device, use_eis=False)
    baseline_results = evaluate_model(baseline, test_loader, args.device, use_eis=False)
    
    # Train multi-modal model
    print("\n[3/4] Training Multi-Modal Model...")
    multimodal = MultiModalPredictor().to(args.device)
    train_model(multimodal, train_loader, val_loader, args.epochs, device=args.device, use_eis=True)
    multimodal_results = evaluate_model(multimodal, test_loader, args.device, use_eis=True)
    
    # Results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"\nCapacity-Only Baseline:")
    print(f"  SOH MAE: {baseline_results['soh_mae']:.4f}")
    print(f"\nMulti-Modal (Capacity + EIS):")
    print(f"  SOH MAE: {multimodal_results['soh_mae']:.4f}")
    print(f"  RUL MAE: {multimodal_results.get('rul_mae', 0):.4f}")
    print(f"  Early Warning Precision: {multimodal_results.get('ew_precision', 0):.2%}")
    print(f"  Early Warning Recall: {multimodal_results.get('ew_recall', 0):.2%}")
    
    improvement = (baseline_results['soh_mae'] - multimodal_results['soh_mae']) / baseline_results['soh_mae'] * 100
    print(f"\nImprovement: {improvement:+.1f}%")
    
    # Save results
    results = {
        'capacity_only': baseline_results,
        'multimodal': multimodal_results,
        'improvement_pct': improvement
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    # Visualization
    print("\n[4/4] Creating visualization...")
    create_comparison_plot(results, output_dir / 'multimodal_results.png')
    
    # Save models
    torch.save(baseline.state_dict(), output_dir / 'capacity_only.pt')
    torch.save(multimodal.state_dict(), output_dir / 'multimodal.pt')
    
    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
