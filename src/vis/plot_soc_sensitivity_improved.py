"""
Improved SOC Sensitivity Analysis Plot

Fixes issues with the original plot:
1. Better SOC binning based on actual data distribution
2. Improved visualization clarity
3. Better labels and annotations
4. Handles edge cases

Author: Battery ML Research
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.train.train_multi_dataset import ContextAwareSOHPredictor
from src.data.unified_pipeline import UnifiedDataPipeline, UnifiedBatteryDataset


def evaluate_for_plotting(model, dataloader, device):
    """Evaluate model and return data for plotting."""
    model.eval()
    
    all_soh_pred = []
    all_soh_true = []
    all_rul_pred = []
    all_rul_true = []
    all_soc = []
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            context = batch['context'].to(device)
            chem_id = batch['chem_id'].to(device)
            soh_true = batch['soh'].to(device)
            rul_norm_true = batch.get('rul_normalized', batch['rul'] / 100.0).to(device)
            rul_abs_true = batch['rul'].to(device)
            eol_cycle = batch.get('eol_cycle', torch.ones_like(rul_abs_true) * 100).to(device)
            
            # Handle NaN
            features = torch.nan_to_num(features, nan=0.0)
            context = torch.nan_to_num(context, nan=0.0)
            
            soh_pred, rul_pred_norm, _ = model(features, context, chem_id)
            
            # Convert normalized RUL back to absolute cycles
            rul_pred_abs = rul_pred_norm.squeeze() * eol_cycle
            
            # Filter valid samples
            valid_mask = ~(torch.isnan(soh_true) | torch.isnan(rul_norm_true))
            if valid_mask.sum() > 0:
                all_soh_pred.append(soh_pred.squeeze()[valid_mask].cpu().numpy())
                all_soh_true.append(soh_true[valid_mask].cpu().numpy())
                all_rul_pred.append(rul_pred_abs[valid_mask].cpu().numpy())
                all_rul_true.append(rul_abs_true[valid_mask].cpu().numpy())
                # SOC is 4th dimension (index 3) in context vector: [Temp, ChargeRate, DischargeRate, SOC]
                soc_values = context[valid_mask, 3].cpu().numpy()
                # Convert from normalized (0-1) back to percentage (0-100) for binning
                soc_values = soc_values * 100.0
                all_soc.append(soc_values)
    
    if len(all_soh_pred) == 0:
        return None
    
    return {
        'soh_pred': np.concatenate(all_soh_pred),
        'soh_true': np.concatenate(all_soh_true),
        'rul_pred': np.concatenate(all_rul_pred),
        'rul_true': np.concatenate(all_rul_true),
        'soc': np.concatenate(all_soc)
    }


def create_improved_soc_plot(model_path, data_root, output_dir, device='cpu'):
    """Create improved SOC sensitivity plot."""
    
    # Load model
    print("Loading model...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Create model
    model = ContextAwareSOHPredictor(
        feature_dim=9,
        context_numeric_dim=5,  # Updated: 5D context [Temp, ChargeRate, DischargeRate, SOC, UsageProfile]
        chem_emb_dim=4,
        hidden_dim=128,
        latent_dim=64
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load test data
    print("Loading test data...")
    pipeline = UnifiedDataPipeline(data_root=data_root, use_lithium_features=False)
    
    # Load NASA from project directory
    nasa_in_project = Path("data") / "nasa_set5" / "raw"
    if nasa_in_project.exists():
        pipeline.data_root = Path("data")
        pipeline._load_nasa()
        pipeline.data_root = Path(data_root)
    
    # Load storage
    try:
        pipeline.load_datasets(['storage'], create_synthetic_if_missing=False)
    except:
        pass
    
    # Create splits
    train_ds, val_ds, test_ds = pipeline.create_splits(
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )
    
    # CRITICAL FIX: Use ALL data (train+val+test) to ensure all SOC levels are represented
    # The test set might not have all SOC levels due to random splitting
    from torch.utils.data import ConcatDataset
    all_ds = ConcatDataset([train_ds, val_ds, test_ds])
    test_loader = DataLoader(all_ds, batch_size=64, shuffle=False)
    
    # Evaluate
    print("Evaluating model...")
    data = evaluate_for_plotting(model, test_loader, device)
    
    if data is None or len(data['soc']) == 0:
        print("⚠️  No data for plotting")
        return
    
    soc = data['soc']
    soh_pred = data['soh_pred']
    soh_true = data['soh_true']
    rul_pred = data['rul_pred']
    rul_true = data['rul_true']
    
    print(f"\nData summary:")
    print(f"  Total samples: {len(soc)}")
    print(f"  SOC range: {soc.min():.3f} to {soc.max():.3f}")
    print(f"  Unique SOC values: {len(np.unique(soc))}")
    print(f"  SOC distribution: {np.unique(soc)}")
    
    # Better SOC binning based on actual data
    # SOC is now in percentage (0-100), so we bin by percentage
    unique_socs = np.unique(soc)
    print(f"\nUnique SOC values (percentage): {unique_socs}")
    
    # Create bins around actual SOC values (in percentage: 0%, 50%, 100%)
    soc_bins = []
    soc_labels = []
    
    # Check for each expected SOC level (in percentage)
    if 0.0 in unique_socs or np.any(np.abs(unique_socs - 0.0) < 10):
        soc_bins.append((0.0, 25.0))
        soc_labels.append('0%')
    
    if 50.0 in unique_socs or np.any(np.abs(unique_socs - 50.0) < 10):
        soc_bins.append((25.0, 75.0))
        soc_labels.append('50%')
    
    if 100.0 in unique_socs or np.any(np.abs(unique_socs - 100.0) < 10):
        soc_bins.append((75.0, 100.1))  # Include 100.0 exactly
        soc_labels.append('100%')
    
    # If no bins created, use default
    if not soc_bins:
        soc_bins = [(0.0, 0.2), (0.4, 0.6), (0.8, 1.0)]
        soc_labels = ['0%', '50%', '100%']
    
    # Group data by SOC
    soc_groups = {}
    colors = ['#e74c3c', '#3498db', '#2ecc71']  # Red, Blue, Green
    
    for i, (bin_range, label) in enumerate(zip(soc_bins, soc_labels)):
        # Use <= for upper bound to include 100.0
        if bin_range[1] >= 100.0:
            mask = (soc >= bin_range[0]) & (soc <= bin_range[1])
        else:
            mask = (soc >= bin_range[0]) & (soc < bin_range[1])
        if np.sum(mask) > 0:
            soc_groups[label] = {
                'mask': mask,
                'color': colors[i % len(colors)],
                'soh_mae': float(np.mean(np.abs(soh_pred[mask] - soh_true[mask]))),
                'soh_rmse': float(np.sqrt(np.mean((soh_pred[mask] - soh_true[mask]) ** 2))),
                'n_samples': int(np.sum(mask))
            }
    
    print(f"\nSOC groups found: {list(soc_groups.keys())}")
    for label, group in soc_groups.items():
        print(f"  {label}: N={group['n_samples']}, MAE={group['soh_mae']:.4f}")
    
    # Create improved plot
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. SOH Predictions by SOC (Top Left, larger)
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    for label, group in soc_groups.items():
        mask = group['mask']
        ax1.scatter(soh_true[mask], soh_pred[mask], 
                   label=f"{label} SOC (N={group['n_samples']})", 
                   color=group['color'], alpha=0.7, s=40, edgecolors='black', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(soh_true.min(), soh_pred.min())
    max_val = max(soh_true.max(), soh_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Prediction', zorder=0)
    ax1.set_xlabel('True SOH', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Predicted SOH', fontsize=12, fontweight='bold')
    ax1.set_title('SOH Predictions by SOC Level', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_aspect('equal', adjustable='box')
    
    # 2. SOH MAE by SOC (Top Right)
    ax2 = fig.add_subplot(gs[0, 2])
    if soc_groups:
        labels = list(soc_groups.keys())
        maes = [soc_groups[l]['soh_mae'] for l in labels]
        colors_bar = [soc_groups[l]['color'] for l in labels]
        bars = ax2.bar(labels, maes, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('SOH MAE', fontsize=11, fontweight='bold')
        ax2.set_title('SOH MAE by SOC Level', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Add value labels on bars
        for bar, mae in zip(bars, maes):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{mae:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 3. SOC Distribution (Middle Right)
    ax3 = fig.add_subplot(gs[1, 2])
    n, bins, patches = ax3.hist(soc, bins=30, color='steelblue', alpha=0.7, edgecolor='black', linewidth=1)
    ax3.set_xlabel('SOC (%)', fontsize=11, fontweight='bold')  # Update label to show percentage
    
    # Color bins by SOC level
    for i, (bin_range, label) in enumerate(zip(soc_bins, soc_labels)):
        if label in soc_groups:
            for j, patch in enumerate(patches):
                bin_center = (bins[j] + bins[j+1]) / 2 if j < len(bins) - 1 else bins[j]
                if bin_range[0] <= bin_center <= bin_range[1]:
                    patch.set_facecolor(soc_groups[label]['color'])
                    patch.set_alpha(0.8)
    
    ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax3.set_title('SOC Distribution in Test Set', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # 4. SOH vs SOC Scatter (Bottom Left)
    ax4 = fig.add_subplot(gs[2, 0])
    scatter = ax4.scatter(soc, soh_true, c=soh_pred, cmap='viridis', 
                         alpha=0.7, s=50, edgecolors='black', linewidth=0.3)
    ax4.set_xlabel('SOC (%)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('True SOH', fontsize=11, fontweight='bold')
    ax4.set_title('True SOH vs SOC\n(colored by Predicted SOH)', fontsize=12, fontweight='bold')
    plt.colorbar(scatter, ax=ax4, label='Predicted SOH', shrink=0.8)
    ax4.grid(True, alpha=0.3, linestyle='--')
    
    # 5. Prediction Error vs SOC (Bottom Middle)
    ax5 = fig.add_subplot(gs[2, 1])
    error = np.abs(soh_pred - soh_true)
    scatter2 = ax5.scatter(soc, error, c=soh_true, cmap='coolwarm', 
                          alpha=0.7, s=50, edgecolors='black', linewidth=0.3)
    ax5.set_xlabel('SOC (%)', fontsize=11, fontweight='bold')
    ax5.set_ylabel('|Predicted - True SOH|', fontsize=11, fontweight='bold')
    ax5.set_title('Prediction Error vs SOC\n(colored by True SOH)', fontsize=12, fontweight='bold')
    plt.colorbar(scatter2, ax=ax5, label='True SOH', shrink=0.8)
    ax5.grid(True, alpha=0.3, linestyle='--')
    
    # 6. Summary Statistics (Bottom Right)
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    
    # Create summary text
    summary_text = "SOC Sensitivity Summary\n" + "="*30 + "\n\n"
    summary_text += f"Total Samples: {len(soc)}\n\n"
    
    for label, group in soc_groups.items():
        summary_text += f"{label} SOC:\n"
        summary_text += f"  Samples: {group['n_samples']}\n"
        summary_text += f"  SOH MAE: {group['soh_mae']:.4f}\n"
        summary_text += f"  SOH RMSE: {group['soh_rmse']:.4f}\n\n"
    
    # Overall metrics
    overall_mae = np.mean(np.abs(soh_pred - soh_true))
    overall_rmse = np.sqrt(np.mean((soh_pred - soh_true) ** 2))
    summary_text += f"Overall:\n"
    summary_text += f"  SOH MAE: {overall_mae:.4f}\n"
    summary_text += f"  SOH RMSE: {overall_rmse:.4f}\n"
    
    ax6.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Overall title
    fig.suptitle('SOC-Aware Battery Health Prediction: Sensitivity Analysis', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_dir / 'soc_sensitivity_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / 'soc_sensitivity_analysis.svg', bbox_inches='tight', facecolor='white')
    print(f"\n✅ Improved SOC sensitivity plot saved to {output_dir}")
    
    plt.close()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, 
                       default='reports/soc_aware_full/best_model.pt',
                       help='Path to trained model')
    parser.add_argument('--data_root', type=str, 
                       default=os.path.expanduser('~/Downloads'),
                       help='Root directory for data')
    parser.add_argument('--output_dir', type=str,
                       default='reports/soc_aware_full',
                       help='Output directory for plots')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu or cuda)')
    
    args = parser.parse_args()
    
    create_improved_soc_plot(
        args.model_path,
        args.data_root,
        args.output_dir,
        args.device
    )

