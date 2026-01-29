"""
Context Ablation Study for Fleet RAD Model

This experiment validates the importance of driving profile context in retrieval.
Compares Context-Aware vs Context-Blind inference on an Aggressive profile vehicle.

Hypothesis: Context-aware retrieval prevents the model from being 'optimistically wrong'
about high-stress vehicles by retrieving similar aggressive-profile neighbors.
"""

import os
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from src.models.rad_model import FleetRADModel, create_context_tensor
from src.memory_bank import encode_context, PROFILE_TYPES


def load_test_vehicle(fleet_dir: str, vehicle_id: str) -> dict:
    """Load a specific test vehicle's data."""
    # Load vehicle CSV
    vehicle_path = os.path.join(fleet_dir, f"{vehicle_id}.csv")
    if not os.path.exists(vehicle_path):
        raise FileNotFoundError(f"Vehicle not found: {vehicle_path}")
    
    df = pd.read_csv(vehicle_path)
    
    # Load metadata
    metadata_path = os.path.join(fleet_dir, 'fleet_metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Find vehicle info
    vehicle_info = None
    for v in metadata['vehicles']:
        if v['vehicle_id'] == vehicle_id:
            vehicle_info = v
            break
    
    if vehicle_info is None:
        raise ValueError(f"Vehicle {vehicle_id} not found in metadata")
    
    print(f"Loaded vehicle: {vehicle_id}")
    print(f"  Profile: {vehicle_info['profile_type']}")
    print(f"  Base battery: {vehicle_info['base_battery']}")
    print(f"  Cycles: {len(df)}")
    
    return df, vehicle_info


def prepare_vehicle_data(df: pd.DataFrame, feature_cols: list) -> tuple:
    """Prepare vehicle data for inference."""
    # Features
    X = df[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Normalize
    X_mean = X.mean(axis=0, keepdims=True)
    X_std = X.std(axis=0, keepdims=True) + 1e-8
    X = (X - X_mean) / X_std
    
    # SOH
    soh_col = 'SOH_Q' if 'SOH_Q' in df.columns else 'SOH_R'
    soh = df[soh_col].values.astype(np.float32)
    soh = np.nan_to_num(soh, nan=0.9)
    
    # RUL
    rul = df['RUL_cycles'].values.astype(np.float32)
    rul = np.maximum(0, rul)
    
    # Mask
    mask = np.ones(len(df), dtype=np.float32)
    
    return X, soh, rul, mask


def run_ablation(
    model: FleetRADModel,
    X: np.ndarray,
    soh_true: np.ndarray,
    rul_true: np.ndarray,
    mask: np.ndarray,
    profile_type: str,
    device: str = 'cpu'
) -> dict:
    """Run context ablation experiment."""
    
    model.eval()
    
    # Prepare tensors
    X_tensor = torch.from_numpy(X).unsqueeze(0).to(device)
    mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(device)
    
    results = {}
    
    with torch.no_grad():
        # Scenario A: Context-Aware (correct context)
        print(f"\nScenario A: Context-Aware ({profile_type})")
        context_correct = create_context_tensor(profile_type, batch_size=1, device=device)
        print(f"  Context vector: {context_correct.squeeze().tolist()}")
        
        soh_pred_a, rul_pred_a = model(
            X_tensor, mask_tensor, 
            context=context_correct, 
            use_retrieval=True
        )
        
        rul_pred_a_cycles = torch.expm1(rul_pred_a.clamp(max=6)).squeeze().cpu().numpy()
        soh_pred_a_np = soh_pred_a.squeeze().cpu().numpy()
        
        # Scenario B: Context-Blind (no context)
        print("\nScenario B: Context-Blind (Pure Physics)")
        print("  Context vector: None (fallback to physics-only)")
        
        soh_pred_b, rul_pred_b = model(
            X_tensor, mask_tensor,
            context=None,  # No context - pure physics retrieval
            use_retrieval=True
        )
        
        rul_pred_b_cycles = torch.expm1(rul_pred_b.clamp(max=6)).squeeze().cpu().numpy()
        soh_pred_b_np = soh_pred_b.squeeze().cpu().numpy()
        
        # Scenario C: Wrong Context (Eco - opposite of Aggressive)
        print("\nScenario C: Wrong Context (Eco - opposite profile)")
        context_wrong = create_context_tensor('Eco', batch_size=1, device=device)
        print(f"  Context vector: {context_wrong.squeeze().tolist()}")
        
        soh_pred_c, rul_pred_c = model(
            X_tensor, mask_tensor,
            context=context_wrong,
            use_retrieval=True
        )
        
        rul_pred_c_cycles = torch.expm1(rul_pred_c.clamp(max=6)).squeeze().cpu().numpy()
        soh_pred_c_np = soh_pred_c.squeeze().cpu().numpy()
    
    # Calculate metrics
    valid_mask = mask.astype(bool)
    
    # RUL metrics
    rul_err_a = rul_pred_a_cycles[valid_mask] - rul_true[valid_mask]
    rul_err_b = rul_pred_b_cycles[valid_mask] - rul_true[valid_mask]
    rul_err_c = rul_pred_c_cycles[valid_mask] - rul_true[valid_mask]
    
    mae_a = np.mean(np.abs(rul_err_a))
    mae_b = np.mean(np.abs(rul_err_b))
    mae_c = np.mean(np.abs(rul_err_c))
    
    # R² scores
    rul_mean = rul_true[valid_mask].mean()
    ss_tot = np.sum((rul_true[valid_mask] - rul_mean)**2)
    
    r2_a = 1 - np.sum(rul_err_a**2) / (ss_tot + 1e-8)
    r2_b = 1 - np.sum(rul_err_b**2) / (ss_tot + 1e-8)
    r2_c = 1 - np.sum(rul_err_c**2) / (ss_tot + 1e-8)
    
    # Bias (positive = optimistic, negative = pessimistic)
    bias_a = np.mean(rul_err_a)
    bias_b = np.mean(rul_err_b)
    bias_c = np.mean(rul_err_c)
    
    results = {
        'rul_true': rul_true,
        'soh_true': soh_true,
        'scenario_a': {
            'name': f'Context-Aware ({profile_type})',
            'rul_pred': rul_pred_a_cycles,
            'soh_pred': soh_pred_a_np,
            'mae': mae_a,
            'r2': r2_a,
            'bias': bias_a
        },
        'scenario_b': {
            'name': 'Context-Blind (Physics Only)',
            'rul_pred': rul_pred_b_cycles,
            'soh_pred': soh_pred_b_np,
            'mae': mae_b,
            'r2': r2_b,
            'bias': bias_b
        },
        'scenario_c': {
            'name': 'Wrong Context (Eco)',
            'rul_pred': rul_pred_c_cycles,
            'soh_pred': soh_pred_c_np,
            'mae': mae_c,
            'r2': r2_c,
            'bias': bias_c
        }
    }
    
    return results


def plot_ablation_results(results: dict, vehicle_id: str, out_dir: str):
    """Create ablation comparison plots."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    cycles = np.arange(len(results['rul_true']))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # RUL Predictions Comparison
    ax1 = axes[0, 0]
    ax1.plot(cycles, results['rul_true'], 'k-', lw=2.5, label='True RUL')
    ax1.plot(cycles, results['scenario_a']['rul_pred'], 'g--', lw=2, 
             label=f"A: {results['scenario_a']['name']}")
    ax1.plot(cycles, results['scenario_b']['rul_pred'], 'r--', lw=2, alpha=0.8,
             label=f"B: {results['scenario_b']['name']}")
    ax1.plot(cycles, results['scenario_c']['rul_pred'], 'b--', lw=2, alpha=0.6,
             label=f"C: {results['scenario_c']['name']}")
    
    ax1.set_xlabel('Cycle', fontsize=12)
    ax1.set_ylabel('RUL (cycles)', fontsize=12)
    ax1.set_title('RUL Prediction Comparison', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # RUL Error Comparison
    ax2 = axes[0, 1]
    err_a = results['scenario_a']['rul_pred'] - results['rul_true']
    err_b = results['scenario_b']['rul_pred'] - results['rul_true']
    err_c = results['scenario_c']['rul_pred'] - results['rul_true']
    
    ax2.plot(cycles, err_a, 'g-', lw=1.5, label='A: Context-Aware', alpha=0.8)
    ax2.plot(cycles, err_b, 'r-', lw=1.5, label='B: Context-Blind', alpha=0.8)
    ax2.plot(cycles, err_c, 'b-', lw=1.5, label='C: Wrong Context', alpha=0.6)
    ax2.axhline(0, color='k', linestyle='--', lw=1)
    ax2.fill_between(cycles, 0, err_b, where=(err_b > 0), alpha=0.2, color='red',
                     label='Optimistic Error (B)')
    
    ax2.set_xlabel('Cycle', fontsize=12)
    ax2.set_ylabel('RUL Error (cycles)', fontsize=12)
    ax2.set_title('RUL Error Analysis\n(Positive = Optimistic, Negative = Pessimistic)', 
                  fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Metrics Bar Chart
    ax3 = axes[1, 0]
    scenarios = ['A: Context-Aware', 'B: Context-Blind', 'C: Wrong Context']
    maes = [results['scenario_a']['mae'], results['scenario_b']['mae'], results['scenario_c']['mae']]
    colors = ['green', 'red', 'blue']
    
    bars = ax3.bar(scenarios, maes, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_ylabel('MAE (cycles)', fontsize=12)
    ax3.set_title('RUL Mean Absolute Error by Scenario', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, mae in zip(bars, maes):
        ax3.annotate(f'{mae:.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=11, fontweight='bold')
    
    # Bias Analysis
    ax4 = axes[1, 1]
    biases = [results['scenario_a']['bias'], results['scenario_b']['bias'], results['scenario_c']['bias']]
    r2s = [results['scenario_a']['r2'], results['scenario_b']['r2'], results['scenario_c']['r2']]
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, biases, width, label='Bias (cycles)', color=['green', 'red', 'blue'], alpha=0.7)
    ax4.axhline(0, color='k', linestyle='--', lw=1)
    
    ax4.set_ylabel('Bias (cycles)', fontsize=12)
    ax4.set_xticks(x)
    ax4.set_xticklabels(['A', 'B', 'C'], fontsize=11)
    ax4.set_title('Prediction Bias\n(Positive = Optimistic)', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add R² as text
    for i, (bias, r2) in enumerate(zip(biases, r2s)):
        ax4.annotate(f'R²={r2:.3f}', xy=(i, bias), xytext=(0, 5 if bias >= 0 else -15),
                    textcoords='offset points', ha='center', fontsize=10)
    
    fig.suptitle(f'Context Ablation Study: {vehicle_id}\n'
                 f'Proving the Value of Fleet Context in Retrieval', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save
    fig.savefig(os.path.join(out_dir, f'context_ablation_{vehicle_id}.png'), dpi=150, bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, f'context_ablation_{vehicle_id}.svg'), bbox_inches='tight')
    plt.close()
    
    print(f"\nPlot saved to {out_dir}/context_ablation_{vehicle_id}.png")


def print_ablation_summary(results: dict, vehicle_id: str):
    """Print detailed ablation summary."""
    print("\n" + "="*70)
    print(f"CONTEXT ABLATION STUDY RESULTS: {vehicle_id}")
    print("="*70)
    
    print("\n📊 METRICS COMPARISON:")
    print("-"*70)
    print(f"{'Scenario':<30} {'MAE':<12} {'R²':<12} {'Bias':<12}")
    print("-"*70)
    
    for key in ['scenario_a', 'scenario_b', 'scenario_c']:
        s = results[key]
        bias_str = f"{s['bias']:+.1f}" if s['bias'] >= 0 else f"{s['bias']:.1f}"
        print(f"{s['name']:<30} {s['mae']:<12.1f} {s['r2']:<12.3f} {bias_str:<12}")
    
    print("-"*70)
    
    # Calculate improvements
    mae_improvement = results['scenario_b']['mae'] - results['scenario_a']['mae']
    r2_improvement = results['scenario_a']['r2'] - results['scenario_b']['r2']
    bias_reduction = abs(results['scenario_b']['bias']) - abs(results['scenario_a']['bias'])
    
    print("\n🎯 KEY FINDINGS:")
    print("-"*70)
    
    if mae_improvement > 0:
        print(f"✅ Context-Aware reduces MAE by {mae_improvement:.1f} cycles ({mae_improvement/results['scenario_b']['mae']*100:.1f}%)")
    else:
        print(f"⚠️ Context-Aware increases MAE by {-mae_improvement:.1f} cycles")
    
    if r2_improvement > 0:
        print(f"✅ Context-Aware improves R² by {r2_improvement:.3f}")
    else:
        print(f"⚠️ Context-Aware reduces R² by {-r2_improvement:.3f}")
    
    if bias_reduction > 0:
        print(f"✅ Context-Aware reduces prediction bias by {bias_reduction:.1f} cycles")
    else:
        print(f"⚠️ Context-Aware increases prediction bias by {-bias_reduction:.1f} cycles")
    
    # Interpretation
    print("\n📝 INTERPRETATION:")
    print("-"*70)
    
    if results['scenario_b']['bias'] > 0:
        print("• Context-Blind (B) is OPTIMISTIC: overestimates RUL for aggressive vehicles")
        print("  → Without context, the model retrieves 'normal' neighbors with longer lifespans")
        print("  → This leads to dangerous over-confidence in battery health")
    else:
        print("• Context-Blind (B) is PESSIMISTIC: underestimates RUL")
    
    if abs(results['scenario_a']['bias']) < abs(results['scenario_b']['bias']):
        print("\n• Context-Aware (A) provides more CALIBRATED predictions")
        print("  → Retrieving similar aggressive-profile neighbors gives realistic estimates")
    
    if results['scenario_c']['bias'] > results['scenario_b']['bias']:
        print("\n• Wrong Context (C) is MOST OPTIMISTIC: Eco context on Aggressive vehicle")
        print("  → Proves that context matters: wrong context is worse than no context")
    
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Context Ablation Study')
    parser.add_argument('--fleet_dir', type=str, default='data/processed/fleet',
                        help='Directory containing fleet data')
    parser.add_argument('--model_dir', type=str, default='reports/fleet_rad',
                        help='Directory containing trained model')
    parser.add_argument('--vehicle_id', type=str, default='B0005_A00',
                        help='Test vehicle ID (should be Aggressive profile)')
    parser.add_argument('--out_dir', type=str, default='experiments/results',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cpu')
    
    args = parser.parse_args()
    
    print("="*70)
    print("CONTEXT ABLATION EXPERIMENT")
    print("="*70)
    print(f"\nTest Vehicle: {args.vehicle_id}")
    print(f"Model: {args.model_dir}")
    
    # Load test vehicle
    print("\n1. Loading test vehicle...")
    df, vehicle_info = load_test_vehicle(args.fleet_dir, args.vehicle_id)
    profile_type = vehicle_info['profile_type']
    
    # Prepare data
    feature_cols = ['Capacity', 'IR', 'Temp_med']
    available_cols = [c for c in feature_cols if c in df.columns]
    X, soh_true, rul_true, mask = prepare_vehicle_data(df, available_cols)
    
    print(f"\n2. Loading trained model...")
    # Create model with same architecture
    model = FleetRADModel(
        input_dim=len(available_cols),
        hidden_dim=256,
        latent_dim=64,
        num_layers=2,
        retrieval_k=5,
        physics_weight=0.7,
        context_weight=0.3,
        device=args.device
    ).to(args.device)
    
    # Load weights
    model_path = os.path.join(args.model_dir, 'fleet_rad_model.pt')
    model.load_state_dict(torch.load(model_path, map_location=args.device, weights_only=True))
    print(f"  Loaded model from {model_path}")
    
    # Load memory bank
    memory_path = os.path.join(args.model_dir, 'fleet_memory_bank')
    model.load_memory_bank(memory_path)
    print(f"  Memory bank size: {model.memory_bank.size()}")
    
    # Run ablation
    print("\n3. Running ablation scenarios...")
    results = run_ablation(model, X, soh_true, rul_true, mask, profile_type, args.device)
    
    # Print summary
    print_ablation_summary(results, args.vehicle_id)
    
    # Plot results
    print("\n4. Generating plots...")
    plot_ablation_results(results, args.vehicle_id, args.out_dir)
    
    # Save metrics (convert numpy types to Python native types)
    metrics = {
        'vehicle_id': args.vehicle_id,
        'profile_type': profile_type,
        'scenario_a_mae': float(results['scenario_a']['mae']),
        'scenario_a_r2': float(results['scenario_a']['r2']),
        'scenario_a_bias': float(results['scenario_a']['bias']),
        'scenario_b_mae': float(results['scenario_b']['mae']),
        'scenario_b_r2': float(results['scenario_b']['r2']),
        'scenario_b_bias': float(results['scenario_b']['bias']),
        'scenario_c_mae': float(results['scenario_c']['mae']),
        'scenario_c_r2': float(results['scenario_c']['r2']),
        'scenario_c_bias': float(results['scenario_c']['bias']),
        'mae_improvement': float(results['scenario_b']['mae'] - results['scenario_a']['mae']),
        'r2_improvement': float(results['scenario_a']['r2'] - results['scenario_b']['r2'])
    }
    
    metrics_path = os.path.join(args.out_dir, f'context_ablation_metrics_{args.vehicle_id}.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")
    
    print("\n✅ Ablation study complete!")


if __name__ == '__main__':
    main()

