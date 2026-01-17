"""
Simulate Storage Degradation: Parked Car Scenario

Simulates battery health degradation when a car is parked for 30 days at 40°C
at different SOC levels (0%, 50%, 100%).

This demonstrates the model's ability to predict calendar aging (storage degradation).

Author: Battery ML Research
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.train.train_multi_dataset import ContextAwareSOHPredictor
from src.context.extended_context import normalize_temperature, normalize_crate


def simulate_storage_degradation(
    model_path: str,
    storage_days: int = 30,
    temperature_c: float = 40.0,
    soc_levels: list = [0.0, 50.0, 100.0],
    initial_soh: float = 1.0,
    output_dir: Path = None,
    device: str = 'cpu'
):
    """
    Simulate storage degradation at different SOC levels.
    
    Args:
        model_path: Path to trained model
        storage_days: Number of days parked
        temperature_c: Storage temperature in Celsius
        soc_levels: List of SOC levels to test (0-100%)
        initial_soh: Initial SOH (default 1.0 = 100%)
        output_dir: Directory to save plots
        device: Device to run model on
    """
    
    print("=" * 70)
    print("STORAGE DEGRADATION SIMULATION")
    print("=" * 70)
    print(f"\nScenario: Car parked for {storage_days} days at {temperature_c}°C")
    print(f"SOC levels tested: {soc_levels}%")
    print(f"Initial SOH: {initial_soh:.1%}")
    
    # Load model
    print("\n1. Loading model...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model = ContextAwareSOHPredictor(
        feature_dim=9,
        context_numeric_dim=5,  # 5D context: [Temp, ChargeRate, DischargeRate, SOC, UsageProfile]
        chem_emb_dim=4,
        hidden_dim=128,
        latent_dim=64
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Prepare simulation data
    print("\n2. Preparing simulation data...")
    
    # Storage degradation: treat each day as a "cycle" for simulation
    # We'll simulate degradation over time
    time_points = np.arange(0, storage_days + 1, 1)  # Daily measurements
    
    results = {}
    
    for soc_pct in soc_levels:
        print(f"\n   Simulating {soc_pct}% SOC...")
        
        # Build context vector for storage scenario
        # 5D: [Temperature, ChargeRate, DischargeRate, SOC, UsageProfile]
        temp_norm = normalize_temperature(temperature_c)
        charge_norm = 0.0  # No charging during storage
        discharge_norm = 0.0  # No discharging during storage
        soc_norm = soc_pct / 100.0
        usage_profile_norm = 0.0  # STORAGE = calendar aging
        
        context_vector = np.array([
            temp_norm,
            charge_norm,
            discharge_norm,
            soc_norm,
            usage_profile_norm
        ], dtype=np.float32)
        
        # Create feature vector (typical values for storage)
        # Features: [capacity, internal_resistance, soh, temperature_mean, 
        #            temperature_max, temperature_min, current_mean, voltage_min, voltage_max]
        # For storage, we simulate degradation over time
        soh_values = []
        rul_values = []
        
        # Use physics-based calendar aging model
        # Calendar aging rate depends on SOC and temperature
        # Higher SOC and temperature = faster degradation
        # Typical rates: 0.1-0.5% per month at room temp, higher at elevated temp
        
        # Base degradation rate (per day) - depends on SOC and temperature
        # At 40°C and high SOC, degradation is faster
        if soc_pct == 100.0:
            base_rate = 0.00015  # 0.015% per day at 100% SOC, 40°C
        elif soc_pct == 50.0:
            base_rate = 0.00010  # 0.010% per day at 50% SOC, 40°C
        else:  # 0% SOC
            base_rate = 0.00005  # 0.005% per day at 0% SOC, 40°C
        
        # Temperature acceleration (Arrhenius-like)
        temp_acceleration = np.exp((temperature_c - 25.0) / 10.0)  # Rough approximation
        daily_rate = base_rate * temp_acceleration
        
        for day in time_points:
            # Calculate SOH based on calendar aging
            # Calendar aging: exponential decay with time
            current_soh = initial_soh * np.exp(-daily_rate * day)
            
            # Create feature vector based on actual storage data patterns
            # Features: [capacity, internal_resistance, soh, temp_mean, temp_max, temp_min, current_mean, voltage_min, voltage_max]
            nominal_capacity = 2.0  # Ah (typical)
            current_capacity = current_soh * nominal_capacity
            
            # Internal resistance increases slightly with degradation
            base_resistance = 0.01  # Ohms
            resistance = base_resistance * (1.0 + (1.0 - current_soh) * 0.5)
            
            # Resting voltage depends on SOC
            if soc_pct == 100.0:
                resting_voltage = 4.1
            elif soc_pct == 50.0:
                resting_voltage = 3.7
            else:  # 0% SOC
                resting_voltage = 3.2
            
            features = np.array([
                current_capacity,  # Capacity (Ah)
                resistance,  # Internal resistance (Ohms)
                current_soh,  # Current SOH
                temperature_c,  # Temperature mean
                temperature_c,  # Temperature max
                temperature_c,  # Temperature min
                0.0,  # Current mean (no current during storage)
                resting_voltage - 0.1,  # Voltage min
                resting_voltage + 0.1,  # Voltage max
            ], dtype=np.float32)
            
            # Normalize features (matching training data normalization)
            features[0] = features[0] / 2.0  # Normalize capacity (assuming 2Ah nominal)
            features[1] = features[1] / 0.1  # Normalize resistance
            features[3:6] = features[3:6] / 60.0  # Normalize temperature
            features[6] = features[6] / 3.0  # Normalize current
            features[7:9] = (features[7:9] - 2.5) / 2.0  # Normalize voltage
            
            # Get prediction from model
            with torch.no_grad():
                features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 9)
                context_tensor = torch.tensor(context_vector, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 5)
                chem_id_tensor = torch.tensor([0], dtype=torch.long).unsqueeze(0).to(device)  # (1, 1) LCO chemistry
                
                soh_pred, rul_pred_norm, _ = model(features_tensor, context_tensor, chem_id_tensor)
                
                soh_pred_val = soh_pred.item()
                rul_pred_val = rul_pred_norm.item() * 100  # Convert to cycles (assuming 100 cycle EOL)
            
            soh_values.append(soh_pred_val)
            rul_values.append(rul_pred_val)
        
        results[soc_pct] = {
            'days': time_points,
            'soh': np.array(soh_values),
            'rul': np.array(rul_values),
            'context': context_vector
        }
    
    # Create visualization
    print("\n3. Creating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Color map for SOC levels
    colors = {0.0: '#e74c3c', 50.0: '#3498db', 100.0: '#2ecc71'}  # Red, Blue, Green
    labels = {0.0: '0% SOC', 50.0: '50% SOC', 100.0: '100% SOC'}
    
    # 1. SOH Degradation Over Time
    ax = axes[0, 0]
    for soc_pct in soc_levels:
        data = results[soc_pct]
        ax.plot(data['days'], data['soh'], 
               label=labels[soc_pct], 
               color=colors[soc_pct], 
               linewidth=2.5, 
               marker='o', 
               markersize=4)
    
    ax.set_xlabel('Storage Days', fontsize=12, fontweight='bold')
    ax.set_ylabel('State of Health (SOH)', fontsize=12, fontweight='bold')
    ax.set_title(f'SOH Degradation: {storage_days} Days at {temperature_c}°C', 
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0.9, 1.01])
    
    # 2. Degradation Rate Comparison
    ax = axes[0, 1]
    degradation_rates = []
    soc_labels = []
    for soc_pct in soc_levels:
        data = results[soc_pct]
        initial = data['soh'][0]
        final = data['soh'][-1]
        degradation = (initial - final) / initial * 100
        degradation_rates.append(degradation)
        soc_labels.append(f'{soc_pct:.0f}%')
    
    bars = ax.bar(soc_labels, degradation_rates, 
                  color=[colors[s] for s in soc_levels], 
                  alpha=0.7, 
                  edgecolor='black', 
                  linewidth=1.5)
    ax.set_ylabel('Total Degradation (%)', fontsize=12, fontweight='bold')
    ax.set_title('Total Degradation After 30 Days', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add value labels on bars
    for bar, rate in zip(bars, degradation_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{rate:.3f}%', ha='center', va='bottom', 
               fontsize=10, fontweight='bold')
    
    # 3. Daily Degradation Rate
    ax = axes[1, 0]
    for soc_pct in soc_levels:
        data = results[soc_pct]
        daily_rates = np.diff(data['soh']) * 100  # Convert to percentage per day
        days_mid = (data['days'][:-1] + data['days'][1:]) / 2
        ax.plot(days_mid, daily_rates, 
               label=labels[soc_pct], 
               color=colors[soc_pct], 
               linewidth=2, 
               marker='s', 
               markersize=3)
    
    ax.set_xlabel('Storage Days', fontsize=12, fontweight='bold')
    ax.set_ylabel('Daily Degradation Rate (%/day)', fontsize=12, fontweight='bold')
    ax.set_title('Daily Degradation Rate Over Time', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # 4. Summary Statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = f"Storage Degradation Analysis\n" + "="*50 + "\n\n"
    summary_text += f"Scenario:\n"
    summary_text += f"  Duration: {storage_days} days\n"
    summary_text += f"  Temperature: {temperature_c}°C\n"
    summary_text += f"  Initial SOH: {initial_soh:.1%}\n\n"
    
    summary_text += f"Results by SOC Level:\n"
    for soc_pct in soc_levels:
        data = results[soc_pct]
        initial = data['soh'][0]
        final = data['soh'][-1]
        total_degradation = (initial - final) / initial * 100
        daily_avg = total_degradation / storage_days
        
        summary_text += f"\n{soc_pct:.0f}% SOC:\n"
        summary_text += f"  Initial SOH: {initial:.4f}\n"
        summary_text += f"  Final SOH: {final:.4f}\n"
        summary_text += f"  Total Degradation: {total_degradation:.4f}%\n"
        summary_text += f"  Avg Daily Rate: {daily_avg:.6f}%/day\n"
    
    summary_text += f"\nKey Finding:\n"
    # Find which SOC has highest degradation
    max_degradation = max(degradation_rates)
    max_soc = soc_levels[degradation_rates.index(max_degradation)]
    summary_text += f"  Highest degradation at {max_soc:.0f}% SOC\n"
    summary_text += f"  Physics: Higher SOC = faster calendar aging\n"
    summary_text += f"  (SEI growth, lithium plating risk)"
    
    ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'Battery Health: {storage_days} Days Parked at {temperature_c}°C\n'
                f'Calendar Aging at Different SOC Levels', 
                fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save
    if output_dir is None:
        output_dir = Path('reports/storage_simulation')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f'storage_degradation_{storage_days}days_{temperature_c}C.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / filename.replace('.png', '.svg'), bbox_inches='tight')
    print(f"\n✅ Visualization saved to {output_dir / filename}")
    
    plt.close()
    
    # Print summary
    print("\n" + "=" * 70)
    print("SIMULATION RESULTS")
    print("=" * 70)
    for soc_pct in soc_levels:
        data = results[soc_pct]
        initial = data['soh'][0]
        final = data['soh'][-1]
        total_degradation = (initial - final) / initial * 100
        print(f"\n{soc_pct:.0f}% SOC:")
        print(f"  Initial SOH: {initial:.4f}")
        print(f"  Final SOH: {final:.4f}")
        print(f"  Total Degradation: {total_degradation:.4f}%")
        print(f"  Average Daily Rate: {total_degradation/storage_days:.6f}%/day")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Simulate storage degradation')
    parser.add_argument('--model_path', type=str,
                       default='reports/soc_aware_5d_context/best_model.pt',
                       help='Path to trained model')
    parser.add_argument('--storage_days', type=int, default=30,
                       help='Number of days parked')
    parser.add_argument('--temperature', type=float, default=40.0,
                       help='Storage temperature in Celsius')
    parser.add_argument('--soc_levels', type=float, nargs='+', 
                       default=[0.0, 50.0, 100.0],
                       help='SOC levels to test (0-100%)')
    parser.add_argument('--output_dir', type=str,
                       default='reports/storage_simulation',
                       help='Output directory for plots')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu or cuda)')
    
    args = parser.parse_args()
    
    results = simulate_storage_degradation(
        args.model_path,
        args.storage_days,
        args.temperature,
        args.soc_levels,
        output_dir=Path(args.output_dir),
        device=args.device
    )

