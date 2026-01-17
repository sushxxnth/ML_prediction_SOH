"""
Run causal attribution on XJTU high C-rate data.

Tests: Does the causal attribution model correctly identify
Active Material Loss as the dominant mechanism for high C-rate cycling?
"""

import numpy as np
import scipy.io
import torch
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.causal_attribution import CausalAttributionModel, DegradationMechanism


def load_xjtu_for_attribution():
    """Load XJTU samples formatted for causal attribution."""
    
    xjtu_dir = Path("data/new_datasets/XJTU/Battery Dataset")
    
    samples = []
    
    for batch_dir in sorted(xjtu_dir.iterdir()):
        if not batch_dir.is_dir():
            continue
            
        batch_name = batch_dir.name
        mat_files = list(batch_dir.glob("*.mat"))
        
        if "Batch-1" in batch_name:
            c_rate = 2.0
        elif "Batch-2" in batch_name:
            c_rate = 3.0
        else:
            c_rate = 2.5
        
        for mat_file in mat_files:
            try:
                data = scipy.io.loadmat(mat_file, simplify_cells=True)
                
                if 'summary' in data and isinstance(data['summary'], dict):
                    summary = data['summary']
                    
                    if 'discharge_capacity_Ah' in summary:
                        capacity = np.array(summary['discharge_capacity_Ah']).flatten()
                    else:
                        continue
                    
                    if len(capacity) < 50:
                        continue
                    
                    initial_capacity = capacity[1] if len(capacity) > 1 and capacity[1] > capacity[0] else capacity[0]
                    if initial_capacity <= 0:
                        initial_capacity = 2.0
                    
                    # Calculate capacity loss at end of life
                    final_capacity = capacity[-10:].mean()
                    total_loss = (initial_capacity - final_capacity) / initial_capacity
                    
                    if total_loss < 0.05:
                        continue
                    
                    cell_name = mat_file.stem
                    
                    # Create features (9-dim to match trained model)
                    features = np.zeros(9, dtype=np.float32)
                    features[0] = capacity.mean() / initial_capacity  # Mean normalized capacity
                    features[1] = capacity.std() / initial_capacity   # Std of capacity
                    features[2] = c_rate / 4.0                        # Normalized C-rate
                    features[3] = total_loss                          # Total capacity loss
                    features[4] = len(capacity) / 1000.0              # Normalized cycles
                    features[5] = final_capacity / initial_capacity   # End SOH
                    features[6] = capacity[0] / initial_capacity      # Start SOH
                    features[7] = 0.0                                 # Placeholder
                    features[8] = 0.0                                 # Placeholder
                    
                    # Context: [temp, charge_rate, discharge_rate, soc, profile, mode]
                    context = np.array([
                        25.0 / 60.0,       # Room temperature (normalized)
                        c_rate / 4.0,      # Charge rate normalized (max 4C)
                        c_rate / 4.0,      # Discharge rate (same as charge for XJTU)
                        0.5,               # Mid SOC
                        0.0,               # Constant current profile
                        0.0                # Cycling mode (not storage)
                    ], dtype=np.float32)
                    
                    samples.append({
                        'cell_id': cell_name,
                        'batch': batch_name,
                        'c_rate': c_rate,
                        'features': features,
                        'context': context,
                        'total_loss': total_loss,
                        'cycles': len(capacity)
                    })
                    
            except Exception as e:
                print(f"Error loading {mat_file.name}: {e}")
    
    return samples


def run_causal_attribution_on_xjtu():
    """Run causal attribution on XJTU high C-rate data."""
    
    print("=" * 60)
    print("CAUSAL ATTRIBUTION ON XJTU HIGH C-RATE DATA")
    print("=" * 60)
    
    # Load causal attribution model
    model_path = Path("reports/causal_attribution/causal_model.pt")
    
    if not model_path.exists():
        print("ERROR: Causal attribution model not found")
        print("Please ensure the model exists at:", model_path)
        return
    
    # Model dimensions from saved checkpoint
    model = CausalAttributionModel(
        feature_dim=9,
        context_dim=6,
        hidden_dim=128
    )
    
    model.load_state_dict(torch.load(model_path, weights_only=False, map_location='cpu'))
    model.eval()
    print(f"✓ Loaded causal attribution model")
    
    # Load XJTU samples
    samples = load_xjtu_for_attribution()
    print(f"\nLoaded {len(samples)} XJTU cells for attribution")
    
    # Group by C-rate
    by_crate = {}
    for s in samples:
        cr = s['c_rate']
        by_crate[cr] = by_crate.get(cr, [])
        by_crate[cr].append(s)
    
    print(f"By C-rate: {[(cr, len(cells)) for cr, cells in by_crate.items()]}")
    
    # Mechanism names (lowercase to match model output)
    mechanism_names = [
        'sei_growth',
        'lithium_plating',
        'am_loss',
        'electrolyte',
        'corrosion'
    ]
    
    # Display names for output
    display_names = {
        'sei_growth': 'SEI_GROWTH',
        'lithium_plating': 'LITHIUM_PLATING',
        'am_loss': 'ACTIVE_MATERIAL_LOSS',
        'electrolyte': 'ELECTROLYTE_DECOMP',
        'corrosion': 'COLLECTOR_CORROSION'
    }
    
    results = {}
    
    for c_rate, cells in sorted(by_crate.items()):
        print(f"\n{'='*60}")
        print(f"{c_rate}C CHARGING ({len(cells)} cells)")
        print("=" * 60)
        
        all_contributions = []
        
        with torch.no_grad():
            for cell in cells:
                features = torch.tensor(cell['features'], dtype=torch.float32).unsqueeze(0)
                context = torch.tensor(cell['context'], dtype=torch.float32).unsqueeze(0)
                total_loss = torch.tensor([cell['total_loss']], dtype=torch.float32)
                
                # Run attribution (model returns dict, not tuple)
                output = model(features, context, use_physics_only=True)
                contributions = output['attributions_pct']
                
                # Store contributions
                contrib_dict = {}
                for name, val in contributions.items():
                    contrib_dict[name] = float(val[0].item())
                
                all_contributions.append(contrib_dict)
        
        # Average contributions across cells
        avg_contributions = {}
        for name in mechanism_names:
            values = [c.get(name, 0) for c in all_contributions]
            avg_contributions[name] = np.mean(values)
        
        # Normalize to sum to 1
        total = sum(avg_contributions.values())
        if total > 0:
            for name in avg_contributions:
                avg_contributions[name] /= total
        
        # Find dominant mechanism
        sorted_mechs = sorted(avg_contributions.items(), key=lambda x: x[1], reverse=True)
        dominant = sorted_mechs[0][0]
        
        print(f"\nMechanism Contributions (averaged across {len(cells)} cells):")
        print("-" * 45)
        for name, val in sorted_mechs:
            disp_name = display_names.get(name, name)
            bar = "█" * int(val * 40)
            marker = " ← DOMINANT" if name == dominant else ""
            print(f"  {disp_name:<25} {val*100:5.1f}% {bar}{marker}")
        
        results[f"{c_rate}C"] = {
            'n_cells': len(cells),
            'avg_contributions': {display_names.get(k, k): v for k, v in avg_contributions.items()},
            'dominant_mechanism': display_names.get(dominant, dominant),
            'dominant_percentage': float(sorted_mechs[0][1])
        }
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: HIGH C-RATE CAUSAL ATTRIBUTION")
    print("=" * 60)
    
    print(f"\n{'C-Rate':<10} {'Cells':<10} {'Dominant Mechanism':<25} {'Contribution':<15}")
    print("-" * 60)
    
    for c_rate, data in sorted(results.items()):
        print(f"{c_rate:<10} {data['n_cells']:<10} {data['dominant_mechanism']:<25} {data['dominant_percentage']*100:.1f}%")
    
    # Expected vs Actual
    print("\n" + "=" * 60)
    print("VALIDATION: EXPECTED vs ACTUAL")
    print("=" * 60)
    
    expected = "ACTIVE_MATERIAL_LOSS"
    for c_rate, data in sorted(results.items()):
        actual = data['dominant_mechanism']
        match = "✓" if actual == expected else "✗"
        print(f"\n{c_rate}: Expected={expected}, Actual={actual} {match}")
        
        # Show top 3
        sorted_contribs = sorted(data['avg_contributions'].items(), key=lambda x: x[1], reverse=True)
        for name, val in sorted_contribs[:3]:
            print(f"    {name}: {val*100:.1f}%")
    
    # Save results
    output_path = Path("reports/xjtu_causal_attribution_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    
    return results


if __name__ == '__main__':
    run_causal_attribution_on_xjtu()
