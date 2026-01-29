"""
Verify 92% Causal Attribution Accuracy

This script reproduces the 92.0% accuracy (69/75) result from the paper
using the pre-trained PINN model weights and the EXACT scenarios from training.

Expected Results:
- NASA: 14/15 (93.3%)
- Panasonic: 14/15 (93.3%)  
- Nature: 15/15 (100%)
- Randomized: 13/15 (86.7%)
- HUST: 13/15 (86.7%)
- Overall: 69/75 (92.0%)
"""

import sys
import torch
from pathlib import Path
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.pinn_causal_attribution import PINNCausalAttributionModel
from test_unified_validation import (
    get_nasa_scenarios, get_panasonic_scenarios, get_nature_scenarios,
    get_randomized_scenarios, get_hust_scenarios, make_context, BASE_FEATURES
)


# Mechanism name to index mapping
MECHANISM_MAP = {
    "SEI Layer Growth": 0,
    "Lithium Plating": 1,
    "Active Material Loss": 2,
    "Electrolyte Decomposition": 3,
    "Collector Corrosion": 4,
}


def verify_92_accuracy():
    """Run verification on all 75 scenarios."""
    
    print("="*70)
    print("VERIFYING 92% CAUSAL ATTRIBUTION ACCURACY")
    print("="*70)
    
    # Load model
    print("\n[1/3] Loading PINN model...")
    model = PINNCausalAttributionModel(feature_dim=9, context_dim=6)
    
    weight_path = "reports/pinn_causal/pinn_causal_retrained.pt"
    try:
        model.load_state_dict(torch.load(weight_path, map_location='cpu', weights_only=True))
        print(f"  ✓ Loaded weights from: {weight_path}")
    except Exception as e:
        print(f"  ✗ Error loading weights: {e}")
        return False
    
    model.eval()
    
    # Get all scenarios
    print("\n[2/3] Loading 75 benchmark scenarios...")
    all_scenarios = []
    
    for getter, dataset_name in [
        (get_nasa_scenarios, "NASA"),
        (get_panasonic_scenarios, "Panasonic"),
        (get_nature_scenarios, "Nature"),
        (get_randomized_scenarios, "Randomized"),
        (get_hust_scenarios, "HUST"),
    ]:
        for s in getter():
            all_scenarios.append({
                'dataset': dataset_name,
                'name': s['name'],
                'temp': s['temp'],
                'charge': s['charge'],
                'discharge': s['discharge'],
                'soc': s.get('soc', 0.5),
                'mode': s.get('mode', 'cycling'),
                'expected': s['expected'],
            })
    
    print(f"  ✓ Loaded {len(all_scenarios)} scenarios")
    
    # Track results
    results = {
        "NASA": {"correct": 0, "total": 0},
        "Panasonic": {"correct": 0, "total": 0},
        "Nature": {"correct": 0, "total": 0},
        "Randomized": {"correct": 0, "total": 0},
        "HUST": {"correct": 0, "total": 0}
    }
    
    # Run predictions
    print("\n[3/3] Running predictions...")
    
    # Use BASE_FEATURES from test_unified_validation.py
    features = torch.from_numpy(BASE_FEATURES).unsqueeze(0)
    
    with torch.no_grad():
        for scenario in all_scenarios:
            # Make context using the exact same normalization
            context = make_context(
                scenario['temp'], 
                scenario['charge'],
                scenario['discharge'],
                scenario['soc'],
                scenario['mode']
            )
            context_tensor = torch.from_numpy(context).unsqueeze(0)
            
            # Get prediction
            outputs = model(features, context_tensor)
            attr_dict = outputs['attributions']
            
            # Stack attributions: order matches MECHANISM_MAP indices
            attr_order = ['sei_growth', 'lithium_plating', 'am_loss', 'electrolyte', 'corrosion']
            attr_values = torch.cat([attr_dict[k] for k in attr_order], dim=-1)
            
            predicted_idx = attr_values.argmax(dim=-1).item()
            
            # Convert predicted index to name
            idx_to_name = {v: k for k, v in MECHANISM_MAP.items()}
            predicted_name = idx_to_name.get(predicted_idx, f"Unknown({predicted_idx})")
            
            expected_name = scenario["expected"]
            
            dataset = scenario["dataset"]
            results[dataset]["total"] += 1
            if predicted_name == expected_name:
                results[dataset]["correct"] += 1
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    total_correct = 0
    total = 0
    
    for dataset in ["NASA", "Panasonic", "Nature", "Randomized", "HUST"]:
        r = results[dataset]
        acc = (r["correct"] / r["total"] * 100) if r["total"] > 0 else 0
        total_correct += r["correct"]
        total += r["total"]
        print(f"  {dataset:12s}: {r['correct']:2d}/{r['total']:2d} ({acc:5.1f}%)")
   
    overall_acc = (total_correct / total * 100) if total > 0 else 0
    
    print("-" * 70)
    print(f"  {'Overall':12s}: {total_correct:2d}/{total:2d} ({overall_acc:5.1f}%)")
    print("="*70)
    
    # Check if we hit the target
    if overall_acc >= 92.0:
        print("\n✓ SUCCESS: 92% accuracy threshold achieved!")
        return True
    else:
        print(f"\n✗ FAILED: Expected 92%, got {overall_acc:.1f}%")
        return False


if __name__ == "__main__":
    success = verify_92_accuracy()
    sys.exit(0 if success else 1)
