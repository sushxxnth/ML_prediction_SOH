"""
Verify Causal Attribution Accuracy - FIXED VERSION

This script validates the causal attribution model using the SAME
methodology as VERIFY_92_ACCURACY.py (which works correctly).

Uses:
- PINNCausalAttributionModel directly (not CausalExplainer)
- Scenarios from test_unified_validation.py
- Retrained weights from reports/pinn_causal/pinn_causal_retrained.pt

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


def verify_causal_accuracy():
    """Run verification on all 75 scenarios - FIXED VERSION."""
    
    print("="*70)
    print("CAUSAL ATTRIBUTION ACCURACY VERIFICATION (Fixed)")
    print("="*70)
    
    # Load model
    print("\\n[1/3] Loading PINN model...")
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
    print("\\n[2/3] Loading 75 benchmark scenarios...")
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
        "NASA": {"correct": 0, "total": 0, "errors": []},
        "Panasonic": {"correct": 0, "total": 0, "errors": []},
        "Nature": {"correct": 0, "total": 0, "errors": []},
        "Randomized": {"correct": 0, "total": 0, "errors": []},
        "HUST": {"correct": 0, "total": 0, "errors": []}
    }
    
    # Run predictions
    print("\\n[3/3] Running predictions...")
    
    # Use BASE_FEATURES from test_unified_validation.py
    features = torch.from_numpy(BASE_FEATURES).unsqueeze(0)
    
    with torch.no_grad():
        for scenario in all_scenarios:
            # Make context vector
            context = make_context(
                scenario['temp'],
                scenario['charge'],
                scenario['discharge'],
                scenario['soc'],
                scenario['mode']
            )
            context_t = torch.from_numpy(context).unsqueeze(0)
            
            # Run model
            outputs = model(features, context_t)
            
            # Extract attributions
            attr_dict = outputs['attributions']
            
            # Stack in correct order
            attr_order = ['sei_growth', 'lithium_plating', 'am_loss', 'electrolyte', 'corrosion']
            attr_tensor = torch.cat([attr_dict[k] for k in attr_order], dim=-1)
            
            # Get highest attribution
            pred_idx = attr_tensor.argmax(dim=-1).item()
            
            # Convert index to mechanism name
            idx_to_name = {v: k for k, v in MECHANISM_MAP.items()}
            predicted = idx_to_name[pred_idx]
            
            # Check if correct
            is_correct = (predicted == scenario['expected'])
            
            dataset = scenario['dataset']
            results[dataset]['total'] += 1
            if is_correct:
                results[dataset]['correct'] += 1
            else:
                results[dataset]['errors'].append({
                    'name': scenario['name'],
                    'expected': scenario['expected'],
                    'predicted': predicted
                })
    
    # Print results
    print("\\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    total_correct = 0
    total = 0
    
    for dataset in ["NASA", "Panasonic", "Nature", "Randomized", "HUST"]:
        r = results[dataset]
        acc = r['correct'] / r['total'] if r['total'] > 0 else 0
        total_correct += r['correct']
        total += r['total']
        
        print(f"  {dataset:<15}: {r['correct']}/{r['total']} ({acc*100:5.1f}%)")
        
        # Show errors
        if r['errors']:
            for error in r['errors']:
                print(f"    ✗ {error['name']}: expected '{error['expected']}', got '{error['predicted']}'")
    
    overall_acc = total_correct / total if total > 0 else 0
    print("-"*70)
    print(f"  {'Overall':<15}: {total_correct}/{total} ({overall_acc*100:5.1f}%)")
    print("="*70)
    
    if overall_acc >= 0.92:
        print("\\n✓ SUCCESS: 92% accuracy threshold achieved!")
        return True
    else:
        print(f"\\n✗ FAILURE: Only {overall_acc*100:.1f}%, expected 92%")
        return False


if __name__ == '__main__':
    success = verify_causal_accuracy()
    sys.exit(0 if success else 1)
