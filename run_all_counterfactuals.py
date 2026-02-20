import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.pinn_causal_attribution import PINNCausalAttributionModel
import torch

try:
    print("Loading actual Hybrid PINN weights...")
    model = PINNCausalAttributionModel()
    model.load_state_dict(torch.load("reports/pinn_causal/pinn_causal_retrained.pt", map_location='cpu'))
    print("Weights loaded successfully.\n")
except Exception as e:
    print("Error loading weights:", e)
    sys.exit(1)

import validate_counterfactual_optimization as vco

# Override simulator instantiation in validation script
original_main = vco.main

def new_main():
    print("=" * 70)
    print("COUNTERFACTUAL INTERVENTION VALIDATION (WITH REAL HYBRID PINN)")
    print("Testing on Real NASA & XJTU Battery Scenarios")
    print("=" * 70)
    
    # Create optimizer with real model
    simulator = vco.CounterfactualSimulator(hybrid_pinn_model=model)
    optimizer = vco.InterventionOptimizer(simulator)
    
    nasa_scenarios = vco.load_nasa_scenarios()
    xjtu_scenarios = vco.load_xjtu_scenarios()
    
    all_scenarios = nasa_scenarios + xjtu_scenarios
    results = []
    
    for scenario in all_scenarios:
        result = vco.validate_scenario(scenario, optimizer)
        results.append(result)
        
    avg_reduction = np.mean([r['mechanism_reduction'] for r in results])
    avg_alignment = np.mean([r['alignment_with_known'] for r in results])
    
    print(f"\n\n{'='*70}")
    print("VALIDATION SUMMARY (WITH REAL PINN)")
    print("=" * 70)
    print(f"\n  ✓ Average dominant mechanism reduction: {avg_reduction:.1f}%")
    print(f"  ✓ Alignment with known optimal strategies: {avg_alignment:.0f}%")

import numpy as np
new_main()
