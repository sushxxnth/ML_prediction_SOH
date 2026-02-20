"""
Verify Pure PINN accuracy by running it on the 75 benchmark scenarios.

This script:
1. Loads the Pure Collocation PINN model
2. Runs it on the 75 benchmark scenarios (same as used for Hybrid PINN)
3. Reports accuracy to determine the correct number for the paper
"""

import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from src.models.pure_collocation_pinn import PureCollocationPINN


def get_all_scenarios():
    """Get all 75 benchmark scenarios with correct context encoding."""
    scenarios = []
    
    # NASA Ames (15 scenarios)
    nasa_scenarios = [
        {"name": "NASA_cold_1C", "temp": 4.0, "charge": 1.0, "discharge": 1.0, "soc": 0.5, "profile": 0.0, "mode": 1.0, "expected": "lithium_plating"},
        {"name": "NASA_cold_1.5C", "temp": 4.0, "charge": 1.5, "discharge": 1.0, "soc": 0.5, "profile": 0.0, "mode": 1.0, "expected": "lithium_plating"},
        {"name": "NASA_cold_2C", "temp": 4.0, "charge": 2.0, "discharge": 2.0, "soc": 0.5, "profile": 0.0, "mode": 1.0, "expected": "lithium_plating"},
        {"name": "NASA_cold_0.5C", "temp": 4.0, "charge": 0.5, "discharge": 0.5, "soc": 0.5, "profile": 0.0, "mode": 1.0, "expected": "lithium_plating"},
        {"name": "NASA_moderate_storage", "temp": 24.0, "charge": 0.0, "discharge": 0.0, "soc": 0.5, "profile": 0.0, "mode": 0.0, "expected": "sei_growth"},
        {"name": "NASA_moderate_gentle", "temp": 24.0, "charge": 0.5, "discharge": 0.5, "soc": 0.5, "profile": 0.0, "mode": 1.0, "expected": "sei_growth"},
        {"name": "NASA_moderate_1C", "temp": 24.0, "charge": 1.0, "discharge": 1.0, "soc": 0.5, "profile": 0.0, "mode": 1.0, "expected": "sei_growth"},
        {"name": "NASA_warm_gentle", "temp": 34.0, "charge": 0.5, "discharge": 0.5, "soc": 0.5, "profile": 0.0, "mode": 1.0, "expected": "sei_growth"},
        {"name": "NASA_warm_1C", "temp": 34.0, "charge": 1.0, "discharge": 1.0, "soc": 0.5, "profile": 0.0, "mode": 1.0, "expected": "sei_growth"},
        {"name": "NASA_hot_1C", "temp": 43.0, "charge": 1.0, "discharge": 1.0, "soc": 0.5, "profile": 0.0, "mode": 1.0, "expected": "sei_growth"},
        {"name": "NASA_hot_2C", "temp": 43.0, "charge": 2.0, "discharge": 2.0, "soc": 0.5, "profile": 0.0, "mode": 1.0, "expected": "am_loss"},
        {"name": "NASA_hot_high_discharge", "temp": 43.0, "charge": 1.0, "discharge": 2.0, "soc": 0.5, "profile": 0.0, "mode": 1.0, "expected": "am_loss"},
        {"name": "NASA_moderate_2C", "temp": 24.0, "charge": 2.0, "discharge": 2.0, "soc": 0.5, "profile": 0.0, "mode": 1.0, "expected": "am_loss"},
        {"name": "NASA_storage_low_soc", "temp": 24.0, "charge": 0.0, "discharge": 0.0, "soc": 0.15, "profile": 0.0, "mode": 0.0, "expected": "corrosion"},
        {"name": "NASA_storage_very_low_soc", "temp": 24.0, "charge": 0.0, "discharge": 0.0, "soc": 0.10, "profile": 0.0, "mode": 0.0, "expected": "corrosion"},
    ]
    
    # TJU/Panasonic NCM/NCA (15 scenarios) - cold weather focus
    panasonic_scenarios = [
        {"name": "Pan_neg20_1C", "temp": -20.0, "charge": 1.0, "discharge": 1.0, "soc": 0.5, "profile": 0.0, "mode": 1.0, "expected": "lithium_plating"},
        {"name": "Pan_neg10_1C", "temp": -10.0, "charge": 1.0, "discharge": 1.0, "soc": 0.5, "profile": 0.0, "mode": 1.0, "expected": "lithium_plating"},
        {"name": "Pan_0C_1C", "temp": 0.0, "charge": 1.0, "discharge": 1.0, "soc": 0.5, "profile": 0.0, "mode": 1.0, "expected": "lithium_plating"},
        {"name": "Pan_neg20_0.5C", "temp": -20.0, "charge": 0.5, "discharge": 0.5, "soc": 0.5, "profile": 0.0, "mode": 1.0, "expected": "lithium_plating"},
        {"name": "Pan_neg10_0.5C", "temp": -10.0, "charge": 0.5, "discharge": 0.5, "soc": 0.5, "profile": 0.0, "mode": 1.0, "expected": "lithium_plating"},
        {"name": "Pan_5C_1C", "temp": 5.0, "charge": 1.0, "discharge": 1.0, "soc": 0.5, "profile": 0.0, "mode": 1.0, "expected": "lithium_plating"},
        {"name": "Pan_25_gentle", "temp": 25.0, "charge": 0.5, "discharge": 0.5, "soc": 0.5, "profile": 0.0, "mode": 1.0, "expected": "sei_growth"},
        {"name": "Pan_25_1C", "temp": 25.0, "charge": 1.0, "discharge": 1.0, "soc": 0.5, "profile": 0.0, "mode": 1.0, "expected": "sei_growth"},
        {"name": "Pan_25_storage", "temp": 25.0, "charge": 0.0, "discharge": 0.0, "soc": 0.5, "profile": 0.0, "mode": 0.0, "expected": "sei_growth"},
        {"name": "Pan_US06", "temp": 25.0, "charge": 1.0, "discharge": 2.0, "soc": 0.5, "profile": 1.0, "mode": 1.0, "expected": "am_loss"},
        {"name": "Pan_HWFET", "temp": 25.0, "charge": 0.5, "discharge": 1.0, "soc": 0.5, "profile": 2.0, "mode": 1.0, "expected": "sei_growth"},
        {"name": "Pan_LA92", "temp": 25.0, "charge": 1.0, "discharge": 1.5, "soc": 0.5, "profile": 3.0, "mode": 1.0, "expected": "am_loss"},
        {"name": "Pan_UDDS", "temp": 25.0, "charge": 0.5, "discharge": 0.5, "soc": 0.5, "profile": 4.0, "mode": 1.0, "expected": "sei_growth"},
        {"name": "Pan_25_2C", "temp": 25.0, "charge": 2.0, "discharge": 2.0, "soc": 0.5, "profile": 0.0, "mode": 1.0, "expected": "am_loss"},
        {"name": "Pan_25_1.5C_high", "temp": 25.0, "charge": 1.5, "discharge": 1.5, "soc": 0.5, "profile": 0.0, "mode": 1.0, "expected": "am_loss"},
    ]
    
    # Nature MATR fast charging (15 scenarios)
    nature_scenarios = [
        {"name": "Nature_1C", "temp": 30.0, "charge": 1.0, "discharge": 1.0, "soc": 0.5, "profile": 0.0, "mode": 1.0, "expected": "sei_growth"},
        {"name": "Nature_2C", "temp": 30.0, "charge": 2.0, "discharge": 1.0, "soc": 0.5, "profile": 0.0, "mode": 1.0, "expected": "am_loss"},
        {"name": "Nature_3C", "temp": 30.0, "charge": 3.0, "discharge": 1.0, "soc": 0.5, "profile": 0.0, "mode": 1.0, "expected": "am_loss"},
        {"name": "Nature_4C", "temp": 30.0, "charge": 4.0, "discharge": 1.0, "soc": 0.5, "profile": 0.0, "mode": 1.0, "expected": "am_loss"},
        {"name": "Nature_5C", "temp": 30.0, "charge": 5.0, "discharge": 1.0, "soc": 0.5, "profile": 0.0, "mode": 1.0, "expected": "am_loss"},
        {"name": "Nature_6C", "temp": 30.0, "charge": 6.0, "discharge": 1.0, "soc": 0.5, "profile": 0.0, "mode": 1.0, "expected": "am_loss"},
        {"name": "Nature_7C", "temp": 30.0, "charge": 7.0, "discharge": 1.0, "soc": 0.5, "profile": 0.0, "mode": 1.0, "expected": "am_loss"},
        {"name": "Nature_8C", "temp": 30.0, "charge": 8.0, "discharge": 1.0, "soc": 0.5, "profile": 0.0, "mode": 1.0, "expected": "am_loss"},
        {"name": "Nature_1.5C", "temp": 30.0, "charge": 1.5, "discharge": 1.0, "soc": 0.5, "profile": 0.0, "mode": 1.0, "expected": "am_loss"},
        {"name": "Nature_2.5C", "temp": 30.0, "charge": 2.5, "discharge": 1.0, "soc": 0.5, "profile": 0.0, "mode": 1.0, "expected": "am_loss"},
        {"name": "Nature_3.5C", "temp": 30.0, "charge": 3.5, "discharge": 1.0, "soc": 0.5, "profile": 0.0, "mode": 1.0, "expected": "am_loss"},
        {"name": "Nature_4.5C", "temp": 30.0, "charge": 4.5, "discharge": 1.0, "soc": 0.5, "profile": 0.0, "mode": 1.0, "expected": "am_loss"},
        {"name": "Nature_5.5C", "temp": 30.0, "charge": 5.5, "discharge": 1.0, "soc": 0.5, "profile": 0.0, "mode": 1.0, "expected": "am_loss"},
        {"name": "Nature_6.5C", "temp": 30.0, "charge": 6.5, "discharge": 1.0, "soc": 0.5, "profile": 0.0, "mode": 1.0, "expected": "am_loss"},
        {"name": "Nature_storage", "temp": 30.0, "charge": 0.0, "discharge": 0.0, "soc": 0.5, "profile": 0.0, "mode": 0.0, "expected": "sei_growth"},
    ]
    
    # Randomized 40C stress (15 scenarios)
    random_scenarios = [
        {"name": "Rand_40C_2C_dis", "temp": 40.0, "charge": 1.0, "discharge": 2.0, "soc": 0.5, "profile": 0.0, "mode": 1.0, "expected": "am_loss"},
        {"name": "Rand_40C_2.5C", "temp": 40.0, "charge": 2.5, "discharge": 2.5, "soc": 0.5, "profile": 0.0, "mode": 1.0, "expected": "am_loss"},
        {"name": "Rand_40C_1.5C", "temp": 40.0, "charge": 1.5, "discharge": 1.5, "soc": 0.5, "profile": 0.0, "mode": 1.0, "expected": "am_loss"},
        {"name": "Rand_40C_3C", "temp": 40.0, "charge": 3.0, "discharge": 3.0, "soc": 0.5, "profile": 0.0, "mode": 1.0, "expected": "am_loss"},
        {"name": "Rand_40C_1C", "temp": 40.0, "charge": 1.0, "discharge": 1.0, "soc": 0.5, "profile": 0.0, "mode": 1.0, "expected": "sei_growth"},
        {"name": "Rand_40C_0.5C", "temp": 40.0, "charge": 0.5, "discharge": 0.5, "soc": 0.5, "profile": 0.0, "mode": 1.0, "expected": "sei_growth"},
        {"name": "Rand_40C_storage", "temp": 40.0, "charge": 0.0, "discharge": 0.0, "soc": 0.5, "profile": 0.0, "mode": 0.0, "expected": "sei_growth"},
        {"name": "Rand_40C_storage_high", "temp": 40.0, "charge": 0.0, "discharge": 0.0, "soc": 0.9, "profile": 0.0, "mode": 0.0, "expected": "sei_growth"},
        {"name": "Rand_40C_storage_low", "temp": 40.0, "charge": 0.0, "discharge": 0.0, "soc": 0.2, "profile": 0.0, "mode": 0.0, "expected": "sei_growth"},
        {"name": "Rand_40C_1C_asym", "temp": 40.0, "charge": 0.5, "discharge": 1.0, "soc": 0.5, "profile": 0.0, "mode": 1.0, "expected": "am_loss"},
        {"name": "Rand_40C_2C_charge", "temp": 40.0, "charge": 2.0, "discharge": 1.0, "soc": 0.5, "profile": 0.0, "mode": 1.0, "expected": "am_loss"},
        {"name": "Rand_40C_high_soc", "temp": 40.0, "charge": 1.0, "discharge": 1.0, "soc": 0.85, "profile": 0.0, "mode": 1.0, "expected": "sei_growth"},
        {"name": "Rand_40C_gentle_cycle", "temp": 40.0, "charge": 0.5, "discharge": 0.5, "soc": 0.3, "profile": 0.0, "mode": 1.0, "expected": "sei_growth"},
        {"name": "Rand_40C_2C_dis_only", "temp": 40.0, "charge": 0.5, "discharge": 2.0, "soc": 0.5, "profile": 0.0, "mode": 1.0, "expected": "am_loss"},
        {"name": "Rand_40C_moderate", "temp": 40.0, "charge": 1.0, "discharge": 1.5, "soc": 0.5, "profile": 0.0, "mode": 1.0, "expected": "am_loss"},
    ]
    
    # HUST LFP (15 scenarios)
    hust_scenarios = [
        {"name": "HUST_25C_1C", "temp": 25.0, "charge": 1.0, "discharge": 1.0, "soc": 0.5, "profile": 0.0, "mode": 1.0, "expected": "sei_growth"},
        {"name": "HUST_25C_0.5C", "temp": 25.0, "charge": 0.5, "discharge": 0.5, "soc": 0.5, "profile": 0.0, "mode": 1.0, "expected": "sei_growth"},
        {"name": "HUST_25C_2C", "temp": 25.0, "charge": 2.0, "discharge": 2.0, "soc": 0.5, "profile": 0.0, "mode": 1.0, "expected": "am_loss"},
        {"name": "HUST_40C_1C", "temp": 40.0, "charge": 1.0, "discharge": 1.0, "soc": 0.5, "profile": 0.0, "mode": 1.0, "expected": "sei_growth"},
        {"name": "HUST_40C_2C", "temp": 40.0, "charge": 2.0, "discharge": 2.0, "soc": 0.5, "profile": 0.0, "mode": 1.0, "expected": "am_loss"},
        {"name": "HUST_25C_storage", "temp": 25.0, "charge": 0.0, "discharge": 0.0, "soc": 0.5, "profile": 0.0, "mode": 0.0, "expected": "sei_growth"},
        {"name": "HUST_25C_1.5C", "temp": 25.0, "charge": 1.5, "discharge": 1.5, "soc": 0.5, "profile": 0.0, "mode": 1.0, "expected": "am_loss"},
        {"name": "HUST_25C_3C", "temp": 25.0, "charge": 3.0, "discharge": 3.0, "soc": 0.5, "profile": 0.0, "mode": 1.0, "expected": "am_loss"},
        {"name": "HUST_25C_2.5C", "temp": 25.0, "charge": 2.5, "discharge": 2.5, "soc": 0.5, "profile": 0.0, "mode": 1.0, "expected": "am_loss"},
        {"name": "HUST_40C_0.5C", "temp": 40.0, "charge": 0.5, "discharge": 0.5, "soc": 0.5, "profile": 0.0, "mode": 1.0, "expected": "sei_growth"},
        {"name": "HUST_25C_charge_2C", "temp": 25.0, "charge": 2.0, "discharge": 1.0, "soc": 0.5, "profile": 0.0, "mode": 1.0, "expected": "am_loss"},
        {"name": "HUST_40C_1.5C", "temp": 40.0, "charge": 1.5, "discharge": 1.5, "soc": 0.5, "profile": 0.0, "mode": 1.0, "expected": "am_loss"},
        {"name": "HUST_25C_asym", "temp": 25.0, "charge": 0.5, "discharge": 1.0, "soc": 0.5, "profile": 0.0, "mode": 1.0, "expected": "sei_growth"},
        {"name": "HUST_40C_storage", "temp": 40.0, "charge": 0.0, "discharge": 0.0, "soc": 0.5, "profile": 0.0, "mode": 0.0, "expected": "sei_growth"},
        {"name": "HUST_25C_high_soc", "temp": 25.0, "charge": 1.0, "discharge": 1.0, "soc": 0.85, "profile": 0.0, "mode": 1.0, "expected": "sei_growth"},
    ]
    
    all_scenarios = nasa_scenarios + panasonic_scenarios + nature_scenarios + random_scenarios + hust_scenarios
    return all_scenarios


# Base features (9 dims - standard battery features)
BASE_FEATURES = [
    0.85,   # SOH
    100.0,  # cycles
    3.7,    # voltage
    1.0,    # current
    0.5,    # capacity
    25.0,   # temperature reference
    0.5,    # dV/dQ
    0.01,   # resistance
    0.95    # coulombic efficiency
]

MECHANISM_IDX = {
    "sei_growth": 0,
    "lithium_plating": 1,
    "am_loss": 2,
    "electrolyte": 3,
    "corrosion": 4,
}


def run_pure_pinn_evaluation():
    """Run Pure Collocation PINN on all 75 scenarios to get its accuracy."""
    print("=" * 70)
    print("PURE COLLOCATION PINN ACCURACY VERIFICATION")
    print("=" * 70)
    
    # Create model
    model = PureCollocationPINN(feature_dim=9, context_dim=6)
    
    # Check if there's a saved model — if not, test with random init
    # The Pure PINN was described as trained on 11,066 samples
    # But we need to test what it achieves
    
    # Check for saved weights
    weight_paths = [
        "reports/pinn_causal/pure_pinn_best.pt",
        "reports/pinn_causal/pure_collocation_pinn.pt",
    ]
    
    loaded = False
    for wpath in weight_paths:
        if os.path.exists(wpath):
            print(f"\nLoading saved weights from: {wpath}")
            state = torch.load(wpath, map_location='cpu')
            if isinstance(state, dict) and 'model_state_dict' in state:
                model.load_state_dict(state['model_state_dict'])
            else:
                model.load_state_dict(state)
            loaded = True
            break
    
    if not loaded:
        print("\nNo saved Pure PINN weights found. Testing with random init as baseline...")
        print("Available weight files in reports/pinn_causal/:")
        import glob
        for f in glob.glob("reports/pinn_causal/*.pt"):
            print(f"  {f}")
    
    model.eval()
    scenarios = get_all_scenarios()
    
    correct = 0
    total = 0
    errors = []
    
    by_dataset = {"NASA": [0, 0], "Panasonic": [0, 0], "Nature": [0, 0], "Randomized": [0, 0], "HUST": [0, 0]}
    by_mechanism = {}
    
    mechanism_names = ["sei_growth", "lithium_plating", "am_loss", "electrolyte", "corrosion"]
    
    with torch.no_grad():
        for i, sc in enumerate(scenarios):
            features = torch.FloatTensor(BASE_FEATURES).unsqueeze(0)
            context = torch.FloatTensor([
                sc["temp"] / 60.0,
                sc["charge"] / 3.0,
                sc["discharge"] / 4.0,
                sc["soc"],
                sc["profile"] / 4.0 if sc["profile"] > 0 else 0.0,
                sc["mode"]
            ]).unsqueeze(0)
            
            output = model(features, context, compute_physics=False)
            probs = output["mechanism_probs"]
            predicted_idx = probs.argmax(dim=1).item()
            predicted_mech = mechanism_names[predicted_idx]
            
            expected_mech = sc["expected"]
            expected_idx = MECHANISM_IDX[expected_mech]
            
            is_correct = (predicted_idx == expected_idx)
            total += 1
            
            # Determine dataset
            if i < 15:
                ds = "NASA"
            elif i < 30:
                ds = "Panasonic"
            elif i < 45:
                ds = "Nature"
            elif i < 60:
                ds = "Randomized"
            else:
                ds = "HUST"
            
            if is_correct:
                correct += 1
                by_dataset[ds][0] += 1
            else:
                errors.append(f"  [{ds}] {sc['name']}: expected={expected_mech}, got={predicted_mech} (T={sc['temp']}°C, C={sc['charge']}C)")
            
            by_dataset[ds][1] += 1
            
            if expected_mech not in by_mechanism:
                by_mechanism[expected_mech] = [0, 0]
            by_mechanism[expected_mech][1] += 1
            if is_correct:
                by_mechanism[expected_mech][0] += 1
    
    accuracy = correct / total * 100
    
    print(f"\n{'='*70}")
    print(f"PURE PINN RESULTS: {correct}/{total} = {accuracy:.1f}%")
    print(f"{'='*70}")
    
    print(f"\nPer-dataset breakdown:")
    for ds, (c, t) in by_dataset.items():
        pct = c/t*100 if t > 0 else 0
        print(f"  {ds}: {c}/{t} = {pct:.0f}%")
    
    print(f"\nPer-mechanism breakdown:")
    for mech, (c, t) in sorted(by_mechanism.items()):
        pct = c/t*100 if t > 0 else 0
        print(f"  {mech}: {c}/{t} = {pct:.0f}%")
    
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors:
            print(e)
    
    print(f"\n{'='*70}")
    print(f"CONCLUSION: Pure PINN accuracy is {accuracy:.1f}%")
    print(f"Paper table says 60.0%, methodology text says 77.3%")
    if abs(accuracy - 60.0) < 2:
        print(f"→ MATCHES TABLE (60.0%)")
    elif abs(accuracy - 77.3) < 2:
        print(f"→ MATCHES METHODOLOGY TEXT (77.3%)")
    else:
        print(f"→ MATCHES NEITHER. This is the actual evaluated value.")
    print(f"{'='*70}")


if __name__ == '__main__':
    run_pure_pinn_evaluation()
