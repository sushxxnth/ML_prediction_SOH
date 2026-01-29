"""
Verify Causal Attribution Accuracy (90.7% from paper)

This script validates the causal attribution model using the SAME
methodology as in the paper - with physics-only mode enabled.

The paper achieved:
- NASA Ames: 87% (13/15)
- Panasonic 18650PF: 87% (13/15)
- Nature MATR: 100% (15/15)
- Randomized: 87% (13/15)
- HUST LFP: 93% (14/15)
- Total: 90.7% (68/75)
"""

import sys
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from src.models.causal_attribution import (
    CausalAttributionModel,
    CausalExplainer,
    DegradationMechanism,
)


def get_test_scenarios():
    """
    Generate the 75 test scenarios matching the paper's validation.
    
    Based on Table 5 in the paper:
    - NASA Ames: 15 scenarios
    - Panasonic 18650PF: 15 scenarios
    - Nature MATR: 15 scenarios
    - Randomized: 15 scenarios
    - HUST LFP: 15 scenarios
    """
    
    scenarios = []
    
    # NASA Ames scenarios
    # Mix of storage and cycling at various conditions
    nasa_scenarios = [
        # SEI Growth scenarios (high temp, high SOC, storage)
        {"name": "NASA_1", "temp": 0.75, "c_rate": 0.0, "d_rate": 0.0, "soc": 0.95, "mode": 1.0, "expected": "SEI Layer Growth"},
        {"name": "NASA_2", "temp": 0.625, "c_rate": 0.0, "d_rate": 0.0, "soc": 0.60, "mode": 1.0, "expected": "SEI Layer Growth"},
        {"name": "NASA_3", "temp": 0.5625, "c_rate": 0.1, "d_rate": 0.1, "soc": 0.80, "mode": 0.7, "expected": "SEI Layer Growth"},
        {"name": "NASA_4", "temp": 0.5625, "c_rate": 0.33, "d_rate": 0.33, "soc": 0.50, "mode": 0.0, "expected": "SEI Layer Growth"},
        # AM Loss scenarios (high C-rate)
        {"name": "NASA_5", "temp": 0.5625, "c_rate": 0.67, "d_rate": 0.67, "soc": 0.50, "mode": 0.0, "expected": "Active Material Loss"},
        {"name": "NASA_6", "temp": 0.625, "c_rate": 0.5, "d_rate": 0.75, "soc": 0.30, "mode": 0.0, "expected": "Active Material Loss"},
        {"name": "NASA_7", "temp": 0.5625, "c_rate": 0.33, "d_rate": 0.9, "soc": 0.50, "mode": 0.0, "expected": "Active Material Loss"},
        # Plating scenarios (cold + fast charge)
        {"name": "NASA_8", "temp": 0.25, "c_rate": 0.5, "d_rate": 0.25, "soc": 0.60, "mode": 0.0, "expected": "Lithium Plating"},
        {"name": "NASA_9", "temp": 0.125, "c_rate": 0.67, "d_rate": 0.25, "soc": 0.50, "mode": 0.0, "expected": "Lithium Plating"},
        # More mixed scenarios
        {"name": "NASA_10", "temp": 0.9, "c_rate": 0.33, "d_rate": 0.33, "soc": 0.70, "mode": 0.3, "expected": "SEI Layer Growth"},
        {"name": "NASA_11", "temp": 0.5, "c_rate": 0.0, "d_rate": 0.0, "soc": 0.10, "mode": 1.0, "expected": "Collector Corrosion"},
        {"name": "NASA_12", "temp": 0.525, "c_rate": 0.2, "d_rate": 0.2, "soc": 0.50, "mode": 0.2, "expected": "SEI Layer Growth"},
        {"name": "NASA_13", "temp": 0.5625, "c_rate": 0.33, "d_rate": 0.4, "soc": 0.55, "mode": 0.1, "expected": "SEI Layer Growth"},
        {"name": "NASA_14", "temp": 0.5625, "c_rate": 0.167, "d_rate": 0.15, "soc": 0.85, "mode": 0.6, "expected": "SEI Layer Growth"},
        {"name": "NASA_15", "temp": 0.6875, "c_rate": 0.8, "d_rate": 0.8, "soc": 0.40, "mode": 0.0, "expected": "Active Material Loss"},
    ]
    
    for s in nasa_scenarios:
        scenarios.append({
            "dataset": "NASA Ames",
            "name": s["name"],
            "context": np.array([s["temp"], s["c_rate"], s["d_rate"], s["soc"], 0.0, s["mode"]], dtype=np.float32),
            "expected": s["expected"],
        })
    
    # Panasonic 18650PF (EV drive cycles)
    panasonic_scenarios = [
        {"name": "Pan_1", "temp": 0.75, "c_rate": 0.5, "d_rate": 0.8, "soc": 0.50, "mode": 0.0, "expected": "Active Material Loss"},
        {"name": "Pan_2", "temp": 0.5625, "c_rate": 0.33, "d_rate": 0.5, "soc": 0.60, "mode": 0.0, "expected": "SEI Layer Growth"},
        {"name": "Pan_3", "temp": 0.3, "c_rate": 0.4, "d_rate": 0.6, "soc": 0.55, "mode": 0.0, "expected": "Lithium Plating"},
        {"name": "Pan_4", "temp": 0.525, "c_rate": 0.25, "d_rate": 0.4, "soc": 0.50, "mode": 0.0, "expected": "SEI Layer Growth"},
        {"name": "Pan_5", "temp": 0.8, "c_rate": 0.0, "d_rate": 0.0, "soc": 0.90, "mode": 1.0, "expected": "SEI Layer Growth"},
        {"name": "Pan_6", "temp": 0.2, "c_rate": 0.0, "d_rate": 0.0, "soc": 0.70, "mode": 1.0, "expected": "SEI Layer Growth"},
        {"name": "Pan_7", "temp": 0.625, "c_rate": 0.9, "d_rate": 0.3, "soc": 0.50, "mode": 0.0, "expected": "Active Material Loss"},
        {"name": "Pan_8", "temp": 0.25, "c_rate": 0.8, "d_rate": 0.3, "soc": 0.50, "mode": 0.0, "expected": "Lithium Plating"},
        {"name": "Pan_9", "temp": 0.5625, "c_rate": 0.2, "d_rate": 0.3, "soc": 0.65, "mode": 0.4, "expected": "SEI Layer Growth"},
        {"name": "Pan_10", "temp": 0.625, "c_rate": 0.5, "d_rate": 0.6, "soc": 0.45, "mode": 0.1, "expected": "Active Material Loss"},
        {"name": "Pan_11", "temp": 0.525, "c_rate": 0.167, "d_rate": 0.2, "soc": 0.80, "mode": 0.7, "expected": "SEI Layer Growth"},
        {"name": "Pan_12", "temp": 0.5625, "c_rate": 0.4, "d_rate": 0.7, "soc": 0.40, "mode": 0.0, "expected": "Active Material Loss"},
        {"name": "Pan_13", "temp": 0.6875, "c_rate": 0.6, "d_rate": 0.5, "soc": 0.55, "mode": 0.0, "expected": "Active Material Loss"},
        {"name": "Pan_14", "temp": 0.525, "c_rate": 0.2, "d_rate": 0.35, "soc": 0.60, "mode": 0.2, "expected": "SEI Layer Growth"},
        {"name": "Pan_15", "temp": 0.1, "c_rate": 0.5, "d_rate": 0.4, "soc": 0.50, "mode": 0.0, "expected": "Lithium Plating"},
    ]
    
    for s in panasonic_scenarios:
        scenarios.append({
            "dataset": "Panasonic 18650PF",
            "name": s["name"],
            "context": np.array([s["temp"], s["c_rate"], s["d_rate"], s["soc"], 0.5, s["mode"]], dtype=np.float32),
            "expected": s["expected"],
        })
    
    # Nature MATR (fast charging focus) - 100% accuracy in paper
    matr_scenarios = [
        {"name": "MATR_1", "temp": 0.5625, "c_rate": 0.33, "d_rate": 0.33, "soc": 0.50, "mode": 0.0, "expected": "SEI Layer Growth"},
        {"name": "MATR_2", "temp": 0.5625, "c_rate": 0.67, "d_rate": 0.33, "soc": 0.50, "mode": 0.0, "expected": "Active Material Loss"},
        {"name": "MATR_3", "temp": 0.625, "c_rate": 1.0, "d_rate": 0.5, "soc": 0.50, "mode": 0.0, "expected": "Active Material Loss"},
        {"name": "MATR_4", "temp": 0.6875, "c_rate": 1.0, "d_rate": 0.5, "soc": 0.50, "mode": 0.0, "expected": "Active Material Loss"},
        {"name": "MATR_5", "temp": 0.75, "c_rate": 1.0, "d_rate": 0.5, "soc": 0.50, "mode": 0.0, "expected": "Active Material Loss"},
        {"name": "MATR_6", "temp": 0.25, "c_rate": 1.0, "d_rate": 0.5, "soc": 0.50, "mode": 0.0, "expected": "Lithium Plating"},
        {"name": "MATR_7", "temp": 0.6875, "c_rate": 1.0, "d_rate": 0.5, "soc": 0.50, "mode": 0.0, "expected": "Active Material Loss"},
        {"name": "MATR_8", "temp": 0.8, "c_rate": 1.0, "d_rate": 0.5, "soc": 0.50, "mode": 0.0, "expected": "Active Material Loss"},
        {"name": "MATR_9", "temp": 0.5625, "c_rate": 0.5, "d_rate": 0.33, "soc": 0.50, "mode": 0.0, "expected": "Active Material Loss"},
        {"name": "MATR_10", "temp": 0.5625, "c_rate": 0.67, "d_rate": 0.33, "soc": 0.50, "mode": 0.0, "expected": "Active Material Loss"},
        {"name": "MATR_11", "temp": 0.5625, "c_rate": 0.8, "d_rate": 0.33, "soc": 0.50, "mode": 0.0, "expected": "Active Material Loss"},
        {"name": "MATR_12", "temp": 0.5625, "c_rate": 0.4, "d_rate": 0.33, "soc": 0.50, "mode": 0.0, "expected": "SEI Layer Growth"},
        {"name": "MATR_13", "temp": 0.5625, "c_rate": 0.45, "d_rate": 0.33, "soc": 0.50, "mode": 0.0, "expected": "Active Material Loss"},
        {"name": "MATR_14", "temp": 0.625, "c_rate": 0.5, "d_rate": 0.33, "soc": 0.50, "mode": 0.0, "expected": "Active Material Loss"},
        {"name": "MATR_15", "temp": 0.5625, "c_rate": 0.33, "d_rate": 0.33, "soc": 0.50, "mode": 0.0, "expected": "SEI Layer Growth"},
    ]
    
    for s in matr_scenarios:
        scenarios.append({
            "dataset": "Nature MATR",
            "name": s["name"],
            "context": np.array([s["temp"], s["c_rate"], s["d_rate"], s["soc"], 0.0, s["mode"]], dtype=np.float32),
            "expected": s["expected"],
        })
    
    # Randomized usage scenarios
    random_scenarios = [
        {"name": "Rand_1", "temp": 0.05, "c_rate": 0.9, "d_rate": 0.5, "soc": 0.50, "mode": 0.0, "expected": "Lithium Plating"},
        {"name": "Rand_2", "temp": 0.95, "c_rate": 0.0, "d_rate": 0.0, "soc": 0.95, "mode": 1.0, "expected": "SEI Layer Growth"},
        {"name": "Rand_3", "temp": 0.5, "c_rate": 0.0, "d_rate": 0.0, "soc": 0.05, "mode": 1.0, "expected": "Collector Corrosion"},
        {"name": "Rand_4", "temp": 0.6, "c_rate": 0.45, "d_rate": 0.6, "soc": 0.55, "mode": 0.3, "expected": "Active Material Loss"},
        {"name": "Rand_5", "temp": 0.4, "c_rate": 0.6, "d_rate": 0.4, "soc": 0.45, "mode": 0.2, "expected": "Active Material Loss"},
        {"name": "Rand_6", "temp": 0.55, "c_rate": 0.3, "d_rate": 0.5, "soc": 0.60, "mode": 0.4, "expected": "SEI Layer Growth"},
        {"name": "Rand_7", "temp": 0.5625, "c_rate": 0.33, "d_rate": 0.4, "soc": 0.50, "mode": 0.1, "expected": "SEI Layer Growth"},
        {"name": "Rand_8", "temp": 0.525, "c_rate": 0.25, "d_rate": 0.35, "soc": 0.55, "mode": 0.15, "expected": "SEI Layer Growth"},
        {"name": "Rand_9", "temp": 0.5625, "c_rate": 0.4, "d_rate": 0.45, "soc": 0.50, "mode": 0.1, "expected": "SEI Layer Growth"},
        {"name": "Rand_10", "temp": 0.7, "c_rate": 0.7, "d_rate": 0.8, "soc": 0.40, "mode": 0.0, "expected": "Active Material Loss"},
        {"name": "Rand_11", "temp": 0.65, "c_rate": 0.8, "d_rate": 0.7, "soc": 0.45, "mode": 0.0, "expected": "Active Material Loss"},
        {"name": "Rand_12", "temp": 0.75, "c_rate": 0.6, "d_rate": 0.9, "soc": 0.35, "mode": 0.0, "expected": "Active Material Loss"},
        {"name": "Rand_13", "temp": 0.5, "c_rate": 0.1, "d_rate": 0.15, "soc": 0.60, "mode": 0.5, "expected": "SEI Layer Growth"},
        {"name": "Rand_14", "temp": 0.525, "c_rate": 0.15, "d_rate": 0.2, "soc": 0.65, "mode": 0.6, "expected": "SEI Layer Growth"},
        {"name": "Rand_15", "temp": 0.55, "c_rate": 0.2, "d_rate": 0.25, "soc": 0.55, "mode": 0.4, "expected": "SEI Layer Growth"},
    ]
    
    for s in random_scenarios:
        scenarios.append({
            "dataset": "Randomized usage",
            "name": s["name"],
            "context": np.array([s["temp"], s["c_rate"], s["d_rate"], s["soc"], 0.5, s["mode"]], dtype=np.float32),
            "expected": s["expected"],
        })
    
    # HUST LFP scenarios
    hust_scenarios = [
        {"name": "HUST_1", "temp": 0.5625, "c_rate": 0.33, "d_rate": 0.33, "soc": 0.50, "mode": 0.0, "expected": "SEI Layer Growth"},
        {"name": "HUST_2", "temp": 0.5625, "c_rate": 0.67, "d_rate": 0.33, "soc": 0.50, "mode": 0.0, "expected": "Active Material Loss"},
        {"name": "HUST_3", "temp": 0.5625, "c_rate": 0.33, "d_rate": 0.5, "soc": 0.30, "mode": 0.0, "expected": "Active Material Loss"},
        {"name": "HUST_4", "temp": 0.25, "c_rate": 0.33, "d_rate": 0.33, "soc": 0.50, "mode": 0.0, "expected": "Lithium Plating"},
        {"name": "HUST_5", "temp": 0.6875, "c_rate": 0.33, "d_rate": 0.33, "soc": 0.50, "mode": 0.0, "expected": "SEI Layer Growth"},
        {"name": "HUST_6", "temp": 0.8, "c_rate": 0.33, "d_rate": 0.33, "soc": 0.50, "mode": 0.0, "expected": "SEI Layer Growth"},
        {"name": "HUST_7", "temp": 0.5625, "c_rate": 0.0, "d_rate": 0.0, "soc": 0.90, "mode": 1.0, "expected": "SEI Layer Growth"},
        {"name": "HUST_8", "temp": 0.5625, "c_rate": 0.0, "d_rate": 0.0, "soc": 0.50, "mode": 1.0, "expected": "SEI Layer Growth"},
        {"name": "HUST_9", "temp": 0.5625, "c_rate": 0.0, "d_rate": 0.0, "soc": 0.20, "mode": 1.0, "expected": "Collector Corrosion"},
        {"name": "HUST_10", "temp": 0.525, "c_rate": 0.2, "d_rate": 0.25, "soc": 0.60, "mode": 0.3, "expected": "SEI Layer Growth"},
        {"name": "HUST_11", "temp": 0.5625, "c_rate": 0.4, "d_rate": 0.45, "soc": 0.50, "mode": 0.1, "expected": "SEI Layer Growth"},
        {"name": "HUST_12", "temp": 0.625, "c_rate": 0.6, "d_rate": 0.7, "soc": 0.40, "mode": 0.0, "expected": "Active Material Loss"},
        {"name": "HUST_13", "temp": 0.55, "c_rate": 0.35, "d_rate": 0.4, "soc": 0.55, "mode": 0.25, "expected": "SEI Layer Growth"},
        {"name": "HUST_14", "temp": 0.6, "c_rate": 0.45, "d_rate": 0.5, "soc": 0.45, "mode": 0.15, "expected": "Active Material Loss"},
        {"name": "HUST_15", "temp": 0.5, "c_rate": 0.25, "d_rate": 0.35, "soc": 0.60, "mode": 0.35, "expected": "SEI Layer Growth"},
    ]
    
    for s in hust_scenarios:
        scenarios.append({
            "dataset": "HUST LFP",
            "name": s["name"],
            "context": np.array([s["temp"], s["c_rate"], s["d_rate"], s["soc"], 0.0, s["mode"]], dtype=np.float32),
            "expected": s["expected"],
        })
    
    return scenarios


def run_validation():
    """Run validation using physics-only mode (as in paper)."""
    
    print("="*70)
    print("CAUSAL ATTRIBUTION VALIDATION (Paper Methodology)")
    print("="*70)
    
    # Create model and explainer with PHYSICS-ONLY mode
    model = CausalAttributionModel(feature_dim=9, context_dim=6)
    
    # CRITICAL: use_physics_only=True as per paper
    explainer = CausalExplainer(model, use_physics_only=True)
    
    print("\\nUsing physics-only mode: TRUE (matching paper methodology)")
    
    # Get test scenarios
    scenarios = get_test_scenarios()
    print(f"\\nTotal scenarios: {len(scenarios)}")
    
    # Results tracking
    results_by_dataset = defaultdict(lambda: {'correct': 0, 'total': 0, 'errors': []})
    
    # Base features (consistent)
    np.random.seed(42)
    base_features = np.random.randn(9).astype(np.float32) * 0.1 + 0.5
    base_features = np.clip(base_features, 0, 1)
    
    for scenario in scenarios:
        context = scenario['context']
        expected = scenario['expected']
        dataset = scenario['dataset']
        
        # Get prediction
        result = explainer.explain(base_features, context)
        predicted = result.primary_mechanism
        
        is_correct = (predicted == expected)
        
        results_by_dataset[dataset]['total'] += 1
        if is_correct:
            results_by_dataset[dataset]['correct'] += 1
        else:
            results_by_dataset[dataset]['errors'].append({
                'scenario': scenario['name'],
                'expected': expected,
                'predicted': predicted,
            })
    
    # Print results
    print("\\n" + "="*70)
    print("RESULTS BY DATASET")
    print("="*70)
    
    total_correct = 0
    total = 0
    
    for dataset in ['NASA Ames', 'Panasonic 18650PF', 'Nature MATR', 'Randomized usage', 'HUST LFP']:
        r = results_by_dataset[dataset]
        acc = r['correct'] / r['total'] if r['total'] > 0 else 0
        total_correct += r['correct']
        total += r['total']
        
        print(f"\\n{dataset}:")
        print(f"  Accuracy: {r['correct']}/{r['total']} ({acc*100:.0f}%)")
        
        if r['errors']:
            print(f"  Errors:")
            for e in r['errors']:
                print(f"    {e['scenario']}: expected '{e['expected']}', got '{e['predicted']}'")
    
    overall_acc = total_correct / total if total > 0 else 0
    print("\\n" + "="*70)
    print(f"OVERALL: {total_correct}/{total} ({overall_acc*100:.1f}%)")
    print("="*70)
    
    # Compare with paper
    print("\\n" + "-"*70)
    print("COMPARISON WITH PAPER")
    print("-"*70)
    print(f"  Paper reported: 68/75 (90.7%)")
    print(f"  This run:       {total_correct}/75 ({overall_acc*100:.1f}%)")
    
    return results_by_dataset


def get_test_scenarios():
    """
    Generate the 75 test scenarios matching the paper's validation.
    
    Based on Table 5 in the paper:
    - NASA Ames: 15 scenarios
    - Panasonic 18650PF: 15 scenarios
    - Nature MATR: 15 scenarios
    - Randomized: 15 scenarios
    - HUST LFP: 15 scenarios
    """
    
    scenarios = []
    
    # NASA Ames scenarios
    # Mix of storage and cycling at various conditions
    nasa_scenarios = [
        # SEI Growth scenarios (high temp, high SOC, storage)
        {"name": "NASA_1", "temp": 0.75, "c_rate": 0.0, "d_rate": 0.0, "soc": 0.95, "mode": 1.0, "expected": "SEI Layer Growth"},
        {"name": "NASA_2", "temp": 0.625, "c_rate": 0.0, "d_rate": 0.0, "soc": 0.60, "mode": 1.0, "expected": "SEI Layer Growth"},
        {"name": "NASA_3", "temp": 0.5625, "c_rate": 0.1, "d_rate": 0.1, "soc": 0.80, "mode": 0.7, "expected": "SEI Layer Growth"},
        {"name": "NASA_4", "temp": 0.5625, "c_rate": 0.33, "d_rate": 0.33, "soc": 0.50, "mode": 0.0, "expected": "SEI Layer Growth"},
        # AM Loss scenarios (high C-rate)
        {"name": "NASA_5", "temp": 0.5625, "c_rate": 0.67, "d_rate": 0.67, "soc": 0.50, "mode": 0.0, "expected": "Active Material Loss"},
        {"name": "NASA_6", "temp": 0.625, "c_rate": 0.5, "d_rate": 0.75, "soc": 0.30, "mode": 0.0, "expected": "Active Material Loss"},
        {"name": "NASA_7", "temp": 0.5625, "c_rate": 0.33, "d_rate": 0.9, "soc": 0.50, "mode": 0.0, "expected": "Active Material Loss"},
        # Plating scenarios (cold + fast charge)
        {"name": "NASA_8", "temp": 0.25, "c_rate": 0.5, "d_rate": 0.25, "soc": 0.60, "mode": 0.0, "expected": "Lithium Plating"},
        {"name": "NASA_9", "temp": 0.125, "c_rate": 0.67, "d_rate": 0.25, "soc": 0.50, "mode": 0.0, "expected": "Lithium Plating"},
        # More mixed scenarios
        {"name": "NASA_10", "temp": 0.9, "c_rate": 0.33, "d_rate": 0.33, "soc": 0.70, "mode": 0.3, "expected": "SEI Layer Growth"},
        {"name": "NASA_11", "temp": 0.5, "c_rate": 0.0, "d_rate": 0.0, "soc": 0.10, "mode": 1.0, "expected": "Collector Corrosion"},
        {"name": "NASA_12", "temp": 0.525, "c_rate": 0.2, "d_rate": 0.2, "soc": 0.50, "mode": 0.2, "expected": "SEI Layer Growth"},
        {"name": "NASA_13", "temp": 0.5625, "c_rate": 0.33, "d_rate": 0.4, "soc": 0.55, "mode": 0.1, "expected": "SEI Layer Growth"},
        {"name": "NASA_14", "temp": 0.5625, "c_rate": 0.167, "d_rate": 0.15, "soc": 0.85, "mode": 0.6, "expected": "SEI Layer Growth"},
        {"name": "NASA_15", "temp": 0.6875, "c_rate": 0.8, "d_rate": 0.8, "soc": 0.40, "mode": 0.0, "expected": "Active Material Loss"},
    ]
    
    for s in nasa_scenarios:
        scenarios.append({
            "dataset": "NASA Ames",
            "name": s["name"],
            "context": np.array([s["temp"], s["c_rate"], s["d_rate"], s["soc"], 0.0, s["mode"]], dtype=np.float32),
            "expected": s["expected"],
        })
    
    # Panasonic 18650PF (EV drive cycles)
    panasonic_scenarios = [
        {"name": "Pan_1", "temp": 0.75, "c_rate": 0.5, "d_rate": 0.8, "soc": 0.50, "mode": 0.0, "expected": "Active Material Loss"},
        {"name": "Pan_2", "temp": 0.5625, "c_rate": 0.33, "d_rate": 0.5, "soc": 0.60, "mode": 0.0, "expected": "SEI Layer Growth"},
        {"name": "Pan_3", "temp": 0.3, "c_rate": 0.4, "d_rate": 0.6, "soc": 0.55, "mode": 0.0, "expected": "Lithium Plating"},
        {"name": "Pan_4", "temp": 0.525, "c_rate": 0.25, "d_rate": 0.4, "soc": 0.50, "mode": 0.0, "expected": "SEI Layer Growth"},
        {"name": "Pan_5", "temp": 0.8, "c_rate": 0.0, "d_rate": 0.0, "soc": 0.90, "mode": 1.0, "expected": "SEI Layer Growth"},
        {"name": "Pan_6", "temp": 0.2, "c_rate": 0.0, "d_rate": 0.0, "soc": 0.70, "mode": 1.0, "expected": "SEI Layer Growth"},
        {"name": "Pan_7", "temp": 0.625, "c_rate": 0.9, "d_rate": 0.3, "soc": 0.50, "mode": 0.0, "expected": "Active Material Loss"},
        {"name": "Pan_8", "temp": 0.25, "c_rate": 0.8, "d_rate": 0.3, "soc": 0.50, "mode": 0.0, "expected": "Lithium Plating"},
        {"name": "Pan_9", "temp": 0.5625, "c_rate": 0.2, "d_rate": 0.3, "soc": 0.65, "mode": 0.4, "expected": "SEI Layer Growth"},
        {"name": "Pan_10", "temp": 0.625, "c_rate": 0.5, "d_rate": 0.6, "soc": 0.45, "mode": 0.1, "expected": "Active Material Loss"},
        {"name": "Pan_11", "temp": 0.525, "c_rate": 0.167, "d_rate": 0.2, "soc": 0.80, "mode": 0.7, "expected": "SEI Layer Growth"},
        {"name": "Pan_12", "temp": 0.5625, "c_rate": 0.4, "d_rate": 0.7, "soc": 0.40, "mode": 0.0, "expected": "Active Material Loss"},
        {"name": "Pan_13", "temp": 0.6875, "c_rate": 0.6, "d_rate": 0.5, "soc": 0.55, "mode": 0.0, "expected": "Active Material Loss"},
        {"name": "Pan_14", "temp": 0.525, "c_rate": 0.2, "d_rate": 0.35, "soc": 0.60, "mode": 0.2, "expected": "SEI Layer Growth"},
        {"name": "Pan_15", "temp": 0.1, "c_rate": 0.5, "d_rate": 0.4, "soc": 0.50, "mode": 0.0, "expected": "Lithium Plating"},
    ]
    
    for s in panasonic_scenarios:
        scenarios.append({
            "dataset": "Panasonic 18650PF",
            "name": s["name"],
            "context": np.array([s["temp"], s["c_rate"], s["d_rate"], s["soc"], 0.5, s["mode"]], dtype=np.float32),
            "expected": s["expected"],
        })
    
    # Nature MATR (fast charging focus) - 100% accuracy in paper
    matr_scenarios = [
        {"name": "MATR_1", "temp": 0.5625, "c_rate": 0.33, "d_rate": 0.33, "soc": 0.50, "mode": 0.0, "expected": "SEI Layer Growth"},
        {"name": "MATR_2", "temp": 0.5625, "c_rate": 0.67, "d_rate": 0.33, "soc": 0.50, "mode": 0.0, "expected": "Active Material Loss"},
        {"name": "MATR_3", "temp": 0.625, "c_rate": 1.0, "d_rate": 0.5, "soc": 0.50, "mode": 0.0, "expected": "Active Material Loss"},
        {"name": "MATR_4", "temp": 0.6875, "c_rate": 1.0, "d_rate": 0.5, "soc": 0.50, "mode": 0.0, "expected": "Active Material Loss"},
        {"name": "MATR_5", "temp": 0.75, "c_rate": 1.0, "d_rate": 0.5, "soc": 0.50, "mode": 0.0, "expected": "Active Material Loss"},
        {"name": "MATR_6", "temp": 0.25, "c_rate": 1.0, "d_rate": 0.5, "soc": 0.50, "mode": 0.0, "expected": "Lithium Plating"},
        {"name": "MATR_7", "temp": 0.6875, "c_rate": 1.0, "d_rate": 0.5, "soc": 0.50, "mode": 0.0, "expected": "Active Material Loss"},
        {"name": "MATR_8", "temp": 0.8, "c_rate": 1.0, "d_rate": 0.5, "soc": 0.50, "mode": 0.0, "expected": "Active Material Loss"},
        {"name": "MATR_9", "temp": 0.5625, "c_rate": 0.5, "d_rate": 0.33, "soc": 0.50, "mode": 0.0, "expected": "Active Material Loss"},
        {"name": "MATR_10", "temp": 0.5625, "c_rate": 0.67, "d_rate": 0.33, "soc": 0.50, "mode": 0.0, "expected": "Active Material Loss"},
        {"name": "MATR_11", "temp": 0.5625, "c_rate": 0.8, "d_rate": 0.33, "soc": 0.50, "mode": 0.0, "expected": "Active Material Loss"},
        {"name": "MATR_12", "temp": 0.5625, "c_rate": 0.4, "d_rate": 0.33, "soc": 0.50, "mode": 0.0, "expected": "SEI Layer Growth"},
        {"name": "MATR_13", "temp": 0.5625, "c_rate": 0.45, "d_rate": 0.33, "soc": 0.50, "mode": 0.0, "expected": "Active Material Loss"},
        {"name": "MATR_14", "temp": 0.625, "c_rate": 0.5, "d_rate": 0.33, "soc": 0.50, "mode": 0.0, "expected": "Active Material Loss"},
        {"name": "MATR_15", "temp": 0.5625, "c_rate": 0.33, "d_rate": 0.33, "soc": 0.50, "mode": 0.0, "expected": "SEI Layer Growth"},
    ]
    
    for s in matr_scenarios:
        scenarios.append({
            "dataset": "Nature MATR",
            "name": s["name"],
            "context": np.array([s["temp"], s["c_rate"], s["d_rate"], s["soc"], 0.0, s["mode"]], dtype=np.float32),
            "expected": s["expected"],
        })
    
    # Randomized usage scenarios
    random_scenarios = [
        {"name": "Rand_1", "temp": 0.05, "c_rate": 0.9, "d_rate": 0.5, "soc": 0.50, "mode": 0.0, "expected": "Lithium Plating"},
        {"name": "Rand_2", "temp": 0.95, "c_rate": 0.0, "d_rate": 0.0, "soc": 0.95, "mode": 1.0, "expected": "SEI Layer Growth"},
        {"name": "Rand_3", "temp": 0.5, "c_rate": 0.0, "d_rate": 0.0, "soc": 0.05, "mode": 1.0, "expected": "Collector Corrosion"},
        {"name": "Rand_4", "temp": 0.6, "c_rate": 0.45, "d_rate": 0.6, "soc": 0.55, "mode": 0.3, "expected": "Active Material Loss"},
        {"name": "Rand_5", "temp": 0.4, "c_rate": 0.6, "d_rate": 0.4, "soc": 0.45, "mode": 0.2, "expected": "Active Material Loss"},
        {"name": "Rand_6", "temp": 0.55, "c_rate": 0.3, "d_rate": 0.5, "soc": 0.60, "mode": 0.4, "expected": "SEI Layer Growth"},
        {"name": "Rand_7", "temp": 0.5625, "c_rate": 0.33, "d_rate": 0.4, "soc": 0.50, "mode": 0.1, "expected": "SEI Layer Growth"},
        {"name": "Rand_8", "temp": 0.525, "c_rate": 0.25, "d_rate": 0.35, "soc": 0.55, "mode": 0.15, "expected": "SEI Layer Growth"},
        {"name": "Rand_9", "temp": 0.5625, "c_rate": 0.4, "d_rate": 0.45, "soc": 0.50, "mode": 0.1, "expected": "SEI Layer Growth"},
        {"name": "Rand_10", "temp": 0.7, "c_rate": 0.7, "d_rate": 0.8, "soc": 0.40, "mode": 0.0, "expected": "Active Material Loss"},
        {"name": "Rand_11", "temp": 0.65, "c_rate": 0.8, "d_rate": 0.7, "soc": 0.45, "mode": 0.0, "expected": "Active Material Loss"},
        {"name": "Rand_12", "temp": 0.75, "c_rate": 0.6, "d_rate": 0.9, "soc": 0.35, "mode": 0.0, "expected": "Active Material Loss"},
        {"name": "Rand_13", "temp": 0.5, "c_rate": 0.1, "d_rate": 0.15, "soc": 0.60, "mode": 0.5, "expected": "SEI Layer Growth"},
        {"name": "Rand_14", "temp": 0.525, "c_rate": 0.15, "d_rate": 0.2, "soc": 0.65, "mode": 0.6, "expected": "SEI Layer Growth"},
        {"name": "Rand_15", "temp": 0.55, "c_rate": 0.2, "d_rate": 0.25, "soc": 0.55, "mode": 0.4, "expected": "SEI Layer Growth"},
    ]
    
    for s in random_scenarios:
        scenarios.append({
            "dataset": "Randomized usage",
            "name": s["name"],
            "context": np.array([s["temp"], s["c_rate"], s["d_rate"], s["soc"], 0.5, s["mode"]], dtype=np.float32),
            "expected": s["expected"],
        })
    
    # HUST LFP scenarios
    hust_scenarios = [
        {"name": "HUST_1", "temp": 0.5625, "c_rate": 0.33, "d_rate": 0.33, "soc": 0.50, "mode": 0.0, "expected": "SEI Layer Growth"},
        {"name": "HUST_2", "temp": 0.5625, "c_rate": 0.67, "d_rate": 0.33, "soc": 0.50, "mode": 0.0, "expected": "Active Material Loss"},
        {"name": "HUST_3", "temp": 0.5625, "c_rate": 0.33, "d_rate": 0.5, "soc": 0.30, "mode": 0.0, "expected": "Active Material Loss"},
        {"name": "HUST_4", "temp": 0.25, "c_rate": 0.33, "d_rate": 0.33, "soc": 0.50, "mode": 0.0, "expected": "Lithium Plating"},
        {"name": "HUST_5", "temp": 0.6875, "c_rate": 0.33, "d_rate": 0.33, "soc": 0.50, "mode": 0.0, "expected": "SEI Layer Growth"},
        {"name": "HUST_6", "temp": 0.8, "c_rate": 0.33, "d_rate": 0.33, "soc": 0.50, "mode": 0.0, "expected": "SEI Layer Growth"},
        {"name": "HUST_7", "temp": 0.5625, "c_rate": 0.0, "d_rate": 0.0, "soc": 0.90, "mode": 1.0, "expected": "SEI Layer Growth"},
        {"name": "HUST_8", "temp": 0.5625, "c_rate": 0.0, "d_rate": 0.0, "soc": 0.50, "mode": 1.0, "expected": "SEI Layer Growth"},
        {"name": "HUST_9", "temp": 0.5625, "c_rate": 0.0, "d_rate": 0.0, "soc": 0.20, "mode": 1.0, "expected": "Collector Corrosion"},
        {"name": "HUST_10", "temp": 0.525, "c_rate": 0.2, "d_rate": 0.25, "soc": 0.60, "mode": 0.3, "expected": "SEI Layer Growth"},
        {"name": "HUST_11", "temp": 0.5625, "c_rate": 0.4, "d_rate": 0.45, "soc": 0.50, "mode": 0.1, "expected": "SEI Layer Growth"},
        {"name": "HUST_12", "temp": 0.625, "c_rate": 0.6, "d_rate": 0.7, "soc": 0.40, "mode": 0.0, "expected": "Active Material Loss"},
        {"name": "HUST_13", "temp": 0.55, "c_rate": 0.35, "d_rate": 0.4, "soc": 0.55, "mode": 0.25, "expected": "SEI Layer Growth"},
        {"name": "HUST_14", "temp": 0.6, "c_rate": 0.45, "d_rate": 0.5, "soc": 0.45, "mode": 0.15, "expected": "Active Material Loss"},
        {"name": "HUST_15", "temp": 0.5, "c_rate": 0.25, "d_rate": 0.35, "soc": 0.60, "mode": 0.35, "expected": "SEI Layer Growth"},
    ]
    
    for s in hust_scenarios:
        scenarios.append({
            "dataset": "HUST LFP",
            "name": s["name"],
            "context": np.array([s["temp"], s["c_rate"], s["d_rate"], s["soc"], 0.0, s["mode"]], dtype=np.float32),
            "expected": s["expected"],
        })
    
    return scenarios


def run_validation():
    """Run validation using Hybrid PINN model (retrained)."""
    
    print("="*70)
    print("PINN CAUSAL ATTRIBUTION VALIDATION (Hybrid Model)")
    print("="*70)
    
    # Create PINN model
    model = PINNCausalAttributionModel(feature_dim=9, context_dim=6)
    
    # Load retrained weights
    weight_path = "reports/pinn_causal/pinn_causal_retrained.pt"
    try:
        model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
        print(f"\\nLoaded weights from: {weight_path}")
    except Exception as e:
        print(f"\\nError loading weights: {e}")
        # Try finding absolute path
        abs_path = "/Users/sushanth.c/physics_informed_model/ML_prediction_SOH/" + weight_path
        try:
             model.load_state_dict(torch.load(abs_path, map_location=torch.device('cpu')))
             print(f"Loaded weights from: {abs_path}")
        except:
             print("Could not load weights. Running with initialized weights (will fail).")
    
    # Create explainer
    explainer = PINNCausalExplainer(model)
    
    print("\\nUsing Hybrid PINN mode: Neural Network + Physics Priors")
    
    # Get test scenarios
    scenarios = get_test_scenarios()
    print(f"\\nTotal scenarios: {len(scenarios)}")
    
    # Results tracking
    results_by_dataset = defaultdict(lambda: {'correct': 0, 'total': 0, 'errors': []})
    
    # Base features (consistent)
    np.random.seed(42)
    base_features = np.random.randn(9).astype(np.float32) * 0.1 + 0.5
    base_features = np.clip(base_features, 0, 1)
    
    for scenario in scenarios:
        context = scenario['context']
        expected = scenario['expected']
        dataset = scenario['dataset']
        
        # Get prediction
        # PINNCausalExplainer returns a DICT
        result = explainer.explain(base_features, context)
        predicted_id = result['primary_mechanism']
        
        # Convert ID to readable name to match expected strings
        predicted = DegradationMechanism.get_readable_name(predicted_id)
        
        is_correct = (predicted == expected)
        
        results_by_dataset[dataset]['total'] += 1
        if is_correct:
            results_by_dataset[dataset]['correct'] += 1
        else:
            results_by_dataset[dataset]['errors'].append({
                'scenario': scenario['name'],
                'expected': expected,
                'predicted': predicted,  # readable
                'id': predicted_id
            })
    
    # Print results
    print("\\n" + "="*70)
    print("RESULTS BY DATASET")
    print("="*70)
    
    total_correct = 0
    total = 0
    
    for dataset in ['NASA Ames', 'Panasonic 18650PF', 'Nature MATR', 'Randomized usage', 'HUST LFP']:
        r = results_by_dataset[dataset]
        acc = r['correct'] / r['total'] if r['total'] > 0 else 0
        total_correct += r['correct']
        total += r['total']
        
        print(f"\\n{dataset}:")
        print(f"  Accuracy: {r['correct']}/{r['total']} ({acc*100:.0f}%)")
        
        if r['errors']:
            print(f"  Errors:")
            for e in r['errors']:
                print(f"    {e['scenario']}: expected '{e['expected']}', got '{e['predicted']}'")
    
    overall_acc = total_correct / total if total > 0 else 0
    print("\\n" + "="*70)
    print(f"OVERALL: {total_correct}/{total} ({overall_acc*100:.1f}%)")
    print("="*70)
    
    # Compare with paper claim
    print("\\n" + "-"*70)
    print("COMPARISON")
    print("-"*70)
    print(f"  Physics-Only (Paper): 90.7% (68/75)")
    print(f"  Hybrid PINN (Claim):  92.0% (target)")
    print(f"  This run:             {overall_acc*100:.1f}% ({total_correct}/{total})")
    
    return results_by_dataset


if __name__ == '__main__':
    run_validation()
