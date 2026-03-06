"""
Verify Causal-Directed Advisory System

This script tests if the advisory system dynamically changes its suggestions 
based on the diagnosed degradation mechanism from the PINN model.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.advisory.battery_advisor import BatteryAdvisor

def test_causal_advice():
    print("="*70)
    print("TESTING CAUSAL-DIRECTED ADVISORY SYSTEM")
    print("="*70)
    
    # Initialize advisor with both models
    advisor = BatteryAdvisor(
        unified_path="reports/causal_attribution/causal_model.pt",
        pinn_path="reports/pinn_causal/pinn_causal_retrained.pt"
    )
    
    if advisor.model is None or advisor.pinn_model is None:
        print(" Error: Could not load one or more models.")
        return

    # Standard "Healthy" features
    features = np.array([0.12, 0.25, 0.82, 0.35, 0.45, 0.08, 0.09, 0.06, 0.25], dtype=np.float32)

    # Scenario 1: Severe Lithium Plating (Cold + High Charge)
    print("\n--- Scenario 1: Cold (-10°C) + High Charge (1.5C) ---")
    # Context: [temp, charge, discharge, soc, profile, mode]
    # temp = (temp_c - 25) / 20  => (-10 - 25) / 20 = -1.75
    # charge = charge_c / 3.0 => 1.5 / 3.0 = 0.5
    context_plating = np.array([-1.75, 0.5, 0.2, 0.6, 0.0, 1.0], dtype=np.float32)
    
    report1 = advisor.analyze(features, context_plating)
    print(advisor.format_report(report1))
    
    has_plating_warning = any("Plating" in s.title for s in report1.suggestions)
    print(f"Causal Suggestion Triggered: {' YES' if has_plating_warning else ' NO'}")

    # Scenario 2: Active Material Loss (High Discharge)
    print("\n--- Scenario 2: Room Temp (25°C) + High Discharge (3C) ---")
    # temp = (25 - 25) / 20 = 0
    # discharge = 3 / 4.0 = 0.75
    context_am = np.array([0.0, 0.1, 0.75, 0.5, 0.0, 1.0], dtype=np.float32)
    
    report2 = advisor.analyze(features, context_am)
    print(advisor.format_report(report2))
    
    has_am_warning = any("Structural Aging" in s.title for s in report2.suggestions)
    print(f"Causal Suggestion Triggered: {' YES' if has_am_warning else ' NO'}")

    # Scenario 3: SEI Growth (High Temp Storage)
    print("\n--- Scenario 3: Hot (45°C) + High SOC Storage (90%) ---")
    # temp = (45 - 25) / 20 = 1.0
    # soc = 0.9
    # mode = 0 (storage)
    context_sei = np.array([1.0, 0.0, 0.0, 0.9, 1.0, 0.0], dtype=np.float32)
    
    report3 = advisor.analyze(features, context_sei)
    print(advisor.format_report(report3))
    
    has_sei_warning = any("Passivation Layer" in s.title for s in report3.suggestions)
    print(f"Causal Suggestion Triggered: {' YES' if has_sei_warning else ' NO'}")

    if has_plating_warning and has_am_warning and has_sei_warning:
        print("\n" + "="*70)
        print(" SUCCESS: Advisory system is dynamically driven by causal diagnostics!")
        print("="*70)
    else:
        print("\n" + "="*70)
        print(" WARNING: Some causal suggestions were not triggered as expected.")
        print("="*70)

if __name__ == "__main__":
    test_causal_advice()
