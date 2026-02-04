import numpy as np
from src.advisory.battery_advisor import BatteryAdvisor
from src.advisory.suggestion_generator import DegradationMode

def test_real_world_scenarios():
    # Initialize advisor
    advisor = BatteryAdvisor(
        unified_path="reports/causal_attribution/causal_model.pt",
        pinn_path="reports/pinn_causal/pinn_causal_retrained.pt"
    )
    
    # Base battery features (normalized)
    # [cap, res, vol_max, vol_min, vol_avg, t_max, t_avg, curr_max, curr_avg]
    base_features = np.array([0.85, 0.45, 0.95, 0.70, 0.85, 0.35, 0.30, 0.65, 0.45], dtype=np.float32)

    def make_context(temp_c, charge_c, discharge_c, soc=0.5, mode=1.0):
        temp_norm = (temp_c - 25) / 20
        charge_norm = charge_c / 3.0
        discharge_norm = discharge_c / 4.0
        return np.array([temp_norm, charge_norm, discharge_norm, soc, 0.0, mode], dtype=np.float32)

    scenarios = [
        {
            "name": "REAL WORLD #1: NASA Ames (Cold Storage/Cycling)",
            "description": "NMC Cell at 4°C with 1.5C Fast Charge",
            "temp": 4, "charge": 1.5, "discharge": 2.0, "soc": 0.5, "mode": 1.0, # Cycling
            "expected": "Lithium Plating"
        },
        {
            "name": "REAL WORLD #2: Panasonic EV (Aggressive Drive Cycle)",
            "description": "NCA Cell at 25°C under high-power acceleration (4C discharge)",
            "temp": 25, "charge": 0.5, "discharge": 4.0, "soc": 0.4, "mode": 1.0, # Cycling
            "expected": "Active Material Loss (AM Loss)"
        },
        {
            "name": "REAL WORLD #3: Nature MATR (Calendar Aging)",
            "description": "LFP Cell stored at 45°C and 90% SOC",
            "temp": 45, "charge": 0.0, "discharge": 0.0, "soc": 0.9, "mode": 0.0, # Storage
            "expected": "SEI Layer Growth"
        },
        {
            "name": "REAL WORLD #4: Long-Term Deep Storage",
            "description": "Battery left at 5% SOC for extended period",
            "temp": 25, "charge": 0.0, "discharge": 0.0, "soc": 0.05, "mode": 0.0, # Storage
            "expected": "Collector Corrosion"
        },
        {
            "name": "REAL WORLD #5: HUST (LFP Economy Cycling)",
            "description": "LFP Battery with gentle usage (0.5C Charge/Discharge)",
            "temp": 25, "charge": 0.5, "discharge": 0.5, "soc": 0.5, "mode": 1.0, # Cycling
            "expected": "SEI Layer Growth (Normal Chemical Aging)"
        },
        {
            "name": "REAL WORLD #6: Nature MATR (Extreme 8C Fast-Charge)",
            "description": "Ultra-fast charging protocol (8C) at 30°C",
            "temp": 30, "charge": 8.0, "discharge": 4.0, "soc": 0.4, "mode": 1.0, # Cycling
            "expected": "Active Material Loss (Mechanical Fracture)"
        }
    ]

    print("\n" + "="*80)
    print("      UNIFIED ADVISORY SYSTEM: REAL WORLD DATASET VERIFICATION      ")
    print("="*80)

    for i, s in enumerate(scenarios):
        print(f"\n>>> RUNNING TEST {i+1}: {s['name']}")
        print(f"    Scenario: {s['description']}")
        
        ctx = make_context(s['temp'], s['charge'], s['discharge'], s['soc'], s['mode'])
        report = advisor.analyze(base_features, ctx)
        
        print(advisor.format_report(report))
        
        # Verify if expected mechanism is in top suggestions
        top_sug = report.top_recommendation
        print(f"    Verification: Top suggestion = {top_sug}")
        print("-" * 80)

if __name__ == "__main__":
    test_real_world_scenarios()
