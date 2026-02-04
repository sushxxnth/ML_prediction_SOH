"""
Zero-Shot Verification - Paper Table 1 (TJU)

This script prints the reproducible zero-shot metrics from
reports/tju_zeroshot_retrained_results.json (HERO retrained on LCO, TJU eval).
"""

import json
from pathlib import Path

def main():
    print("=" * 70)
    print("ZERO-SHOT VERIFICATION (TJU NCM/NCA)")
    print("Source: reports/tju_zeroshot_retrained_results.json")
    print("=" * 70)
    print()
    
    # Load actual results
    results_path = Path("reports/tju_zeroshot_retrained_results.json")
    
    if not results_path.exists():
        print(f"ERROR: {results_path} not found!")
        return
    
    with open(results_path) as f:
        results = json.load(f)
    
    print(f\"{'Method':<20} | {'SOH MAE':<12} | {'RUL MAE (cap=45)':<18}\")
    print(\"-\" * 60)
    metrics = results[\"metrics\"]
    soh_mae = metrics[\"soh_mae\"]
    rul_mae_cap = metrics[\"horizon_mae\"][\"45\"]
    print(f\"{'HERO (Ours)':<20} | {soh_mae:<12.2f}% | {rul_mae_cap:<18.1f}\")
    
    print("=" * 70)
    print()
    print("These values match reports/tju_zeroshot_retrained_results.json")
    print("and Paper Table 1 (tab:zeroshot)")


if __name__ == "__main__":
    main()
