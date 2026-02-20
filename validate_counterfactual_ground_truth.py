#!/usr/bin/env python3
"""
Retrospective Ground-Truth Validation of Counterfactual Intervention Framework

Methodology (Natural Experiment / Observational Causal Inference):
=================================================================
NASA Ames tested identical 18650 Li-ion batteries under systematically varied
operating conditions. We exploit matched pairs as "natural experiments" to 
validate our counterfactual predictions against actual outcomes.

Experiment 1: Elevated Temperature → SEI Acceleration
------------------------------------------------------
  Factual:     B0029 at 43°C, 4A discharge, 2.0V cutoff (accelerated SEI growth)
  Counterfact: B0033 at 24°C, 4A discharge, 2.0V cutoff (baseline SEI rate)
  
  Intervention: "Cool from 43°C to 24°C" 
  Ground truth: B0033's actual degradation trajectory
  
  Expected: 43°C battery degrades faster due to Arrhenius-governed SEI kinetics.
  Our simulator should predict that cooling to 24°C reduces SEI contribution,
  matching B0033's actual (slower) trajectory.

Experiment 2: Current Reduction → Reduced Mechanical Stress
------------------------------------------------------------
  Factual:     B0033 at 24°C, 4A discharge, 2.0V cutoff (high mechanical stress)
  Counterfact: B0036 at 24°C, 2A discharge, 2.7V cutoff (reduced stress)
  
  Intervention: "Reduce discharge current from 4A to 2A"
  Ground truth: B0036's actual degradation trajectory

  Expected: higher current causes more active-material-loss from particle cracking.
  Simulator should predict improvement from current reduction.

Experiment 3: Combined (High Temp + High Current) → Baseline
-------------------------------------------------------------
  Factual:     B0029 at 43°C, 4A discharge  (worst case: hot + high current)
  Counterfact: B0005 at 24°C, 2A discharge  (best case: room temp + moderate current)
  
  This tests the combined counterfactual: "cool AND reduce current."

References:
  - Waldmann et al. (2014). Temperature dependent ageing mechanisms. J. Power Sources.
  - Birkl et al. (2017). Degradation diagnostics for lithium ion cells. J. Power Sources.
  - Arrhenius (1889). Über die Dissociationswärme.
"""

import numpy as np
import scipy.io as sio
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.optimization.counterfactual_intervention import (
    CounterfactualSimulator,
    InterventionOptimizer,
    BatteryState,
    CausalAttribution,
    Intervention,
)


# =============================================================================
# Data Loading
# =============================================================================

def load_capacity_trajectory(mat_path: str) -> dict:
    """
    Load cycle-by-cycle discharge capacity from a NASA battery .mat file.
    
    Returns dict with capacity trajectory and operating conditions.
    """
    mat = sio.loadmat(mat_path, simplify_cells=True)
    
    # The battery data is stored under a key like 'B0005'
    battery_key = None
    for key in mat.keys():
        if key.startswith('B0'):
            battery_key = key
            break
    
    if battery_key is None:
        # Fallback: use last non-dunder key
        for key in mat.keys():
            if not key.startswith('__'):
                battery_key = key
        
    battery_data = mat[battery_key]
    
    # Navigate to cycle data
    if isinstance(battery_data, np.ndarray) and battery_data.dtype.names:
        cycles = battery_data['cycle'][0, 0][0]
    elif isinstance(battery_data, dict):
        cycles = battery_data.get('cycle', [])
    else:
        cycles = battery_data
        
    capacities = []
    cycle_numbers = []
    temperatures = []
    currents = []
    
    cycle_idx = 0
    for cycle in cycles:
        try:
            # Handle structured numpy array
            if isinstance(cycle, np.void) or (isinstance(cycle, np.ndarray) and cycle.dtype.names):
                cycle_type = str(cycle['type']).strip()
                ambient_temp = float(cycle['ambient_temperature'])
                data = cycle['data']
            elif isinstance(cycle, dict):
                cycle_type = cycle.get('type', '')
                ambient_temp = cycle.get('ambient_temperature', 24.0)
                data = cycle.get('data', {})
            else:
                continue
            
            if cycle_type == 'discharge':
                # Extract capacity
                if isinstance(data, np.void) or (isinstance(data, np.ndarray) and data.dtype.names):
                    if 'Capacity' in data.dtype.names:
                        cap_data = data['Capacity']
                        if isinstance(cap_data, np.ndarray):
                            cap_data = cap_data.flatten()
                            if len(cap_data) > 0:
                                cap = float(cap_data[-1])
                            else:
                                continue
                        else:
                            cap = float(cap_data)
                    else:
                        continue
                elif isinstance(data, dict):
                    if 'Capacity' in data:
                        cap_data = data['Capacity']
                        if hasattr(cap_data, '__len__') and len(cap_data) > 0:
                            cap = float(cap_data[-1]) if hasattr(cap_data, '__len__') else float(cap_data)
                        else:
                            cap = float(cap_data)
                    else:
                        continue
                else:
                    continue
                
                # Filter anomalous values
                if 0.5 < cap < 2.5:
                    capacities.append(cap)
                    cycle_numbers.append(cycle_idx)
                    temperatures.append(float(ambient_temp))
                    
                    # Get current
                    try:
                        if isinstance(data, np.void) or (isinstance(data, np.ndarray) and data.dtype.names):
                            if 'Current_measured' in data.dtype.names:
                                curr = data['Current_measured'].flatten()
                                avg_curr = float(np.mean(np.abs(curr)))
                            else:
                                avg_curr = 2.0
                        elif isinstance(data, dict) and 'Current_measured' in data:
                            curr = data['Current_measured']
                            avg_curr = float(np.mean(np.abs(curr)))
                        else:
                            avg_curr = 2.0
                    except:
                        avg_curr = 2.0
                    
                    currents.append(avg_curr)
                    cycle_idx += 1
                    
        except (KeyError, TypeError, ValueError, IndexError):
            continue
    
    if len(capacities) == 0:
        raise ValueError(f"No valid capacity data found in {mat_path}")
    
    capacities = np.array(capacities)
    
    return {
        'cycles': np.array(cycle_numbers),
        'capacities': capacities,
        'temperatures': np.array(temperatures),
        'currents': np.array(currents),
        'initial_capacity': float(capacities[0]),
        'final_capacity': float(capacities[-1]),
        'total_cycles': len(capacities),
        'soh_trajectory': capacities / capacities[0],
        'capacity_fade_pct': (1 - capacities[-1] / capacities[0]) * 100,
    }


def compute_degradation_rate(capacities: np.ndarray) -> dict:
    """
    Compute degradation metrics from a capacity trajectory.
    
    Returns:
        dict with 'rate_pct_per_cycle', 'total_fade_pct', 'rate_Ah_per_cycle'
    """
    initial = capacities[0]
    final = capacities[-1]
    n_cycles = len(capacities)
    
    # Total fade 
    total_fade_pct = (1 - final / initial) * 100
    
    # Linear rate (slope from linear regression)
    x = np.arange(n_cycles)
    coeffs = np.polyfit(x, capacities, 1)
    slope_Ah = coeffs[0]  # Ah per cycle (negative = degradation)
    rate_pct_per_cycle = abs(slope_Ah / initial) * 100
    
    # Average rate
    avg_rate = total_fade_pct / max(n_cycles, 1)
    
    return {
        'rate_pct_per_cycle': rate_pct_per_cycle,
        'avg_rate_pct_per_cycle': avg_rate,
        'total_fade_pct': total_fade_pct,
        'rate_Ah_per_cycle': abs(slope_Ah),
        'n_cycles': n_cycles,
    }


# =============================================================================
# Ground-Truth Experiments
# =============================================================================

def run_experiment_1_temperature():
    """
    Experiment 1: Elevated Temperature → SEI Acceleration
    B0029 (43°C, 4A) vs B0033 (24°C, 4A)
    """
    data_dir = Path(__file__).parent / "data" / "nasa_set5" / "raw"
    
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: Temperature Effect on SEI Growth")
    print("  Factual:        B0029 at 43°C, 4A discharge, 2.0V cutoff")
    print("  Ground Truth CF: B0033 at 24°C, 4A discharge, 2.0V cutoff")
    print("  Intervention:   Cool from 43°C to 24°C")
    print("=" * 80)
    
    factual = load_capacity_trajectory(str(data_dir / "B0029.mat"))
    ground_truth = load_capacity_trajectory(str(data_dir / "B0033.mat"))
    
    fact_metrics = compute_degradation_rate(factual['capacities'])
    gt_metrics = compute_degradation_rate(ground_truth['capacities'])
    
    print(f"\n  📊 B0029 (43°C): {fact_metrics['rate_pct_per_cycle']:.4f} %SOH/cycle "
          f"({fact_metrics['total_fade_pct']:.1f}% total fade over {fact_metrics['n_cycles']} cycles)")
    print(f"  📊 B0033 (24°C): {gt_metrics['rate_pct_per_cycle']:.4f} %SOH/cycle "
          f"({gt_metrics['total_fade_pct']:.1f}% total fade over {gt_metrics['n_cycles']} cycles)")
    
    rate_ratio = fact_metrics['rate_pct_per_cycle'] / max(gt_metrics['rate_pct_per_cycle'], 1e-8)
    print(f"  📊 Actual rate ratio: {rate_ratio:.2f}x (43°C is {rate_ratio:.2f}x faster)")
    
    # Run counterfactual simulator
    simulator = CounterfactualSimulator()
    
    factual_state = BatteryState(
        soc=0.5, temperature=43.0, current=4.0, voltage=3.5,
        cycle_count=fact_metrics['n_cycles'] // 2, c_rate=2.0, capacity=2.0
    )
    
    # At 43°C: SEI dominant, some electrolyte decomposition
    factual_attr = CausalAttribution(
        sei_growth=0.55, lithium_plating=0.05, active_material_loss=0.20,
        electrolyte_loss=0.15, corrosion=0.05
    )
    
    intervention = Intervention(
        action_type="adjust_temperature", parameter="temperature",
        current_value=43.0, target_value=24.0,
        description="Cool from 43°C to 24°C"
    )
    
    cf_attr = simulator.simulate_counterfactual(factual_state, factual_attr, intervention)
    
    # Metrics
    sei_reduction = factual_attr.sei_growth - cf_attr.sei_growth
    direction_correct = (sei_reduction > 0) == (fact_metrics['rate_pct_per_cycle'] > gt_metrics['rate_pct_per_cycle'])
    
    # Predicted rate improvement based on mechanism redistribution
    total_reduction = sum([
        max(0, factual_attr.sei_growth - cf_attr.sei_growth),
        max(0, factual_attr.electrolyte_loss - cf_attr.electrolyte_loss),
    ])
    predicted_rate_ratio = 1 / max(1 - total_reduction, 0.1)
    ratio_error = abs(predicted_rate_ratio - rate_ratio) / rate_ratio * 100
    
    print(f"\n  🔮 Simulator Predictions (cool 43°C → 24°C):")
    print(f"     SEI:     {factual_attr.sei_growth*100:.1f}% → {cf_attr.sei_growth*100:.1f}% "
          f"(Δ = {sei_reduction*100:+.1f}%)")
    print(f"     Electrolyte: {factual_attr.electrolyte_loss*100:.1f}% → "
          f"{cf_attr.electrolyte_loss*100:.1f}%")
    print(f"     AM Loss: {factual_attr.active_material_loss*100:.1f}% → "
          f"{cf_attr.active_material_loss*100:.1f}%")
    
    print(f"\n  ✅ Validation:")
    print(f"     Direction: {'✓ CORRECT' if direction_correct else '✗ WRONG'} "
          f"(simulator predicts cooling reduces degradation: {sei_reduction > 0})")
    print(f"     Predicted rate improvement: {predicted_rate_ratio:.2f}x")  
    print(f"     Actual rate improvement:    {rate_ratio:.2f}x")
    print(f"     Ratio estimation error:     {ratio_error:.1f}%")
    
    return {
        'experiment': 'Temperature Effect on SEI Growth',
        'factual': 'B0029 (43°C, 4A, 2.0V)',
        'ground_truth': 'B0033 (24°C, 4A, 2.0V)',
        'intervention': 'Cool from 43°C to 24°C',
        'factual_rate': fact_metrics['rate_pct_per_cycle'],
        'gt_rate': gt_metrics['rate_pct_per_cycle'],
        'actual_rate_ratio': rate_ratio,
        'predicted_rate_ratio': predicted_rate_ratio,
        'direction_correct': direction_correct,
        'ratio_estimation_error_pct': ratio_error,
        'sei_reduction_pct': sei_reduction * 100,
        'factual_cycles': fact_metrics['n_cycles'],
        'gt_cycles': gt_metrics['n_cycles'],
        'factual_total_fade_pct': fact_metrics['total_fade_pct'],
        'gt_total_fade_pct': gt_metrics['total_fade_pct'],
    }


def run_experiment_2_current():
    """
    Experiment 2: Current Reduction → Reduced Mechanical Stress
    B0033 (24°C, 4A) vs B0036 (24°C, 2A)
    """
    data_dir = Path(__file__).parent / "data" / "nasa_set5" / "raw"
    
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Discharge Current Effect on Active Material Loss")
    print("  Factual:        B0033 at 24°C, 4A discharge, 2.0V cutoff")
    print("  Ground Truth CF: B0036 at 24°C, 2A discharge, 2.7V cutoff")
    print("  Intervention:   Reduce current from 4A to 2A")
    print("=" * 80)
    
    factual = load_capacity_trajectory(str(data_dir / "B0033.mat"))
    ground_truth = load_capacity_trajectory(str(data_dir / "B0036.mat"))
    
    fact_metrics = compute_degradation_rate(factual['capacities'])
    gt_metrics = compute_degradation_rate(ground_truth['capacities'])
    
    print(f"\n  📊 B0033 (4A): {fact_metrics['rate_pct_per_cycle']:.4f} %SOH/cycle "
          f"({fact_metrics['total_fade_pct']:.1f}% total fade over {fact_metrics['n_cycles']} cycles)")
    print(f"  📊 B0036 (2A): {gt_metrics['rate_pct_per_cycle']:.4f} %SOH/cycle "
          f"({gt_metrics['total_fade_pct']:.1f}% total fade over {gt_metrics['n_cycles']} cycles)")
    
    rate_ratio = fact_metrics['rate_pct_per_cycle'] / max(gt_metrics['rate_pct_per_cycle'], 1e-8)
    print(f"  📊 Actual rate ratio: {rate_ratio:.2f}x (4A is {rate_ratio:.2f}x faster)")
    
    # Run counterfactual simulator
    simulator = CounterfactualSimulator()
    
    factual_state = BatteryState(
        soc=0.5, temperature=24.0, current=4.0, voltage=3.5,
        cycle_count=fact_metrics['n_cycles'] // 2, c_rate=2.0, capacity=2.0
    )
    
    # At 4A / 2C: active material loss is dominant
    factual_attr = CausalAttribution(
        sei_growth=0.20, lithium_plating=0.05, active_material_loss=0.55,
        electrolyte_loss=0.10, corrosion=0.10
    )
    
    intervention = Intervention(
        action_type="reduce_current", parameter="current",
        current_value=4.0, target_value=2.0,
        description="Reduce discharge current from 4A to 2A"
    )
    
    cf_attr = simulator.simulate_counterfactual(factual_state, factual_attr, intervention)
    
    am_reduction = factual_attr.active_material_loss - cf_attr.active_material_loss
    direction_correct = (am_reduction > 0) == (fact_metrics['rate_pct_per_cycle'] > gt_metrics['rate_pct_per_cycle'])
    
    total_reduction = max(0, factual_attr.active_material_loss - cf_attr.active_material_loss)
    predicted_rate_ratio = 1 / max(1 - total_reduction, 0.1)
    ratio_error = abs(predicted_rate_ratio - rate_ratio) / max(rate_ratio, 1e-8) * 100
    
    print(f"\n  🔮 Simulator Predictions (reduce 4A → 2A):")
    print(f"     AM Loss: {factual_attr.active_material_loss*100:.1f}% → "
          f"{cf_attr.active_material_loss*100:.1f}% (Δ = {am_reduction*100:+.1f}%)")
    print(f"     SEI:     {factual_attr.sei_growth*100:.1f}% → {cf_attr.sei_growth*100:.1f}%")
    print(f"     Plating: {factual_attr.lithium_plating*100:.1f}% → "
          f"{cf_attr.lithium_plating*100:.1f}%")
    
    print(f"\n  ✅ Validation:")
    print(f"     Direction: {'✓ CORRECT' if direction_correct else '✗ WRONG'}")
    print(f"     Predicted rate improvement: {predicted_rate_ratio:.2f}x")
    print(f"     Actual rate improvement:    {rate_ratio:.2f}x")
    print(f"     Ratio estimation error:     {ratio_error:.1f}%")
    
    return {
        'experiment': 'Current Reduction Effect on AM Loss',
        'factual': 'B0033 (24°C, 4A, 2.0V)',
        'ground_truth': 'B0036 (24°C, 2A, 2.7V)',
        'intervention': 'Reduce current from 4A to 2A',
        'factual_rate': fact_metrics['rate_pct_per_cycle'],
        'gt_rate': gt_metrics['rate_pct_per_cycle'],
        'actual_rate_ratio': rate_ratio,
        'predicted_rate_ratio': predicted_rate_ratio,
        'direction_correct': direction_correct,
        'ratio_estimation_error_pct': ratio_error,
        'am_reduction_pct': am_reduction * 100,
        'factual_cycles': fact_metrics['n_cycles'],
        'gt_cycles': gt_metrics['n_cycles'],
        'factual_total_fade_pct': fact_metrics['total_fade_pct'],
        'gt_total_fade_pct': gt_metrics['total_fade_pct'],
    }


def run_experiment_3_combined():
    """
    Experiment 3: Combined (Hot + High Current) → Baseline (Room + Moderate)
    B0029 (43°C, 4A) vs B0005 (24°C, 2A)
    """
    data_dir = Path(__file__).parent / "data" / "nasa_set5" / "raw"
    
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: Combined Temperature + Current Effect")
    print("  Factual:        B0029 at 43°C, 4A discharge")
    print("  Ground Truth CF: B0005 at 24°C, 2A discharge")
    print("  Intervention:   Cool to 24°C AND reduce to 2A")
    print("=" * 80)
    
    factual = load_capacity_trajectory(str(data_dir / "B0029.mat"))
    ground_truth = load_capacity_trajectory(str(data_dir / "B0005.mat"))
    
    fact_metrics = compute_degradation_rate(factual['capacities'])
    gt_metrics = compute_degradation_rate(ground_truth['capacities'])
    
    print(f"\n  📊 B0029 (43°C/4A): {fact_metrics['rate_pct_per_cycle']:.4f} %SOH/cycle "
          f"({fact_metrics['total_fade_pct']:.1f}% fade over {fact_metrics['n_cycles']} cycles)")
    print(f"  📊 B0005 (24°C/2A): {gt_metrics['rate_pct_per_cycle']:.4f} %SOH/cycle "
          f"({gt_metrics['total_fade_pct']:.1f}% fade over {gt_metrics['n_cycles']} cycles)")
    
    rate_ratio = fact_metrics['rate_pct_per_cycle'] / max(gt_metrics['rate_pct_per_cycle'], 1e-8)
    print(f"  📊 Actual rate ratio: {rate_ratio:.2f}x (harsh is {rate_ratio:.2f}x faster)")
    
    # Run simulator with two sequential interventions
    simulator = CounterfactualSimulator()
    
    factual_state = BatteryState(
        soc=0.5, temperature=43.0, current=4.0, voltage=3.5,
        cycle_count=fact_metrics['n_cycles'] // 2, c_rate=2.0, capacity=2.0
    )
    
    # Hot + high current: mixed SEI + AM loss
    factual_attr = CausalAttribution(
        sei_growth=0.40, lithium_plating=0.05, active_material_loss=0.35,
        electrolyte_loss=0.15, corrosion=0.05
    )
    
    # First intervention: cool down
    int_cool = Intervention(
        action_type="adjust_temperature", parameter="temperature",
        current_value=43.0, target_value=24.0,
        description="Cool from 43°C to 24°C"
    )
    cf_attr_1 = simulator.simulate_counterfactual(factual_state, factual_attr, int_cool)
    
    # Second intervention: reduce current (applied to cooled state)
    cooled_state = BatteryState(
        soc=0.5, temperature=24.0, current=4.0, voltage=3.5,
        cycle_count=fact_metrics['n_cycles'] // 2, c_rate=2.0, capacity=2.0
    )
    int_reduce = Intervention(
        action_type="reduce_current", parameter="current",
        current_value=4.0, target_value=2.0,
        description="Reduce from 4A to 2A"
    )
    cf_attr_final = simulator.simulate_counterfactual(cooled_state, cf_attr_1, int_reduce)
    
    # Combined reduction
    total_sei_reduction = factual_attr.sei_growth - cf_attr_final.sei_growth
    total_am_reduction = factual_attr.active_material_loss - cf_attr_final.active_material_loss
    total_reduction = max(0, total_sei_reduction) + max(0, total_am_reduction)
    
    direction_correct = (total_reduction > 0) == (fact_metrics['rate_pct_per_cycle'] > gt_metrics['rate_pct_per_cycle'])
    predicted_rate_ratio = 1 / max(1 - total_reduction, 0.1)
    ratio_error = abs(predicted_rate_ratio - rate_ratio) / max(rate_ratio, 1e-8) * 100
    
    print(f"\n  🔮 Simulator Predictions (cool + reduce current):")
    print(f"     SEI:     {factual_attr.sei_growth*100:.1f}% → {cf_attr_final.sei_growth*100:.1f}% "
          f"(Δ = {total_sei_reduction*100:+.1f}%)")
    print(f"     AM Loss: {factual_attr.active_material_loss*100:.1f}% → "
          f"{cf_attr_final.active_material_loss*100:.1f}% (Δ = {total_am_reduction*100:+.1f}%)")
    print(f"     Combined mechanism reduction: {total_reduction*100:.1f}%")
    
    print(f"\n  ✅ Validation:")
    print(f"     Direction: {'✓ CORRECT' if direction_correct else '✗ WRONG'}")
    print(f"     Predicted rate improvement: {predicted_rate_ratio:.2f}x")
    print(f"     Actual rate improvement:    {rate_ratio:.2f}x")
    print(f"     Ratio estimation error:     {ratio_error:.1f}%")
    
    # SOH correlation
    n_overlap = min(len(factual['soh_trajectory']), len(ground_truth['soh_trajectory']))
    if n_overlap > 10:
        soh_corr = float(np.corrcoef(
            factual['soh_trajectory'][:n_overlap],
            ground_truth['soh_trajectory'][:n_overlap]
        )[0, 1])
    else:
        soh_corr = None
    
    print(f"     SOH trajectory correlation: {soh_corr:.3f}" if soh_corr else "")
    
    return {
        'experiment': 'Combined Temperature + Current Effect',
        'factual': 'B0029 (43°C, 4A)',
        'ground_truth': 'B0005 (24°C, 2A)',
        'intervention': 'Cool to 24°C + reduce to 2A',
        'factual_rate': fact_metrics['rate_pct_per_cycle'],
        'gt_rate': gt_metrics['rate_pct_per_cycle'],
        'actual_rate_ratio': rate_ratio,
        'predicted_rate_ratio': predicted_rate_ratio,
        'direction_correct': direction_correct,
        'ratio_estimation_error_pct': ratio_error,
        'combined_mechanism_reduction_pct': total_reduction * 100,
        'soh_correlation': soh_corr,
        'factual_cycles': fact_metrics['n_cycles'],
        'gt_cycles': gt_metrics['n_cycles'],
        'factual_total_fade_pct': fact_metrics['total_fade_pct'],
        'gt_total_fade_pct': gt_metrics['total_fade_pct'],
    }


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 80)
    print("RETROSPECTIVE GROUND-TRUTH VALIDATION")
    print("Counterfactual Intervention Framework")
    print("Using Matched NASA Battery Pairs as Natural Experiments")
    print("=" * 80)
    
    results = []
    
    try:
        r1 = run_experiment_1_temperature()
        results.append(r1)
    except Exception as e:
        print(f"  ⚠ Experiment 1 failed: {e}")
    
    try:
        r2 = run_experiment_2_current()
        results.append(r2)
    except Exception as e:
        print(f"  ⚠ Experiment 2 failed: {e}")
    
    try:
        r3 = run_experiment_3_combined()
        results.append(r3)
    except Exception as e:
        print(f"  ⚠ Experiment 3 failed: {e}")
    
    # Aggregate
    print(f"\n\n{'='*80}")
    print("AGGREGATE RESULTS")
    print(f"{'='*80}")
    
    if results:
        direction_acc = sum(r['direction_correct'] for r in results) / len(results) * 100
        avg_ratio_error = np.mean([r['ratio_estimation_error_pct'] for r in results])
        
        print(f"\n  Experiments completed:    {len(results)}")
        print(f"  Direction accuracy:       {direction_acc:.0f}% "
              f"({sum(r['direction_correct'] for r in results)}/{len(results)})")
        print(f"  Avg ratio estimation err: {avg_ratio_error:.1f}%")
        
        print(f"\n  {'Experiment':<45} {'Dir':>5} {'Pred':>8} {'Actual':>8} {'Err':>8}")
        print(f"  {'-'*74}")
        for r in results:
            dir_sym = '✓' if r['direction_correct'] else '✗'
            print(f"  {r['experiment']:<45} {dir_sym:>5} "
                  f"{r['predicted_rate_ratio']:>7.2f}x "
                  f"{r['actual_rate_ratio']:>7.2f}x "
                  f"{r['ratio_estimation_error_pct']:>7.1f}%")
        
        # Save
        output_dir = Path("reports")
        output_dir.mkdir(exist_ok=True)
        
        # Convert numpy types to native Python for JSON serialization
        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            elif isinstance(obj, (np.bool_,)):
                return bool(obj)
            elif isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            return obj
        
        save_data = convert_types({
            'experiment': 'Retrospective Ground-Truth Validation',
            'methodology': 'Observational causal inference using matched NASA battery pairs',
            'date': '2026-02-18',
            'pairs_tested': len(results),
            'aggregate_metrics': {
                'direction_accuracy_pct': direction_acc,
                'avg_ratio_estimation_error_pct': avg_ratio_error,
            },
            'detailed_results': results,
        })
        
        output_path = output_dir / "counterfactual_ground_truth_validation.json"
        with open(output_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"\n  Results saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    results = main()
