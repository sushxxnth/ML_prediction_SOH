"""
Ablation Study: Verify the 38.7% claim from the paper.

This script systematically ablates physics priors from the Hybrid PINN
and measures accuracy on the 75 benchmark scenarios.

Configurations tested:
1. Full Hybrid Model (all priors + NN) → expect 92.0%
2. Remove ALL priors (NN only) → expect ~38.7%
3. Remove cold-temperature (plating) prior only → measure
4. Remove high-discharge (AM loss) prior only → measure
5. Remove SEI prior only → measure
6. NN logits only (zero out priors) → same as #2
7. Random baseline → expect 20%
"""

import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).parent))

from src.models.pinn_causal_attribution import PINNCausalAttributionModel
from src.models.causal_attribution import DegradationMechanism
from test_unified_validation import (
    get_nasa_scenarios, get_panasonic_scenarios, get_nature_scenarios,
    get_randomized_scenarios, get_hust_scenarios, make_context, BASE_FEATURES
)

MECHANISM_MAP = {
    "SEI Layer Growth": 0,
    "Lithium Plating": 1,
    "Active Material Loss": 2,
    "Electrolyte Decomposition": 3,
    "Collector Corrosion": 4,
}

MECHANISM_NAMES = {v: k for k, v in MECHANISM_MAP.items()}

MECHANISM_LIST = [
    DegradationMechanism.SEI_GROWTH,
    DegradationMechanism.LITHIUM_PLATING,
    DegradationMechanism.ACTIVE_MATERIAL_LOSS,
    DegradationMechanism.ELECTROLYTE_DECOMP,
    DegradationMechanism.COLLECTOR_CORROSION,
]


def load_scenarios():
    """Load all 75 benchmark scenarios."""
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
    return all_scenarios


def run_inference(model, scenarios, prior_mask=None, use_nn=True, random_mode=False):
    """
    Run inference with optional prior ablation.
    
    Args:
        model: The Hybrid PINN model
        scenarios: List of scenario dicts
        prior_mask: Dict mapping mechanism name to bool (True=keep, False=ablate).
                    If None, all priors are kept.
        use_nn: If False, zero out NN logits (prior-only mode)
        random_mode: If True, return random predictions
    
    Returns:
        accuracy, per_dataset_results, misclassified_details
    """
    features = torch.from_numpy(BASE_FEATURES).unsqueeze(0)
    
    results = {ds: {"correct": 0, "total": 0} for ds in ["NASA", "Panasonic", "Nature", "Randomized", "HUST"]}
    misclassified = []
    
    with torch.no_grad():
        for scenario in scenarios:
            if random_mode:
                pred_idx = np.random.randint(0, 5)
                predicted = MECHANISM_NAMES[pred_idx]
            else:
                context = make_context(
                    scenario['temp'], scenario['charge'],
                    scenario['discharge'], scenario['soc'],
                    scenario['mode']
                )
                context_t = torch.from_numpy(context).unsqueeze(0)
                
                # We need to intercept the forward pass to ablate priors
                # Store original forward
                model_copy = model
                
                # Manually reproduce forward with ablation
                h_feat = model_copy.feature_encoder(features)
                h_ctx = model_copy.context_encoder(context_t)
                h_combined = torch.cat([h_feat, h_ctx], dim=-1)
                h_fused = model_copy.fusion(h_combined)
                
                # NN logits
                raw_scores = {}
                for mech in model_copy.mechanisms:
                    raw_scores[mech] = model_copy.mechanism_heads[mech](h_fused)
                nn_logits = torch.cat([raw_scores[m] for m in model_copy.mechanisms], dim=-1)
                
                if not use_nn:
                    nn_logits = torch.zeros_like(nn_logits)
                
                # Physics prior logits (reproduce compute_prior)
                temp = context_t[:, 0]
                charge = context_t[:, 1]
                discharge = context_t[:, 2]
                soc = context_t[:, 3]
                mode = context_t[:, 5]
                
                is_cycling = mode > 0.7
                is_storage = mode < 0.3
                
                prior_scores = []
                for mech in MECHANISM_LIST:
                    score = torch.zeros_like(temp)
                    
                    should_ablate = (prior_mask is not None and not prior_mask.get(mech, True))
                    
                    if should_ablate:
                        prior_scores.append(score.unsqueeze(-1))
                        continue
                    
                    if mech == DegradationMechanism.SEI_GROWTH:
                        score = torch.where(is_storage, (temp + 0.5) * 1.5, score)
                        score = torch.where(is_storage & (soc > 0.7), score * 1.3, score)
                        gentle = is_cycling & (charge < 1.0) & (discharge < 1.0)
                        score = torch.where(gentle, (temp + 0.5) * 2.0, score)
                        very_gentle = is_cycling & (charge < 0.5) & (discharge < 0.5)
                        score = torch.where(very_gentle, (temp + 0.5) * 2.5, score)
                    
                    elif mech == DegradationMechanism.LITHIUM_PLATING:
                        cold = temp <= -0.75
                        score = torch.where(is_cycling & cold, (0.5 - temp) * 5.0, score)
                    
                    elif mech == DegradationMechanism.ACTIVE_MATERIAL_LOSS:
                        high_discharge = discharge > 0.5
                        score = torch.where(is_cycling & high_discharge, discharge * 3.0, score)
                        stress = (discharge * 1.5 + charge) > 1.2
                        score = torch.where(is_cycling & stress, torch.max(score, (discharge + charge) * 2.0), score)
                        very_moderate = (charge < 0.5) & (discharge < 0.5)
                        score = torch.where(very_moderate, score * 0.3, score)
                    
                    elif mech == DegradationMechanism.COLLECTOR_CORROSION:
                        low_soc = is_storage & (soc <= 0.25)
                        score = torch.where(low_soc, (0.3 - soc) * 20.0, score)
                    
                    elif mech == DegradationMechanism.ELECTROLYTE_DECOMP:
                        hot = temp > 1.0
                        score = torch.where(hot, temp * 2.0, score)
                    
                    prior_scores.append(score.unsqueeze(-1))
                
                prior_logits = torch.cat(prior_scores, dim=-1)
                
                # Combine
                final_logits = prior_logits + nn_logits
                probs = F.softmax(final_logits, dim=-1)
                pred_idx = probs.argmax(dim=-1).item()
                predicted = MECHANISM_NAMES[pred_idx]
            
            is_correct = (predicted == scenario['expected'])
            dataset = scenario['dataset']
            results[dataset]['total'] += 1
            if is_correct:
                results[dataset]['correct'] += 1
            else:
                misclassified.append({
                    'dataset': dataset,
                    'name': scenario['name'],
                    'expected': scenario['expected'],
                    'predicted': predicted,
                })
    
    total_correct = sum(r['correct'] for r in results.values())
    total = sum(r['total'] for r in results.values())
    accuracy = total_correct / total if total > 0 else 0
    
    return accuracy, results, misclassified


def main():
    print("=" * 70)
    print("ABLATION STUDY: Verifying Paper Claims")
    print("=" * 70)
    
    # Load model
    model = PINNCausalAttributionModel(feature_dim=9, context_dim=6)
    weight_path = "reports/pinn_causal/pinn_causal_retrained.pt"
    model.load_state_dict(torch.load(weight_path, map_location='cpu', weights_only=True))
    model.eval()
    print(f" Loaded model from {weight_path}")
    
    # Load scenarios
    scenarios = load_scenarios()
    print(f" Loaded {len(scenarios)} scenarios\n")
    
    # Define ablation configurations
    configs = [
        {
            "name": "Full Hybrid Model (all priors + NN)",
            "prior_mask": None,  # All priors active
            "use_nn": True,
            "random": False,
        },
        {
            "name": "Remove ALL physics priors (NN only)",
            "prior_mask": {m: False for m in MECHANISM_LIST},
            "use_nn": True,
            "random": False,
        },
        {
            "name": "Remove cold-temperature (plating) prior only",
            "prior_mask": {
                DegradationMechanism.SEI_GROWTH: True,
                DegradationMechanism.LITHIUM_PLATING: False,  # ABLATED
                DegradationMechanism.ACTIVE_MATERIAL_LOSS: True,
                DegradationMechanism.ELECTROLYTE_DECOMP: True,
                DegradationMechanism.COLLECTOR_CORROSION: True,
            },
            "use_nn": True,
            "random": False,
        },
        {
            "name": "Remove high-discharge (AM loss) prior only",
            "prior_mask": {
                DegradationMechanism.SEI_GROWTH: True,
                DegradationMechanism.LITHIUM_PLATING: True,
                DegradationMechanism.ACTIVE_MATERIAL_LOSS: False,  # ABLATED
                DegradationMechanism.ELECTROLYTE_DECOMP: True,
                DegradationMechanism.COLLECTOR_CORROSION: True,
            },
            "use_nn": True,
            "random": False,
        },
        {
            "name": "Remove SEI prior only",
            "prior_mask": {
                DegradationMechanism.SEI_GROWTH: False,  # ABLATED
                DegradationMechanism.LITHIUM_PLATING: True,
                DegradationMechanism.ACTIVE_MATERIAL_LOSS: True,
                DegradationMechanism.ELECTROLYTE_DECOMP: True,
                DegradationMechanism.COLLECTOR_CORROSION: True,
            },
            "use_nn": True,
            "random": False,
        },
        {
            "name": "Remove corrosion prior only",
            "prior_mask": {
                DegradationMechanism.SEI_GROWTH: True,
                DegradationMechanism.LITHIUM_PLATING: True,
                DegradationMechanism.ACTIVE_MATERIAL_LOSS: True,
                DegradationMechanism.ELECTROLYTE_DECOMP: True,
                DegradationMechanism.COLLECTOR_CORROSION: False,  # ABLATED
            },
            "use_nn": True,
            "random": False,
        },
        {
            "name": "Remove electrolyte decomp prior only",
            "prior_mask": {
                DegradationMechanism.SEI_GROWTH: True,
                DegradationMechanism.LITHIUM_PLATING: True,
                DegradationMechanism.ACTIVE_MATERIAL_LOSS: True,
                DegradationMechanism.ELECTROLYTE_DECOMP: False,  # ABLATED
                DegradationMechanism.COLLECTOR_CORROSION: True,
            },
            "use_nn": True,
            "random": False,
        },
        {
            "name": "Random baseline",
            "prior_mask": None,
            "use_nn": True,
            "random": True,
        },
    ]
    
    # Run each configuration
    print(f"{'Configuration':<50} {'Accuracy':>10} {'Δ from Full':>12}")
    print("-" * 74)
    
    full_accuracy = None
    all_results = []
    
    for config in configs:
        acc, per_ds, misclassified = run_inference(
            model, scenarios,
            prior_mask=config["prior_mask"],
            use_nn=config["use_nn"],
            random_mode=config["random"],
        )
        
        if full_accuracy is None:
            full_accuracy = acc
            delta = "---"
        else:
            delta = f"{(acc - full_accuracy)*100:+.1f}%"
        
        print(f"  {config['name']:<48} {acc*100:>8.1f}%  {delta:>10}")
        all_results.append((config['name'], acc, per_ds, misclassified))
    
    # Print detailed breakdowns for interesting cases
    print("\n" + "=" * 70)
    print("DETAILED PER-DATASET BREAKDOWN")
    print("=" * 70)
    
    for name, acc, per_ds, misclassified in all_results:
        if "Random" in name:
            continue
        print(f"\n  {name} ({acc*100:.1f}%)")
        for ds in ["NASA", "Panasonic", "Nature", "Randomized", "HUST"]:
            r = per_ds[ds]
            ds_acc = r['correct']/r['total']*100 if r['total'] > 0 else 0
            print(f"    {ds:<15}: {r['correct']}/{r['total']} ({ds_acc:.1f}%)")
        
        if misclassified:
            print(f"    Errors ({len(misclassified)}):")
            for m in misclassified[:5]:
                print(f"      {m['dataset']}/{m['name']}: expected {m['expected']}, got {m['predicted']}")
            if len(misclassified) > 5:
                print(f"      ... and {len(misclassified) - 5} more")
    
    # Conclusion
    print("\n" + "=" * 70)
    nn_only_acc = all_results[1][1]
    print(f"KEY FINDING: Full model = {full_accuracy*100:.1f}%, NN-only = {nn_only_acc*100:.1f}%")
    print(f"Physics priors contribute: {(full_accuracy - nn_only_acc)*100:.1f} percentage points")
    
    # Check if single-prior ablations all collapse to same value
    single_ablation_accs = [r[1] for r in all_results[2:7]]
    all_same = len(set([round(a, 3) for a in single_ablation_accs])) == 1
    print(f"All single-prior ablations identical: {all_same}")
    if all_same:
        print(f"  → All collapse to {single_ablation_accs[0]*100:.1f}% (paper claims 38.7%)")
    else:
        print(f"  → Different values: {[f'{a*100:.1f}%' for a in single_ablation_accs]}")
    print("=" * 70)


if __name__ == '__main__':
    np.random.seed(42)
    main()
