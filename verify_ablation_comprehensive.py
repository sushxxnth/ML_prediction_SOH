"""
Comprehensive ablation re-test across ALL available model weight files.

Tests each weight file with:
1. Full model (NN + priors)
2. NN-only (all priors zeroed)
3. Each individual prior removed
4. Priors-only (NN zeroed)

This will definitively show which weight file, if any, matches the paper's 92% claim.
"""

import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.models.pinn_causal_attribution import PINNCausalAttributionModel
from src.models.causal_attribution import DegradationMechanism
from test_unified_validation import (
    get_nasa_scenarios, get_panasonic_scenarios, get_nature_scenarios,
    get_randomized_scenarios, get_hust_scenarios, make_context, BASE_FEATURES
)

MECHANISM_NAMES = {
    0: "SEI Layer Growth",
    1: "Lithium Plating",
    2: "Active Material Loss",
    3: "Electrolyte Decomposition",
    4: "Collector Corrosion",
}

MECHANISM_LIST = [
    DegradationMechanism.SEI_GROWTH,
    DegradationMechanism.LITHIUM_PLATING,
    DegradationMechanism.ACTIVE_MATERIAL_LOSS,
    DegradationMechanism.ELECTROLYTE_DECOMP,
    DegradationMechanism.COLLECTOR_CORROSION,
]


def load_scenarios():
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


def compute_prior_score(mech, temp, charge, discharge, soc, mode):
    """Compute physics prior score for a single mechanism — mirrors model forward."""
    is_cycling = mode > 0.7
    is_storage = mode < 0.3
    score = torch.zeros_like(temp)

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

    return score.unsqueeze(-1)


def run_with_ablation(model, scenarios, ablate_mechanisms=None, zero_nn=False, zero_priors=False):
    """
    Run inference with surgical ablation.
    
    ablate_mechanisms: set of mechanism enums to zero out their prior
    zero_nn: if True, zero out all NN logits
    zero_priors: if True, zero out all prior logits
    """
    features = torch.from_numpy(BASE_FEATURES).unsqueeze(0)
    if ablate_mechanisms is None:
        ablate_mechanisms = set()

    per_ds = {ds: {"correct": 0, "total": 0} for ds in ["NASA", "Panasonic", "Nature", "Randomized", "HUST"]}
    errors = []

    with torch.no_grad():
        for scenario in scenarios:
            context = make_context(
                scenario['temp'], scenario['charge'],
                scenario['discharge'], scenario['soc'],
                scenario['mode']
            )
            context_t = torch.from_numpy(context).unsqueeze(0)

            # Step 1: Encode
            h_feat = model.feature_encoder(features)
            h_ctx = model.context_encoder(context_t)
            h_combined = torch.cat([h_feat, h_ctx], dim=-1)
            h_fused = model.fusion(h_combined)

            # Step 2: NN logits
            nn_logits = torch.cat(
                [model.mechanism_heads[m](h_fused) for m in model.mechanisms], dim=-1
            )
            if zero_nn:
                nn_logits = torch.zeros_like(nn_logits)

            # Step 3: Prior logits
            temp = context_t[:, 0]
            charge = context_t[:, 1]
            discharge = context_t[:, 2]
            soc_val = context_t[:, 3]
            mode_val = context_t[:, 5]

            prior_scores = []
            for mech in MECHANISM_LIST:
                if zero_priors or mech in ablate_mechanisms:
                    prior_scores.append(torch.zeros(1, 1))
                else:
                    prior_scores.append(
                        compute_prior_score(mech, temp, charge, discharge, soc_val, mode_val)
                    )
            prior_logits = torch.cat(prior_scores, dim=-1)

            # Step 4: Combine
            final_logits = prior_logits + nn_logits
            pred_idx = final_logits.argmax(dim=-1).item()
            predicted = MECHANISM_NAMES[pred_idx]

            is_correct = (predicted == scenario['expected'])
            ds = scenario['dataset']
            per_ds[ds]['total'] += 1
            if is_correct:
                per_ds[ds]['correct'] += 1
            else:
                errors.append(f"  {ds}/{scenario['name']}: expected={scenario['expected']}, got={predicted}")

    total_c = sum(r['correct'] for r in per_ds.values())
    total_n = sum(r['total'] for r in per_ds.values())
    acc = total_c / total_n
    return acc, per_ds, errors


def run_with_model_native(model, scenarios):
    """Run using the model's own forward() — as a sanity check."""
    features = torch.from_numpy(BASE_FEATURES).unsqueeze(0)
    per_ds = {ds: {"correct": 0, "total": 0} for ds in ["NASA", "Panasonic", "Nature", "Randomized", "HUST"]}

    with torch.no_grad():
        for scenario in scenarios:
            context = make_context(
                scenario['temp'], scenario['charge'],
                scenario['discharge'], scenario['soc'],
                scenario['mode']
            )
            context_t = torch.from_numpy(context).unsqueeze(0)
            outputs = model(features, context_t)
            attr_dict = outputs['attributions']
            attr_order = ['sei_growth', 'lithium_plating', 'am_loss', 'electrolyte', 'corrosion']
            attr_tensor = torch.cat([attr_dict[k] for k in attr_order], dim=-1)
            pred_idx = attr_tensor.argmax(dim=-1).item()
            predicted = MECHANISM_NAMES[pred_idx]

            ds = scenario['dataset']
            per_ds[ds]['total'] += 1
            if predicted == scenario['expected']:
                per_ds[ds]['correct'] += 1

    total_c = sum(r['correct'] for r in per_ds.values())
    total_n = sum(r['total'] for r in per_ds.values())
    return total_c / total_n, per_ds


def main():
    scenarios = load_scenarios()
    print(f"Loaded {len(scenarios)} benchmark scenarios.\n")

    weight_files = [
        "reports/pinn_causal/pinn_causal_retrained.pt",
        "reports/pinn_causal/pinn_causal_best.pt",
        "reports/pinn_causal/pinn_real_data_best.pt",
    ]

    for wf in weight_files:
        print("=" * 80)
        print(f"MODEL WEIGHTS: {wf}")
        print("=" * 80)

        model = PINNCausalAttributionModel(feature_dim=9, context_dim=6)
        try:
            model.load_state_dict(torch.load(wf, map_location='cpu', weights_only=True))
        except Exception as e:
            print(f"  ✗ Failed to load: {e}\n")
            continue
        model.eval()

        # Sanity check: native forward
        native_acc, native_ds = run_with_model_native(model, scenarios)
        print(f"\n  [SANITY CHECK] model.forward() accuracy: {native_acc*100:.1f}%")
        for ds in ["NASA", "Panasonic", "Nature", "Randomized", "HUST"]:
            r = native_ds[ds]
            print(f"    {ds:<15}: {r['correct']}/{r['total']}")

        # Ablation configs
        configs = [
            ("Full model (NN + all priors)", {}, False, False),
            ("NN only (all priors zeroed)", {}, False, True),
            ("Priors only (NN zeroed)", {}, True, False),
            ("Remove plating prior", {DegradationMechanism.LITHIUM_PLATING}, False, False),
            ("Remove AM loss prior", {DegradationMechanism.ACTIVE_MATERIAL_LOSS}, False, False),
            ("Remove SEI prior", {DegradationMechanism.SEI_GROWTH}, False, False),
            ("Remove corrosion prior", {DegradationMechanism.COLLECTOR_CORROSION}, False, False),
            ("Remove electrolyte prior", {DegradationMechanism.ELECTROLYTE_DECOMP}, False, False),
        ]

        print(f"\n  {'Configuration':<45} {'Acc':>8}  Per-dataset")
        print("  " + "-" * 78)

        for name, ablate, zero_nn, zero_priors in configs:
            acc, per_ds, errs = run_with_ablation(
                model, scenarios,
                ablate_mechanisms=ablate,
                zero_nn=zero_nn,
                zero_priors=zero_priors,
            )
            ds_str = "  ".join(
                f"{per_ds[d]['correct']}/{per_ds[d]['total']}"
                for d in ["NASA", "Panasonic", "Nature", "Randomized", "HUST"]
            )
            print(f"  {name:<45} {acc*100:>6.1f}%  [{ds_str}]")
            if errs:
                for e in errs:
                    print(f"      ↳ {e.strip()}")

        # Random baseline (10 trials averaged)
        rand_accs = []
        for seed in range(10):
            np.random.seed(seed)
            c = sum(1 for s in scenarios if MECHANISM_NAMES[np.random.randint(0,5)] == s['expected'])
            rand_accs.append(c / len(scenarios))
        print(f"  {'Random baseline (avg of 10)':<45} {np.mean(rand_accs)*100:>6.1f}%")

        print()

    # Also check paper's claimed per-dataset numbers
    print("=" * 80)
    print("PAPER'S CLAIMED RESULTS (for comparison):")
    print("  NASA=14/15, TJU=14/15, Nature=15/15, Random=13/15, HUST=13/15 → 69/75 = 92.0%")
    print("=" * 80)


if __name__ == '__main__':
    main()
