"""
Boundary-Aware PINN Training Experiment

Hypothesis: Pure PINN can learn SEI/AM Loss boundary if given enough 
training data specifically at the decision boundary.

Approach:
1. Generate synthetic "boundary data" with explicit labels
2. Train pure PINN without any hard-coded rules
3. Test if it learns the boundary

Author: Battery ML Research
"""

import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from src.models.pure_collocation_pinn import PureCollocationPINN, PureCollocationLoss
from test_unified_validation import (
    get_nasa_scenarios, get_panasonic_scenarios, get_nature_scenarios,
    get_randomized_scenarios, get_hust_scenarios, make_context, BASE_FEATURES
)

MECHANISM_MAP = {
    "SEI Layer Growth": "sei",
    "Lithium Plating": "plating",
    "Active Material Loss": "am_loss",
    "Electrolyte Decomposition": "electrolyte",
    "Collector Corrosion": "corrosion",
}
MECHANISM_IDX = {"sei": 0, "plating": 1, "am_loss": 2, "electrolyte": 3, "corrosion": 4}
IDX_TO_MECH = {v: k for k, v in MECHANISM_IDX.items()}


def generate_boundary_data(n_samples=10000):
    """
    Generate synthetic data with EXPLICIT SEI/AM Loss boundary.
    
    The key insight:
    - C_rate < 0.5 → SEI dominates (gentle cycling)
    - C_rate > 0.7 → AM Loss dominates (high stress)
    - 0.5 < C_rate < 0.7 → BOUNDARY ZONE (augmented heavily)
    """
    
    print("=" * 70)
    print("GENERATING BOUNDARY-AWARE TRAINING DATA")
    print("=" * 70)
    
    samples = []
    mechanism_counts = defaultdict(int)
    
    # =========================================================================
    # 1. SEI Growth examples (low C-rate / storage)
    # =========================================================================
    n_sei = n_samples // 4
    print(f"\n  Generating {n_sei} SEI samples...")
    
    for _ in range(n_sei):
        # Storage mode
        if np.random.rand() < 0.5:
            temp = np.random.uniform(-0.5, 1.0)  # Various temps
            soc = np.random.uniform(0.3, 0.9)
            charge = 0.0
            discharge = 0.0
            mode = 0.0  # Storage
        # Gentle cycling
        else:
            temp = np.random.uniform(-0.3, 0.8)
            soc = np.random.uniform(0.2, 0.8)
            # KEY: Low C-rates → SEI
            charge = np.random.uniform(0.1, 0.4)
            discharge = np.random.uniform(0.1, 0.4)
            mode = 1.0  # Cycling
        
        context = np.array([temp, charge, discharge, soc, 0.0, mode], dtype=np.float32)
        features = np.random.randn(9).astype(np.float32) * 0.1
        features[0] = np.random.uniform(0.75, 0.95)
        
        samples.append({
            'features': features,
            'context': context,
            'mechanism': MECHANISM_IDX["sei"],
            'capacity_loss': np.random.uniform(0.02, 0.15),
        })
        mechanism_counts[MECHANISM_IDX["sei"]] += 1
    
    # =========================================================================
    # 2. AM Loss examples (high C-rate cycling)
    # =========================================================================
    n_am = n_samples // 4
    print(f"  Generating {n_am} AM Loss samples...")
    
    for _ in range(n_am):
        temp = np.random.uniform(-0.3, 1.0)
        soc = np.random.uniform(0.2, 0.8)
        # KEY: High C-rates → AM Loss
        charge = np.random.uniform(0.6, 1.0)
        discharge = np.random.uniform(0.6, 1.0)
        mode = 1.0  # Cycling
        
        context = np.array([temp, charge, discharge, soc, 0.0, mode], dtype=np.float32)
        features = np.random.randn(9).astype(np.float32) * 0.1
        features[0] = np.random.uniform(0.65, 0.9)
        
        samples.append({
            'features': features,
            'context': context,
            'mechanism': MECHANISM_IDX["am_loss"],
            'capacity_loss': np.random.uniform(0.05, 0.25),
        })
        mechanism_counts[MECHANISM_IDX["am_loss"]] += 1
    
    # =========================================================================
    # 3. BOUNDARY ZONE: The critical region between SEI and AM Loss
    # =========================================================================
    n_boundary = n_samples // 4
    print(f"  Generating {n_boundary} BOUNDARY samples (0.4 < C-rate < 0.7)...")
    
    for _ in range(n_boundary):
        temp = np.random.uniform(-0.3, 0.8)
        soc = np.random.uniform(0.2, 0.8)
        
        # BOUNDARY: C-rate between 0.4 and 0.7
        c_rate = np.random.uniform(0.4, 0.7)
        charge = c_rate + np.random.uniform(-0.1, 0.1)
        discharge = c_rate + np.random.uniform(-0.1, 0.1)
        charge = np.clip(charge, 0.3, 0.8)
        discharge = np.clip(discharge, 0.3, 0.8)
        mode = 1.0
        
        context = np.array([temp, charge, discharge, soc, 0.0, mode], dtype=np.float32)
        features = np.random.randn(9).astype(np.float32) * 0.1
        features[0] = np.random.uniform(0.7, 0.9)
        
        # DECISION RULE: charge + discharge < 1.0 → SEI, else AM Loss
        total_rate = charge + discharge
        if total_rate < 1.0:
            mechanism = MECHANISM_IDX["sei"]
        else:
            mechanism = MECHANISM_IDX["am_loss"]
        
        samples.append({
            'features': features,
            'context': context,
            'mechanism': mechanism,
            'capacity_loss': np.random.uniform(0.03, 0.18),
        })
        mechanism_counts[mechanism] += 1
    
    # =========================================================================
    # 4. Other mechanisms (plating, corrosion, electrolyte)
    # =========================================================================
    n_other = n_samples // 4
    n_each = n_other // 3
    
    # Plating: Cold + charge
    print(f"  Generating {n_each} Plating samples...")
    for _ in range(n_each):
        temp = np.random.uniform(-1.0, -0.3)
        charge = np.random.uniform(0.4, 1.0)
        discharge = np.random.uniform(0.3, 0.8)
        soc = np.random.uniform(0.3, 0.7)
        mode = 1.0
        
        context = np.array([temp, charge, discharge, soc, 0.0, mode], dtype=np.float32)
        features = np.random.randn(9).astype(np.float32) * 0.1
        features[0] = np.random.uniform(0.7, 0.9)
        
        samples.append({
            'features': features,
            'context': context,
            'mechanism': MECHANISM_IDX["plating"],
            'capacity_loss': np.random.uniform(0.05, 0.2),
        })
        mechanism_counts[MECHANISM_IDX["plating"]] += 1
    
    # Corrosion: Low SOC storage
    print(f"  Generating {n_each} Corrosion samples...")
    for _ in range(n_each):
        temp = np.random.uniform(-0.5, 0.5)
        soc = np.random.uniform(0.05, 0.2)
        mode = 0.0
        
        context = np.array([temp, 0.0, 0.0, soc, 0.0, mode], dtype=np.float32)
        features = np.random.randn(9).astype(np.float32) * 0.1
        features[0] = np.random.uniform(0.6, 0.85)
        
        samples.append({
            'features': features,
            'context': context,
            'mechanism': MECHANISM_IDX["corrosion"],
            'capacity_loss': np.random.uniform(0.1, 0.3),
        })
        mechanism_counts[MECHANISM_IDX["corrosion"]] += 1
    
    # Electrolyte: High temp
    print(f"  Generating {n_each} Electrolyte samples...")
    for _ in range(n_each):
        temp = np.random.uniform(1.2, 2.0)  # >50°C
        soc = np.random.uniform(0.3, 0.7)
        mode = np.random.choice([0.0, 1.0])
        
        context = np.array([temp, 0.3, 0.3, soc, 0.0, mode], dtype=np.float32)
        features = np.random.randn(9).astype(np.float32) * 0.1
        features[0] = np.random.uniform(0.65, 0.85)
        
        samples.append({
            'features': features,
            'context': context,
            'mechanism': MECHANISM_IDX["electrolyte"],
            'capacity_loss': np.random.uniform(0.1, 0.25),
        })
        mechanism_counts[MECHANISM_IDX["electrolyte"]] += 1
    
    print(f"\n  Total samples: {len(samples)}")
    print("\n  By mechanism:")
    for idx in range(5):
        name = IDX_TO_MECH[idx]
        count = mechanism_counts[idx]
        pct = count / len(samples) * 100
        print(f"    {name}: {count} ({pct:.1f}%)")
    
    return samples


class BoundaryDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'features': torch.FloatTensor(s['features']),
            'context': torch.FloatTensor(s['context']),
            'mechanism': torch.tensor(s['mechanism'], dtype=torch.long),
            'capacity_loss': torch.tensor([s['capacity_loss']], dtype=torch.float32),
        }


def get_test_scenarios():
    scenarios = []
    for getter, ds in [(get_nasa_scenarios, "NASA"), (get_panasonic_scenarios, "Panasonic"),
                       (get_nature_scenarios, "Nature"), (get_randomized_scenarios, "Randomized"),
                       (get_hust_scenarios, "HUST")]:
        for s in getter():
            context = make_context(s['temp'], s['charge'], s['discharge'], s.get('soc', 0.5), s.get('mode', 'cycling'))
            exp = MECHANISM_MAP.get(s['expected'], s['expected'].lower().replace(" ", "_"))
            if exp == "sei_growth": exp = "sei"
            if exp == "lithium_plating": exp = "plating"
            scenarios.append({
                'name': s['name'], 'dataset': ds, 'context': context,
                'expected': exp, 'expected_idx': MECHANISM_IDX.get(exp, 0),
            })
    return scenarios


def train_boundary_pinn(n_samples=10000, epochs=150, batch_size=128, lr=0.002):
    """Train pure PINN with boundary-aware data."""
    
    print("\n" + "=" * 70)
    print("BOUNDARY-AWARE PINN TRAINING")
    print("Hypothesis: Can learn SEI/AM Loss boundary without rules?")
    print("=" * 70)
    
    # Generate boundary-aware data
    samples = generate_boundary_data(n_samples)
    
    # Balanced sampling
    labels = [s['mechanism'] for s in samples]
    class_counts = defaultdict(int)
    for l in labels: class_counts[l] += 1
    weights = [1.0 / class_counts[l] for l in labels]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    
    dataset = BoundaryDataset(samples)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    
    # Create model
    model = PureCollocationPINN(feature_dim=9, context_dim=6)
    print(f"\n  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = PureCollocationLoss(lambda_physics=0.3, lambda_param_reg=0.1)
    
    print(f"\n  Training for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in loader:
            optimizer.zero_grad()
            output = model(batch['features'], batch['context'])
            loss, _ = loss_fn(output, batch['mechanism'], batch['capacity_loss'])
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item() * batch['features'].shape[0]
            pred = output['mechanism_probs'].argmax(dim=1)
            correct += (pred == batch['mechanism']).sum().item()
            total += batch['features'].shape[0]
        
        scheduler.step()
        
        if (epoch + 1) % 15 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: loss={total_loss/total:.4f}, acc={correct/total*100:.1f}%")
    
    # Evaluate on 75 benchmark
    print("\n  Evaluating on 75 benchmark scenarios...")
    model.eval()
    scenarios = get_test_scenarios()
    
    results_by_mech = defaultdict(lambda: {'correct': 0, 'total': 0})
    correct = 0
    errors = []
    
    with torch.no_grad():
        for s in scenarios:
            features = torch.FloatTensor(BASE_FEATURES).unsqueeze(0)
            context = torch.FloatTensor(s['context']).unsqueeze(0)
            output = model(features, context)
            pred = output['mechanism_probs'].argmax().item()
            
            if pred == s['expected_idx']:
                correct += 1
                results_by_mech[s['expected']]['correct'] += 1
            else:
                errors.append({
                    'name': s['name'],
                    'expected': s['expected'],
                    'predicted': IDX_TO_MECH[pred],
                    'context': s['context'].tolist(),
                })
            results_by_mech[s['expected']]['total'] += 1
    
    print("\n" + "=" * 70)
    print("BOUNDARY-AWARE PINN - FINAL RESULTS")
    print("=" * 70)
    print(f"\n  Overall: {correct}/75 ({correct/75*100:.1f}%)")
    
    print("\n  By Mechanism:")
    for mech in ['sei', 'plating', 'am_loss', 'electrolyte', 'corrosion']:
        r = results_by_mech.get(mech, {'correct': 0, 'total': 0})
        if r['total'] > 0:
            print(f"    {mech:15}: {r['correct']:2}/{r['total']:2} ({r['correct']/r['total']*100:.0f}%)")
    
    # Analyze errors at the boundary
    print("\n  Analyzing SEI vs AM Loss errors...")
    sei_am_errors = [e for e in errors if e['expected'] in ['sei', 'am_loss'] or e['predicted'] in ['sei', 'am_loss']]
    
    for e in sei_am_errors[:10]:
        c = e['context']
        total_rate = c[1] + c[2]
        print(f"    {e['name'][:30]:30}: {e['expected']} → {e['predicted']} | C_total={total_rate:.2f}")
    
    # Save
    output_dir = Path('reports/boundary_pinn')
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / 'boundary_pinn.pt')
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump({
            'accuracy': correct/75,
            'by_mechanism': {k: dict(v) for k, v in results_by_mech.items()},
            'errors': errors,
        }, f, indent=2)
    
    print(f"\n✓ Model saved to: {output_dir / 'boundary_pinn.pt'}")
    
    return model, correct/75


if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    # Test with different amounts of boundary data
    print("\n" + "=" * 70)
    print("EXPERIMENT: Does more boundary data help?")
    print("=" * 70)
    
    results = []
    for n in [5000, 10000, 20000]:
        print(f"\n\n>>> TRAINING WITH {n} SAMPLES <<<")
        _, acc = train_boundary_pinn(n_samples=n, epochs=150)
        results.append((n, acc))
    
    print("\n\n" + "=" * 70)
    print("EXPERIMENT RESULTS")
    print("=" * 70)
    for n, acc in results:
        print(f"  {n:6d} samples: {acc*100:.1f}%")
    
    print("\n  Hybrid PINN baseline: 90.7%")
    print("=" * 70)
