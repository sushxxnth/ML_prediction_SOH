"""
Train Pure Collocation PINN on Full Dataset (~3.8k trajectories)

This script:
1. Loads ALL available battery trajectories from the unified pipeline
2. Assigns mechanism labels based on physics rules
3. Trains the Pure Collocation PINN (no hard-coded rules)
4. Tests on the 75 benchmark scenarios

Author: Battery ML Research
"""

import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.pure_collocation_pinn import PureCollocationPINN, PureCollocationLoss
from src.data.unified_pipeline import UnifiedDataPipeline
from test_unified_validation import (
    get_nasa_scenarios, get_panasonic_scenarios, get_nature_scenarios,
    get_randomized_scenarios, get_hust_scenarios, make_context, BASE_FEATURES
)


# Mechanism mapping
MECHANISM_MAP = {
    "SEI Layer Growth": "sei",
    "Lithium Plating": "plating",
    "Active Material Loss": "am_loss",
    "Electrolyte Decomposition": "electrolyte",
    "Collector Corrosion": "corrosion",
}
MECHANISM_IDX = {
    "sei": 0,
    "plating": 1,
    "am_loss": 2,
    "electrolyte": 3,
    "corrosion": 4,
}
IDX_TO_MECH = {v: k for k, v in MECHANISM_IDX.items()}


def assign_mechanism_from_context(context: np.ndarray) -> int:
    """
    Assign ground truth mechanism label based on operating conditions.
    
    Context: [temp_norm, charge_rate, discharge_rate, soc, profile, mode]
    """
    temp = context[0]  # Normalized: (T - 25) / 20
    charge = context[1]
    discharge = context[2]
    soc = context[3]
    mode = context[5]  # 1.0 = cycling, 0.0 = storage
    
    is_cycling = mode > 0.7
    is_storage = mode < 0.3
    
    # Denormalize temperature
    temp_celsius = temp * 20 + 25
    
    # Priority-based assignment (physics-driven)
    
    # 1. Lithium Plating: Cold + Fast charge
    if temp_celsius < 15 and is_cycling and charge > 0.3:
        return MECHANISM_IDX["plating"]
    
    # 2. Electrolyte Decomposition: Very high temperature
    if temp_celsius > 50:
        return MECHANISM_IDX["electrolyte"]
    
    # 3. Collector Corrosion: Low SOC storage
    if is_storage and soc < 0.25:
        return MECHANISM_IDX["corrosion"]
    
    # 4. Active Material Loss: High C-rate cycling
    if is_cycling and (discharge > 0.5 or charge > 0.5):
        return MECHANISM_IDX["am_loss"]
    
    # 5. SEI Growth: Default (calendar aging or gentle cycling)
    return MECHANISM_IDX["sei"]


class FullDataset(Dataset):
    """Dataset from unified pipeline."""
    
    def __init__(self, samples_list):
        self.samples = samples_list
        
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


def load_full_dataset(data_root: str = 'data'):
    """Load all available trajectories from the unified pipeline."""
    
    print("=" * 70)
    print("LOADING FULL DATASET (~3.8k trajectories)")
    print("=" * 70)
    
    samples = []
    source_counts = defaultdict(int)
    mechanism_counts = defaultdict(int)
    
    try:
        pipeline = UnifiedDataPipeline(data_root=data_root, use_lithium_features=False)
        
        # Load all available datasets
        available = ['nasa', 'calce', 'oxford', 'xjtu', 'panasonic', 'randomized', 'storage']
        for ds in available:
            try:
                pipeline.load_datasets([ds])
                print(f"  ✓ Loaded {ds}")
            except Exception as e:
                print(f"  ✗ {ds}: {e}")
        
        print(f"\n  Total raw samples: {len(pipeline.samples)}")
        
        # Convert to training samples
        for sample in pipeline.samples:
            # Skip invalid samples
            if not np.isfinite(sample.soh) or sample.soh < 0.5 or sample.soh > 1.1:
                continue
            if not np.all(np.isfinite(sample.features[:9])):
                continue
            
            # Features (9D)
            features = sample.features[:9].astype(np.float32)
            
            # Build context vector [temp, charge, discharge, soc, profile, mode]
            source = sample.source_dataset.lower()
            is_storage = 'storage' in source
            
            context = np.zeros(6, dtype=np.float32)
            
            if len(sample.context_vector) >= 1:
                context[0] = sample.context_vector[0]  # temp
            else:
                context[0] = 0.0  # 25°C default
                
            if len(sample.context_vector) >= 2:
                context[1] = sample.context_vector[1]  # charge rate
            else:
                context[1] = 0.33
                
            if len(sample.context_vector) >= 3:
                context[2] = sample.context_vector[2]  # discharge rate
            else:
                context[2] = 0.33
                
            if len(sample.context_vector) >= 4:
                context[3] = sample.context_vector[3]  # soc
            else:
                context[3] = 0.5
                
            context[4] = 0.0  # profile
            context[5] = 0.0 if is_storage else 1.0  # mode
            
            # Special handling for XJTU (high C-rate)
            if 'xjtu' in source:
                context[1] = 0.75  # ~2.5C
                context[2] = 0.75
            
            # Assign mechanism
            mechanism = assign_mechanism_from_context(context)
            
            # Capacity loss
            cap_loss = max(0.0, 1.0 - sample.soh)
            
            samples.append({
                'features': features,
                'context': context,
                'mechanism': mechanism,
                'capacity_loss': cap_loss,
                'source': source,
            })
            
            source_counts[source] += 1
            mechanism_counts[mechanism] += 1
            
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print(f"\n  Processed samples: {len(samples)}")
    print("\n  By source:")
    for src, count in sorted(source_counts.items()):
        print(f"    {src}: {count}")
    
    print("\n  By mechanism:")
    for idx in range(5):
        name = IDX_TO_MECH.get(idx, f"Unknown({idx})")
        count = mechanism_counts.get(idx, 0)
        pct = count / len(samples) * 100 if samples else 0
        print(f"    {name}: {count} ({pct:.1f}%)")
    
    return samples


def get_validation_scenarios():
    """Get the 75 benchmark validation scenarios."""
    scenarios = []
    
    for getter, dataset_name in [
        (get_nasa_scenarios, "NASA"),
        (get_panasonic_scenarios, "Panasonic"),
        (get_nature_scenarios, "Nature"),
        (get_randomized_scenarios, "Randomized"),
        (get_hust_scenarios, "HUST"),
    ]:
        for s in getter():
            context = make_context(
                s['temp'], s['charge'], s['discharge'],
                s.get('soc', 0.5), s.get('mode', 'cycling')
            )
            expected_name = s['expected']
            expected_key = MECHANISM_MAP.get(expected_name, expected_name.lower().replace(" ", "_"))
            # Handle naming differences
            if expected_key == "sei_growth":
                expected_key = "sei"
            if expected_key == "lithium_plating":
                expected_key = "plating"
            
            scenarios.append({
                'name': s['name'],
                'dataset': dataset_name,
                'context': context,
                'expected': expected_key,
                'expected_idx': MECHANISM_IDX.get(expected_key, 0),
            })
    
    return scenarios


def train_pure_pinn(epochs=100, batch_size=64, lr=0.001):
    """Train Pure Collocation PINN on full dataset."""
    
    print("\n" + "=" * 70)
    print("PURE COLLOCATION PINN TRAINING")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    print("\n[1/4] Loading full dataset...")
    train_samples = load_full_dataset('data')
    
    if len(train_samples) < 100:
        print("ERROR: Not enough training samples!")
        return None
    
    # Create dataset and dataloader
    train_dataset = FullDataset(train_samples)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    print(f"\n[2/4] Creating Pure Collocation PINN...")
    model = PureCollocationPINN(feature_dim=9, context_dim=6)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    loss_fn = PureCollocationLoss(lambda_physics=0.5, lambda_param_reg=0.1)
    
    # Training loop
    print(f"\n[3/4] Training for {epochs} epochs...")
    best_val_acc = 0
    history = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            features = batch['features']
            context = batch['context']
            target = batch['mechanism']
            cap_loss = batch['capacity_loss']
            
            optimizer.zero_grad()
            
            output = model(features, context)
            loss, losses_dict = loss_fn(output, target, cap_loss)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item() * features.shape[0]
            pred = output['mechanism_probs'].argmax(dim=1)
            correct += (pred == target).sum().item()
            total += features.shape[0]
        
        scheduler.step()
        
        train_acc = correct / total
        avg_loss = total_loss / total
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': avg_loss,
            'train_acc': train_acc,
        })
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: loss={avg_loss:.4f}, train_acc={train_acc*100:.1f}%")
    
    # Save model
    output_dir = Path('reports/pure_pinn')
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / 'pure_pinn_full.pt')
    print(f"\n  Model saved to: {output_dir / 'pure_pinn_full.pt'}")
    
    # Evaluate on 75 benchmark scenarios
    print(f"\n[4/4] Evaluating on 75 benchmark scenarios...")
    model.eval()
    
    scenarios = get_validation_scenarios()
    results_by_mech = defaultdict(lambda: {'correct': 0, 'total': 0})
    results_by_dataset = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    correct = 0
    with torch.no_grad():
        for s in scenarios:
            features = torch.FloatTensor(BASE_FEATURES).unsqueeze(0)
            context = torch.FloatTensor(s['context']).unsqueeze(0)
            
            output = model(features, context)
            pred = output['mechanism_probs'].argmax(dim=1).item()
            
            is_correct = (pred == s['expected_idx'])
            if is_correct:
                correct += 1
                results_by_mech[s['expected']]['correct'] += 1
            results_by_mech[s['expected']]['total'] += 1
            
            if is_correct:
                results_by_dataset[s['dataset']]['correct'] += 1
            results_by_dataset[s['dataset']]['total'] += 1
    
    val_acc = correct / len(scenarios)
    
    # Print results
    print("\n" + "=" * 70)
    print("PURE COLLOCATION PINN - FINAL RESULTS")
    print("=" * 70)
    
    print(f"\n  Overall Accuracy: {correct}/75 ({val_acc*100:.1f}%)")
    
    print("\n  By Dataset:")
    for ds in ['NASA', 'Panasonic', 'Nature', 'Randomized', 'HUST']:
        r = results_by_dataset.get(ds, {'correct': 0, 'total': 0})
        if r['total'] > 0:
            acc = r['correct'] / r['total'] * 100
            print(f"    {ds:15}: {r['correct']:2}/{r['total']:2} ({acc:.0f}%)")
    
    print("\n  By Mechanism:")
    for mech in ['sei', 'plating', 'am_loss', 'electrolyte', 'corrosion']:
        r = results_by_mech.get(mech, {'correct': 0, 'total': 0})
        if r['total'] > 0:
            acc = r['correct'] / r['total'] * 100
            print(f"    {mech:20}: {r['correct']:2}/{r['total']:2} ({acc:.0f}%)")
    
    # Get learned physics parameters
    with torch.no_grad():
        sample_out = model(
            torch.FloatTensor(BASE_FEATURES).unsqueeze(0),
            torch.zeros(1, 6)
        )
        params = sample_out['params']
        print("\n  Learned Physics Parameters:")
        print(f"    E_a_SEI: {params['E_a_sei'].mean().item()/1000:.1f} kJ/mol (lit: 35-60)")
        print(f"    beta (C-rate exp): {params['beta'].mean().item():.2f} (lit: ~1.5)")
        print(f"    gamma (cycle exp): {params['gamma'].mean().item():.2f} (lit: 0.3-1.0)")
    
    # Save results
    results = {
        'date': datetime.now().isoformat(),
        'epochs': epochs,
        'train_samples': len(train_samples),
        'val_accuracy': val_acc,
        'by_dataset': {k: dict(v) for k, v in results_by_dataset.items()},
        'by_mechanism': {k: dict(v) for k, v in results_by_mech.items()},
    }
    
    with open(output_dir / 'pure_pinn_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_dir / 'pure_pinn_results.json'}")
    print("=" * 70)
    
    return model, results


if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    train_pure_pinn(epochs=100, batch_size=64, lr=0.001)
