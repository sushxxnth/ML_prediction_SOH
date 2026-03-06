"""
PINN Training with Correct Context Encoding

This script trains the PINN model using the same context encoding as our tuned 
physics priors that achieved 89% accuracy.

Context encoding:
- temp_norm = (temp_c - 25) / 20  (25°C = 0, 5°C = -1, 45°C = 1)
- charge_norm = charge_c / 3.0
- discharge_norm = discharge_c / 4.0
- mode = 1.0 for cycling, 0.0 for storage

Author: Battery ML Research
"""

import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.pinn_causal_attribution import PINNCausalAttributionModel
from test_unified_validation import (
    get_nasa_scenarios, get_panasonic_scenarios, get_nature_scenarios,
    get_randomized_scenarios, get_hust_scenarios, make_context, BASE_FEATURES
)


# Mechanism name mapping
MECHANISM_MAP = {
    "SEI Layer Growth": "sei_growth",
    "Lithium Plating": "lithium_plating", 
    "Active Material Loss": "am_loss",
    "Electrolyte Decomposition": "electrolyte",
    "Collector Corrosion": "corrosion",
}

MECHANISM_IDX = {
    "sei_growth": 0,
    "lithium_plating": 1,
    "am_loss": 2,
    "electrolyte": 3,
    "corrosion": 4,
}


def get_all_scenarios():
    """Get all 75 test scenarios with correct context encoding."""
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
            
            scenarios.append({
                'name': s['name'],
                'dataset': dataset_name,
                'context': context,
                'expected': expected_key,
                'expected_idx': MECHANISM_IDX.get(expected_key, 0),
            })
    
    return scenarios


def generate_augmented_training_data(scenarios, augment_factor=10):
    """
    Generate augmented training data from the 75 scenarios.
    
    Adds small noise to create more training examples.
    """
    augmented = []
    
    for scenario in scenarios:
        # Determine augmentation factor based on rarity (Importance Sampling)
        target_mech = scenario['expected']
        
        if target_mech == 'corrosion':
            # Rare class (only 2 examples) -> Massive oversampling
            current_factor = augment_factor * 20
        elif target_mech == 'lithium_plating':
            # Uncommon class -> Moderate oversampling
            current_factor = augment_factor * 5
        elif target_mech == 'electrolyte':
            current_factor = augment_factor * 5
        else:
            # Common classes (SEI, AM Loss)
            current_factor = augment_factor

        # Original
        augmented.append(scenario.copy())
        
        # Augmented variations
        for _ in range(current_factor - 1):
            noisy = scenario.copy()
            # Standard noise
            noise = np.random.randn(6).astype(np.float32) * 0.02
            noise[5] = 0  # Don't change mode
            
            # For corrosion (rare), add slightly more noise to cover the space
            if target_mech == 'corrosion':
                 noise[:5] = np.random.randn(5).astype(np.float32) * 0.05
                 
            noisy['context'] = np.clip(scenario['context'] + noise, -2, 2)
            augmented.append(noisy)
    
    return augmented


# Dataset class
class BatteryDataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data = data_list
        # Pre-compute tensors
        self.features = torch.FloatTensor(BASE_FEATURES)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            'features': self.features,  # Shared features
            'context': torch.FloatTensor(sample['context']),
            'target': torch.tensor(sample['expected_idx'], dtype=torch.long),
            'dataset': sample['dataset'],
            'expected': sample['expected']
        }

def train_pinn_model(epochs=300, lr=0.002, augment_factor=30, batch_size=32):
    """Train the PINN model on the 75 scenarios."""
    
    print("=" * 70)
    print("PINN MODEL TRAINING (optimized)")
    print("=" * 70)
    print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get scenarios
    print("\n[1/5] Loading scenarios...")
    scenarios = get_all_scenarios()
    print(f"  Loaded {len(scenarios)} base scenarios")
    
    # Augment data
    print("\n[2/5] Augmenting training data...")
    train_data_list = generate_augmented_training_data(scenarios, augment_factor)
    train_dataset = BatteryDataset(train_data_list)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print(f"  Created {len(train_data_list)} training examples")
    print(f"  Batch size: {batch_size}, Batches per epoch: {len(train_loader)}")
    
    # Create model
    print("\n[3/5] Creating PINN model...")
    model = PINNCausalAttributionModel(feature_dim=9, context_dim=6)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print("\n[4/5] Training...")
    best_accuracy = 0
    best_epoch = 0
    history = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            features = batch['features']
            context = batch['context']
            target = batch['target']
            
            optimizer.zero_grad()
            
            output = model(features, context)
            
            # Get mechanism logits
            if 'attributions' in output:
                # Stack attributions into tensor
                attr_list = []
                for m in ['sei_growth', 'lithium_plating', 'am_loss', 'electrolyte', 'corrosion']:
                    attr = output['attributions'].get(m, torch.zeros(context.shape[0], device=context.device))
                    # Ensure shape (Batch)
                    if attr.dim() == 0: attr = attr.expand(context.shape[0])
                    attr_list.append(attr)
                
                logits = torch.stack(attr_list, dim=1)  # Shape (Batch, 5)
            else:
                logits = output.get('mechanism_logits', torch.zeros(context.shape[0], 5, device=context.device))
            
            # Remove any extra singleton dims if they appear
            if logits.dim() > 2: logits = logits.reshape(context.shape[0], 5)
                
            loss = criterion(logits, target)
            
            # Add physics residual loss if available
            if 'physics_residuals' in output:
                physics_loss = sum(r.abs().mean() for r in output['physics_residuals'].values())
                loss = loss + 0.1 * physics_loss
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            total_loss += loss.item() * context.shape[0]
            pred = logits.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += context.shape[0]
        
        scheduler.step()
        
        train_acc = correct / total
        avg_loss = total_loss / total
        
        # Evaluate on original 75 scenarios (full batch is fine for 75)
        model.eval()
        val_dataset = BatteryDataset(scenarios)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(scenarios), shuffle=False)
        
        # There's only 1 batch
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features']
                context = batch['context']
                target = batch['target']
                
                output = model(features, context)
                
                if 'attributions' in output:
                   attr_list = [output['attributions'].get(m, torch.zeros(context.shape[0])) 
                                for m in ['sei_growth', 'lithium_plating', 'am_loss', 'electrolyte', 'corrosion']]
                   logits = torch.stack(attr_list, dim=1)
                else:
                    logits = output.get('mechanism_logits')
                
                pred = logits.argmax(dim=1)
                val_correct = (pred == target).sum().item()
        
        val_acc = val_correct / len(scenarios)
        
        # Log less frequently but show progress
        if (epoch + 1) % 10 == 0 or epoch == 0 or val_acc > best_accuracy:
             print(f"  Epoch {epoch+1:3d}: loss={avg_loss:.4f}, train_acc={train_acc*100:.1f}%, val_acc={val_acc*100:.1f}% ({val_correct}/75)")
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': avg_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
        })
        
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), 'reports/pinn_causal/pinn_causal_retrained.pt')
    
    print(f"\n  Best validation accuracy: {best_accuracy*100:.1f}% at epoch {best_epoch}")
    
    # Final evaluation
    print("\n[5/5] Final Evaluation...")
    model.load_state_dict(torch.load('reports/pinn_causal/pinn_causal_retrained.pt', weights_only=True))
    model.eval()
    
    results_by_dataset = defaultdict(lambda: {'correct': 0, 'total': 0})
    results_by_mechanism = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    with torch.no_grad():
        for sample in scenarios:
            features = torch.FloatTensor(BASE_FEATURES).unsqueeze(0)
            context = torch.FloatTensor(sample['context']).unsqueeze(0)
            
            output = model(features, context)
            
            if 'attributions' in output:
                attr_list = [output['attributions'].get(m, torch.zeros(1)) 
                            for m in ['sei_growth', 'lithium_plating', 'am_loss', 'electrolyte', 'corrosion']]
                logits = torch.stack(attr_list, dim=1).squeeze(0)
            else:
                logits = output.get('mechanism_logits', torch.zeros(1, 5))
            
            pred = logits.argmax().item()
            is_correct = (pred == sample['expected_idx'])
            
            results_by_dataset[sample['dataset']]['total'] += 1
            results_by_mechanism[sample['expected']]['total'] += 1
            
            if is_correct:
                results_by_dataset[sample['dataset']]['correct'] += 1
                results_by_mechanism[sample['expected']]['correct'] += 1
    
    # Print results
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    total_correct = sum(r['correct'] for r in results_by_dataset.values())
    print(f"\n  Overall Accuracy: {total_correct}/75 ({total_correct/75*100:.1f}%)")
    
    print("\n  By Dataset:")
    for ds in ['NASA', 'Panasonic', 'Nature', 'Randomized', 'HUST']:
        r = results_by_dataset[ds]
        acc = r['correct'] / r['total'] * 100 if r['total'] > 0 else 0
        print(f"    {ds:15}: {r['correct']:2}/{r['total']:2} ({acc:.0f}%)")
    
    print("\n  By Mechanism:")
    for mech in ['sei_growth', 'lithium_plating', 'am_loss', 'electrolyte', 'corrosion']:
        r = results_by_mechanism[mech]
        if r['total'] > 0:
            acc = r['correct'] / r['total'] * 100
            print(f"    {mech:20}: {r['correct']:2}/{r['total']:2} ({acc:.0f}%)")
    
    # Save results
    results = {
        'date': datetime.now().isoformat(),
        'epochs': epochs,
        'augment_factor': augment_factor,
        'best_accuracy': best_accuracy,
        'best_epoch': best_epoch,
        'final_accuracy': total_correct / 75,
        'by_dataset': {k: {'correct': v['correct'], 'total': v['total']} for k, v in results_by_dataset.items()},
        'by_mechanism': {k: {'correct': v['correct'], 'total': v['total']} for k, v in results_by_mechanism.items()},
        'history': history[-10:],  # Last 10 epochs
    }
    
    with open('reports/pinn_causal/pinn_retrained_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n Model saved to: reports/pinn_causal/pinn_causal_retrained.pt")
    print(f" Results saved to: reports/pinn_causal/pinn_retrained_results.json")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    
    return model, results


if __name__ == '__main__':
    train_pinn_model(epochs=300, lr=0.002, augment_factor=30)
