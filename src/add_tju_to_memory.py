"""
Add TJU samples to HERO memory bank with chemistry tags.

This populates the memory bank with TJU NCM/NCA trajectories for improved retrieval.
"""

import numpy as np
import torch
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.train.hero_rad_decoupled import RADDecoupledModel


def load_tju_samples():
    """Load TJU samples for memory bank."""
    
    data_path = Path("data/new_datasets/RUL-Mamba/data/TJU data/Dataset_3_NCM_NCA_battery_1C.npy")
    data = np.load(data_path, allow_pickle=True).item()
    
    samples = []
    
    for cell_name, df in data.items():
        capacity = df['Capacity'].values
        initial_capacity = capacity[0]
        soh = capacity / initial_capacity
        
        feature_cols = [
            'voltage mean', 'voltage std', 'voltage kurtosis', 'voltage skewness',
            'CC Q', 'CC charge time', 'voltage slope', 'voltage entropy',
            'current mean', 'current std', 'current kurtosis', 'current skewness',
            'CV Q', 'CV charge time', 'current slope', 'current entropy'
        ]
        
        # Sample every 10th cycle to avoid memory explosion
        for i in range(0, len(df), 10):
            features = np.zeros(20, dtype=np.float32)
            for j, col in enumerate(feature_cols):
                if col in df.columns:
                    val = df[col].iloc[i]
                    features[j] = float(val) if not np.isnan(val) else 0.0
            
            features = np.clip(features / (np.abs(features).max() + 1e-8), -1, 1)
            
            # Calculate RUL
            eol_threshold = 0.8
            rul = 0
            for future_i in range(i, len(df)):
                if soh[future_i] < eol_threshold:
                    rul = future_i - i
                    break
            else:
                rul = len(df) - i
            
            rul_normalized = min(rul / 1000.0, 1.0)
            
            context = np.array([25.0/60.0, 1.0/3.0, 1.0/4.0, 0.5, 0.0], dtype=np.float32)
            
            samples.append({
                'cell_id': cell_name,
                'cycle': i,
                'features': features,
                'context': context,
                'soh': float(soh[i]),
                'rul_normalized': rul_normalized,
                'chem_id': 1,  # NMC/NCA
                'source': 'TJU'
            })
    
    return samples


def add_tju_to_memory_bank():
    """Add TJU samples to HERO memory bank."""
    
    print("=" * 60)
    print("ADDING TJU TO HERO MEMORY BANK")
    print("=" * 60)
    
    # Load fine-tuned model
    model_path = Path("reports/hero_model/hero_model_tju_finetuned.pt")
    
    if not model_path.exists():
        print(f"Error: Fine-tuned model not found at {model_path}")
        print("Please run src/finetune_tju.py first")
        return
    
    model = RADDecoupledModel(
        feature_dim=20,
        context_dim=5,
        hidden_dim=128,
        latent_dim=64,
        n_chemistries=5,
        device='cpu'
    )
    
    model.load_state_dict(torch.load(model_path, weights_only=False, map_location='cpu'))
    model.eval()
    print(f"✓ Loaded fine-tuned model from {model_path}")
    
    # Get current memory bank size
    initial_size = model.memory_bank.size()
    print(f"\nInitial memory bank size: {initial_size}")
    
    # Load TJU samples
    samples = load_tju_samples()
    print(f"TJU samples to add: {len(samples)}")
    
    # Add samples to memory bank
    print("\nAdding samples to memory bank...")
    
    with torch.no_grad():
        for i, sample in enumerate(samples):
            features = torch.tensor(sample['features'], dtype=torch.float32).unsqueeze(0)
            context = torch.tensor(sample['context'], dtype=torch.float32).unsqueeze(0)
            chem_id = torch.tensor([sample['chem_id']], dtype=torch.long)
            
            features = torch.nan_to_num(features, nan=0.0)
            context = torch.nan_to_num(context, nan=0.0)
            
            # Get latent representation
            soh_pred, rul_pred, _, latent = model(features, context, chem_id, use_retrieval=False)
            
            # Add to memory bank with chemistry tag
            model.memory_bank.add(
                latent=latent[0],
                soh=sample['soh'],
                rul=sample['rul_normalized'],
                chem_id=sample['chem_id'],
                source=sample['source']
            )
            
            if (i + 1) % 50 == 0:
                print(f"  Added {i + 1}/{len(samples)} samples...")
    
    # Summary
    final_size = model.memory_bank.size()
    size_by_source = model.memory_bank.size_by_source()
    
    print("\n" + "=" * 60)
    print("MEMORY BANK UPDATED")
    print("=" * 60)
    print(f"\nFinal memory bank size: {final_size}")
    print(f"Added: {final_size - initial_size} TJU samples")
    print(f"\nBy source: {size_by_source}")
    
    # Save updated model with populated memory bank
    output_path = Path("reports/hero_model/hero_model_with_tju_memory.pt")
    torch.save(model.state_dict(), output_path)
    print(f"\n✓ Model with TJU memory saved to {output_path}")
    
    # Save memory bank stats
    stats = {
        'initial_size': initial_size,
        'final_size': final_size,
        'tju_samples_added': len(samples),
        'by_source': size_by_source,
        'chemistry_ids': {1: 'NMC/NCA (TJU)'}
    }
    
    stats_path = Path("reports/hero_model/memory_bank_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Memory bank stats saved to {stats_path}")
    
    return stats


if __name__ == '__main__':
    add_tju_to_memory_bank()
