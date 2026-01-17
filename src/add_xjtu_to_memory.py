"""
Add XJTU samples to HERO memory bank with C-rate tags.

Extends the memory bank with high C-rate (2C, 3C) trajectories.
"""

import numpy as np
import scipy.io
import torch
import pickle
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.train.hero_rad_decoupled import RADDecoupledModel


def load_xjtu_samples():
    """Load XJTU samples for memory bank."""
    
    xjtu_dir = Path("data/new_datasets/XJTU/Battery Dataset")
    
    samples = []
    
    for batch_dir in sorted(xjtu_dir.iterdir()):
        if not batch_dir.is_dir():
            continue
            
        batch_name = batch_dir.name
        mat_files = list(batch_dir.glob("*.mat"))
        
        # Determine C-rate from batch name
        if "Batch-1" in batch_name:
            c_rate = 2.0
        elif "Batch-2" in batch_name:
            c_rate = 3.0
        else:
            c_rate = 2.5
        
        for mat_file in mat_files:
            try:
                data = scipy.io.loadmat(mat_file, simplify_cells=True)
                
                if 'summary' in data:
                    summary = data['summary']
                    
                    if isinstance(summary, dict) and 'discharge_capacity_Ah' in summary:
                        capacity = np.array(summary['discharge_capacity_Ah']).flatten()
                    else:
                        continue
                    
                    if len(capacity) == 0:
                        continue
                    
                    # Normalize capacity to SOH
                    initial_capacity = capacity[1] if len(capacity) > 1 and capacity[1] > capacity[0] else capacity[0]
                    if initial_capacity <= 0:
                        initial_capacity = 2.0
                    soh = capacity / initial_capacity
                    
                    # Filter valid SOH values
                    valid_mask = (soh > 0.5) & (soh <= 1.1) & ~np.isnan(soh)
                    soh = soh[valid_mask]
                    
                    if len(soh) < 10:
                        continue
                    
                    cell_name = mat_file.stem
                    
                    # Sample every 10th cycle
                    for i in range(0, len(soh), 10):
                        features = np.zeros(20, dtype=np.float32)
                        features[0] = soh[i]
                        features[1] = c_rate / 4.0
                        features[2] = i / len(soh)
                        
                        eol_threshold = 0.8
                        rul = 0
                        for future_i in range(i, len(soh)):
                            if soh[future_i] < eol_threshold:
                                rul = future_i - i
                                break
                        else:
                            rul = len(soh) - i
                        
                        rul_normalized = min(rul / 1000.0, 1.0)
                        
                        context = np.array([
                            25.0 / 60.0,
                            c_rate / 3.0,
                            c_rate / 4.0,
                            0.5,
                            0.0
                        ], dtype=np.float32)
                        
                        samples.append({
                            'cell_id': cell_name,
                            'batch': batch_name,
                            'c_rate': c_rate,
                            'cycle': i,
                            'features': features,
                            'context': context,
                            'soh': float(soh[i]),
                            'rul_normalized': rul_normalized,
                            'chem_id': 1,  # NCM
                            'source': f'XJTU_{c_rate}C'
                        })
                        
            except Exception as e:
                print(f"Error loading {mat_file.name}: {e}")
    
    return samples


def add_xjtu_to_memory_bank():
    """Add XJTU samples to HERO memory bank."""
    
    print("=" * 60)
    print("ADDING XJTU TO HERO MEMORY BANK (with C-rate tags)")
    print("=" * 60)
    
    # Load model with TJU memory
    model_path = Path("reports/hero_model/hero_model_with_tju_memory.pt")
    
    if not model_path.exists():
        # Fall back to fine-tuned model
        model_path = Path("reports/hero_model/hero_model_tju_finetuned.pt")
    
    if not model_path.exists():
        print("Error: No model found")
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
    print(f"✓ Loaded model from {model_path}")
    
    # Load existing TJU entries
    tju_entries_path = Path("data/memory_bank_tju_entries.pkl")
    if tju_entries_path.exists():
        with open(tju_entries_path, 'rb') as f:
            existing_entries = pickle.load(f)
        print(f"✓ Loaded {len(existing_entries)} existing TJU entries")
    else:
        existing_entries = []
    
    # Load XJTU samples
    samples = load_xjtu_samples()
    print(f"\nXJTU samples to add: {len(samples)}")
    
    # Group by C-rate
    by_crate = {}
    for s in samples:
        cr = s['c_rate']
        by_crate[cr] = by_crate.get(cr, 0) + 1
    print(f"By C-rate: {by_crate}")
    
    # Add samples to memory bank
    print("\nGenerating latent representations...")
    
    new_entries = []
    with torch.no_grad():
        for i, sample in enumerate(samples):
            features = torch.tensor(sample['features'], dtype=torch.float32).unsqueeze(0)
            context = torch.tensor(sample['context'], dtype=torch.float32).unsqueeze(0)
            chem_id = torch.tensor([sample['chem_id']], dtype=torch.long)
            
            features = torch.nan_to_num(features, nan=0.0)
            context = torch.nan_to_num(context, nan=0.0)
            
            # Get latent representation
            _, _, _, latent = model(features, context, chem_id, use_retrieval=False)
            
            new_entries.append({
                'latent': latent[0].detach().cpu(),
                'soh': sample['soh'],
                'rul': sample['rul_normalized'],
                'chem_id': sample['chem_id'],
                'source': sample['source'],
                'c_rate': sample['c_rate'],
                'cell_id': sample['cell_id'],
                'cycle': sample['cycle']
            })
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(samples)} samples...")
    
    # Combine with existing entries
    all_entries = existing_entries + new_entries
    
    # Save combined memory bank
    output_path = Path("data/memory_bank_combined_entries.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(all_entries, f)
    
    # Summary
    print("\n" + "=" * 60)
    print("MEMORY BANK UPDATED")
    print("=" * 60)
    
    # Count by source
    by_source = {}
    for e in all_entries:
        src = e.get('source', 'unknown')
        by_source[src] = by_source.get(src, 0) + 1
    
    print(f"\nTotal entries: {len(all_entries)}")
    print(f"By source: {by_source}")
    
    # Calculate new paper numbers
    original = 2816
    tju = len([e for e in all_entries if e.get('source', '').startswith('TJU')])
    xjtu = len([e for e in all_entries if e.get('source', '').startswith('XJTU')])
    
    print(f"\n=== UPDATED PAPER NUMBERS ===")
    print(f"Original trajectories: {original}")
    print(f"TJU trajectories: {tju}")
    print(f"XJTU trajectories: {xjtu}")
    print(f"TOTAL: {original + tju + xjtu:,}")
    
    # Save stats
    stats = {
        'total_entries': len(all_entries),
        'by_source': by_source,
        'paper_numbers': {
            'original': original,
            'tju': tju,
            'xjtu': xjtu,
            'total': original + tju + xjtu
        }
    }
    
    stats_path = Path("reports/hero_model/memory_bank_combined_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✓ Combined entries saved to {output_path}")
    print(f"✓ Stats saved to {stats_path}")
    
    return stats


if __name__ == '__main__':
    add_xjtu_to_memory_bank()
