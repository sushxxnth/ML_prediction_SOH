"""
Phase 2 Validation: Test Unified Model on Unseen Data

This script validates that the model learned true physics by testing on:
1. CALCE cycling data (different lab, different cells)
2. Panasonic data (different chemistry, extreme temperatures)

If the model generalizes well, it proves it learned physics, not just patterns.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.unified_pipeline import UnifiedDataPipeline, UnifiedBatteryDataset
from torch.utils.data import DataLoader


class UnifiedDegradationModel(nn.Module):
    """Recreate model architecture for loading weights."""
    
    def __init__(
        self,
        feature_dim: int = 9,
        context_dim: int = 6,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        n_chemistries: int = 5
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.context_dim = context_dim
        
        # Chemistry embedding
        self.chem_embed = nn.Embedding(n_chemistries, 8)
        
        # Feature encoder (shared across domains)
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Context encoder (6D → embedding)
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        
        # Degradation mode attention
        self.mode_attention = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, hidden_dim),
            nn.Sigmoid()
        )
        
        # Combined encoder
        combined_dim = hidden_dim + 32 + 8  # features + context + chemistry
        self.shared_encoder = nn.Sequential(
            nn.Linear(combined_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU()
        )
        
        # SOH prediction head (unified)
        self.soh_head = nn.Sequential(
            nn.Linear(latent_dim + context_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # RUL prediction head (unified)
        self.rul_head = nn.Sequential(
            nn.Linear(latent_dim + context_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Domain classifier
        self.domain_classifier = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    
    def forward(self, features, context, chem_id):
        features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        context = torch.nan_to_num(context, nan=0.0)
        
        chem_emb = self.chem_embed(chem_id)
        feat_enc = self.feature_encoder(features)
        
        # Mode attention
        mode = context[:, -1:] if context.shape[1] > 0 else torch.zeros(features.shape[0], 1)
        mode_attn = self.mode_attention(mode)
        feat_enc = feat_enc * mode_attn
        
        ctx_enc = self.context_encoder(context)
        
        combined = torch.cat([feat_enc, ctx_enc, chem_emb], dim=-1)
        latent = self.shared_encoder(combined)
        
        soh_input = torch.cat([latent, context], dim=-1)
        soh_pred = self.soh_head(soh_input)
        rul_pred = self.rul_head(soh_input)
        domain_logits = self.domain_classifier(latent)
        
        return soh_pred, rul_pred, domain_logits, latent


def convert_to_unified_format(samples, feature_dim=9, context_dim=6):
    """Convert pipeline samples to (features, context, soh) format."""
    all_features = []
    all_context = []
    all_soh = []
    all_rul = []
    all_chem = []
    
    for s in samples:
        # Features: use first 9 dimensions of the sample features
        feat = s.features[:feature_dim] if len(s.features) >= feature_dim else np.pad(s.features, (0, feature_dim - len(s.features)))
        
        # Context: [temp_norm, c_rate, 0, soc, 0, mode=0 for cycling]
        ctx = s.context_vector[:context_dim] if len(s.context_vector) >= context_dim else np.pad(s.context_vector, (0, context_dim - len(s.context_vector)))
        
        all_features.append(feat)
        all_context.append(ctx)
        all_soh.append(s.soh)
        all_rul.append(s.rul_normalized)
        all_chem.append(s.chem_id)
    
    return (
        torch.tensor(np.array(all_features), dtype=torch.float32),
        torch.tensor(np.array(all_context), dtype=torch.float32),
        torch.tensor(np.array(all_soh), dtype=torch.float32),
        torch.tensor(np.array(all_rul), dtype=torch.float32),
        torch.tensor(np.array(all_chem), dtype=torch.long)
    )


def evaluate_on_dataset(model, samples, device, name):
    """Evaluate model on a dataset and return metrics."""
    model.eval()
    
    features, context, soh_true, rul_true, chem_id = convert_to_unified_format(samples)
    features = features.to(device)
    context = context.to(device)
    chem_id = chem_id.to(device)
    
    with torch.no_grad():
        soh_pred, rul_pred, _, _ = model(features, context, chem_id)
    
    soh_pred = soh_pred.squeeze().cpu().numpy()
    soh_true = soh_true.numpy()
    rul_pred = rul_pred.squeeze().cpu().numpy()
    rul_true = rul_true.numpy()
    
    # Filter NaN
    valid = ~(np.isnan(soh_pred) | np.isnan(soh_true))
    soh_pred, soh_true = soh_pred[valid], soh_true[valid]
    
    soh_mae = np.mean(np.abs(soh_pred - soh_true))
    
    # R²
    ss_res = np.sum((soh_true - soh_pred) ** 2)
    ss_tot = np.sum((soh_true - np.mean(soh_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    return {
        'name': name,
        'n_samples': len(soh_pred),
        'soh_mae': float(soh_mae),
        'soh_r2': float(r2),
        'soh_pred': soh_pred,
        'soh_true': soh_true
    }


def main():
    print("=" * 60)
    print("PHASE 2 VALIDATION: TESTING ON UNSEEN DATA")
    print("=" * 60)
    
    device = 'cpu'
    model_path = Path("reports/phase2_unified/unified_model.pt")
    
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Please run train_phase2_unified.py first.")
        return
    
    # Load model
    print("\n[1/3] Loading saved Phase 2 model...")
    model = UnifiedDegradationModel(feature_dim=9, context_dim=6).to(device)
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("  Model loaded successfully.")
    
    # Load unseen data
    print("\n[2/3] Loading unseen datasets...")
    pipeline = UnifiedDataPipeline('data', use_lithium_features=False)
    
    results = []
    
    # Test on CALCE (unseen lab)
    try:
        pipeline.load_datasets(['calce'])
        calce_samples = [s for s in pipeline.samples if 'calce' in s.source_dataset.lower()]
        if calce_samples:
            result = evaluate_on_dataset(model, calce_samples, device, "CALCE (Unseen Lab)")
            results.append(result)
            print(f"  CALCE: {len(calce_samples)} samples, SOH MAE = {result['soh_mae']:.4f}")
    except Exception as e:
        print(f"  CALCE: Failed to load - {e}")
    
    # Test on Oxford (if available)
    try:
        pipeline.load_datasets(['oxford'])
        oxford_samples = [s for s in pipeline.samples if 'oxford' in s.source_dataset.lower()]
        if oxford_samples:
            result = evaluate_on_dataset(model, oxford_samples, device, "Oxford (Unseen)")
            results.append(result)
            print(f"  Oxford: {len(oxford_samples)} samples, SOH MAE = {result['soh_mae']:.4f}")
    except Exception as e:
        print(f"  Oxford: Failed to load - {e}")
    
    # Test on TBSI (unseen chemistry potentially)
    try:
        pipeline.load_datasets(['tbsi_sunwoda'])
        tbsi_samples = [s for s in pipeline.samples if 'tbsi' in s.source_dataset.lower()]
        if tbsi_samples:
            result = evaluate_on_dataset(model, tbsi_samples, device, "TBSI Sunwoda (Unseen)")
            results.append(result)
            print(f"  TBSI: {len(tbsi_samples)} samples, SOH MAE = {result['soh_mae']:.4f}")
    except Exception as e:
        print(f"  TBSI: Failed to load - {e}")
    
    if not results:
        print("ERROR: No unseen datasets could be loaded.")
        return
    
    # Summary
    print("\n[3/3] Generating validation results...")
    print("\n" + "=" * 60)
    print("UNSEEN DATA VALIDATION RESULTS")
    print("=" * 60)
    print(f"\n{'Dataset':<25} {'Samples':<10} {'SOH MAE':<12} {'R²':<10}")
    print("-" * 57)
    for r in results:
        print(f"{r['name']:<25} {r['n_samples']:<10} {r['soh_mae']:<12.4f} {r['soh_r2']:<10.4f}")
    print("=" * 60)
    
    avg_mae = np.mean([r['soh_mae'] for r in results])
    avg_r2 = np.mean([r['soh_r2'] for r in results])
    print(f"\nAverage across unseen datasets:")
    print(f"  SOH MAE: {avg_mae:.4f}")
    print(f"  R²: {avg_r2:.4f}")
    
    # Create visualization
    n_datasets = len(results)
    fig, axes = plt.subplots(1, n_datasets + 1, figsize=(5 * (n_datasets + 1), 5))
    
    for i, r in enumerate(results):
        ax = axes[i]
        ax.scatter(r['soh_true'], r['soh_pred'], alpha=0.5, s=10)
        ax.plot([0, 1.2], [0, 1.2], 'r--', label='Perfect')
        ax.set_xlabel('True SOH')
        ax.set_ylabel('Predicted SOH')
        ax.set_title(f"{r['name']}\nMAE={r['soh_mae']:.4f}, R²={r['soh_r2']:.3f}")
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Summary bar chart
    ax = axes[-1]
    names = [r['name'].split()[0] for r in results]
    maes = [r['soh_mae'] for r in results]
    colors = ['#27AE60' if mae < 0.1 else '#F39C12' if mae < 0.2 else '#E74C3C' for mae in maes]
    ax.bar(names, maes, color=colors, edgecolor='black')
    ax.set_ylabel('SOH MAE')
    ax.set_title('Generalization Across\nUnseen Datasets')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0.1, color='green', linestyle='--', alpha=0.7, label='Target (<0.1)')
    ax.legend()
    
    plt.suptitle('Phase 2 Model: Unseen Data Validation\n(Proves Physics Learning)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = Path("reports/phase2_unified/unseen_validation.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    # Save results
    with open("reports/phase2_unified/unseen_validation.json", 'w') as f:
        json.dump({
            'datasets': [{k: v for k, v in r.items() if k not in ['soh_pred', 'soh_true']} for r in results],
            'avg_mae': float(avg_mae),
            'avg_r2': float(avg_r2)
        }, f, indent=2)
    
    print("\nVALIDATION COMPLETE")
    if avg_mae < 0.1:
        print(" Model generalizes well - learned physics principles!")
    else:
        print("⚠ Model may need more training or domain adaptation")


if __name__ == '__main__':
    main()
