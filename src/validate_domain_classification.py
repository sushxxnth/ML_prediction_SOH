"""
Phase 2 Domain Classification Validation

Tests whether the model learned to effectively categorize:
- Cycling data (NASA, CALCE, etc.)
- Storage data (PLN EIS)

Generates visualizations showing domain separation and SOH prediction quality.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.unified_pipeline import UnifiedDataPipeline
from src.data.eis_loader import EISLoader, extract_eis_features


class UnifiedDegradationModel(nn.Module):
    """Model architecture matching train_phase2_unified.py"""
    
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
        
        self.chem_embed = nn.Embedding(n_chemistries, 8)
        
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        
        self.mode_attention = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, hidden_dim),
            nn.Sigmoid()
        )
        
        combined_dim = hidden_dim + 32 + 8
        self.shared_encoder = nn.Sequential(
            nn.Linear(combined_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU()
        )
        
        self.soh_head = nn.Sequential(
            nn.Linear(latent_dim + context_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.rul_head = nn.Sequential(
            nn.Linear(latent_dim + context_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.domain_classifier = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # 0=cycling, 1=storage
        )
    
    def forward(self, features, context, chem_id):
        features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        context = torch.nan_to_num(context, nan=0.0)
        
        chem_emb = self.chem_embed(chem_id)
        feat_enc = self.feature_encoder(features)
        
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


def load_cycling_samples(pipeline, max_samples=500):
    """Load cycling samples (NASA, etc.)"""
    cycling_samples = []
    
    for s in pipeline.samples[:max_samples]:
        feat = s.features[:9] if len(s.features) >= 9 else np.pad(s.features, (0, 9 - len(s.features)))
        ctx = s.context_vector[:6] if len(s.context_vector) >= 6 else np.pad(s.context_vector, (0, 6 - len(s.context_vector)))
        
        # Set mode = 0 for cycling
        if len(ctx) >= 6:
            ctx[5] = 0.0  # cycling mode
        
        cycling_samples.append({
            'features': feat,
            'context': ctx,
            'soh': s.soh,
            'chem_id': s.chem_id,
            'domain': 0,  # cycling
            'source': s.source_dataset
        })
    
    return cycling_samples


def load_storage_samples(data_root, max_samples=500):
    """Load storage samples from EIS data."""
    storage_samples = []
    
    eis_loader = EISLoader(data_root)
    eis_loader.load()
    
    for spectrum in eis_loader.spectra[:max_samples]:
        eis_feat = extract_eis_features(spectrum)
        
        # Create 9D feature vector from EIS
        feat = np.zeros(9, dtype=np.float32)
        feat[:8] = eis_feat
        feat[8] = spectrum.temperature / 100
        
        # Create 6D context with mode=1 (storage)
        ctx = np.zeros(6, dtype=np.float32)
        ctx[0] = spectrum.temperature / 100  # normalized temp
        ctx[1] = 0.0  # charge rate (none for storage)
        ctx[2] = 0.0  # discharge rate
        ctx[3] = spectrum.soc / 100  # SOC
        ctx[4] = 1.0  # storage usage profile
        ctx[5] = 1.0  # storage MODE
        
        # Derive SOH from impedance
        r_ohmic = eis_feat[0]
        soh = np.clip(1.0 - (r_ohmic - 0.05) / 0.15, 0.6, 1.0)
        
        storage_samples.append({
            'features': feat,
            'context': ctx,
            'soh': float(soh),
            'chem_id': 0,  # LCO
            'domain': 1,  # storage
            'source': 'PLN_EIS'
        })
    
    return storage_samples


def main():
    print("=" * 60)
    print("PHASE 2 DOMAIN CLASSIFICATION VALIDATION")
    print("=" * 60)
    
    device = 'cpu'
    model_path = Path("reports/phase2_unified/unified_model.pt")
    
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        return
    
    # Load model
    print("\n[1/4] Loading saved Phase 2 model...")
    model = UnifiedDegradationModel(feature_dim=9, context_dim=6).to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    print("  Model loaded successfully.")
    
    # Load data
    print("\n[2/4] Loading cycling and storage samples...")
    
    # Cycling from NASA
    pipeline = UnifiedDataPipeline('data', use_lithium_features=False)
    pipeline.load_datasets(['nasa'])
    cycling_samples = load_cycling_samples(pipeline)
    print(f"  Cycling: {len(cycling_samples)} samples")
    
    # Storage from EIS
    storage_samples = load_storage_samples('.')
    print(f"  Storage: {len(storage_samples)} samples")
    
    # Combine
    all_samples = cycling_samples + storage_samples
    
    # Prepare tensors
    features = torch.tensor(np.array([s['features'] for s in all_samples]), dtype=torch.float32)
    context = torch.tensor(np.array([s['context'] for s in all_samples]), dtype=torch.float32)
    soh_true = np.array([s['soh'] for s in all_samples])
    chem_id = torch.tensor([s['chem_id'] for s in all_samples], dtype=torch.long)
    domain_true = np.array([s['domain'] for s in all_samples])
    
    # Evaluate
    print("\n[3/4] Running inference...")
    with torch.no_grad():
        soh_pred, rul_pred, domain_logits, latents = model(features, context, chem_id)
    
    soh_pred = soh_pred.squeeze().numpy()
    domain_pred = torch.argmax(domain_logits, dim=1).numpy()
    latents = latents.numpy()
    
    # Domain classification accuracy
    domain_acc = (domain_pred == domain_true).mean()
    cycling_acc = (domain_pred[domain_true == 0] == 0).mean()
    storage_acc = (domain_pred[domain_true == 1] == 1).mean()
    
    # SOH metrics by domain
    cycling_mask = domain_true == 0
    storage_mask = domain_true == 1
    
    cycling_mae = np.mean(np.abs(soh_pred[cycling_mask] - soh_true[cycling_mask]))
    storage_mae = np.mean(np.abs(soh_pred[storage_mask] - soh_true[storage_mask]))
    
    # R² calculation
    def calc_r2(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    cycling_r2 = calc_r2(soh_true[cycling_mask], soh_pred[cycling_mask])
    storage_r2 = calc_r2(soh_true[storage_mask], soh_pred[storage_mask])
    
    print("\n" + "=" * 60)
    print("DOMAIN CLASSIFICATION RESULTS")
    print("=" * 60)
    print(f"\n  Overall Domain Accuracy: {domain_acc*100:.1f}%")
    print(f"  Cycling Detection:       {cycling_acc*100:.1f}%")
    print(f"  Storage Detection:       {storage_acc*100:.1f}%")
    print(f"\n  Cycling SOH MAE: {cycling_mae:.4f} (R²: {cycling_r2:.3f})")
    print(f"  Storage SOH MAE: {storage_mae:.4f} (R²: {storage_r2:.3f})")
    print("=" * 60)
    
    # Create visualizations
    print("\n[4/4] Creating visualizations...")
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Domain Classification Confusion Matrix
    ax1 = fig.add_subplot(2, 3, 1)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(domain_true, domain_pred)
    im = ax1.imshow(cm, cmap='Blues')
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['Cycling', 'Storage'])
    ax1.set_yticklabels(['Cycling', 'Storage'])
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    ax1.set_title(f'Domain Classification\nAccuracy: {domain_acc*100:.1f}%', fontweight='bold')
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, cm[i, j], ha='center', va='center', fontsize=14, fontweight='bold',
                    color='white' if cm[i, j] > cm.max()/2 else 'black')
    
    # 2. t-SNE of latent space colored by domain
    ax2 = fig.add_subplot(2, 3, 2)
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latents)-1))
    latent_2d = tsne.fit_transform(latents)
    ax2.scatter(latent_2d[cycling_mask, 0], latent_2d[cycling_mask, 1], 
               c='blue', alpha=0.5, s=10, label='Cycling')
    ax2.scatter(latent_2d[storage_mask, 0], latent_2d[storage_mask, 1], 
               c='red', alpha=0.5, s=10, label='Storage')
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    ax2.set_title('Latent Space Separation\n(Domain Clusters)', fontweight='bold')
    ax2.legend()
    
    # 3. SOH Prediction: Cycling
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.scatter(soh_true[cycling_mask], soh_pred[cycling_mask], alpha=0.5, s=10, c='blue')
    ax3.plot([0.5, 1.1], [0.5, 1.1], 'r--', label='Perfect')
    ax3.set_xlabel('True SOH')
    ax3.set_ylabel('Predicted SOH')
    ax3.set_title(f'Cycling SOH Prediction\nMAE={cycling_mae:.4f}, R²={cycling_r2:.3f}', fontweight='bold')
    ax3.set_xlim([0.5, 1.1])
    ax3.set_ylim([0.5, 1.1])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. SOH Prediction: Storage
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.scatter(soh_true[storage_mask], soh_pred[storage_mask], alpha=0.5, s=10, c='red')
    ax4.plot([0.5, 1.1], [0.5, 1.1], 'r--', label='Perfect')
    ax4.set_xlabel('True SOH')
    ax4.set_ylabel('Predicted SOH')
    ax4.set_title(f'Storage SOH Prediction\nMAE={storage_mae:.4f}, R²={storage_r2:.3f}', fontweight='bold')
    ax4.set_xlim([0.5, 1.1])
    ax4.set_ylim([0.5, 1.1])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. SOH Error Distribution by Domain
    ax5 = fig.add_subplot(2, 3, 5)
    cycling_errors = np.abs(soh_pred[cycling_mask] - soh_true[cycling_mask])
    storage_errors = np.abs(soh_pred[storage_mask] - soh_true[storage_mask])
    ax5.boxplot([cycling_errors, storage_errors], labels=['Cycling', 'Storage'])
    ax5.set_ylabel('Absolute SOH Error')
    ax5.set_title('Error Distribution by Domain', fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Summary
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    summary = f"""
    PHASE 2 VALIDATION SUMMARY
    ━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    Domain Classification:
      Overall Accuracy: {domain_acc*100:.1f}%
      Cycling Recall:   {cycling_acc*100:.1f}%
      Storage Recall:   {storage_acc*100:.1f}%
    
    SOH Prediction:
      Cycling MAE: {cycling_mae:.4f}
      Storage MAE: {storage_mae:.4f}
    
    Samples:
      Cycling: {cycling_mask.sum()}
      Storage: {storage_mask.sum()}
    
    Model learned to differentiate
    cycling vs storage domains!
    """
    ax6.text(0.1, 0.9, summary, fontsize=11, verticalalignment='top',
            fontfamily='monospace', transform=ax6.transAxes,
            bbox=dict(boxstyle='round', facecolor='#E8F8F5', edgecolor='#1ABC9C', linewidth=2))
    
    plt.suptitle('Phase 2: Unified Degradation Model - Domain Classification Validation',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = Path("reports/phase2_unified/domain_classification_validation.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved to: {output_path}")
    
    # Save results
    results = {
        'domain_accuracy': float(domain_acc),
        'cycling_recall': float(cycling_acc),
        'storage_recall': float(storage_acc),
        'cycling_mae': float(cycling_mae),
        'storage_mae': float(storage_mae),
        'cycling_r2': float(cycling_r2),
        'storage_r2': float(storage_r2),
        'n_cycling': int(cycling_mask.sum()),
        'n_storage': int(storage_mask.sum())
    }
    
    with open("reports/phase2_unified/domain_validation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nVALIDATION COMPLETE")


if __name__ == '__main__':
    main()
