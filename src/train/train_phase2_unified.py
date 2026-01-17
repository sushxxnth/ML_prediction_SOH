"""
Phase 2: Unified Degradation Model

A single model for BOTH cycling and storage (calendar) aging.
Novel Contribution #2: First unified cycling + calendar aging model.

Key Features:
- 6D context: [Temp, ChargeRate, DischargeRate, SOC, UsageProfile, DegradationMode]
- Shared encoder learns common degradation patterns
- Cross-domain transfer: cycling↔storage
- Joint training on NASA (cycling) + PLN (storage)

Author: Battery ML Research
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import matplotlib.pyplot as plt

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.unified_pipeline import UnifiedDataPipeline
from src.data.eis_impedance_loader import EISImpedanceLoader


# =============================================================================
# Unified Sample Format
# =============================================================================

@dataclass
class UnifiedDegradationSample:
    """
    Unified sample format that works for both cycling and storage.
    
    Key innovation: degradation_mode explicitly encodes cycling vs storage.
    """
    cell_id: str
    sample_idx: int
    source: str  # 'cycling' or 'storage'
    
    # Features (normalized)
    features: np.ndarray  # shape: (feature_dim,)
    
    # Labels
    soh: float           # 0-1
    rul_normalized: float  # 0-1 (fraction of life remaining)
    
    # 6D Context Vector
    # [temp_norm, charge_rate, discharge_rate, soc_norm, usage_profile, degradation_mode]
    context: np.ndarray  # shape: (6,)
    
    # Metadata
    temperature: float   # Original temperature in °C
    soc: float          # Original SOC in %
    chemistry_id: int    # 0=LCO, 1=NMC, etc.


class UnifiedDegradationDataset(Dataset):
    """PyTorch Dataset for unified degradation samples."""
    
    def __init__(self, samples: List[UnifiedDegradationSample]):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'features': torch.tensor(s.features, dtype=torch.float32),
            'context': torch.tensor(s.context, dtype=torch.float32),
            'soh': torch.tensor(s.soh, dtype=torch.float32),
            'rul': torch.tensor(s.rul_normalized, dtype=torch.float32),
            'chem_id': torch.tensor(s.chemistry_id, dtype=torch.long),
            'source': s.source,
            'degradation_mode': torch.tensor(s.context[5], dtype=torch.float32)
        }


# =============================================================================
# Unified Model Architecture
# =============================================================================

class UnifiedDegradationModel(nn.Module):
    """
    Unified model for cycling + storage degradation prediction.
    
    Architecture:
    - Shared feature encoder (learns domain-invariant features)
    - 6D context encoder (includes degradation mode)
    - Degradation mode attention (soft routing)
    - Unified prediction heads
    """
    
    def __init__(
        self,
        feature_dim: int = 9,
        context_dim: int = 6,  # 6D: temp, charge, discharge, soc, usage, mode
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
        # Learn to weight features differently for cycling vs storage
        self.mode_attention = nn.Sequential(
            nn.Linear(1, 16),  # degradation_mode input
            nn.ReLU(),
            nn.Linear(16, hidden_dim),
            nn.Sigmoid()  # Attention weights
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
            nn.Linear(latent_dim + context_dim, 32),  # Context residual
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
        
        # Domain classifier (for domain adversarial training)
        self.domain_classifier = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # cycling vs storage
        )
    
    def forward(
        self,
        features: torch.Tensor,
        context: torch.Tensor,
        chem_id: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            features: (B, feature_dim)
            context: (B, 6) - 6D context with degradation mode as last dim
            chem_id: (B,) - chemistry IDs
            
        Returns:
            soh_pred: (B, 1)
            rul_pred: (B, 1)
            domain_logits: (B, 2)
            latent: (B, latent_dim)
        """
        batch_size = features.shape[0]
        
        # Encode features
        feat_enc = self.feature_encoder(features)
        
        # Degradation mode attention
        degradation_mode = context[:, 5:6]  # Last dimension
        mode_attn = self.mode_attention(degradation_mode)
        feat_enc = feat_enc * mode_attn  # Soft attention
        
        # Encode context and chemistry
        ctx_enc = self.context_encoder(context)
        chem_enc = self.chem_embed(chem_id)
        
        # Combine and encode
        combined = torch.cat([feat_enc, ctx_enc, chem_enc], dim=-1)
        latent = self.shared_encoder(combined)
        
        # Predictions with context residual
        soh_input = torch.cat([latent, context], dim=-1)
        soh_pred = self.soh_head(soh_input)
        
        rul_input = torch.cat([latent, context], dim=-1)
        rul_pred = self.rul_head(rul_input)
        
        # Domain classification
        domain_logits = self.domain_classifier(latent)
        
        return soh_pred, rul_pred, domain_logits, latent


# =============================================================================
# Data Loading
# =============================================================================

def load_cycling_samples(pipeline: UnifiedDataPipeline) -> List[UnifiedDegradationSample]:
    """Convert NASA cycling data to unified format."""
    samples = []
    
    for s in pipeline.samples:
        # 6D context: [temp, charge, discharge, soc, usage, mode]
        # mode = 1.0 for cycling
        context = np.array([
            s.context_vector[0],  # temp_norm
            s.context_vector[1],  # charge_rate
            s.context_vector[2],  # discharge_rate
            s.context_vector[3],  # soc_norm
            s.context_vector[4],  # usage_profile
            1.0  # degradation_mode = CYCLING
        ], dtype=np.float32)
        
        sample = UnifiedDegradationSample(
            cell_id=s.cell_id,
            sample_idx=s.cycle_idx,
            source='cycling',
            features=s.features[:9] if len(s.features) >= 9 else np.pad(s.features, (0, 9-len(s.features))),
            soh=s.soh,
            rul_normalized=s.rul_normalized,
            context=context,
            temperature=25.0,  # NASA is ~room temp
            soc=50.0,  # Default
            chemistry_id=s.chem_id
        )
        samples.append(sample)
    
    return samples


def load_storage_samples(project_root: Path) -> List[UnifiedDegradationSample]:
    """Load PLN EIS storage data in unified format."""
    loader = EISImpedanceLoader(str(project_root))
    loader.load()
    features_df = loader.get_all_features()
    
    samples = []
    
    # Feature columns
    feature_cols = ['R0', 'Rct_estimate', 'warburg_slope', 'z_real_mean', 
                   'z_imag_mean', 'z_imag_min', 'z_mag_max', 'z_mag_min', 'phase_min']
    
    for idx, row in features_df.iterrows():
        # Extract features
        feats = np.array([row.get(c, 0.0) for c in feature_cols], dtype=np.float32)
        feats = np.nan_to_num(feats, nan=0.0)
        
        # Normalize temperature: -40 to 50 -> 0 to 1
        temp = row.get('temperature', 25.0)
        temp_norm = (temp + 40) / 90.0
        
        # SOC normalized
        soc = row.get('soc', 50.0)
        soc_norm = soc / 100.0
        
        # Estimate SOH from R0 (storage degradation)
        r0 = row.get('R0', 0.1)
        # Higher R0 = more degradation = lower SOH
        soh = np.clip(1.0 - 0.1 * r0, 0.7, 1.0)
        
        # Estimate RUL from storage period
        period = row.get('storage_period', '3W')
        rul_map = {'3W': 0.8, '3M': 0.5, '6M': 0.2}
        rul_norm = rul_map.get(period, 0.5)
        
        # 6D context: mode = 0.0 for storage
        context = np.array([
            temp_norm,
            0.0,  # charge_rate (storage = no charging)
            0.0,  # discharge_rate (storage = no discharging)
            soc_norm,
            0.0,  # usage_profile = storage
            0.0   # degradation_mode = STORAGE
        ], dtype=np.float32)
        
        sample = UnifiedDegradationSample(
            cell_id=row.get('cell_id', f'PLN_{idx}'),
            sample_idx=idx,
            source='storage',
            features=feats,
            soh=soh,
            rul_normalized=rul_norm,
            context=context,
            temperature=temp,
            soc=soc,
            chemistry_id=0  # LCO
        )
        samples.append(sample)
    
    return samples


def normalize_features(samples: List[UnifiedDegradationSample]) -> List[UnifiedDegradationSample]:
    """Normalize features across all samples."""
    features = np.array([s.features for s in samples])
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-8
    
    for s in samples:
        s.features = (s.features - mean) / std
    
    return samples


# =============================================================================
# Training
# =============================================================================

def train_unified_model(
    cycling_samples: List[UnifiedDegradationSample],
    storage_samples: List[UnifiedDegradationSample],
    output_dir: Path,
    epochs: int = 100,
    device: str = 'cpu'
) -> Tuple[UnifiedDegradationModel, Dict]:
    """Train the unified degradation model."""
    
    print("\n" + "=" * 60)
    print("PHASE 2: UNIFIED DEGRADATION MODEL TRAINING")
    print("=" * 60)
    
    # Combine and normalize
    all_samples = cycling_samples + storage_samples
    all_samples = normalize_features(all_samples)
    
    # Split
    np.random.seed(42)
    np.random.shuffle(all_samples)
    
    n = len(all_samples)
    train_samples = all_samples[:int(0.7 * n)]
    val_samples = all_samples[int(0.7 * n):int(0.85 * n)]
    test_samples = all_samples[int(0.85 * n):]
    
    print(f"\nData split:")
    print(f"  Train: {len(train_samples)} ({sum(1 for s in train_samples if s.source == 'cycling')} cycling, {sum(1 for s in train_samples if s.source == 'storage')} storage)")
    print(f"  Val: {len(val_samples)}")
    print(f"  Test: {len(test_samples)}")
    
    # Balance training with WeightedRandomSampler
    train_weights = []
    n_cycling = sum(1 for s in train_samples if s.source == 'cycling')
    n_storage = sum(1 for s in train_samples if s.source == 'storage')
    
    for s in train_samples:
        if s.source == 'cycling':
            train_weights.append(1.0 / n_cycling)
        else:
            train_weights.append(1.0 / n_storage)
    
    sampler = WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_samples),
        replacement=True
    )
    
    # Data loaders
    train_loader = DataLoader(
        UnifiedDegradationDataset(train_samples),
        batch_size=64,
        sampler=sampler
    )
    val_loader = DataLoader(
        UnifiedDegradationDataset(val_samples),
        batch_size=64,
        shuffle=False
    )
    test_loader = DataLoader(
        UnifiedDegradationDataset(test_samples),
        batch_size=64,
        shuffle=False
    )
    
    # Model
    model = UnifiedDegradationModel(
        feature_dim=9,
        context_dim=6,
        hidden_dim=128,
        latent_dim=64
    ).to(device)
    
    print(f"\nModel: UnifiedDegradationModel")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Context: 6D (temp, charge, discharge, soc, usage, mode)")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    history = {'train_loss': [], 'val_cycling_mae': [], 'val_storage_mae': []}
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        
        for batch in train_loader:
            features = batch['features'].to(device)
            context = batch['context'].to(device)
            chem_id = batch['chem_id'].to(device)
            soh_true = batch['soh'].to(device)
            rul_true = batch['rul'].to(device)
            deg_mode = batch['degradation_mode'].to(device)
            
            # Handle NaN values
            features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            context = torch.nan_to_num(context, nan=0.0)
            soh_true = torch.nan_to_num(soh_true, nan=0.9)
            rul_true = torch.nan_to_num(rul_true, nan=0.5)
            
            # Forward
            soh_pred, rul_pred, domain_logits, _ = model(features, context, chem_id)
            
            # Losses
            soh_loss = F.mse_loss(soh_pred.squeeze(), soh_true)
            rul_loss = F.mse_loss(rul_pred.squeeze(), rul_true)
            
            # Domain classification loss (for domain adaptation)
            domain_labels = (deg_mode < 0.5).long()  # 0=storage, 1=cycling
            domain_loss = F.cross_entropy(domain_logits, domain_labels)
            
            # Combined loss
            loss = soh_loss + 0.3 * rul_loss + 0.1 * domain_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        history['train_loss'].append(avg_loss)
        
        # Validation
        model.eval()
        cycling_preds, cycling_true = [], []
        storage_preds, storage_true = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                context = batch['context'].to(device)
                chem_id = batch['chem_id'].to(device)
                soh_true = batch['soh']
                sources = batch['source']
                
                # Handle NaN
                features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
                context = torch.nan_to_num(context, nan=0.0)
                
                soh_pred, _, _, _ = model(features, context, chem_id)
                soh_pred = soh_pred.squeeze().cpu().numpy()
                soh_true = soh_true.numpy()
                
                for i, src in enumerate(sources):
                    if src == 'cycling':
                        cycling_preds.append(soh_pred[i])
                        cycling_true.append(soh_true[i])
                    else:
                        storage_preds.append(soh_pred[i])
                        storage_true.append(soh_true[i])
        
        cycling_mae = np.mean(np.abs(np.array(cycling_preds) - np.array(cycling_true))) if cycling_preds else 0
        storage_mae = np.mean(np.abs(np.array(storage_preds) - np.array(storage_true))) if storage_preds else 0
        
        history['val_cycling_mae'].append(cycling_mae)
        history['val_storage_mae'].append(storage_mae)
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, "
                  f"cycling_mae={cycling_mae:.4f}, storage_mae={storage_mae:.4f}")
        
        # Save best
        val_loss = cycling_mae + storage_mae
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / 'unified_model.pt')
    
    # Load best model
    model.load_state_dict(torch.load(output_dir / 'unified_model.pt', weights_only=True))
    
    # Test evaluation
    print("\n" + "=" * 60)
    print("TEST EVALUATION")
    print("=" * 60)
    
    model.eval()
    test_results = {'cycling': {'pred': [], 'true': [], 'temp': []},
                    'storage': {'pred': [], 'true': [], 'temp': [], 'soc': []}}
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            context = batch['context'].to(device)
            chem_id = batch['chem_id'].to(device)
            soh_true = batch['soh']
            sources = batch['source']
            
            # Handle NaN
            features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            context = torch.nan_to_num(context, nan=0.0)
            
            soh_pred, _, _, _ = model(features, context, chem_id)
            soh_pred = soh_pred.squeeze().cpu().numpy()
            
            for i, src in enumerate(sources):
                pred_val = float(soh_pred[i]) if soh_pred.ndim > 0 else float(soh_pred)
                true_val = float(soh_true[i].item())
                if not np.isnan(pred_val) and not np.isnan(true_val):
                    test_results[src]['pred'].append(pred_val)
                    test_results[src]['true'].append(true_val)
    
    # Compute metrics
    metrics = {}
    for domain in ['cycling', 'storage']:
        if test_results[domain]['pred']:
            preds = np.array(test_results[domain]['pred'])
            trues = np.array(test_results[domain]['true'])
            mae = np.mean(np.abs(preds - trues))
            rmse = np.sqrt(np.mean((preds - trues) ** 2))
            ss_res = np.sum((trues - preds) ** 2)
            ss_tot = np.sum((trues - np.mean(trues)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            
            metrics[domain] = {'mae': mae, 'rmse': rmse, 'r2': r2, 'n': len(preds)}
            print(f"\n{domain.upper()}:")
            print(f"  MAE: {mae:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  R²: {r2:.4f}")
            print(f"  N: {len(preds)}")
    
    return model, {'history': history, 'metrics': metrics, 'test_results': test_results}


def create_phase2_visualizations(
    results: Dict,
    output_dir: Path
):
    """Generate Phase 2 result visualizations."""
    
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # ===== Plot 1: Training History =====
    ax = axes[0, 0]
    epochs = range(1, len(results['history']['train_loss']) + 1)
    ax.plot(epochs, results['history']['train_loss'], 'b-', linewidth=2, label='Train Loss')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # ===== Plot 2: Validation MAE by Domain =====
    ax = axes[0, 1]
    ax.plot(epochs, results['history']['val_cycling_mae'], 'b-', linewidth=2, label='Cycling')
    ax.plot(epochs, results['history']['val_storage_mae'], 'r-', linewidth=2, label='Storage')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('SOH MAE', fontsize=12)
    ax.set_title('Validation MAE: Cycling vs Storage', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # ===== Plot 3: Prediction Scatter =====
    ax = axes[1, 0]
    
    for domain, color, marker in [('cycling', '#3498DB', 'o'), ('storage', '#E74C3C', 's')]:
        if results['test_results'][domain]['pred']:
            preds = results['test_results'][domain]['pred']
            trues = results['test_results'][domain]['true']
            ax.scatter(trues, preds, c=color, marker=marker, alpha=0.6, s=30, label=domain.title())
    
    ax.plot([0.6, 1.1], [0.6, 1.1], 'k--', linewidth=2, label='Perfect')
    ax.set_xlabel('True SOH', fontsize=12)
    ax.set_ylabel('Predicted SOH', fontsize=12)
    ax.set_title('Unified Model: SOH Predictions', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.6, 1.1)
    ax.set_ylim(0.6, 1.1)
    
    # ===== Plot 4: Performance Comparison =====
    ax = axes[1, 1]
    
    domains = ['Cycling', 'Storage']
    maes = [results['metrics'].get('cycling', {}).get('mae', 0),
            results['metrics'].get('storage', {}).get('mae', 0)]
    r2s = [results['metrics'].get('cycling', {}).get('r2', 0),
           results['metrics'].get('storage', {}).get('r2', 0)]
    
    x = np.arange(len(domains))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, maes, width, label='MAE', color='#3498DB')
    
    ax.set_xlabel('Domain', fontsize=12)
    ax.set_ylabel('SOH MAE', fontsize=12)
    ax.set_title('Unified Model Performance by Domain', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(domains)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, mae, r2 in zip(bars1, maes, r2s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
               f'MAE: {mae:.4f}\nR²: {r2:.3f}', ha='center', fontsize=9)
    
    plt.suptitle('Phase 2: Unified Degradation Model Results\n(Cycling + Storage in Single Model)', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = output_dir / 'phase2_unified_results.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--output_dir', type=str, default='reports/phase2_unified')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--device', type=str, default='cpu')
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).parent.parent.parent
    
    print("=" * 60)
    print("PHASE 2: UNIFIED DEGRADATION MODEL")
    print("=" * 60)
    print("\nNovel Contribution #2: Single model for cycling + storage aging")
    
    # Load cycling data
    print("\n[1/4] Loading cycling data (NASA)...")
    pipeline = UnifiedDataPipeline(args.data_root, use_lithium_features=False)
    pipeline.load_datasets(['nasa'])
    cycling_samples = load_cycling_samples(pipeline)
    print(f"  Loaded {len(cycling_samples)} cycling samples")
    
    # Load storage data
    print("\n[2/4] Loading storage data (PLN EIS)...")
    storage_samples = load_storage_samples(project_root)
    print(f"  Loaded {len(storage_samples)} storage samples")
    
    # Train
    print("\n[3/4] Training unified model...")
    model, results = train_unified_model(
        cycling_samples, storage_samples, output_dir, args.epochs, args.device
    )
    
    # Visualize
    print("\n[4/4] Creating visualizations...")
    viz_path = create_phase2_visualizations(results, output_dir)
    
    # Save results
    save_results = {
        'metrics': results['metrics'],
        'config': vars(args),
        'data_counts': {
            'cycling': len(cycling_samples),
            'storage': len(storage_samples)
        }
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(save_results, f, indent=2, default=str)
    
    print("\n" + "=" * 60)
    print("PHASE 2 COMPLETE")
    print("=" * 60)
    print(f"\nFinal Metrics:")
    for domain in ['cycling', 'storage']:
        if domain in results['metrics']:
            m = results['metrics'][domain]
            print(f"  {domain.upper()}: MAE={m['mae']:.4f}, R²={m['r2']:.4f}")
    print(f"\n✅ Results saved to {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
