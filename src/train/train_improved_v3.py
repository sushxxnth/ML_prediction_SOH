"""
Improved Model Training v3: SOH-Balanced

Fixes the SOH prediction bias by:
1. Stratified sampling across SOH ranges
2. Weighted loss for minority SOH ranges
3. Data augmentation for low-SOH samples
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.unified_pipeline import UnifiedDataPipeline
from src.data.eis_loader import EISLoader, extract_eis_features


@dataclass
class BalancedSample:
    """Sample with SOH bin for stratified sampling."""
    features: np.ndarray
    context: np.ndarray
    soh: float
    rul_normalized: float
    domain: int
    chem_id: int
    soh_bin: int  # 0=low, 1=mid, 2=high for stratification


class SOHBalancedDataset(Dataset):
    """Dataset with SOH-balanced sampling."""
    
    def __init__(self, samples: List[BalancedSample]):
        self.samples = samples
        
        # Count samples per SOH bin
        self.bin_counts = {}
        for s in samples:
            self.bin_counts[s.soh_bin] = self.bin_counts.get(s.soh_bin, 0) + 1
        
        print(f"  SOH bins: {self.bin_counts}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'features': torch.tensor(s.features, dtype=torch.float32),
            'context': torch.tensor(s.context, dtype=torch.float32),
            'soh': torch.tensor(s.soh, dtype=torch.float32),
            'rul': torch.tensor(s.rul_normalized, dtype=torch.float32),
            'domain': torch.tensor(s.domain, dtype=torch.long),
            'chem_id': torch.tensor(s.chem_id, dtype=torch.long),
            'soh_bin': torch.tensor(s.soh_bin, dtype=torch.long)
        }
    
    def get_balanced_sampler(self):
        """Create sampler that balances across SOH bins."""
        # Inverse frequency weighting
        weights = []
        for s in self.samples:
            w = 1.0 / self.bin_counts[s.soh_bin]
            weights.append(w)
        
        return WeightedRandomSampler(weights, len(self.samples), replacement=True)


class ImprovedUnifiedModel(nn.Module):
    """Improved model with better SOH regression."""
    
    def __init__(
        self,
        feature_dim: int = 9,
        context_dim: int = 6,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        n_chemistries: int = 5
    ):
        super().__init__()
        
        self.chem_embed = nn.Embedding(n_chemistries, 8)
        
        # Deeper feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )
        
        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, 32),
            nn.GELU(),
            nn.Linear(32, 32)
        )
        
        # Mode attention
        self.mode_attention = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, hidden_dim),
            nn.Sigmoid()
        )
        
        # Latent encoder
        combined_dim = hidden_dim + 32 + 8
        self.latent_encoder = nn.Sequential(
            nn.Linear(combined_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU()
        )
        
        # SOH head - NO sigmoid, use linear output for full range
        self.soh_head = nn.Sequential(
            nn.Linear(latent_dim + context_dim, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1)
            # No sigmoid! We'll clamp output instead
        )
        
        # RUL head
        self.rul_head = nn.Sequential(
            nn.Linear(latent_dim + context_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Domain classifier
        self.domain_classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 2)
        )
    
    def forward(self, features, context, chem_id):
        features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        context = torch.nan_to_num(context, nan=0.0)
        
        chem_emb = self.chem_embed(chem_id)
        feat_enc = self.feature_encoder(features)
        
        mode = context[:, -1:] if context.shape[1] > 0 else torch.zeros(features.shape[0], 1, device=features.device)
        mode_attn = self.mode_attention(mode)
        feat_enc = feat_enc * mode_attn
        
        ctx_enc = self.context_encoder(context)
        combined = torch.cat([feat_enc, ctx_enc, chem_emb], dim=-1)
        latent = self.latent_encoder(combined)
        
        soh_input = torch.cat([latent, context], dim=-1)
        soh_raw = self.soh_head(soh_input)
        soh_pred = torch.clamp(soh_raw, 0.0, 1.2)  # Clamp instead of sigmoid
        
        rul_pred = self.rul_head(soh_input)
        domain_logits = self.domain_classifier(latent)
        
        return soh_pred, rul_pred, domain_logits, latent


def get_soh_bin(soh: float) -> int:
    """Assign SOH to bin for stratification."""
    if soh < 0.75:
        return 0  # Degraded
    elif soh < 0.90:
        return 1  # Moderate
    else:
        return 2  # Healthy


def create_balanced_samples(pipeline) -> List[BalancedSample]:
    """Create samples with SOH bins."""
    samples = []
    
    for s in pipeline.samples:
        # Skip outliers
        if s.soh < 0.5 or s.soh > 1.1:
            continue
        
        feat = s.features[:9] if len(s.features) >= 9 else np.pad(s.features, (0, 9 - len(s.features)))
        ctx = s.context_vector[:6] if len(s.context_vector) >= 6 else np.pad(s.context_vector, (0, 6 - len(s.context_vector)))
        
        if len(ctx) >= 6:
            ctx = ctx.copy()
            ctx[5] = 0.0  # Cycling mode
        
        samples.append(BalancedSample(
            features=feat.astype(np.float32),
            context=ctx.astype(np.float32),
            soh=float(s.soh),
            rul_normalized=float(s.rul_normalized),
            domain=0,
            chem_id=s.chem_id,
            soh_bin=get_soh_bin(s.soh)
        ))
    
    return samples


def augment_low_soh_samples(samples: List[BalancedSample], target_count: int) -> List[BalancedSample]:
    """Augment low-SOH samples to balance dataset."""
    low_soh = [s for s in samples if s.soh_bin == 0]
    
    if not low_soh or len(low_soh) >= target_count:
        return samples
    
    # Augment by adding noise
    augmented = []
    while len(low_soh) + len(augmented) < target_count:
        s = np.random.choice(low_soh)
        new_sample = BalancedSample(
            features=s.features + np.random.randn(9).astype(np.float32) * 0.05,
            context=s.context.copy(),
            soh=float(np.clip(s.soh + np.random.randn() * 0.02, 0.5, 0.75)),
            rul_normalized=float(np.clip(s.rul_normalized + np.random.randn() * 0.05, 0, 1)),
            domain=s.domain,
            chem_id=s.chem_id,
            soh_bin=0
        )
        augmented.append(new_sample)
    
    return samples + augmented


def train_improved_model(
    train_dataset: SOHBalancedDataset,
    val_samples: List[BalancedSample],
    output_dir: Path,
    epochs: int = 150,
    device: str = 'cpu'
):
    """Train with SOH-balanced sampling and weighted loss."""
    
    print("\n" + "=" * 60)
    print("TRAINING IMPROVED MODEL (SOH-BALANCED)")
    print("=" * 60)
    
    # Balanced sampler
    sampler = train_dataset.get_balanced_sampler()
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    
    val_loader = DataLoader(SOHBalancedDataset(val_samples), batch_size=64)
    
    model = ImprovedUnifiedModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = {'train_loss': [], 'val_mae': [], 'val_correlation': []}
    best_mae = float('inf')
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            features = batch['features'].to(device)
            context = batch['context'].to(device)
            soh_true = batch['soh'].to(device)
            rul_true = batch['rul'].to(device)
            domain_true = batch['domain'].to(device)
            chem_id = batch['chem_id'].to(device)
            soh_bin = batch['soh_bin'].to(device)
            
            soh_pred, rul_pred, domain_logits, _ = model(features, context, chem_id)
            
            # Weighted MSE loss - higher weight for low-SOH samples
            bin_weights = torch.ones_like(soh_true)
            bin_weights[soh_bin == 0] = 3.0  # 3x weight for degraded
            bin_weights[soh_bin == 1] = 2.0  # 2x weight for moderate
            
            soh_loss = (bin_weights * (soh_pred.squeeze() - soh_true) ** 2).mean()
            rul_loss = F.mse_loss(rul_pred.squeeze(), rul_true)
            domain_loss = F.cross_entropy(domain_logits, domain_true)
            
            loss = soh_loss + 0.3 * rul_loss + 0.2 * domain_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_loss)
        
        # Validation
        val_metrics = evaluate_model(model, val_loader, device)
        history['val_mae'].append(val_metrics['mae'])
        history['val_correlation'].append(val_metrics['correlation'])
        
        if val_metrics['mae'] < best_mae:
            best_mae = val_metrics['mae']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, mae={val_metrics['mae']:.4f}, "
                  f"corr={val_metrics['correlation']:.3f}, pred_range=[{val_metrics['min_pred']:.2%}, {val_metrics['max_pred']:.2%}]")
    
    if best_state:
        model.load_state_dict(best_state)
    
    return model, history


def evaluate_model(model, dataloader, device):
    """Evaluate model with detailed metrics."""
    model.eval()
    
    all_pred, all_true = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            context = batch['context'].to(device)
            chem_id = batch['chem_id'].to(device)
            
            soh_pred, _, _, _ = model(features, context, chem_id)
            all_pred.extend(soh_pred.squeeze().cpu().numpy())
            all_true.extend(batch['soh'].numpy())
    
    pred = np.array(all_pred)
    true = np.array(all_true)
    
    mae = np.mean(np.abs(pred - true))
    correlation = np.corrcoef(pred, true)[0, 1] if len(pred) > 1 else 0
    
    return {
        'mae': mae,
        'correlation': correlation,
        'min_pred': pred.min(),
        'max_pred': pred.max(),
        'std_pred': pred.std()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--output_dir', type=str, default='reports/improved_model')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--device', type=str, default='cpu')
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("IMPROVED MODEL TRAINING v3")
    print("SOH-Balanced with Weighted Loss")
    print("=" * 60)
    
    # Load data
    print("\n[1/4] Loading data...")
    pipeline = UnifiedDataPipeline(args.data_root, use_lithium_features=False)
    pipeline.load_datasets(['nasa'])
    
    samples = create_balanced_samples(pipeline)
    print(f"  Total samples: {len(samples)}")
    
    # Augment low-SOH samples
    mid_count = len([s for s in samples if s.soh_bin == 1])
    samples = augment_low_soh_samples(samples, target_count=mid_count)
    print(f"  After augmentation: {len(samples)}")
    
    # Split
    np.random.seed(42)
    np.random.shuffle(samples)
    
    n_train = int(0.8 * len(samples))
    train_samples = samples[:n_train]
    val_samples = samples[n_train:]
    
    train_dataset = SOHBalancedDataset(train_samples)
    
    # Train
    print("\n[2/4] Training...")
    model, history = train_improved_model(
        train_dataset, val_samples, output_dir, args.epochs, args.device
    )
    
    # Final evaluation
    print("\n[3/4] Final evaluation...")
    val_loader = DataLoader(SOHBalancedDataset(val_samples), batch_size=64)
    final_metrics = evaluate_model(model, val_loader, args.device)
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"  SOH MAE:        {final_metrics['mae']:.4f}")
    print(f"  Correlation:    {final_metrics['correlation']:.3f}")
    print(f"  Prediction Range: [{final_metrics['min_pred']:.2%}, {final_metrics['max_pred']:.2%}]")
    print(f"  Prediction Std:   {final_metrics['std_pred']:.3f}")
    
    # Save
    print("\n[4/4] Saving...")
    torch.save(model.state_dict(), output_dir / 'improved_model.pt')
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(final_metrics, f, indent=2, default=float)
    
    print(f"  Saved to {output_dir}")


if __name__ == '__main__':
    main()
