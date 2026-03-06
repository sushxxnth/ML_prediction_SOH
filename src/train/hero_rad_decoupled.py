"""
Combined RAD (Memory Retrieval) + Decoupled RUL Model.

Integrates:
- Retrieval Augmented Dynamics (RAD) for SOH calibration
- Decoupled RUL head for independent lifetime prediction
- Domain adversarial training for feature invariance
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Function
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.unified_pipeline import UnifiedDataPipeline, UnifiedBatteryDataset


# Gradient Reversal (for domain adversarial)

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


# Memory Bank for Retrieval


class SimpleMemoryBank:
    """Simplified memory bank for retrieval during inference with chemistry tagging."""
    
    def __init__(self, latent_dim: int, device: str = 'cpu'):
        self.latent_dim = latent_dim
        self.device = device
        self.entries = []  # List of (latent, soh, rul, chem_id, source)
        self.cached_latents = None
        self.cached_sohs = None
        self.cached_ruls = None
        self.cached_chems = None
    
    def add(self, latent: torch.Tensor, soh: float, rul: float, chem_id: int = -1, source: str = 'unknown'):
        """Add a single entry with optional chemistry tag and source."""
        self.entries.append({
            'latent': latent.detach().cpu(),
            'soh': soh,
            'rul': rul,
            'chem_id': chem_id,
            'source': source
        })
        self.cached_latents = None  # Invalidate cache
    
    def retrieve(self, query: torch.Tensor, k: int = 5, filter_chem: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve k nearest neighbors, optionally filtered by chemistry."""
        if len(self.entries) == 0:
            return None, None
        
        # Build cache if needed
        if self.cached_latents is None:
            self.cached_latents = torch.stack([e['latent'] for e in self.entries]).to(self.device)
            self.cached_sohs = torch.tensor([e['soh'] for e in self.entries]).to(self.device)
            self.cached_ruls = torch.tensor([e['rul'] for e in self.entries]).to(self.device)
            self.cached_chems = torch.tensor([e.get('chem_id', -1) for e in self.entries]).to(self.device)
        
        # Apply chemistry filter if specified
        if filter_chem is not None:
            chem_mask = self.cached_chems == filter_chem
            if chem_mask.sum() == 0:
                chem_mask = torch.ones(len(self.entries), dtype=torch.bool)
            filtered_latents = self.cached_latents[chem_mask]
            filtered_sohs = self.cached_sohs[chem_mask]
            filtered_ruls = self.cached_ruls[chem_mask]
        else:
            filtered_latents = self.cached_latents
            filtered_sohs = self.cached_sohs
            filtered_ruls = self.cached_ruls
        
        query = query.to(self.device)
        if query.dim() == 1:
            query = query.unsqueeze(0)
        
        query_norm = F.normalize(query, dim=-1)
        memory_norm = F.normalize(filtered_latents, dim=-1)
        similarities = torch.mm(query_norm, memory_norm.t())
        
        k = min(k, len(filtered_latents))
        _, indices = similarities.topk(k, dim=-1)
        
        retrieved_sohs = filtered_sohs[indices]
        retrieved_ruls = filtered_ruls[indices]
        
        return retrieved_sohs, retrieved_ruls
    
    def size(self):
        return len(self.entries)
    
    def size_by_source(self) -> Dict[str, int]:
        """Get count of entries by source."""
        counts = {}
        for e in self.entries:
            src = e.get('source', 'unknown')
            counts[src] = counts.get(src, 0) + 1
        return counts
    
    def clear(self):
        self.entries = []
        self.cached_latents = None


# Combined RAD + Decoupled RUL Model

class RADDecoupledModel(nn.Module):
    """
    Combined Hero Model:
    - RAD-style memory retrieval for SOH calibration
    - Decoupled RUL prediction (independent from SOH)
    - Domain adversarial for cross-chemistry transfer
    - Chemistry-specific calibration
    """
    
    def __init__(
        self,
        feature_dim: int = 20,
        context_dim: int = 5,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        n_chemistries: int = 5,
        retrieval_k: int = 5,
        dropout: float = 0.2,
        device: str = 'cpu'
    ):
        super().__init__()
        
        self.retrieval_k = retrieval_k
        self.device = device
        
        # Memory bank for retrieval
        self.memory_bank = SimpleMemoryBank(latent_dim, device)
        
        # Chemistry embedding
        self.chem_embed = nn.Embedding(n_chemistries, 16)
        
        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        
        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim + 16, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Latent encoder
        self.latent_encoder = nn.Sequential(
            nn.Linear(hidden_dim + 64, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.GELU()
        )
        
        # SOH prediction head (with retrieval augmentation)
        self.soh_head = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Retrieval fusion for SOH calibration
        self.retrieval_fusion = nn.Sequential(
            nn.Linear(latent_dim + 1, 32),  # latent + mean retrieved SOH
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()  # Correction factor in [-1, 1]
        )
        
        # RUL prediction head (DECOUPLED - independent path!)
        self.rul_encoder = nn.Sequential(
            nn.Linear(hidden_dim + 64, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.rul_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Domain adversarial (gradient reversal)
        self.alpha = 1.0
        self.domain_classifier = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_chemistries)
        )
        
        # Chemistry-specific calibration
        self.chemistry_scale = nn.Parameter(torch.ones(n_chemistries))
        self.chemistry_bias = nn.Parameter(torch.zeros(n_chemistries))
    
    def set_alpha(self, alpha: float):
        self.alpha = alpha
    
    def forward(
        self,
        features: torch.Tensor,
        context: torch.Tensor,
        chem_id: torch.Tensor,
        use_retrieval: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            soh_pred: (B, 1)
            rul_pred: (B, 1) - DECOUPLED from SOH
            domain_logits: (B, n_chemistries)
            latent: (B, latent_dim)
        """
        batch_size = features.shape[0]
        
        # Get chemistry embedding
        chem_emb = self.chem_embed(chem_id)
        
        # Encode features
        feat_enc = self.feature_encoder(features)
        
        # Encode context
        ctx_input = torch.cat([context, chem_emb], dim=-1)
        ctx_enc = self.context_encoder(ctx_input)
        
        # Create combined representation
        combined = torch.cat([feat_enc, ctx_enc], dim=-1)
        
        # === SOH Path (with retrieval) ===
        latent = self.latent_encoder(combined)
        soh_raw = self.soh_head(latent)
        
        # Retrieval augmentation for SOH
        if use_retrieval and self.memory_bank.size() > 0:
            retrieved_sohs, _ = self.memory_bank.retrieve(latent, k=self.retrieval_k)
            if retrieved_sohs is not None:
                mean_retrieved_soh = retrieved_sohs.mean(dim=-1, keepdim=True)
                fusion_input = torch.cat([latent, mean_retrieved_soh], dim=-1)
                soh_correction = self.retrieval_fusion(fusion_input) * 0.1  # Small correction
                soh_raw = soh_raw + soh_correction
        
        # Apply chemistry-specific calibration
        scale = self.chemistry_scale[chem_id].unsqueeze(-1)
        bias = self.chemistry_bias[chem_id].unsqueeze(-1)
        soh_pred = torch.clamp(soh_raw * scale + bias, 0, 1)
        
        # === RUL Path (DECOUPLED - separate encoder!) ===
        rul_latent = self.rul_encoder(combined)  # Independent from SOH latent
        rul_pred = self.rul_head(rul_latent)
        
        # === Domain Adversarial ===
        reversed_latent = GradientReversalFunction.apply(latent, self.alpha)
        domain_logits = self.domain_classifier(reversed_latent)
        
        return soh_pred, rul_pred, domain_logits, latent


# Training

def train_combined_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: RADDecoupledModel,
    device: str = 'cpu',
    epochs: int = 100,
    lr: float = 3e-4
) -> Dict:
    """Train the combined RAD + Decoupled RUL model."""
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = {'train_loss': [], 'val_soh_mae': [], 'val_rul_mae': []}
    best_val_loss = float('inf')
    best_state = None
    
    print("\n" + "=" * 60)
    print("TRAINING COMBINED RAD + DECOUPLED RUL MODEL")
    print("=" * 60)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        
        # Gradually increase gradient reversal
        alpha = min(1.0, 2.0 * epoch / epochs)
        model.set_alpha(alpha)
        
        for batch in train_loader:
            features = batch['features'].to(device)
            context = batch['context'].to(device)
            chem_id = batch['chem_id'].to(device)
            soh_true = batch['soh'].to(device)
            rul_true = batch['rul_normalized'].to(device)
            
            # Handle NaN
            features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            context = torch.nan_to_num(context, nan=0.0)
            soh_true = torch.nan_to_num(soh_true, nan=0.9)
            rul_true = torch.nan_to_num(rul_true, nan=0.5)
            
            # Forward
            soh_pred, rul_pred, domain_logits, latent = model(features, context, chem_id)
            
            # Losses
            soh_loss = F.mse_loss(soh_pred.squeeze(), soh_true)
            rul_loss = F.mse_loss(rul_pred.squeeze(), rul_true)
            domain_loss = F.cross_entropy(domain_logits, chem_id)
            
            loss = soh_loss + 0.5 * rul_loss + 0.1 * domain_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Update memory bank with good samples
            if epoch > 10 and n_batches % 10 == 0:
                for i in range(min(5, batch_size := features.shape[0])):
                    if not torch.isnan(soh_true[i]) and not torch.isnan(rul_true[i]):
                        model.memory_bank.add(latent[i], float(soh_true[i]), float(rul_true[i]))
            
            total_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        history['train_loss'].append(avg_loss)
        
        # Validation
        val_metrics = evaluate_model(model, val_loader, device)
        history['val_soh_mae'].append(val_metrics['soh_mae'])
        history['val_rul_mae'].append(val_metrics['rul_mae'])
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, "
                  f"soh_mae={val_metrics['soh_mae']:.4f}, rul_mae={val_metrics['rul_mae']:.1f}, "
                  f"memory={model.memory_bank.size()}")
        
        # Save best
        val_loss = val_metrics['soh_mae'] + val_metrics['rul_mae'] / 100
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
    
    if best_state:
        model.load_state_dict(best_state)
    
    return history


def evaluate_model(model: nn.Module, dataloader: DataLoader, device: str) -> Dict:
    """Evaluate model."""
    model.eval()
    
    all_soh_pred, all_soh_true = [], []
    all_rul_pred, all_rul_true = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            context = batch['context'].to(device)
            chem_id = batch['chem_id'].to(device)
            
            features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            context = torch.nan_to_num(context, nan=0.0)
            
            soh_pred, rul_pred, _, _ = model(features, context, chem_id)
            
            all_soh_pred.extend(soh_pred.squeeze().cpu().numpy())
            all_soh_true.extend(batch['soh'].numpy())
            all_rul_pred.extend(rul_pred.squeeze().cpu().numpy())
            all_rul_true.extend(batch['rul_normalized'].numpy())
    
    soh_pred = np.array(all_soh_pred)
    soh_true = np.array(all_soh_true)
    rul_pred = np.array(all_rul_pred)
    rul_true = np.array(all_rul_true)
    
    valid = ~(np.isnan(soh_pred) | np.isnan(soh_true) | np.isnan(rul_pred) | np.isnan(rul_true))
    soh_pred, soh_true = soh_pred[valid], soh_true[valid]
    rul_pred, rul_true = rul_pred[valid], rul_true[valid]
    
    soh_mae = np.mean(np.abs(soh_pred - soh_true)) if len(soh_pred) > 0 else float('nan')
    rul_mae = np.mean(np.abs(rul_pred - rul_true)) * 100 if len(rul_pred) > 0 else float('nan')
    
    # R² for SOH
    if len(soh_pred) > 1:
        ss_res = np.sum((soh_true - soh_pred) ** 2)
        ss_tot = np.sum((soh_true - np.mean(soh_true)) ** 2)
        soh_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    else:
        soh_r2 = 0
    
    return {'soh_mae': soh_mae, 'soh_r2': soh_r2, 'rul_mae': rul_mae, 'n': len(soh_pred)}


def finetune_on_target(
    model: RADDecoupledModel,
    target_loader: DataLoader,
    device: str,
    epochs: int = 30
) -> Dict:
    """Fine-tune on target chemistry."""
    
    print("\n" + "=" * 60)
    print("FINE-TUNING ON TARGET (50%)")
    print("=" * 60)
    
    # Freeze feature encoder
    for name, param in model.named_parameters():
        if 'feature_encoder' in name:
            param.requires_grad = False
    
    model.set_alpha(0.0)  # Disable domain adversarial
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4, weight_decay=1e-5
    )
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        
        for batch in target_loader:
            features = batch['features'].to(device)
            context = batch['context'].to(device)
            chem_id = batch['chem_id'].to(device)
            soh_true = batch['soh'].to(device)
            rul_true = batch['rul_normalized'].to(device)
            
            features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            context = torch.nan_to_num(context, nan=0.0)
            soh_true = torch.nan_to_num(soh_true, nan=0.9)
            rul_true = torch.nan_to_num(rul_true, nan=0.5)
            
            soh_pred, rul_pred, _, latent = model(features, context, chem_id)
            
            soh_loss = F.mse_loss(soh_pred.squeeze(), soh_true)
            rul_loss = F.mse_loss(rul_pred.squeeze(), rul_true)
            loss = soh_loss + 0.5 * rul_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update memory with target samples
            for i in range(min(3, features.shape[0])):
                if not torch.isnan(soh_true[i]):
                    model.memory_bank.add(latent[i], float(soh_true[i]), float(rul_true[i]))
            
            total_loss += loss.item()
            n_batches += 1
        
        if epoch % 5 == 0 or epoch == epochs - 1:
            metrics = evaluate_model(model, target_loader, device)
            print(f"  Epoch {epoch+1}/{epochs}: soh_mae={metrics['soh_mae']:.4f}, rul_mae={metrics['rul_mae']:.1f}")
    
    # Unfreeze
    for param in model.parameters():
        param.requires_grad = True


def create_hero_visualization(results: Dict, output_dir: Path):
    """Create visualization for hero model results."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    approaches = ['Baseline\n(Original)', 'Decoupled\nRUL Only', 'Combined\nHero Model']
    colors = ['#E74C3C', '#F39C12', '#27AE60']
    
    # SOH MAE
    ax = axes[0]
    soh_vals = [results['baseline']['soh_mae'], 
                results['decoupled_only']['soh_mae'],
                results['hero']['soh_mae']]
    bars = ax.bar(approaches, soh_vals, color=colors, edgecolor='black')
    ax.set_ylabel('SOH MAE', fontsize=12)
    ax.set_title('SOH Prediction (Lower = Better)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, soh_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
    
    # RUL MAE
    ax = axes[1]
    rul_vals = [results['baseline']['rul_mae'],
                results['decoupled_only']['rul_mae'],
                results['hero']['rul_mae']]
    bars = ax.bar(approaches, rul_vals, color=colors, edgecolor='black')
    ax.set_ylabel('RUL MAE', fontsize=12)
    ax.set_title('RUL Prediction (Lower = Better)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, rul_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
               f'{val:.1f}', ha='center', fontsize=10, fontweight='bold')
    
    # Summary
    ax = axes[2]
    ax.axis('off')
    
    soh_imp = (results['baseline']['soh_mae'] - results['hero']['soh_mae']) / results['baseline']['soh_mae'] * 100
    rul_imp = (results['baseline']['rul_mae'] - results['hero']['rul_mae']) / results['baseline']['rul_mae'] * 100
    
    summary = f"""
    HERO MODEL RESULTS
    
    SOH Improvement: {soh_imp:+.1f}%
    RUL Improvement: {rul_imp:+.1f}%
    
    Key Innovations:
    - RAD Memory Retrieval
    - Decoupled RUL Path
    - Domain Adversarial
    - Chemistry Calibration
    """
    
    ax.text(0.1, 0.9, summary, fontsize=11, verticalalignment='top',
           fontfamily='monospace', transform=ax.transAxes,
           bbox=dict(boxstyle='round', facecolor='#D5F5E3', edgecolor='green'))
    
    plt.suptitle('Combined Hero Model: RAD + Decoupled RUL',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = output_dir / 'hero_model_results.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--output_dir', type=str, default='reports/hero_model')
    parser.add_argument('--pretrain_epochs', type=int, default=100)
    parser.add_argument('--finetune_epochs', type=int, default=30)
    parser.add_argument('--device', type=str, default='cpu')
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("HERO MODEL: RAD + DECOUPLED RUL")
    print("=" * 60)
    
    # Load data
    print("\n[1/5] Loading data...")
    pipeline = UnifiedDataPipeline(args.data_root, use_lithium_features=True)
    pipeline.load_datasets(['nasa', 'calce', 'oxford', 'tbsi_sunwoda'])
    
    source_samples = [s for s in pipeline.samples if s.source_dataset in ['nasa', 'calce', 'oxford']]
    target_samples = [s for s in pipeline.samples if s.source_dataset == 'tbsi_sunwoda']
    
    np.random.seed(42)
    np.random.shuffle(source_samples)
    np.random.shuffle(target_samples)
    
    source_train = source_samples[:int(0.85 * len(source_samples))]
    source_val = source_samples[int(0.85 * len(source_samples)):]
    target_finetune = target_samples[:len(target_samples)//2]
    target_test = target_samples[len(target_samples)//2:]
    
    print(f"  Source: {len(source_train)} train, {len(source_val)} val")
    print(f"  Target: {len(target_finetune)} fine-tune, {len(target_test)} test")
    
    # Create loaders
    train_loader = DataLoader(UnifiedBatteryDataset(source_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(UnifiedBatteryDataset(source_val), batch_size=64, shuffle=False)
    finetune_loader = DataLoader(UnifiedBatteryDataset(target_finetune), batch_size=32, shuffle=True)
    test_loader = DataLoader(UnifiedBatteryDataset(target_test), batch_size=64, shuffle=False)
    
    # Create model
    print("\n[2/5] Creating Hero Model...")
    model = RADDecoupledModel(
        feature_dim=20,
        context_dim=5,
        hidden_dim=128,
        latent_dim=64,
        device=args.device
    ).to(args.device)
    
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Pre-train
    print("\n[3/5] Pre-training on source...")
    train_combined_model(train_loader, val_loader, model, args.device, args.pretrain_epochs)
    
    # Zero-shot evaluation
    zeroshot_metrics = evaluate_model(model, test_loader, args.device)
    print(f"\n  Zero-shot: SOH MAE={zeroshot_metrics['soh_mae']:.4f}, RUL MAE={zeroshot_metrics['rul_mae']:.1f}")
    
    # Fine-tune
    print("\n[4/5] Fine-tuning on 50% target...")
    finetune_on_target(model, finetune_loader, args.device, args.finetune_epochs)
    
    # Final evaluation
    print("\n[5/5] Final evaluation...")
    final_metrics = evaluate_model(model, test_loader, args.device)
    
    print("\n" + "=" * 60)
    print("HERO MODEL FINAL RESULTS")
    print("=" * 60)
    print(f"\n  SOH MAE: {final_metrics['soh_mae']:.4f}")
    print(f"  SOH R²:  {final_metrics['soh_r2']:.4f}")
    print(f"  RUL MAE: {final_metrics['rul_mae']:.1f}")
    
    # Results comparison
    results = {
        'baseline': {'soh_mae': 0.157, 'rul_mae': 366.3},
        'decoupled_only': {'soh_mae': 0.141, 'rul_mae': 7.7},
        'hero': {
            'soh_mae': float(final_metrics['soh_mae']),
            'soh_r2': float(final_metrics['soh_r2']),
            'rul_mae': float(final_metrics['rul_mae'])
        }
    }
    
    # Visualization
    viz_path = create_hero_visualization(results, output_dir)
    print(f"\n  Visualization: {viz_path}")
    
    # Save
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    torch.save(model.state_dict(), output_dir / 'hero_model.pt')
    
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"\n{'Model':<25} {'SOH MAE':<12} {'RUL MAE':<12}")
    print("-" * 49)
    print(f"{'Baseline':<25} {0.157:<12.4f} {366.3:<12.1f}")
    print(f"{'Decoupled RUL':<25} {0.141:<12.4f} {7.7:<12.1f}")
    print(f"{'Hero (RAD+Decoupled)':<25} {final_metrics['soh_mae']:<12.4f} {final_metrics['rul_mae']:<12.1f}")
    print("=" * 60)
    print(f"\n  Results saved to {output_dir}")


if __name__ == '__main__':
    main()
