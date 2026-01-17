"""
Improved Multi-Modal Training v2

Fixes the original issues:
1. Uses REAL capacity data from NASA (not synthetic)
2. Integrates EIS for early warning (complementary modality)
3. Proper multi-task learning with shared representations

Architecture:
- Capacity features → for SOH/RUL prediction (cycling)
- EIS features → for early warning and safe zone detection (storage)
- Fusion → combines both for enhanced predictions
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.unified_pipeline import UnifiedDataPipeline
from src.data.eis_loader import EISLoader, extract_eis_features


@dataclass
class ImprovedMultiModalSample:
    """Sample with real capacity data + EIS features."""
    cell_id: str
    
    # Real capacity data (from NASA cycling)
    capacity_features: np.ndarray  # (9,) - real lithium features
    
    # EIS data (from storage)
    eis_features: np.ndarray  # (8,) - extracted EIS physics features
    has_eis: bool  # Whether EIS data is available
    
    # Context
    context: np.ndarray  # (6,)
    mode: int  # 0=cycling, 1=storage
    
    # Labels
    soh: float
    rul_normalized: float
    early_warning: float  # 1 if SOH < 0.8 (approaching failure)


class ImprovedMultiModalDataset(Dataset):
    """Dataset supporting real multi-modal data."""
    
    def __init__(self, samples: List[ImprovedMultiModalSample]):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'capacity_features': torch.tensor(s.capacity_features, dtype=torch.float32),
            'eis_features': torch.tensor(s.eis_features, dtype=torch.float32),
            'has_eis': torch.tensor(s.has_eis, dtype=torch.float32),
            'context': torch.tensor(s.context, dtype=torch.float32),
            'mode': torch.tensor(s.mode, dtype=torch.long),
            'soh': torch.tensor(s.soh, dtype=torch.float32),
            'rul': torch.tensor(s.rul_normalized, dtype=torch.float32),
            'early_warning': torch.tensor(s.early_warning, dtype=torch.float32)
        }


class ImprovedMultiModalModel(nn.Module):
    """
    Improved Multi-Modal Fusion Model.
    
    - Capacity encoder: processes cycling data
    - EIS encoder: processes storage/impedance data
    - Mode-aware fusion: adapts based on available data
    """
    
    def __init__(
        self,
        capacity_dim: int = 9,
        eis_dim: int = 8,
        context_dim: int = 6,
        hidden_dim: int = 64,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # Capacity encoder (for cycling data)
        self.capacity_encoder = nn.Sequential(
            nn.Linear(capacity_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # EIS encoder (for storage data)
        self.eis_encoder = nn.Sequential(
            nn.Linear(eis_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        
        # Mode-aware attention for fusion
        self.mode_attention = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, hidden_dim),
            nn.Sigmoid()
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 32, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Task-specific heads
        self.soh_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.rul_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.early_warning_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.mode_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    
    def forward(self, capacity_features, eis_features, context, has_eis=None):
        """
        Forward pass supporting both modalities.
        
        Args:
            capacity_features: (B, 9)
            eis_features: (B, 8)
            context: (B, 6) - last dim is mode (0=cycling, 1=storage)
            has_eis: (B,) - whether sample has EIS data
        """
        # Handle NaN
        capacity_features = torch.nan_to_num(capacity_features, nan=0.0)
        eis_features = torch.nan_to_num(eis_features, nan=0.0)
        context = torch.nan_to_num(context, nan=0.0)
        
        # Encode each modality
        cap_enc = self.capacity_encoder(capacity_features)  # (B, H)
        eis_enc = self.eis_encoder(eis_features)  # (B, H)
        ctx_enc = self.context_encoder(context)  # (B, 32)
        
        # Mode-aware attention
        mode = context[:, -1:]  # Last element is mode
        mode_attn = self.mode_attention(mode)  # (B, H)
        
        # Apply attention: cycling emphasizes capacity, storage emphasizes EIS
        # mode=0 (cycling): weight capacity more
        # mode=1 (storage): weight EIS more
        cap_weighted = cap_enc * (1 - mode_attn * 0.5)
        eis_weighted = eis_enc * (0.5 + mode_attn * 0.5)
        
        # If no EIS data, rely fully on capacity
        if has_eis is not None:
            eis_mask = has_eis.unsqueeze(-1)
            eis_weighted = eis_weighted * eis_mask
        
        # Fuse
        combined = torch.cat([cap_weighted, eis_weighted, ctx_enc], dim=-1)
        fused = self.fusion(combined)
        
        # Predictions
        soh = self.soh_head(fused)
        rul = self.rul_head(fused)
        early_warning = self.early_warning_head(fused)
        mode_logits = self.mode_classifier(fused)
        
        return {
            'soh': soh,
            'rul': rul,
            'early_warning': early_warning,
            'mode_logits': mode_logits,
            'fused': fused
        }


def create_improved_dataset(data_root: str) -> List[ImprovedMultiModalSample]:
    """Create dataset with REAL capacity data + EIS data."""
    samples = []
    
    # Load REAL capacity data from NASA
    print("\n[1/3] Loading real NASA capacity data...")
    pipeline = UnifiedDataPipeline(data_root, use_lithium_features=False)
    pipeline.load_datasets(['nasa'])
    
    # Create cycling samples (real capacity data, no EIS)
    for s in pipeline.samples[:500]:  # Use 500 NASA samples
        feat = s.features[:9] if len(s.features) >= 9 else np.pad(s.features, (0, 9 - len(s.features)))
        ctx = s.context_vector[:6] if len(s.context_vector) >= 6 else np.pad(s.context_vector, (0, 6 - len(s.context_vector)))
        ctx = ctx.copy()
        ctx[5] = 0.0  # Cycling mode
        
        samples.append(ImprovedMultiModalSample(
            cell_id=s.cell_id,
            capacity_features=feat.astype(np.float32),
            eis_features=np.zeros(8, dtype=np.float32),  # No EIS for cycling
            has_eis=False,
            context=ctx.astype(np.float32),
            mode=0,
            soh=float(s.soh),
            rul_normalized=float(s.rul_normalized),
            early_warning=1.0 if s.soh < 0.8 else 0.0
        ))
    
    print(f"  Created {len(samples)} cycling samples (real capacity)")
    
    # Load REAL EIS data for storage samples
    print("\n[2/3] Loading real EIS storage data...")
    # EIS data is at project root (Impedance_* folders), not in data/
    project_root = str(Path(data_root).parent) if 'data' in data_root else data_root
    eis_loader = EISLoader(project_root)
    eis_loader.load()
    
    eis_samples_count = 0
    for spectrum in eis_loader.spectra[:500]:  # Use 500 EIS samples
        eis_feat = extract_eis_features(spectrum)
        
        # Create capacity proxy from EIS (physics-based, not random)
        cap_feat = np.zeros(9, dtype=np.float32)
        cap_feat[0] = eis_feat[0]  # R_ohmic
        cap_feat[1] = eis_feat[1]  # R_ct
        cap_feat[2] = eis_feat[6]  # Z_imag_min
        cap_feat[3] = spectrum.temperature / 100
        cap_feat[4] = spectrum.soc / 100
        cap_feat[5] = eis_feat[2]  # Z_warburg
        cap_feat[6] = eis_feat[3]  # Phase_max
        cap_feat[7] = eis_feat[7]  # Nyquist_area
        cap_feat[8] = eis_feat[5]  # Z_real_slope
        
        # Context for storage
        ctx = np.zeros(6, dtype=np.float32)
        ctx[0] = spectrum.temperature / 100
        ctx[3] = spectrum.soc / 100
        ctx[4] = 1.0  # Storage profile
        ctx[5] = 1.0  # Storage mode
        
        # Derive SOH from resistance
        soh = np.clip(1.0 - (eis_feat[0] - 0.05) / 0.15, 0.6, 1.0)
        
        # RUL based on storage period
        period_rul = {'3W': 0.8, '3M': 0.5, '6M': 0.2}
        rul = period_rul.get(spectrum.storage_period, 0.5)
        
        samples.append(ImprovedMultiModalSample(
            cell_id=spectrum.cell_id,
            capacity_features=cap_feat,
            eis_features=eis_feat,
            has_eis=True,
            context=ctx,
            mode=1,
            soh=float(soh),
            rul_normalized=float(rul),
            early_warning=1.0 if soh < 0.8 else 0.0
        ))
        eis_samples_count += 1
    
    print(f"  Created {eis_samples_count} storage samples (real EIS)")
    print(f"\n[3/3] Total samples: {len(samples)}")
    
    return samples


def train_improved_multimodal(
    samples: List[ImprovedMultiModalSample],
    output_dir: Path,
    epochs: int = 100,
    device: str = 'cpu'
) -> Tuple[nn.Module, Dict]:
    """Train the improved multi-modal model."""
    
    # Split
    np.random.seed(42)
    np.random.shuffle(samples)
    
    n_train = int(0.7 * len(samples))
    n_val = int(0.15 * len(samples))
    
    train_samples = samples[:n_train]
    val_samples = samples[n_train:n_train + n_val]
    test_samples = samples[n_train + n_val:]
    
    train_loader = DataLoader(ImprovedMultiModalDataset(train_samples), batch_size=32, shuffle=True)
    val_loader = DataLoader(ImprovedMultiModalDataset(val_samples), batch_size=32)
    test_loader = DataLoader(ImprovedMultiModalDataset(test_samples), batch_size=32)
    
    print(f"\nDataset: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")
    
    # Model
    model = ImprovedMultiModalModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float('inf')
    best_state = None
    history = {'train_loss': [], 'val_soh_mae': [], 'val_ew_recall': []}
    
    print(f"\nTraining for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            cap_feat = batch['capacity_features'].to(device)
            eis_feat = batch['eis_features'].to(device)
            context = batch['context'].to(device)
            has_eis = batch['has_eis'].to(device)
            soh_true = batch['soh'].to(device)
            rul_true = batch['rul'].to(device)
            ew_true = batch['early_warning'].to(device)
            mode_true = batch['mode'].to(device)
            
            outputs = model(cap_feat, eis_feat, context, has_eis)
            
            # Multi-task loss
            soh_loss = F.mse_loss(outputs['soh'].squeeze(), soh_true)
            rul_loss = F.mse_loss(outputs['rul'].squeeze(), rul_true)
            ew_loss = F.binary_cross_entropy(outputs['early_warning'].squeeze(), ew_true)
            mode_loss = F.cross_entropy(outputs['mode_logits'], mode_true)
            
            # Balanced loss
            loss = soh_loss + 0.3 * rul_loss + 0.3 * ew_loss + 0.2 * mode_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_loss)
        
        # Validation
        val_metrics = evaluate_improved_model(model, val_loader, device)
        history['val_soh_mae'].append(val_metrics['soh_mae'])
        history['val_ew_recall'].append(val_metrics.get('ew_recall', 0))
        
        if val_metrics['soh_mae'] < best_val_loss:
            best_val_loss = val_metrics['soh_mae']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, soh_mae={val_metrics['soh_mae']:.4f}, "
                  f"ew_prec={val_metrics.get('ew_precision', 0):.2%}, ew_rec={val_metrics.get('ew_recall', 0):.2%}")
    
    if best_state:
        model.load_state_dict(best_state)
    
    # Final test
    test_metrics = evaluate_improved_model(model, test_loader, device)
    
    return model, {'history': history, 'test_metrics': test_metrics}


def evaluate_improved_model(model, dataloader, device) -> Dict:
    """Evaluate the model."""
    model.eval()
    
    all_soh_pred, all_soh_true = [], []
    all_ew_pred, all_ew_true = [], []
    all_mode_pred, all_mode_true = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            cap_feat = batch['capacity_features'].to(device)
            eis_feat = batch['eis_features'].to(device)
            context = batch['context'].to(device)
            has_eis = batch['has_eis'].to(device)
            
            outputs = model(cap_feat, eis_feat, context, has_eis)
            
            all_soh_pred.extend(outputs['soh'].squeeze().cpu().numpy())
            all_soh_true.extend(batch['soh'].numpy())
            all_ew_pred.extend((outputs['early_warning'].squeeze() > 0.5).float().cpu().numpy())
            all_ew_true.extend(batch['early_warning'].numpy())
            all_mode_pred.extend(torch.argmax(outputs['mode_logits'], dim=1).cpu().numpy())
            all_mode_true.extend(batch['mode'].numpy())
    
    soh_pred = np.array(all_soh_pred)
    soh_true = np.array(all_soh_true)
    ew_pred = np.array(all_ew_pred)
    ew_true = np.array(all_ew_true)
    mode_pred = np.array(all_mode_pred)
    mode_true = np.array(all_mode_true)
    
    metrics = {
        'soh_mae': np.mean(np.abs(soh_pred - soh_true)),
        'mode_acc': (mode_pred == mode_true).mean()
    }
    
    if ew_true.sum() > 0:
        metrics['ew_recall'] = (ew_pred * ew_true).sum() / ew_true.sum()
    if ew_pred.sum() > 0:
        metrics['ew_precision'] = (ew_pred * ew_true).sum() / ew_pred.sum()
    
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--output_dir', type=str, default='reports/multimodal_v2')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--device', type=str, default='cpu')
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("IMPROVED MULTI-MODAL TRAINING v2")
    print("=" * 70)
    
    # Create dataset with REAL data
    samples = create_improved_dataset(args.data_root)
    
    # Train
    model, results = train_improved_multimodal(
        samples, output_dir, args.epochs, args.device
    )
    
    # Results
    test = results['test_metrics']
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"\nSOH Prediction:")
    print(f"  MAE: {test['soh_mae']:.4f}")
    print(f"\nMode Classification:")
    print(f"  Accuracy: {test['mode_acc']:.1%}")
    print(f"\nEarly Warning:")
    print(f"  Precision: {test.get('ew_precision', 0):.1%}")
    print(f"  Recall: {test.get('ew_recall', 0):.1%}")
    
    # Save
    torch.save(model.state_dict(), output_dir / 'improved_multimodal.pt')
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(test, f, indent=2, default=float)
    
    print(f"\nSaved to {output_dir}")


if __name__ == '__main__':
    main()
