"""
Multi-Modal Fusion Model for Battery SOH/RUL Prediction.

Combines:
- Capacity features (20D lithium inventory)
- EIS impedance spectra (34×4 Nyquist curves)

Using dual encoders with cross-attention fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import numpy as np


class EISEncoder(nn.Module):
    """
    1D-CNN encoder for EIS Nyquist spectra.
    
    Input: (B, 34, 4) - 34 frequency points × 4 channels (Z_real, Z_imag, Z_mag, Z_phase)
    Output: (B, latent_dim)
    """
    
    def __init__(self, n_freq: int = 34, n_channels: int = 4, latent_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        
        self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        
        self.fc = nn.Linear(128, latent_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n_freq, n_channels) EIS spectrum
        Returns:
            latent: (B, latent_dim)
        """
        # Transpose to (B, C, L) for Conv1d
        x = x.transpose(1, 2)
        
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = F.gelu(self.bn3(self.conv3(x)))
        
        x = self.pool(x).squeeze(-1)  # (B, 128)
        x = self.dropout(x)
        
        return self.fc(x)


class CapacityEncoder(nn.Module):
    """
    MLP encoder for capacity/lithium features.
    
    Input: (B, 20) - 20D feature vector
    Output: (B, latent_dim)
    """
    
    def __init__(self, feature_dim: int = 20, latent_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Linear(64, latent_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention module to fuse capacity and EIS representations.
    
    Learns which modality is more informative at each degradation stage.
    """
    
    def __init__(self, latent_dim: int = 64, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        # Capacity attends to EIS
        self.cap_to_eis = nn.MultiheadAttention(
            embed_dim=latent_dim, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        
        # EIS attends to capacity
        self.eis_to_cap = nn.MultiheadAttention(
            embed_dim=latent_dim, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        
        # Fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 4, latent_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim * 2)
        )
        
        self.norm = nn.LayerNorm(latent_dim * 2)
    
    def forward(self, cap_latent: torch.Tensor, eis_latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cap_latent: (B, latent_dim) capacity encoding
            eis_latent: (B, latent_dim) EIS encoding
        Returns:
            fused: (B, latent_dim * 2) fused representation
        """
        # Add sequence dimension for attention
        cap_seq = cap_latent.unsqueeze(1)  # (B, 1, D)
        eis_seq = eis_latent.unsqueeze(1)  # (B, 1, D)
        
        # Cross-attention
        cap_attended, _ = self.cap_to_eis(cap_seq, eis_seq, eis_seq)
        eis_attended, _ = self.eis_to_cap(eis_seq, cap_seq, cap_seq)
        
        # Remove sequence dimension
        cap_attended = cap_attended.squeeze(1)
        eis_attended = eis_attended.squeeze(1)
        
        # Concatenate original + attended
        combined = torch.cat([cap_latent, cap_attended, eis_latent, eis_attended], dim=-1)
        
        # Fuse
        fused = self.fusion(combined)
        fused = self.norm(fused)
        
        return fused


class MultiModalPredictor(nn.Module):
    """
    Multi-modal SOH/RUL predictor with early warning capability.
    
    Combines capacity and EIS data for:
    1. SOH prediction (improved accuracy)
    2. RUL prediction (decoupled)
    3. Early warning (predict failure N cycles ahead)
    """
    
    def __init__(
        self,
        capacity_dim: int = 20,
        eis_freqs: int = 34,
        eis_channels: int = 4,
        latent_dim: int = 64,
        n_heads: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # Encoders
        self.capacity_encoder = CapacityEncoder(capacity_dim, latent_dim, dropout)
        self.eis_encoder = EISEncoder(eis_freqs, eis_channels, latent_dim, dropout)
        
        # Fusion
        self.fusion = CrossAttentionFusion(latent_dim, n_heads, dropout)
        
        # Prediction heads
        self.soh_head = nn.Sequential(
            nn.Linear(latent_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # RUL head (decoupled from SOH)
        self.rul_head = nn.Sequential(
            nn.Linear(latent_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Early warning head (binary: will fail within N cycles?)
        self.early_warning_head = nn.Sequential(
            nn.Linear(latent_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        capacity_features: torch.Tensor,
        eis_spectrum: torch.Tensor,
        return_latents: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            capacity_features: (B, 20) lithium inventory features
            eis_spectrum: (B, 34, 4) EIS Nyquist data
            return_latents: whether to return intermediate representations
        
        Returns:
            Dict with 'soh', 'rul', 'early_warning', and optionally latents
        """
        # Handle NaN
        capacity_features = torch.nan_to_num(capacity_features, nan=0.0)
        eis_spectrum = torch.nan_to_num(eis_spectrum, nan=0.0)
        
        # Encode
        cap_latent = self.capacity_encoder(capacity_features)
        eis_latent = self.eis_encoder(eis_spectrum)
        
        # Fuse
        fused = self.fusion(cap_latent, eis_latent)
        
        # Predict
        soh = self.soh_head(fused)
        rul = self.rul_head(fused)
        early_warning = self.early_warning_head(fused)
        
        outputs = {
            'soh': soh,
            'rul': rul,
            'early_warning': early_warning
        }
        
        if return_latents:
            outputs['cap_latent'] = cap_latent
            outputs['eis_latent'] = eis_latent
            outputs['fused'] = fused
        
        return outputs


class CapacityOnlyPredictor(nn.Module):
    """Capacity-only baseline for comparison."""
    
    def __init__(self, capacity_dim: int = 20, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(capacity_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        
        self.soh_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, capacity_features: torch.Tensor) -> torch.Tensor:
        capacity_features = torch.nan_to_num(capacity_features, nan=0.0)
        latent = self.encoder(capacity_features)
        return self.soh_head(latent)


if __name__ == '__main__':
    # Test the models
    batch_size = 8
    
    # Create dummy data
    capacity_features = torch.randn(batch_size, 20)
    eis_spectrum = torch.randn(batch_size, 34, 4)
    
    # Test multi-modal model
    model = MultiModalPredictor()
    outputs = model(capacity_features, eis_spectrum, return_latents=True)
    
    print("Multi-Modal Predictor:")
    print(f"  SOH shape: {outputs['soh'].shape}")
    print(f"  RUL shape: {outputs['rul'].shape}")
    print(f"  Early Warning shape: {outputs['early_warning'].shape}")
    print(f"  Fused latent shape: {outputs['fused'].shape}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test capacity-only baseline
    baseline = CapacityOnlyPredictor()
    soh_baseline = baseline(capacity_features)
    print(f"\nCapacity-Only Baseline:")
    print(f"  SOH shape: {soh_baseline.shape}")
    print(f"  Parameters: {sum(p.numel() for p in baseline.parameters()):,}")
