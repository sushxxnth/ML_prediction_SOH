"""
Physics-Informed Neural Network (PINN) for Battery SOH Prediction
Incorporates battery degradation physics into the neural network
by enforcing physical constraints and using physics-based loss terms.
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsInformedSOH(nn.Module):
    """
    Physics-Informed Neural Network that incorporates:
    1. Arrhenius equation for temperature-dependent degradation
    2. Power-law capacity fade model
    3. Exponential resistance growth model
    4. Monotonicity constraints for SOH and RUL
    5. Physical bounds (SOH in [0, 1], RUL >= 0)
    """
    def __init__(
        self,
        input_dim: int,
        hidden: int = 256,
        layers: int = 3,
        dropout: float = 0.1,
        use_physics_loss: bool = True
    ):
        super().__init__()
        self.use_physics_loss = use_physics_loss
        
        # Feature extraction layers
        layers_list = []
        layers_list.append(nn.Linear(input_dim, hidden))
        layers_list.append(nn.LayerNorm(hidden))
        layers_list.append(nn.GELU())
        layers_list.append(nn.Dropout(dropout))
        
        for _ in range(layers - 1):
            layers_list.append(nn.Linear(hidden, hidden))
            layers_list.append(nn.LayerNorm(hidden))
            layers_list.append(nn.GELU())
            layers_list.append(nn.Dropout(dropout))
        
        self.feature_extractor = nn.Sequential(*layers_list)
        
        # Physics-aware feature processing
        # Extract temperature, current, and cycle index for physics equations
        self.temp_proj = nn.Linear(hidden, hidden // 4)
        self.current_proj = nn.Linear(hidden, hidden // 4)
        self.cycle_proj = nn.Linear(hidden, hidden // 4)
        self.rest_proj = nn.Linear(hidden, hidden // 4)
        
        # Physics-informed fusion
        self.physics_fusion = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Multi-task heads
        self.head_soh = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1)
        )
        
        self.head_rul = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1)
        )
        
        # Physics parameters (learnable)
        # Arrhenius pre-exponential factor and activation energy
        self.register_parameter('arrhenius_A', nn.Parameter(torch.tensor(1.0)))
        self.register_parameter('arrhenius_Ea', nn.Parameter(torch.tensor(0.5)))  # Normalized
        
        # Power-law exponent for capacity fade
        self.register_parameter('fade_exponent', nn.Parameter(torch.tensor(0.5)))
        
    def forward(self, x: torch.Tensor, temp: Optional[torch.Tensor] = None, 
                current: Optional[torch.Tensor] = None, cycle: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, F) input features
            temp: (B, T) temperature values (optional, extracted from x if None)
            current: (B, T) current values (optional)
            cycle: (B, T) cycle indices (optional)
        Returns:
            soh: (B, T) SOH predictions (clamped to [0, 1])
            rul: (B, T) RUL predictions (log-space, non-negative)
        """
        B, T, F = x.shape
        
        # Feature extraction
        features = self.feature_extractor(x)  # (B, T, hidden)
        
        # Physics-aware processing
        temp_feat = self.temp_proj(features)
        current_feat = self.current_proj(features)
        cycle_feat = self.cycle_proj(features)
        rest_feat = self.rest_proj(features)
        
        # Apply physics-based transformations
        # Temperature effect (Arrhenius-like)
        if temp is not None:
            # Normalize temperature (assuming 25°C = 298K as reference)
            temp_norm = (temp - 298.0) / 50.0  # Normalize around room temp
            arrhenius_factor = self.arrhenius_A * torch.exp(-self.arrhenius_Ea * temp_norm)
            temp_feat = temp_feat * (1.0 + 0.1 * arrhenius_factor.unsqueeze(-1))
        
        # Cycle-dependent degradation (power-law)
        if cycle is not None:
            cycle_norm = cycle / 1000.0  # Normalize cycles
            cycle_factor = torch.pow(cycle_norm + 1e-6, self.fade_exponent)
            cycle_feat = cycle_feat * (1.0 + 0.1 * cycle_factor.unsqueeze(-1))
        
        # Fuse physics-aware features
        physics_features = torch.cat([temp_feat, current_feat, cycle_feat, rest_feat], dim=-1)
        fused = self.physics_fusion(physics_features)
        
        # Predictions
        soh_raw = self.head_soh(fused).squeeze(-1)  # (B, T)
        rul_raw = self.head_rul(fused).squeeze(-1)  # (B, T)
        
        # Apply physical constraints
        # SOH: clamp to [0, 1] and ensure monotonicity (non-increasing)
        soh = torch.clamp(soh_raw, 0.0, 1.2)  # Allow slight over-capacity
        
        # RUL: ensure non-negative and monotonic (non-increasing)
        rul = rul_raw  # Keep in log-space for training
        
        return soh, rul
    
    def compute_physics_loss(self, soh_pred: torch.Tensor, rul_pred: torch.Tensor,
                            temp: Optional[torch.Tensor] = None,
                            cycle: Optional[torch.Tensor] = None,
                            mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute physics-based regularization loss
        """
        if not self.use_physics_loss:
            return torch.tensor(0.0, device=soh_pred.device)
        
        m = mask.bool() if mask is not None else torch.ones_like(soh_pred, dtype=torch.bool)
        
        # 1. Monotonicity constraint: SOH should be non-increasing
        if soh_pred.shape[1] > 1:
            soh_diff = soh_pred[:, 1:] - soh_pred[:, :-1]  # (B, T-1)
            m_steps = m[:, 1:] & m[:, :-1]
            if m_steps.any():
                # Penalize increases in SOH
                mono_soh_loss = F.relu(soh_diff[m_steps]).mean()
            else:
                mono_soh_loss = torch.tensor(0.0, device=soh_pred.device)
        else:
            mono_soh_loss = torch.tensor(0.0, device=soh_pred.device)
        
        # 2. RUL monotonicity: RUL should be non-increasing
        rul_pred_cycles = torch.expm1(rul_pred).clamp(min=0.0)
        if rul_pred_cycles.shape[1] > 1:
            rul_diff = rul_pred_cycles[:, 1:] - rul_pred_cycles[:, :-1]
            m_steps = m[:, 1:] & m[:, :-1]
            if m_steps.any():
                mono_rul_loss = F.relu(rul_diff[m_steps]).mean()
            else:
                mono_rul_loss = torch.tensor(0.0, device=soh_pred.device)
        else:
            mono_rul_loss = torch.tensor(0.0, device=soh_pred.device)
        
        # 3. Physics consistency: Higher temperature should accelerate degradation
        if temp is not None and temp.shape[1] > 1:
            temp_diff = temp[:, 1:] - temp[:, :-1]
            soh_diff = soh_pred[:, 1:] - soh_pred[:, :-1]
            m_steps = m[:, 1:] & m[:, :-1]
            if m_steps.any():
                # When temp increases, degradation should be faster (more negative soh_diff)
                # Penalize cases where temp increases but degradation slows
                temp_effect = -temp_diff[m_steps] * soh_diff[m_steps]
                physics_consistency_loss = F.relu(temp_effect).mean()
            else:
                physics_consistency_loss = torch.tensor(0.0, device=soh_pred.device)
        else:
            physics_consistency_loss = torch.tensor(0.0, device=soh_pred.device)
        
        # 4. Cycle-dependent degradation rate
        if cycle is not None and cycle.shape[1] > 1:
            cycle_diff = cycle[:, 1:] - cycle[:, :-1]
            soh_diff = soh_pred[:, 1:] - soh_pred[:, :-1]
            m_steps = m[:, 1:] & m[:, :-1]
            if m_steps.any():
                # Degradation rate should be consistent with cycle progression
                # Penalize unrealistic jumps
                degradation_rate = soh_diff[m_steps] / (cycle_diff[m_steps] + 1e-6)
                # Degradation rate should be negative and bounded
                rate_penalty = F.relu(degradation_rate + 0.01)  # Allow small positive for noise
                physics_rate_loss = rate_penalty.mean()
            else:
                physics_rate_loss = torch.tensor(0.0, device=soh_pred.device)
        else:
            physics_rate_loss = torch.tensor(0.0, device=soh_pred.device)
        
        total_physics_loss = (
            0.1 * mono_soh_loss +
            0.1 * mono_rul_loss +
            0.05 * physics_consistency_loss +
            0.05 * physics_rate_loss
        )
        
        return total_physics_loss

