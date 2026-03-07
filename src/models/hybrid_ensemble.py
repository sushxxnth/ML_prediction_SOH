"""
Hybrid Ensemble Model combining multiple novel approaches
Combines Transformer, Physics-Informed, and Uncertainty Quantification models
"""
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.transformer_attention import TransformerMultiTask
from src.models.physics_informed import PhysicsInformedSOH
from src.models.uncertainty_quantification import BayesianGRU, EvidentialMultiTask


class HybridEnsembleSOH(nn.Module):
    """
    Hybrid ensemble that combines:
    1. Transformer with attention (captures long-range dependencies)
    2. Physics-Informed model (enforces physical constraints)
    3. Bayesian/Evidential model (provides uncertainty estimates)
    
    Uses learnable weights to combine predictions
    """
    def __init__(
        self,
        input_dim: int,
        transformer_config: dict = None,
        physics_config: dict = None,
        bayesian_config: dict = None,
        use_uncertainty: bool = True
    ):
        super().__init__()
        
        # Default configurations
        transformer_config = transformer_config or {
            'd_model': 256,
            'nhead': 8,
            'num_layers': 4,
            'dim_feedforward': 1024,
            'dropout': 0.1
        }
        
        physics_config = physics_config or {
            'hidden': 256,
            'layers': 3,
            'dropout': 0.1
        }
        
        bayesian_config = bayesian_config or {
            'hidden': 256,
            'layers': 2,
            'dropout': 0.3
        }
        
        # Initialize sub-models
        self.transformer = TransformerMultiTask(input_dim, **transformer_config)
        self.physics = PhysicsInformedSOH(input_dim, **physics_config)
        self.bayesian = BayesianGRU(input_dim, **bayesian_config)
        
        if use_uncertainty:
            self.evidential = EvidentialMultiTask(input_dim, hidden=256, layers=2, dropout=0.2)
        else:
            self.evidential = None
        
        # Learnable ensemble weights
        self.weights_transformer = nn.Parameter(torch.tensor(0.4))
        self.weights_physics = nn.Parameter(torch.tensor(0.3))
        self.weights_bayesian = nn.Parameter(torch.tensor(0.3))
        
        # Cross-model attention for fusion
        self.fusion_dim = transformer_config['d_model']
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.fusion_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # Final prediction heads with residual connections
        self.final_soh = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim // 2),
            nn.LayerNorm(self.fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.fusion_dim // 2, 1)
        )
        
        self.final_rul = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim // 2),
            nn.LayerNorm(self.fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.fusion_dim // 2, 1)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        temp: Optional[torch.Tensor] = None,
        current: Optional[torch.Tensor] = None,
        cycle: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_uncertainty: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with ensemble prediction
        """
        # Get predictions from each model
        soh_t, rul_t = self.transformer(x, mask)
        soh_p, rul_p = self.physics(x, temp, current, cycle)
        soh_b, rul_b = self.bayesian(x, return_samples=False)
        
        # Normalize weights
        w_sum = torch.abs(self.weights_transformer) + torch.abs(self.weights_physics) + torch.abs(self.weights_bayesian)
        w_t = torch.abs(self.weights_transformer) / (w_sum + 1e-8)
        w_p = torch.abs(self.weights_physics) / (w_sum + 1e-8)
        w_b = torch.abs(self.weights_bayesian) / (w_sum + 1e-8)
        
        # Weighted ensemble
        soh_ensemble = w_t * soh_t + w_p * soh_p + w_b * soh_b
        rul_ensemble = w_t * rul_t + w_p * rul_p + w_b * rul_b
        
        # Apply physical constraints
        soh_ensemble = torch.clamp(soh_ensemble, 0.0, 1.2)
        
        if return_uncertainty and self.evidential is not None:
            # Get uncertainty estimates
            soh_params, rul_params = self.evidential(x)
            mu_soh, nu_soh, alpha_soh, beta_soh = soh_params
            mu_rul, nu_rul, alpha_rul, beta_rul = rul_params
            
            # Combine with ensemble prediction
            soh_final = 0.7 * soh_ensemble + 0.3 * mu_soh
            rul_final = 0.7 * rul_ensemble + 0.3 * mu_rul
            
            # Compute uncertainty
            soh_var = beta_soh / (alpha_soh - 1.0 + 1e-6)
            soh_std = torch.sqrt(soh_var)
            
            rul_var = beta_rul / (alpha_rul - 1.0 + 1e-6)
            rul_std = torch.sqrt(rul_var)
            
            return (soh_final, soh_std), (rul_final, rul_std)
        
        return soh_ensemble, rul_ensemble
    
    def compute_physics_loss(self, soh_pred: torch.Tensor, rul_pred: torch.Tensor,
                            temp: Optional[torch.Tensor] = None,
                            cycle: Optional[torch.Tensor] = None,
                            mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute physics-based regularization"""
        return self.physics.compute_physics_loss(soh_pred, rul_pred, temp, cycle, mask)

