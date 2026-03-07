"""
Uncertainty Quantification for Battery SOH Prediction
Provides prediction uncertainty estimates using:
1. Ensemble methods (multiple models)
2. Monte Carlo Dropout (Bayesian approximation)
3. Evidential deep learning (Dirichlet distribution)
"""
import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class BayesianGRU(nn.Module):
    """
    Bayesian GRU with Monte Carlo Dropout for uncertainty quantification
    """
    def __init__(
        self,
        input_dim: int,
        hidden: int = 256,
        layers: int = 2,
        dropout: float = 0.3,
        num_samples: int = 10
    ):
        super().__init__()
        self.num_samples = num_samples
        self.dropout_p = dropout
        self.hidden = hidden
        self.num_layers = layers
        
        self.gru = nn.GRU(
            input_dim, hidden, num_layers=layers,
            batch_first=True, dropout=0.0  # We'll apply dropout manually
        )
        
        # Initialize hidden state from first timestep features
        # Small linear layer to project first timestep to hidden state
        self.h0_proj = nn.Linear(input_dim, hidden * layers)
        
        # Apply dropout to GRU outputs
        self.dropout = nn.Dropout(dropout)
        
        self.head_soh = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1)
        )
        
        self.head_rul = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1)
        )
    
    def forward(self, x: torch.Tensor, return_samples: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, F) input features
            return_samples: If True, return multiple samples for uncertainty estimation
        Returns:
            soh: (B, T) or (num_samples, B, T) if return_samples
            rul: (B, T) or (num_samples, B, T) if return_samples
        """
        if return_samples and self.training:
            # During training, just return single prediction
            return_samples = False
        
        # Initialize hidden state from first timestep
        first_timestep = x[:, 0, :]  # (B, F)
        h0_flat = self.h0_proj(first_timestep)  # (B, hidden * layers)
        h0 = h0_flat.view(x.size(0), self.num_layers, self.hidden).transpose(0, 1).contiguous()  # (num_layers, B, hidden)
        
        if return_samples:
            # Monte Carlo sampling for uncertainty
            soh_samples = []
            rul_samples = []
            
            self.train()  # Enable dropout
            for _ in range(self.num_samples):
                y, _ = self.gru(x, h0)
                y = self.dropout(y)
                
                soh = self.head_soh(y).squeeze(-1)
                rul = self.head_rul(y).squeeze(-1)
                
                soh_samples.append(soh)
                rul_samples.append(rul)
            
            soh = torch.stack(soh_samples, dim=0)  # (num_samples, B, T)
            rul = torch.stack(rul_samples, dim=0)
            return soh, rul
        else:
            # Standard forward pass
            y, _ = self.gru(x, h0)
            y = self.dropout(y)
            soh = self.head_soh(y).squeeze(-1)
            rul = self.head_rul(y).squeeze(-1)
            return soh, rul


class EvidentialHead(nn.Module):
    """
    Evidential deep learning head using Dirichlet distribution
    For regression, we use Normal-Inverse-Gamma (NIG) distribution
    """
    def __init__(self, input_dim: int, hidden: int = 128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 4)  # [mu, nu, alpha, beta] for NIG
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns parameters of Normal-Inverse-Gamma distribution:
        - mu: mean prediction
        - nu: evidence (inverse variance)
        - alpha: shape parameter
        - beta: scale parameter
        """
        params = self.fc(x)  # (B, T, 4)
        mu = params[..., 0:1].squeeze(-1)  # (B, T)
        nu = F.softplus(params[..., 1:2]).squeeze(-1) + 1e-6  # Evidence (positive)
        alpha = F.softplus(params[..., 2:3]).squeeze(-1) + 1.0  # Shape (>= 1)
        beta = F.softplus(params[..., 3:4]).squeeze(-1) + 1e-6  # Scale (positive)
        
        return mu, nu, alpha, beta


class EvidentialMultiTask(nn.Module):
    """
    Evidential deep learning model for uncertainty quantification
    Uses Normal-Inverse-Gamma distribution for regression uncertainty
    """
    def __init__(
        self,
        input_dim: int,
        hidden: int = 256,
        layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        
        self.head_soh = EvidentialHead(hidden, hidden // 2)
        self.head_rul = EvidentialHead(hidden, hidden // 2)
    
    def forward(self, x: torch.Tensor) -> Tuple[Tuple, Tuple]:
        """
        Returns:
            soh_params: (mu, nu, alpha, beta) for SOH
            rul_params: (mu, nu, alpha, beta) for RUL
        """
        y, _ = self.gru(x)
        soh_params = self.head_soh(y)
        rul_params = self.head_rul(y)
        return soh_params, rul_params
    
    def predict_with_uncertainty(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns predictions with uncertainty estimates
        Returns:
            soh_mean: (B, T) mean prediction
            soh_std: (B, T) prediction uncertainty (standard deviation)
            rul_mean: (B, T) mean prediction
            rul_std: (B, T) prediction uncertainty
        """
        soh_params, rul_params = self.forward(x)
        mu_soh, nu_soh, alpha_soh, beta_soh = soh_params
        mu_rul, nu_rul, alpha_rul, beta_rul = rul_params
        
        # For NIG: variance = beta / (alpha - 1) for alpha > 1
        soh_var = beta_soh / (alpha_soh - 1.0 + 1e-6)
        soh_std = torch.sqrt(soh_var)
        
        rul_var = beta_rul / (alpha_rul - 1.0 + 1e-6)
        rul_std = torch.sqrt(rul_var)
        
        return mu_soh, soh_std, mu_rul, rul_std


def evidential_loss(mu: torch.Tensor, nu: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor,
                    target: torch.Tensor, mask: torch.Tensor, lambda_reg: float = 0.1) -> torch.Tensor:
    """
    Evidential loss for regression (Normal-Inverse-Gamma)
    Combines negative log-likelihood with regularization
    """
    m = mask.bool()
    if not m.any():
        return torch.tensor(0.0, device=mu.device)
    
    # Ensure alpha > 1 for valid variance
    alpha = torch.clamp(alpha, min=1.01)
    
    # Negative log-likelihood
    error = target - mu
    nll = 0.5 * torch.log(math.pi / nu) + alpha * torch.log(beta) - \
          torch.lgamma(alpha) + (alpha + 0.5) * torch.log(beta + 0.5 * nu * error**2)
    
    # Regularization: encourage higher evidence (lower uncertainty) for confident predictions
    reg = lambda_reg * torch.abs(alpha - 2.0)  # Penalize low evidence
    
    loss = (nll[m] + reg[m]).mean()
    return loss


class EnsembleModel(nn.Module):
    """
    Ensemble of multiple models for uncertainty quantification
    """
    def __init__(self, models: List[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns mean and std across ensemble
        """
        soh_preds = []
        rul_preds = []
        
        for model in self.models:
            soh, rul = model(x)
            soh_preds.append(soh)
            rul_preds.append(rul)
        
        soh_stack = torch.stack(soh_preds, dim=0)  # (num_models, B, T)
        rul_stack = torch.stack(rul_preds, dim=0)
        
        soh_mean = soh_stack.mean(dim=0)
        soh_std = soh_stack.std(dim=0)
        
        rul_mean = rul_stack.mean(dim=0)
        rul_std = rul_stack.std(dim=0)
        
        return soh_mean, soh_std, rul_mean, rul_std

