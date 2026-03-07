import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class GRUMultiTask(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128, layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.hidden = hidden
        self.num_layers = layers
        
        # Initialize hidden state from first timestep features
        # Small linear layer to project first timestep to hidden state
        self.h0_proj = nn.Linear(input_dim, hidden * layers)
        
        self.head_soh = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1)
        )
        self.head_rul = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1)
        )

    def forward(self, x, h0: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, F)
        # Initialize hidden state from first timestep if not provided
        if h0 is None:
            # Extract first timestep: (B, F)
            first_timestep = x[:, 0, :]
            # Project to hidden state shape: (B, hidden * layers)
            h0_flat = self.h0_proj(first_timestep)
            # Reshape to (num_layers, B, hidden)
            h0 = h0_flat.view(x.size(0), self.num_layers, self.hidden).transpose(0, 1).contiguous()
        
        y, _ = self.gru(x, h0)
        soh = self.head_soh(y).squeeze(-1)
        rul = self.head_rul(y).squeeze(-1)
        return soh, rul


def multitask_loss(soh_pred, rul_pred, soh_tgt, rul_tgt, mask,
                   k: float = 1.0, tau: float = 100.0, use_eol_weight: bool = True,
                   lam_mono: float = 0.1, lam_step: float = 0.01):
    # mask: (B, T) where True means valid target
    # We assume rul_tgt is log1p(RUL_cycles). We'll compute weights and constraints in original space.
    mse = nn.MSELoss(reduction='none')
    huber = nn.SmoothL1Loss(reduction='none')

    m = mask.bool()
    # SOH loss (MSE)
    l_soh_all = mse(soh_pred, soh_tgt)
    l_soh = (l_soh_all[m]).mean() if m.any() else torch.tensor(0.0, device=soh_pred.device)

    # RUL loss in log-space with optional EoL weighting computed from true cycles
    rul_true_cycles = torch.expm1(rul_tgt).clamp(min=0.0)
    w = 1.0 + k * torch.exp(-rul_true_cycles / tau) if use_eol_weight else torch.ones_like(rul_true_cycles)
    # Log-space Huber (as before)
    l_rul_log_all = huber(rul_pred, rul_tgt)
    l_rul_log = (l_rul_log_all[m] * w[m]).mean() if m.any() else torch.tensor(0.0, device=rul_pred.device)

    # Add scale-invariant relative error in original cycle space for stability and better R2
    rul_pred_cycles = torch.expm1(rul_pred).clamp(min=0.0)
    rel_denom = (rul_true_cycles + 1.0)  # avoids division by zero and de-emphasizes very large values
    l_rul_rel_all = huber(rul_pred_cycles / rel_denom, rul_true_cycles / rel_denom)
    l_rul_rel = (l_rul_rel_all[m] * w[m]).mean() if m.any() else torch.tensor(0.0, device=rul_pred.device)

    # Blend losses: prioritize log loss; reduce relative term to avoid instability
    alpha = 0.0
    l_rul = (1.0 - alpha) * l_rul_log + alpha * l_rul_rel

    # Constraints in original RUL space
    rul_pred_cycles = torch.expm1(rul_pred)
    # Monotonic non-increasing: penalize upward steps
    if rul_pred_cycles.shape[1] >= 2:
        diffs = rul_pred_cycles[:, 1:] - rul_pred_cycles[:, :-1]
        mono_pen = torch.clamp(diffs, min=0.0)
        # Remove step-size penalty; it's too prescriptive and can harm R2
        # Mask for consecutive steps (both timesteps valid)
        m_steps = (m[:, 1:] & m[:, :-1])
        # Heavier weights near EoL for constraints as well
        w_steps = ((w[:, 1:] + w[:, :-1]) / 2.0)
        if m_steps.any():
            mono_loss = (mono_pen[m_steps] * w_steps[m_steps]).mean()
            step_loss = torch.tensor(0.0, device=rul_pred.device)
        else:
            mono_loss = torch.tensor(0.0, device=rul_pred.device)
            step_loss = torch.tensor(0.0, device=rul_pred.device)
    else:
        mono_loss = torch.tensor(0.0, device=rul_pred.device)
        step_loss = torch.tensor(0.0, device=rul_pred.device)

    loss = l_soh + l_rul + lam_mono * mono_loss + lam_step * step_loss
    return l_soh, l_rul, loss
