"""
Advanced Multi-Network PINN with Hard Constraints

Architecture:
1. Separate physics network per mechanism (Multi-Network)
2. Hard constraint: Physics equations embedded in network structure
3. Class-balanced sampling to handle imbalanced training data
4. Physics-aware attention mechanism


Author: Battery ML Research
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List
import math

# Physical constants
R = 8.314  # J/(mol·K)


class PhysicsHardConstraint(nn.Module):
    """
    Hard constraint network that FORCES physics equations.
    
    Instead of learning to predict Q directly, it learns the parameters
    that go INTO the physics equations. The physics is then computed
    analytically, ensuring 100% physics compliance.
    """
    
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Parameter predictors (learn physics coefficients)
        self.k_net = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        self.extra_net = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 2),  # For mechanism-specific params
        )
        
    def sei_growth(self, h: torch.Tensor, t: torch.Tensor, T: torch.Tensor, soc: torch.Tensor) -> torch.Tensor:
        """SEI: Q = k * sqrt(t) * exp(-E_a/2RT) * f(SOC) - HARD CODED PHYSICS"""
        k = F.softplus(self.k_net(h)) * 0.01
        params = self.extra_net(h)
        E_a = 35000 + 25000 * torch.sigmoid(params[:, 0:1])  # [35, 60] kJ/mol
        
        # HARD CONSTRAINT: Physics equation enforced exactly
        arrhenius = torch.exp(-E_a / (2 * R * T))
        sqrt_t = torch.sqrt(t.clamp(min=1e-6))
        soc_factor = 1.0 + 0.5 * (soc - 0.5) ** 2
        
        Q = k * sqrt_t * arrhenius * soc_factor
        return Q, {'k': k, 'E_a': E_a}
    
    def lithium_plating(self, h: torch.Tensor, T: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """Plating: Butler-Volmer kinetics at cold temps - HARD CODED PHYSICS"""
        k = F.softplus(self.k_net(h)) * 0.01
        params = self.extra_net(h)
        alpha = 0.3 + 0.4 * torch.sigmoid(params[:, 0:1])
        
        # HARD CONSTRAINT: Cold enhancement
        T_crit = 278.15  # 5°C
        cold_factor = torch.sigmoid((T_crit - T) / 5.0)
        temp_enhancement = torch.exp((298.15 - T) / 10.0).clamp(max=10.0)
        c_factor = C.clamp(min=0.1) ** alpha
        
        Q = k * cold_factor * temp_enhancement * c_factor
        return Q, {'k': k, 'alpha': alpha}
    
    def active_material_loss(self, h: torch.Tensor, C: torch.Tensor, N: torch.Tensor) -> torch.Tensor:
        """AM Loss: Q = k * C^β * N^γ - HARD CODED PHYSICS"""
        k = F.softplus(self.k_net(h)) * 0.01
        params = self.extra_net(h)
        beta = 1.0 + 1.0 * torch.sigmoid(params[:, 0:1])   # [1, 2]
        gamma = 0.3 + 0.7 * torch.sigmoid(params[:, 1:2])  # [0.3, 1]
        
        # HARD CONSTRAINT: Power law stress
        Q = k * (C.clamp(min=0.1) ** beta) * (N.clamp(min=1.0) ** gamma)
        return Q, {'k': k, 'beta': beta, 'gamma': gamma}


class MultiNetworkPINN(nn.Module):
    """
    Multi-Network PINN with Hard Physics Constraints.
    
    Architecture:
    - Shared feature encoder
    - Separate physics network per mechanism
    - Each mechanism network has HARD CODED physics
    - Attention-based mechanism selection based on physics fit
    """
    
    def __init__(
        self,
        feature_dim: int = 9,
        context_dim: int = 6,
        hidden_dim: int = 128,
        n_mechanisms: int = 5,
    ):
        super().__init__()
        
        self.n_mechanisms = n_mechanisms
        self.mechanism_names = ['sei', 'plating', 'am_loss', 'electrolyte', 'corrosion']
        
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim + context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        
        # Mechanism-specific physics networks (Multi-Network)
        self.sei_net = PhysicsHardConstraint(hidden_dim)
        self.plating_net = PhysicsHardConstraint(hidden_dim)
        self.am_net = PhysicsHardConstraint(hidden_dim)
        
        # Electrolyte and Corrosion (simpler models)
        self.electrolyte_net = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        self.corrosion_net = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        
        # Physics-aware attention for mechanism selection
        self.mechanism_attention = nn.Sequential(
            nn.Linear(hidden_dim + n_mechanisms * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_mechanisms),
        )
        
        # Context-based prior (learned, NOT hard-coded rules)
        self.context_prior = nn.Sequential(
            nn.Linear(context_dim, 64),
            nn.GELU(),
            nn.Linear(64, n_mechanisms),
        )
        
    def forward(
        self,
        features: torch.Tensor,
        context: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with hard physics constraints.
        """
        batch_size = features.shape[0]
        device = features.device
        
        # Encode
        x = torch.cat([features, context], dim=-1)
        h = self.encoder(x)
        
        # Extract physics variables
        temp_norm = context[:, 0:1]
        charge_rate = context[:, 1:2]
        discharge_rate = context[:, 2:3]
        soc = context[:, 3:4]
        mode = context[:, 5:6]
        
        # Denormalize
        temp_celsius = temp_norm * 20 + 25
        temp_kelvin = temp_celsius + 273.15
        
        # Default time and cycles
        time_months = torch.ones(batch_size, 1, device=device) * 6
        n_cycles = torch.ones(batch_size, 1, device=device) * 300
        
        # Compute physics for each mechanism (HARD CONSTRAINTS)
        physics_Q = {}
        physics_params = {}
        
        # 1. SEI Growth
        Q_sei, params_sei = self.sei_net.sei_growth(h, time_months, temp_kelvin, soc)
        physics_Q['sei'] = Q_sei
        physics_params['sei'] = params_sei
        
        # 2. Lithium Plating
        c_rate = charge_rate * 3.0
        Q_plating, params_plating = self.plating_net.lithium_plating(h, temp_kelvin, c_rate)
        physics_Q['plating'] = Q_plating
        physics_params['plating'] = params_plating
        
        # 3. Active Material Loss
        c_stress = discharge_rate * 4.0
        Q_am, params_am = self.am_net.active_material_loss(h, c_stress, n_cycles)
        physics_Q['am_loss'] = Q_am
        physics_params['am_loss'] = params_am
        
        # 4. Electrolyte (temperature dependent)
        Q_electrolyte = F.softplus(self.electrolyte_net(h)) * 0.01
        Q_electrolyte = Q_electrolyte * torch.sigmoid((temp_kelvin - 313.15) / 5.0)  # >40°C
        physics_Q['electrolyte'] = Q_electrolyte
        
        # 5. Corrosion (low SOC storage)
        Q_corrosion = F.softplus(self.corrosion_net(h)) * 0.01
        is_storage = (mode < 0.5).float()
        low_soc = torch.sigmoid((0.25 - soc) / 0.05)  # SOC < 25%
        Q_corrosion = Q_corrosion * is_storage * low_soc
        physics_Q['corrosion'] = Q_corrosion
        
        # Stack all Q values
        Q_stack = torch.cat([
            physics_Q['sei'],
            physics_Q['plating'],
            physics_Q['am_loss'],
            physics_Q['electrolyte'],
            physics_Q['corrosion'],
        ], dim=-1)
        
        # Physics-aware mechanism selection
        # Use both hidden state and physics outputs
        Q_normalized = Q_stack / (Q_stack.sum(dim=-1, keepdim=True) + 1e-6)
        attention_input = torch.cat([h, Q_stack, Q_normalized], dim=-1)
        
        # Get attention scores
        attention_logits = self.mechanism_attention(attention_input)
        
        # Add context-based prior (LEARNED, not hard-coded)
        context_prior = self.context_prior(context)
        
        # Final logits = attention + prior (both learned)
        final_logits = attention_logits + context_prior * 0.5
        
        # Probabilities
        mechanism_probs = F.softmax(final_logits, dim=-1)
        
        return {
            'mechanism_probs': mechanism_probs,
            'logits': final_logits,
            'Q_physics': Q_stack,
            'physics_params': physics_params,
        }


class BalancedSampler(torch.utils.data.Sampler):
    """Class-balanced sampler for handling imbalanced datasets."""
    
    def __init__(self, labels: List[int], samples_per_class: int = 100):
        self.labels = labels
        self.samples_per_class = samples_per_class
        
        # Group indices by class
        self.class_indices = {}
        for idx, label in enumerate(labels):
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)
        
        self.n_classes = len(self.class_indices)
        
    def __iter__(self):
        indices = []
        for cls_id, cls_indices in self.class_indices.items():
            if len(cls_indices) > 0:
                # Sample with replacement if class is small
                sampled = torch.randint(
                    0, len(cls_indices), 
                    (self.samples_per_class,)
                ).tolist()
                indices.extend([cls_indices[i] for i in sampled])
        
        # Shuffle
        import random
        random.shuffle(indices)
        return iter(indices)
    
    def __len__(self):
        return self.samples_per_class * self.n_classes


class MultiNetworkLoss(nn.Module):
    """Loss for Multi-Network PINN."""
    
    def __init__(
        self,
        lambda_physics: float = 0.5,
        lambda_diversity: float = 0.1,
    ):
        super().__init__()
        self.lambda_physics = lambda_physics
        self.lambda_diversity = lambda_diversity
        
    def forward(
        self,
        output: Dict[str, torch.Tensor],
        target: torch.Tensor,
        cap_loss: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        losses = {}
        
        # 1. Classification loss
        L_class = F.cross_entropy(output['logits'], target)
        losses['classification'] = L_class
        
        # 2. Physics consistency: predicted mechanism's Q should match observed
        Q = output['Q_physics']
        batch_size = target.shape[0]
        indices = target.view(-1, 1)
        Q_pred = torch.gather(Q, 1, indices)
        
        L_physics = F.mse_loss(Q_pred, cap_loss)
        losses['physics'] = L_physics
        
        # 3. Diversity loss: encourage mechanism networks to specialize
        probs = output['mechanism_probs']
        entropy = -(probs * torch.log(probs + 1e-6)).sum(dim=-1).mean()
        L_diversity = -entropy * 0.1  # Encourage confident predictions
        losses['diversity'] = L_diversity
        
        total = L_class + self.lambda_physics * L_physics + self.lambda_diversity * L_diversity
        losses['total'] = total
        
        return total, losses


if __name__ == '__main__':
    print("="*70)
    print("Multi-Network PINN with Hard Constraints - Unit Test")
    print("="*70)
    
    model = MultiNetworkPINN(feature_dim=9, context_dim=6)
    print(f"\n Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward
    batch_size = 8
    features = torch.randn(batch_size, 9)
    context = torch.rand(batch_size, 6)
    
    output = model(features, context)
    
    print(f" Forward pass successful")
    print(f"  Mechanism probs shape: {output['mechanism_probs'].shape}")
    print(f"  Q_physics shape: {output['Q_physics'].shape}")
    
    # Test loss
    loss_fn = MultiNetworkLoss()
    target = torch.randint(0, 5, (batch_size,))
    cap_loss = torch.rand(batch_size, 1) * 0.2
    
    total_loss, losses = loss_fn(output, target, cap_loss)
    print(f"\n Loss computation successful")
    print(f"  Total: {total_loss.item():.4f}")
    
    print("\n" + "="*70)
    print(" Multi-Network PINN verified!")
    print("="*70)
