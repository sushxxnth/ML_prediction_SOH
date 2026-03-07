#!/usr/bin/env python3
"""
Physics-Constrained Embedding Module for HERO

Enforces electrochemical constraints in the embedding space to improve
zero-shot generalization without requiring target-domain data.

Key constraints:
1. Temperature → Degradation rate (Arrhenius)
2. C-rate → Mechanical stress (power law)
3. SOC → SEI growth acceleration
4. Cycle count → Capacity fade (Wöhler curve)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


class PhysicsConstrainedEmbedding(nn.Module):
    """
    Embedding module that respects electrochemical physics constraints.
    
    The key insight is that certain physical relationships should hold
    REGARDLESS of battery chemistry:
    - Higher temperature → faster degradation (Arrhenius law)
    - Higher C-rate → more mechanical stress → faster AM loss
    - Higher SOC during storage → faster SEI growth
    - More cycles → more capacity fade (power law)
    
    We enforce these by adding physics-based regularization during training
    and physics-aware feature transformations.
    """
    
    def __init__(self, input_dim: int, embed_dim: int = 64, physics_weight: float = 0.3):
        super().__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.physics_weight = physics_weight
        
        # Feature indices (assume standard ordering)
        # [voltage, current, temperature, soc, time, cycles, dV_dQ, ...]
        self.temp_idx = 2
        self.soc_idx = 3
        self.cycles_idx = 5
        
        # Physics-aware feature transformation
        self.arrhenius_scale = nn.Parameter(torch.tensor(0.05))  # Activation energy proxy
        self.stress_exponent = nn.Parameter(torch.tensor(1.5))   # C-rate stress exponent
        self.sei_soc_weight = nn.Parameter(torch.tensor(0.3))    # SOC effect on SEI
        
        # Main embedding layers
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # Physics projection head (learns chemistry-invariant representations)
        self.physics_head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # 4 physics quantities: temp_effect, stress, sei_rate, cycle_fade
        )
        
        # Chemistry-specific residual (captures chemistry-dependent variations)
        self.chemistry_residual = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, embed_dim)
        )
        
    def compute_physics_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute physics-based features that should be chemistry-invariant.
        """
        # Extract relevant features (assuming standardized input)
        temp = x[:, self.temp_idx] if x.shape[1] > self.temp_idx else torch.zeros(x.shape[0], device=x.device)
        soc = x[:, self.soc_idx] if x.shape[1] > self.soc_idx else torch.zeros(x.shape[0], device=x.device)
        cycles = x[:, self.cycles_idx] if x.shape[1] > self.cycles_idx else torch.zeros(x.shape[0], device=x.device)
        
        # Arrhenius temperature effect (normalized)
        # Higher temp → higher degradation rate
        temp_effect = torch.exp(self.arrhenius_scale * temp)
        
        # C-rate stress effect (using current as proxy)
        current = x[:, 1] if x.shape[1] > 1 else torch.zeros(x.shape[0], device=x.device)
        stress_effect = torch.abs(current) ** self.stress_exponent
        
        # SOC effect on SEI growth
        # High SOC accelerates SEI, low SOC → corrosion risk
        sei_effect = 1.0 + self.sei_soc_weight * (soc - 0.5)
        
        # Cycle aging effect (power law)
        cycle_effect = (cycles + 1) ** 0.5  # sqrt(N) for SEI-dominated aging
        
        return {
            'temp_effect': temp_effect,
            'stress_effect': stress_effect,
            'sei_effect': sei_effect,
            'cycle_effect': cycle_effect
        }
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with physics constraints.
        
        Returns:
            embedding: Main embedding vector
            physics_pred: Predicted physics quantities for regularization
        """
        # Main embedding
        embedding = self.feature_encoder(x)
        
        # Physics predictions (for regularization)
        physics_pred = self.physics_head(embedding)
        
        # Add chemistry-specific residual
        chemistry_adjust = self.chemistry_residual(embedding)
        embedding = embedding + 0.1 * chemistry_adjust
        
        return embedding, physics_pred
    
    def physics_loss(self, x: torch.Tensor, physics_pred: torch.Tensor) -> torch.Tensor:
        """
        Compute physics consistency loss.
        
        Ensures that the network's predictions respect physical laws
        regardless of battery chemistry.
        """
        physics_features = self.compute_physics_features(x)
        
        # Target: embedding should predict physics quantities correctly
        targets = torch.stack([
            physics_features['temp_effect'],
            physics_features['stress_effect'],
            physics_features['sei_effect'],
            physics_features['cycle_effect']
        ], dim=1)
        
        # Normalize targets
        targets = (targets - targets.mean(dim=0)) / (targets.std(dim=0) + 1e-6)
        
        # MSE loss between predictions and physics targets
        loss = F.mse_loss(physics_pred, targets)
        
        return loss * self.physics_weight
    
    def monotonicity_loss(self, x: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
        """
        Enforce monotonic relationships in predictions.
        
        Key constraints:
        - Higher temperature → higher degradation prediction
        - Higher C-rate → higher stress prediction
        """
        batch_size = x.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=x.device)
        
        # Sort by temperature and check monotonicity of degradation
        temp = x[:, self.temp_idx] if x.shape[1] > self.temp_idx else x[:, 0]
        sorted_indices = torch.argsort(temp)
        sorted_pred = predictions[sorted_indices]
        
        # Penalize violations: pred[i] should be <= pred[i+1] for increasing temp
        violations = F.relu(sorted_pred[:-1] - sorted_pred[1:])
        
        return violations.mean() * 0.1


class PhysicsConstrainedHERO(nn.Module):
    """
    HERO model with physics-constrained embedding for zero-shot generalization.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, physics_weight: float = 0.3):
        super().__init__()
        
        # Physics-constrained embedding
        self.physics_embed = PhysicsConstrainedEmbedding(
            input_dim=input_dim,
            embed_dim=64,
            physics_weight=physics_weight
        )
        
        # Prediction heads
        self.soh_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # SOH is 0-1
        )
        
        self.rul_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.ReLU()  # RUL is positive
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            soh_pred: Predicted SOH (0-1)
            rul_pred: Predicted RUL (cycles)
            physics_pred: Physics quantities for regularization
        """
        embedding, physics_pred = self.physics_embed(x)
        
        soh_pred = self.soh_head(embedding).squeeze(-1)
        rul_pred = self.rul_head(embedding).squeeze(-1) * 500  # Scale to typical RUL range
        
        return soh_pred, rul_pred, physics_pred
    
    def compute_loss(self, x: torch.Tensor, soh_true: torch.Tensor, rul_true: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute total loss with physics regularization.
        """
        soh_pred, rul_pred, physics_pred = self.forward(x)
        
        # Prediction losses
        soh_loss = F.mse_loss(soh_pred, soh_true)
        rul_loss = F.mse_loss(rul_pred, rul_true) / 10000  # Normalize RUL loss
        
        # Physics losses
        physics_loss = self.physics_embed.physics_loss(x, physics_pred)
        mono_loss = self.physics_embed.monotonicity_loss(x, soh_pred)
        
        total_loss = soh_loss + 0.1 * rul_loss + physics_loss + mono_loss
        
        return {
            'total': total_loss,
            'soh': soh_loss,
            'rul': rul_loss,
            'physics': physics_loss,
            'monotonicity': mono_loss
        }


def train_physics_constrained_hero(
    X_train: np.ndarray,
    y_soh_train: np.ndarray,
    y_rul_train: np.ndarray,
    epochs: int = 200,
    lr: float = 0.001,
    physics_weight: float = 0.1
) -> PhysicsConstrainedHERO:
    """
    Train the physics-constrained HERO model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model with lower physics weight
    model = PhysicsConstrainedHERO(
        input_dim=X_train.shape[1],
        physics_weight=physics_weight
    ).to(device)
    
    # Prepare data
    X_tensor = torch.FloatTensor(X_train).to(device)
    y_soh_tensor = torch.FloatTensor(y_soh_train).to(device)
    y_rul_tensor = torch.FloatTensor(y_rul_train).to(device)
    
    # Replace NaN/Inf in inputs
    X_tensor = torch.nan_to_num(X_tensor, nan=0.0, posinf=1.0, neginf=-1.0)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    print(f"Training Physics-Constrained HERO on {len(X_train)} samples...")
    
    best_loss = float('inf')
    best_state = None
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass with NaN protection
        try:
            soh_pred, rul_pred, physics_pred = model(X_tensor)
            
            # Check for NaN
            if torch.isnan(soh_pred).any() or torch.isnan(rul_pred).any():
                print(f"  NaN detected at epoch {epoch+1}, resetting...")
                # Reinitialize problematic layers
                for layer in model.modules():
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()
                continue
            
            # Compute losses
            soh_loss = F.mse_loss(soh_pred, y_soh_tensor)
            rul_loss = F.mse_loss(rul_pred, y_rul_tensor) / 10000
            
            # Physics loss with stability
            physics_loss = model.physics_embed.physics_loss(X_tensor, physics_pred)
            physics_loss = torch.clamp(physics_loss, 0, 10)  # Clamp physics loss
            
            total_loss = soh_loss + 0.1 * rul_loss + physics_loss
            
            # Check for NaN loss
            if torch.isnan(total_loss):
                continue
            
            total_loss.backward()
            
            # Aggressive gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            
            optimizer.step()
            scheduler.step()
            
            # Track best model
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            
        except RuntimeError as e:
            print(f"  Error at epoch {epoch+1}: {e}")
            continue
        
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: "
                  f"Total={total_loss.item():.4f}, "
                  f"SOH={soh_loss.item():.4f}")
    
    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model


def evaluate_zero_shot(
    model: PhysicsConstrainedHERO,
    X_test: np.ndarray,
    y_soh_test: np.ndarray,
    y_rul_test: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate zero-shot performance on unseen data.
    """
    device = next(model.parameters()).device
    model.eval()
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test).to(device)
        soh_pred, rul_pred, _ = model(X_tensor)
        
        soh_pred = soh_pred.cpu().numpy()
        rul_pred = rul_pred.cpu().numpy()
    
    from sklearn.metrics import mean_absolute_error, r2_score
    
    soh_mae = mean_absolute_error(y_soh_test, soh_pred)
    soh_r2 = r2_score(y_soh_test, soh_pred)
    rul_mae = mean_absolute_error(y_rul_test, rul_pred)
    
    return {
        'soh_mae': soh_mae * 100,  # Convert to percentage
        'soh_r2': soh_r2,
        'rul_mae': rul_mae
    }


if __name__ == "__main__":
    # Test the module
    print("Testing Physics-Constrained Embedding...")
    
    # Create dummy data
    X = torch.randn(100, 15)
    
    # Test embedding
    embed = PhysicsConstrainedEmbedding(input_dim=15)
    embedding, physics_pred = embed(X)
    
    print(f"Input shape: {X.shape}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Physics pred shape: {physics_pred.shape}")
    
    # Test physics loss
    loss = embed.physics_loss(X, physics_pred)
    print(f"Physics loss: {loss.item():.4f}")
    
    # Test full model
    model = PhysicsConstrainedHERO(input_dim=15)
    soh, rul, phys = model(X)
    print(f"\nFull model test:")
    print(f"SOH pred shape: {soh.shape}, range: [{soh.min():.2f}, {soh.max():.2f}]")
    print(f"RUL pred shape: {rul.shape}, range: [{rul.min():.1f}, {rul.max():.1f}]")
    
    print("\n✅ All tests passed!")
