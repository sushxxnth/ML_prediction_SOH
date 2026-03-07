"""
Physics-Aware Temporal Transformer (PATT) for battery domain classification.

Classifies battery operation as storage vs. cycling using a transformer with
physics-informed positional encoding (Arrhenius + sqrt(t) scaling) and
physics-biased attention:

    PE_physics(t, T) = PE_sin(t) + alpha * exp(-Ea/RT) + beta * sqrt(t)
    Attention(Q,K,V) = softmax(QK^T/sqrt(d_k) + M_physics) * V
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass


# Configuration

@dataclass
class PATTConfig:
    """Configuration for Physics-Aware Temporal Transformer."""
    d_model: int = 64           # Model dimension
    n_heads: int = 4            # Number of attention heads
    n_layers: int = 2           # Number of transformer blocks
    d_ff: int = 128             # Feed-forward dimension
    dropout: float = 0.1        # Dropout rate
    max_seq_len: int = 100      # Maximum sequence length
    n_classes: int = 2          # Storage (0) vs Cycling (1)
    
    # Physics parameters
    E_a: float = 50000.0        # Activation energy for SEI (J/mol)
    R: float = 8.314            # Gas constant (J/(mol·K))
    physics_alpha: float = 0.5  # Arrhenius scaling factor (learnable)
    physics_beta: float = 0.3   # √t scaling factor (learnable)
    physics_gamma: float = 0.1  # Attention bias strength


# Physics-Aware Positional Encoding

class PhysicsAwarePositionalEncoding(nn.Module):
    """
    Positional encoding that incorporates electrochemical time scales.
    
    Standard PE: sin/cos based on position
    Physics PE: + Arrhenius temperature term + √t SEI growth term
    
    PE_physics(t, T) = PE_sin(t) + α·exp(-Ea/RT) + β·√t
    
    Where:
        Ea = activation energy (~50 kJ/mol for SEI growth)
        R = gas constant (8.314 J/(mol·K))
        T = temperature (Kelvin)
        α, β = learnable scaling factors
    """
    
    def __init__(self, d_model: int, max_len: int = 100, 
                 E_a: float = 50000.0, R: float = 8.314):
        super().__init__()
        self.d_model = d_model
        self.E_a = E_a
        self.R = R
        
        # Learnable physics scaling factors
        self.alpha = nn.Parameter(torch.tensor(0.5))  # Arrhenius weight
        self.beta = nn.Parameter(torch.tensor(0.3))   # √t weight
        
        # Standard sinusoidal positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)
        
        # Learnable projection for physics terms
        self.physics_proj = nn.Linear(2, d_model)
    
    def forward(self, x: torch.Tensor, temp_kelvin: Optional[torch.Tensor] = None,
                time_fraction: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply physics-aware positional encoding.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            temp_kelvin: Temperature in Kelvin (batch,) or (batch, seq_len)
            time_fraction: Normalized time [0, 1] (batch,) or (batch, seq_len)
        
        Returns:
            x + PE_physics: Position-encoded tensor
        """
        batch_size, seq_len, _ = x.shape
        
        # Standard sinusoidal encoding
        pe_sin = self.pe[:, :seq_len, :]  # (1, seq_len, d_model)
        
        # Physics-based encoding components
        if temp_kelvin is not None:
            # Ensure proper shape
            if temp_kelvin.dim() == 1:
                temp_kelvin = temp_kelvin.unsqueeze(1).expand(-1, seq_len)
            
            # Arrhenius term: exp(-Ea/RT)
            # Clamp temperature to avoid numerical issues
            temp_kelvin = torch.clamp(temp_kelvin, min=250.0, max=350.0)
            arrhenius = torch.exp(-self.E_a / (self.R * temp_kelvin))  # (batch, seq_len)
        else:
            arrhenius = torch.zeros(batch_size, seq_len, device=x.device)
        
        if time_fraction is not None:
            if time_fraction.dim() == 1:
                time_fraction = time_fraction.unsqueeze(1).expand(-1, seq_len)
            
            # √t term for SEI growth scaling
            sqrt_t = torch.sqrt(torch.clamp(time_fraction, min=0.0))  # (batch, seq_len)
        else:
            # Use position as proxy for time
            sqrt_t = torch.sqrt(torch.arange(seq_len, device=x.device, dtype=torch.float) / seq_len)
            sqrt_t = sqrt_t.unsqueeze(0).expand(batch_size, -1)
        
        # Combine physics terms
        physics_features = torch.stack([
            self.alpha * arrhenius,
            self.beta * sqrt_t
        ], dim=-1)  # (batch, seq_len, 2)
        
        pe_physics = self.physics_proj(physics_features)  # (batch, seq_len, d_model)
        
        # Combined encoding
        return x + pe_sin + pe_physics


# Physics-Biased Multi-Head Attention

class PhysicsBiasedAttention(nn.Module):
    """
    Multi-head attention with physics-informed attention bias.
    
    Standard attention: softmax(QK^T / √d_k)
    Physics attention: softmax(QK^T / √d_k + M_physics)
    
    Where M_physics encodes degradation time scaling:
        M_ij = -γ · |t_i - t_j|^0.5
    
    This bias encourages attention to respect the √t scaling of calendar aging.
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 gamma: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.gamma = nn.Parameter(torch.tensor(gamma))
        
        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def _compute_physics_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Compute physics-informed attention bias matrix.
        
        M_ij = -γ · |i - j|^0.5 / √seq_len
        
        This encodes the √t relationship of degradation processes.
        """
        positions = torch.arange(seq_len, device=device, dtype=torch.float)
        # Distance matrix
        dist = torch.abs(positions.unsqueeze(0) - positions.unsqueeze(1))
        # √t scaling with normalization
        bias = -self.gamma * torch.sqrt(dist) / math.sqrt(seq_len)
        return bias  # (seq_len, seq_len)
    
    def forward(self, x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with physics-biased attention.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional attention mask
            return_attention: Whether to return attention weights
        
        Returns:
            output: Attended tensor (batch, seq_len, d_model)
            attn_weights: Optional attention weights (batch, n_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        # (batch, n_heads, seq_len, d_k)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch, n_heads, seq_len, seq_len)
        
        # Add physics bias
        physics_bias = self._compute_physics_bias(seq_len, x.device)
        scores = scores + physics_bias.unsqueeze(0).unsqueeze(0)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)  # (batch, n_heads, seq_len, d_k)
        
        # Concatenate heads and project
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(context)
        
        if return_attention:
            return output, attn_weights
        return output, None


# Transformer Encoder Block

class TransformerEncoderBlock(nn.Module):
    """
    Transformer encoder block with physics-biased attention.
    
    Architecture:
        x -> LayerNorm -> PhysicsBiasedAttention -> Dropout -> Residual
        -> LayerNorm -> FeedForward -> Dropout -> Residual
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, 
                 dropout: float = 0.1, gamma: float = 0.1):
        super().__init__()
        
        self.attention = PhysicsBiasedAttention(d_model, n_heads, dropout, gamma)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, 
                return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with pre-norm architecture."""
        # Self-attention with residual
        normed = self.norm1(x)
        attn_out, attn_weights = self.attention(normed, return_attention=return_attention)
        x = x + self.dropout(attn_out)
        
        # Feed-forward with residual
        x = x + self.ff(self.norm2(x))
        
        return x, attn_weights


# Main Model: Physics-Aware Temporal Transformer

class PATTDomainClassifier(nn.Module):
    """
    Physics-Aware Temporal Transformer for Battery Domain Classification.
    
    Classifies battery degradation patterns as either:
    - Storage (0): Calendar aging, primarily SEI growth
    - Cycling (1): Cycle aging, mechanical stress + SEI
    
    Key innovations:
    1. Physics-aware positional encoding (Arrhenius + √t)
    2. Physics-biased attention mechanism
    3. [CLS] token for classification
    4. Interpretable attention weights
    
    Mathematical formulation in paper:
        PE_physics(t, T) = PE_sin(t) + α·exp(-Ea/RT) + β·√t
        Attention(Q,K,V) = softmax(QK^T/√d_k + M_physics)·V
    """
    
    def __init__(self, input_dim: int = 5, config: Optional[PATTConfig] = None):
        super().__init__()
        
        self.config = config or PATTConfig()
        
        # Input embedding
        self.input_proj = nn.Linear(input_dim, self.config.d_model)
        
        # Learnable [CLS] token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.config.d_model) * 0.02)
        
        # Physics-aware positional encoding
        self.pos_encoder = PhysicsAwarePositionalEncoding(
            d_model=self.config.d_model,
            max_len=self.config.max_seq_len + 1,  # +1 for CLS token
            E_a=self.config.E_a,
            R=self.config.R
        )
        
        # Transformer encoder blocks
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=self.config.d_model,
                n_heads=self.config.n_heads,
                d_ff=self.config.d_ff,
                dropout=self.config.dropout,
                gamma=self.config.physics_gamma
            )
            for _ in range(self.config.n_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(self.config.d_model)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.d_model // 2),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.d_model // 2, self.config.n_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, 
                temp_kelvin: Optional[torch.Tensor] = None,
                time_fraction: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> dict:
        """
        Forward pass for domain classification.
        
        Args:
            x: Battery features (batch, seq_len, input_dim) or (batch, input_dim)
            temp_kelvin: Temperature in Kelvin (optional)
            time_fraction: Normalized time fraction (optional)
            return_attention: Whether to return attention weights
        
        Returns:
            dict with:
                - logits: Classification logits (batch, n_classes)
                - probs: Classification probabilities (batch, n_classes)
                - cls_embedding: [CLS] token embedding (batch, d_model)
                - attention_weights: List of attention weights (optional)
        """
        # Handle single-step input (batch, input_dim) -> (batch, 1, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        batch_size, seq_len, _ = x.shape
        
        # Project input to model dimension
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        
        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, seq_len+1, d_model)
        
        # Apply physics-aware positional encoding
        x = self.pos_encoder(x, temp_kelvin, time_fraction)
        
        # Pass through transformer encoder blocks
        attention_weights = []
        for block in self.encoder_blocks:
            x, attn = block(x, return_attention=return_attention)
            if return_attention and attn is not None:
                attention_weights.append(attn)
        
        # Final layer norm
        x = self.final_norm(x)
        
        # Extract [CLS] token representation
        cls_embedding = x[:, 0, :]  # (batch, d_model)
        
        # Classification
        logits = self.classifier(cls_embedding)  # (batch, n_classes)
        probs = F.softmax(logits, dim=-1)
        
        output = {
            'logits': logits,
            'probs': probs,
            'cls_embedding': cls_embedding,
            'prediction': torch.argmax(probs, dim=-1)
        }
        
        if return_attention:
            output['attention_weights'] = attention_weights
        
        return output
    
    def get_physics_parameters(self) -> dict:
        """Return learned physics parameters for interpretation."""
        return {
            'alpha_arrhenius': self.pos_encoder.alpha.item(),
            'beta_sqrt_t': self.pos_encoder.beta.item(),
            'gamma_attention': [block.attention.gamma.item() for block in self.encoder_blocks]
        }


# Physics-Informed Loss Function

class PhysicsInformedLoss(nn.Module):
    """
    Combined loss with physics regularization.
    
    L_total = L_CE + λ1·L_temporal + λ2·L_physics
    
    Where:
    - L_CE: Cross-entropy classification loss
    - L_temporal: Temporal consistency regularization
    - L_physics: Physics constraint (storage = slower degradation)
    """
    
    def __init__(self, lambda_temporal: float = 0.1, lambda_physics: float = 0.1):
        super().__init__()
        self.lambda_temporal = lambda_temporal
        self.lambda_physics = lambda_physics
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, outputs: dict, labels: torch.Tensor,
                degradation_rates: Optional[torch.Tensor] = None) -> dict:
        """
        Compute combined loss.
        
        Args:
            outputs: Model outputs dict with 'logits'
            labels: Ground truth labels (0=storage, 1=cycling)
            degradation_rates: Optional degradation rates for physics loss
        
        Returns:
            dict with loss components
        """
        logits = outputs['logits']
        
        # Cross-entropy loss
        ce_loss = self.ce_loss(logits, labels)
        
        # Temporal consistency (attention should be smooth)
        temporal_loss = torch.tensor(0.0, device=logits.device)
        if 'attention_weights' in outputs and outputs['attention_weights']:
            for attn in outputs['attention_weights']:
                # Encourage smooth attention patterns
                diff = attn[:, :, 1:, :] - attn[:, :, :-1, :]
                temporal_loss = temporal_loss + torch.mean(diff ** 2)
        
        # Physics loss: storage should have lower predicted degradation rate
        physics_loss = torch.tensor(0.0, device=logits.device)
        if degradation_rates is not None:
            probs = outputs['probs']
            storage_prob = probs[:, 0]  # Probability of storage
            # Storage samples should have lower degradation rates
            # Penalize high degradation rates with high storage probability
            physics_loss = torch.mean(storage_prob * degradation_rates)
        
        # Total loss
        total_loss = (ce_loss + 
                      self.lambda_temporal * temporal_loss + 
                      self.lambda_physics * physics_loss)
        
        return {
            'total': total_loss,
            'ce': ce_loss.item(),
            'temporal': temporal_loss.item(),
            'physics': physics_loss.item()
        }


# Module Test

if __name__ == '__main__':
    print("=" * 60)
    print("Physics-Aware Temporal Transformer (PATT) - Module Test")
    print("=" * 60)
    
    # Create model
    config = PATTConfig()
    model = PATTDomainClassifier(input_dim=5, config=config)

    print(f"\nModel configuration:")
    print(f"  d_model: {config.d_model}")
    print(f"  n_heads: {config.n_heads}")
    print(f"  n_layers: {config.n_layers}")
    print(f"  Activation energy (E_a): {config.E_a} J/mol")
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {n_params:,}")

    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 4
    seq_len = 10
    input_dim = 5
    
    # Random input
    x = torch.randn(batch_size, seq_len, input_dim)
    temp = torch.full((batch_size,), 298.0)  # 25°C in Kelvin
    time_frac = torch.linspace(0, 1, steps=batch_size)
    
    # Forward pass
    outputs = model(x, temp_kelvin=temp, time_fraction=time_frac, return_attention=True)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output logits shape: {outputs['logits'].shape}")
    print(f"  Output probs shape: {outputs['probs'].shape}")
    print(f"  CLS embedding shape: {outputs['cls_embedding'].shape}")
    print(f"  Predictions: {outputs['prediction'].tolist()}")

    # Check attention weights
    if 'attention_weights' in outputs:
        print(f"  Attention weights: {len(outputs['attention_weights'])} layers")
        print(f"  Attention shape: {outputs['attention_weights'][0].shape}")
    
    # Test with single-step input
    print("\nTesting single-step input...")
    x_single = torch.randn(batch_size, input_dim)  # (batch, input_dim)
    outputs_single = model(x_single)
    print(f"  Single-step input shape: {x_single.shape}")
    print(f"  Output probs: {outputs_single['probs']}")
    
    # Physics parameters
    print("\nLearned physics parameters:")
    physics_params = model.get_physics_parameters()
    print(f"  α (Arrhenius): {physics_params['alpha_arrhenius']:.4f}")
    print(f"  β (√t scaling): {physics_params['beta_sqrt_t']:.4f}")
    print(f"  γ (Attention bias): {physics_params['gamma_attention']}")
    
    # Test loss function
    print("\nTesting physics-informed loss...")
    criterion = PhysicsInformedLoss()
    labels = torch.randint(0, 2, (batch_size,))
    deg_rates = torch.rand(batch_size) * 0.1
    
    loss_dict = criterion(outputs, labels, deg_rates)
    print(f"  Total loss: {loss_dict['total']:.4f}")
    print(f"  CE loss: {loss_dict['ce']:.4f}")
    print(f"  Temporal loss: {loss_dict['temporal']:.4f}")
    print(f"  Physics loss: {loss_dict['physics']:.4f}")
    
    print("\n" + "=" * 60)
    print(" PATT module test passed!")
    print("=" * 60)
