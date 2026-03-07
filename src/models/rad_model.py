"""
Fleet-Aware Retrieval Augmented Dynamics (RAD) Model

Enhanced RAD model with context-aware retrieval using FleetMemoryBank.
Supports driving profile context (Aggressive, Normal, Eco) for hybrid retrieval.

Key Features:
- Context-aware retrieval with weighted scoring
- Hybrid retrieval: 0.7 * physics_score + 0.3 * context_score
- Fallback to pure physics retrieval when context is not provided
- Integration with FleetMemoryBank for fleet-level learning
"""

import math
from typing import Optional, Tuple, List, Dict, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.memory_bank import (
    FleetMemoryBank, 
    encode_context, 
    context_to_index,
    NUM_PROFILES,
    PROFILE_TYPES
)


class TrajectoryEncoder(nn.Module):
    """
    Encodes battery trajectory sequences into latent representations.
    Uses bidirectional GRU for temporal encoding.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Temporal encoder (Bidirectional GRU)
        self.gru = nn.GRU(
            hidden_dim,
            hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )
        
        # Projection to latent space
        self.latent_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim)
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_dim) input features
            mask: (B, T) boolean mask (True = valid)
        Returns:
            latent: (B, latent_dim) trajectory representation
        """
        B, T, _ = x.shape
        x = self.input_proj(x)  # (B, T, hidden_dim)
        
        if mask is not None:
            lengths = mask.sum(dim=1).long()
            lengths = torch.clamp(lengths, min=1)
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            output, hidden = self.gru(packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(
                output, batch_first=True, total_length=T
            )
            # Get last valid hidden state
            last_idx = (lengths - 1).clamp(min=0)
            batch_idx = torch.arange(B, device=x.device)
            h = output[batch_idx, last_idx]  # (B, hidden_dim)
        else:
            output, hidden = self.gru(x)
            h = output[:, -1]  # (B, hidden_dim)
        
        latent = self.latent_proj(h)  # (B, latent_dim)
        latent = F.normalize(latent, p=2, dim=1)
        
        return latent


class RetrievalAttentionFusion(nn.Module):
    """
    Cross-attention mechanism to fuse retrieved historical trajectories
    with the current trajectory's prediction.
    """
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        self.retrieval_proj = nn.Linear(latent_dim, hidden_dim)
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(
        self,
        current_latent: torch.Tensor,
        retrieved_latents: torch.Tensor,
        current_features: torch.Tensor,
        debug_mode: bool = False
    ) -> torch.Tensor:
        """
        Fuse retrieved trajectories with current trajectory.
        """
        B, T, _ = current_features.shape
        k = retrieved_latents.size(1)
        
        if debug_mode:
            retrieved_proj = self.retrieval_proj(retrieved_latents)
            retrieved_mean = retrieved_proj.mean(dim=1, keepdim=True)
            retrieved_mean = retrieved_mean.expand(-1, T, -1)
            fused = 0.5 * current_features + 0.5 * retrieved_mean
            return fused
        
        retrieved_proj = self.retrieval_proj(retrieved_latents)
        current_query = current_features
        
        fused, attn_weights = self.cross_attention(
            query=current_query,
            key=retrieved_proj,
            value=retrieved_proj
        )
        
        fused = fused + current_features
        fused = self.ffn(fused) + fused
        
        return fused


class FleetRADModel(nn.Module):
    """
    Fleet-Aware Retrieval Augmented Dynamics (RAD) Model
    
    Enhanced RAD model with context-aware retrieval for fleet-level learning.
    Supports driving profile context (Aggressive, Normal, Eco) for hybrid retrieval.
    
    Hybrid Retrieval Logic:
        - physics_score = Cosine similarity of GRU embeddings
        - context_score = Cosine similarity of context vectors
        - total_score = 0.7 * physics_score + 0.3 * context_score
    
    Args:
        input_dim: Number of input features
        hidden_dim: Hidden dimension for GRU and attention
        latent_dim: Dimension of latent space for retrieval
        num_layers: Number of GRU layers
        num_heads: Number of attention heads
        dropout: Dropout rate
        retrieval_k: Number of neighbors to retrieve
        physics_weight: Weight for physics-based retrieval (default: 0.7)
        context_weight: Weight for context-based retrieval (default: 0.3)
        device: Device for computation
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        retrieval_k: int = 5,
        physics_weight: float = 0.7,
        context_weight: float = 0.3,
        device: str = 'cpu'
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.retrieval_k = retrieval_k
        self.physics_weight = physics_weight
        self.context_weight = context_weight
        self.device = device
        
        # Trajectory encoder for latent representation
        self.encoder = TrajectoryEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Feature encoder (bidirectional GRU)
        self.feature_proj = nn.Linear(input_dim, hidden_dim)
        self.feature_encoder = nn.GRU(
            hidden_dim,
            hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )
        
        # Context encoder: project one-hot context to latent space
        self.context_encoder = nn.Sequential(
            nn.Linear(NUM_PROFILES, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
        # Retrieval attention fusion
        self.fusion = RetrievalAttentionFusion(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Prediction heads
        self.head_soh = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # RUL head with SOH input
        self.head_rul = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Fleet-aware memory bank
        self.memory_bank = FleetMemoryBank(latent_dim=latent_dim, device=device)
        
    def _hybrid_retrieve(
        self,
        query_latent: torch.Tensor,
        query_context: Optional[torch.Tensor] = None,
        k: int = 5,
        soh_hint: Optional[float] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], 
               Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform hybrid retrieval using physics and context scores.
        
        Args:
            query_latent: (latent_dim,) query latent vector
            query_context: (NUM_PROFILES,) one-hot context vector (optional)
            k: Number of neighbors to retrieve
            soh_hint: Optional SOH value for filtering
            
        Returns:
            Tuple of (retrieved_keys, retrieved_ruls, retrieved_sohs, total_scores)
        """
        if self.memory_bank.size() == 0:
            return None, None, None, None
        
        # Ensure cache is updated
        self.memory_bank._update_cache()
        
        # Get cached tensors
        keys = self.memory_bank._keys_tensor.to(query_latent.device)
        values = self.memory_bank._values_tensor.to(query_latent.device)
        sohs = self.memory_bank._soh_tensor.to(query_latent.device)
        contexts = self.memory_bank._context_onehot_tensor.to(query_latent.device)
        
        # Compute physics score (cosine similarity of latent embeddings)
        query_norm = F.normalize(query_latent.unsqueeze(0), dim=-1)
        keys_norm = F.normalize(keys, dim=-1)
        physics_score = torch.mm(query_norm, keys_norm.t()).squeeze(0)  # (N,)
        
        # Compute context score
        if query_context is not None:
            # Encode query context to latent space
            query_ctx = query_context.to(query_latent.device)
            if query_ctx.dim() == 1:
                query_ctx = query_ctx.unsqueeze(0)
            
            # Simple cosine similarity of one-hot vectors
            query_ctx_norm = F.normalize(query_ctx.float(), dim=-1)
            contexts_norm = F.normalize(contexts.float(), dim=-1)
            context_score = torch.mm(query_ctx_norm, contexts_norm.t()).squeeze(0)  # (N,)
        else:
            # Fallback: no context, use zero context score
            context_score = torch.zeros_like(physics_score)
        
        # Compute total score: weighted combination
        total_score = (self.physics_weight * physics_score + 
                      self.context_weight * context_score)
        
        # Apply SOH constraint if provided
        if soh_hint is not None:
            soh_diff = (sohs - soh_hint).abs()
            soh_mask = soh_diff <= 0.1  # 10% tolerance
            if soh_mask.sum() >= k:
                total_score = total_score.masked_fill(~soh_mask, -float('inf'))
        
        # Get top-k
        actual_k = min(k, self.memory_bank.size())
        if actual_k <= 0:
            return None, None, None, None
        
        top_k = torch.topk(total_score, actual_k)
        indices = top_k.indices
        
        return (
            keys[indices],
            values[indices],
            sohs[indices],
            top_k.values
        )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        use_retrieval: bool = True,
        debug_fusion: bool = False,
        target_soh: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with context-aware retrieval augmentation.
        
        Args:
            x: (B, T, F) input features
            mask: (B, T) boolean mask (True = valid)
            context: (B, NUM_PROFILES) one-hot context vectors for driving profile
                     If None, falls back to pure physics retrieval (ctx_score = 0)
            use_retrieval: Whether to use retrieval augmentation
            debug_fusion: If True, use simple averaging instead of attention
            target_soh: (B, T) Optional True SOH for teacher forcing
            
        Returns:
            soh_pred: (B, T) SOH predictions
            rul_pred: (B, T) RUL predictions (log-space)
        """
        B, T, F = x.shape
        
        # Encode current trajectory to latent space
        current_latent = self.encoder(x, mask)  # (B, latent_dim)
        
        # Encode features
        x_proj = self.feature_proj(x)  # (B, T, hidden_dim)
        
        if mask is not None:
            lengths = mask.sum(dim=1).long()
            lengths = torch.clamp(lengths, min=1)
            packed = nn.utils.rnn.pack_padded_sequence(
                x_proj, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            encoded_features, _ = self.feature_encoder(packed)
            encoded_features, _ = nn.utils.rnn.pad_packed_sequence(
                encoded_features, batch_first=True, total_length=T
            )
        else:
            encoded_features, _ = self.feature_encoder(x_proj)
        
        # Retrieval with hybrid scoring
        retrieved_latents = None
        retrieved_ruls = None
        
        if use_retrieval and self.memory_bank.size() > 0:
            # Get SOH hint for retrieval
            prelim_soh = self.head_soh(encoded_features).squeeze(-1)
            soh_for_retrieval = target_soh if target_soh is not None else prelim_soh.detach()
            
            retrieved_latents_list = []
            retrieved_ruls_list = []
            
            for b in range(B):
                # Get SOH hint for this sample
                if mask is not None:
                    valid_mask = mask[b].bool()
                    soh_hint = soh_for_retrieval[b, valid_mask].mean().item() if valid_mask.sum() > 0 else 0.9
                else:
                    soh_hint = soh_for_retrieval[b].mean().item()
                
                # Get context for this sample (if provided)
                sample_context = context[b] if context is not None else None
                
                try:
                    # Hybrid retrieval
                    ret_keys, ret_ruls, ret_sohs, scores = self._hybrid_retrieve(
                        query_latent=current_latent[b],
                        query_context=sample_context,
                        k=self.retrieval_k,
                        soh_hint=soh_hint
                    )
                    
                    if ret_keys is not None:
                        retrieved_latents_list.append(ret_keys)
                        retrieved_ruls_list.append(ret_ruls)
                    else:
                        # Fallback: zeros
                        retrieved_latents_list.append(
                            torch.zeros(self.retrieval_k, self.latent_dim, device=x.device)
                        )
                        retrieved_ruls_list.append(
                            torch.zeros(self.retrieval_k, device=x.device)
                        )
                except Exception as e:
                    # Fallback on error
                    retrieved_latents_list.append(
                        torch.zeros(self.retrieval_k, self.latent_dim, device=x.device)
                    )
                    retrieved_ruls_list.append(
                        torch.zeros(self.retrieval_k, device=x.device)
                    )
            
            retrieved_latents = torch.stack(retrieved_latents_list)  # (B, k, latent_dim)
            retrieved_ruls = torch.stack(retrieved_ruls_list)  # (B, k)
            
            # Apply fusion
            fused_features = self.fusion(
                current_latent=current_latent,
                retrieved_latents=retrieved_latents,
                current_features=encoded_features,
                debug_mode=debug_fusion
            )
        else:
            fused_features = encoded_features
        
        # Predictions
        soh_pred = self.head_soh(fused_features).squeeze(-1)  # (B, T)
        
        # RUL with SOH input
        soh_input = soh_pred.unsqueeze(-1)  # (B, T, 1)
        rul_input = torch.cat([fused_features, soh_input], dim=-1)
        rul_pred = self.head_rul(rul_input).squeeze(-1)  # (B, T)
        
        return soh_pred, rul_pred
    
    def add_to_memory(
        self,
        x: torch.Tensor,
        soh: torch.Tensor,
        rul: torch.Tensor,
        context: Union[str, int, torch.Tensor, np.ndarray],
        mask: Optional[torch.Tensor] = None,
        cell_id: Optional[str] = None,
        store_per_cycle: bool = True,
        cycle_stride: int = 5
    ):
        """
        Add trajectories to the fleet memory bank.
        
        Args:
            x: (B, T, F) input features
            soh: (B, T) SOH values
            rul: (B, T) RUL values (log-space)
            context: Driving profile context - can be:
                - String: 'Normal', 'Aggressive', 'Eco'
                - Integer: 0, 1, 2
                - Tensor/Array: one-hot vector (NUM_PROFILES,)
            mask: (B, T) boolean mask
            cell_id: Optional cell identifier
            store_per_cycle: If True, store individual cycles; else store full trajectory
            cycle_stride: Stride for storing cycles (to control memory size)
        """
        with torch.no_grad():
            B, T, F = x.shape
            
            # Parse context
            if isinstance(context, str):
                context_onehot = encode_context(context)
            elif isinstance(context, int):
                context_onehot = np.zeros(NUM_PROFILES, dtype=np.float32)
                context_onehot[context] = 1.0
            elif isinstance(context, torch.Tensor):
                context_onehot = context.cpu().numpy()
            else:
                context_onehot = np.array(context, dtype=np.float32)
            
            if store_per_cycle:
                for b in range(B):
                    m = mask[b] if mask is not None else torch.ones(T, dtype=torch.bool, device=x.device)
                    
                    for t in range(0, T, cycle_stride):
                        if t < len(m) and m[t]:
                            # Get context window
                            window = min(10, t + 1)
                            start_idx = max(0, t - window + 1)
                            
                            x_context = x[b:b+1, start_idx:t+1, :]
                            m_context = m[start_idx:t+1]
                            
                            # Encode
                            latent = self.encoder(x_context, m_context.unsqueeze(0))
                            
                            # Get RUL in linear space
                            rul_val = float(torch.expm1(rul[b, t].clamp(min=-1.0)))
                            soh_val = float(soh[b, t])
                            
                            # Add to memory bank
                            self.memory_bank.add_to_memory(
                                key=latent.squeeze(0),
                                value=rul_val,
                                soh=soh_val,
                                context_label=context_onehot,
                                cell_id=f"{cell_id}_cycle_{t}" if cell_id else f"cell_{b}_cycle_{t}"
                            )
            else:
                # Store full trajectory
                latent = self.encoder(x, mask)
                
                for b in range(B):
                    m = mask[b] if mask is not None else torch.ones(T, dtype=torch.bool, device=x.device)
                    valid_idx = m.nonzero().squeeze(-1)
                    
                    if len(valid_idx) > 0:
                        last_idx = valid_idx[-1].item()
                        rul_val = float(torch.expm1(rul[b, last_idx].clamp(min=-1.0)))
                        soh_val = float(soh[b, last_idx])
                        
                        self.memory_bank.add_to_memory(
                            key=latent[b],
                            value=rul_val,
                            soh=soh_val,
                            context_label=context_onehot,
                            cell_id=cell_id or f"cell_{b}"
                        )
    
    def save_memory_bank(self, path: str):
        """Save the memory bank to disk."""
        self.memory_bank.save(path)
    
    def load_memory_bank(self, path: str):
        """Load a memory bank from disk."""
        self.memory_bank.load(path)
    
    def clear_memory_bank(self):
        """Clear the memory bank."""
        self.memory_bank.clear()
    
    def get_memory_stats(self) -> Dict:
        """Get memory bank statistics."""
        return self.memory_bank.get_statistics()


# Convenience function to create context tensor from profile string
def create_context_tensor(
    profile: Union[str, List[str]], 
    batch_size: int = 1,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Create context tensor from profile string(s).
    
    Args:
        profile: Single profile string or list of profiles
        batch_size: Batch size (used if single profile provided)
        device: Device for tensor
        
    Returns:
        Context tensor of shape (batch_size, NUM_PROFILES)
    """
    if isinstance(profile, str):
        one_hot = encode_context(profile)
        context = torch.tensor(one_hot, device=device).unsqueeze(0)
        context = context.expand(batch_size, -1)
    else:
        one_hots = [encode_context(p) for p in profile]
        context = torch.tensor(np.stack(one_hots), device=device)
    
    return context


if __name__ == '__main__':
    # Test the model
    print("Testing FleetRADModel...")
    
    # Create model
    model = FleetRADModel(
        input_dim=10,
        hidden_dim=128,
        latent_dim=64,
        num_layers=2,
        retrieval_k=5,
        device='cpu'
    )
    
    # Test forward pass without context
    x = torch.randn(2, 50, 10)
    mask = torch.ones(2, 50, dtype=torch.bool)
    
    print("\n1. Forward pass without context:")
    soh, rul = model(x, mask, context=None, use_retrieval=False)
    print(f"   SOH shape: {soh.shape}, RUL shape: {rul.shape}")
    
    # Add some entries to memory bank
    print("\n2. Adding entries to memory bank:")
    for profile in PROFILE_TYPES:
        model.add_to_memory(
            x=torch.randn(1, 30, 10),
            soh=torch.linspace(1.0, 0.7, 30).unsqueeze(0),
            rul=torch.log1p(torch.linspace(100, 0, 30)).unsqueeze(0),
            context=profile,
            mask=torch.ones(1, 30, dtype=torch.bool),
            cell_id=f"test_{profile}",
            cycle_stride=10
        )
    print(f"   Memory bank size: {model.memory_bank.size()}")
    print(f"   Stats: {model.get_memory_stats()}")
    
    # Test forward pass with context
    print("\n3. Forward pass with context (Aggressive):")
    context = create_context_tensor('Aggressive', batch_size=2)
    soh, rul = model(x, mask, context=context, use_retrieval=True)
    print(f"   SOH shape: {soh.shape}, RUL shape: {rul.shape}")
    
    # Test forward pass with mixed contexts
    print("\n4. Forward pass with mixed contexts:")
    context = create_context_tensor(['Aggressive', 'Eco'])
    soh, rul = model(x, mask, context=context, use_retrieval=True)
    print(f"   SOH shape: {soh.shape}, RUL shape: {rul.shape}")
    
    # Test fallback (no context)
    print("\n5. Forward pass with retrieval but no context (fallback):")
    soh, rul = model(x, mask, context=None, use_retrieval=True)
    print(f"   SOH shape: {soh.shape}, RUL shape: {rul.shape}")
    
    print("\nAll tests passed!")

