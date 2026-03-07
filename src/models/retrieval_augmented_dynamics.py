"""
Retrieval Augmented Dynamics (RAD) model for battery SOH prediction.

Maintains a memory bank of historical degradation trajectories. During inference,
retrieves the most similar historical cases and fuses them via cross-attention
to improve SOH and RUL estimates.
"""
import math
from typing import Optional, Tuple, List, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TrajectoryEncoder(nn.Module):
    """
    Encodes battery trajectory sequences into latent representations.
    Uses GRU for temporal encoding with optional multi-scale features.
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
        
        # Temporal encoder (GRU)
        self.gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
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
            x: (B, T, F) input features
            mask: (B, T) boolean mask (True = valid)
        Returns:
            latent: (B, latent_dim) trajectory representation
        """
        # Project input
        x = self.input_proj(x)  # (B, T, hidden_dim)
        
        # Encode with GRU
        # Use last valid timestep for representation
        if mask is not None:
            # Get lengths for each sequence
            lengths = mask.sum(dim=1).long()  # (B,)
            lengths = torch.clamp(lengths, min=1)
            
            # Pack sequence for efficient processing
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            output, hidden = self.gru(packed)
            # hidden: (num_layers, B, hidden_dim)
            # Use last layer's hidden state
            h = hidden[-1]  # (B, hidden_dim)
        else:
            # Use last timestep
            output, hidden = self.gru(x)
            h = hidden[-1]  # (B, hidden_dim)
        
        # Project to latent space
        latent = self.latent_proj(h)  # (B, latent_dim)
        
        # L2 normalize for cosine similarity
        latent = F.normalize(latent, p=2, dim=1)
        
        return latent


class MemoryBank:
    """
    Non-parametric memory bank storing historical battery trajectories.
    Acts as a vector database for retrieval.
    """
    def __init__(self, latent_dim: int, device: str = 'cpu'):
        self.latent_dim = latent_dim
        self.device = device
        self.memory_keys = []  # List of (B, latent_dim) tensors
        
        # Store data in lists first, then stack when full
        self.storage_soh = []
        self.storage_rul = []
        self.storage_feats = []
        
        self.soh_values = [] # List of scalar SOH values for fast filtering
        self.cell_ids = []  # Track which cell each memory belongs to
        
        # Cache for vectorized operations
        self.cached_memory_stack = None
        self.cached_soh_values = None
        
    def _update_cache(self):
        """Update cached tensors for fast retrieval."""
        if len(self.memory_keys) > 0:
            # Stack on CPU then move to device
            self.cached_memory_stack = torch.stack(self.memory_keys).to(self.device)
            self.cached_soh_values = torch.tensor(self.soh_values, device=self.device)
        else:
            self.cached_memory_stack = None
            self.cached_soh_values = None
            
    def add(self, latent: torch.Tensor, soh: torch.Tensor, rul: torch.Tensor,
            trajectory_features: torch.Tensor, cell_id: Optional[str] = None, 
            store_linear_rul: bool = True):
        """
        Add trajectory to memory bank.
        """
        # Invalidate cache
        self.cached_memory_stack = None
        self.cached_soh_values = None
        
        # Handle both batched and unbatched inputs
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
        if soh.dim() == 1:
            soh = soh.unsqueeze(0)
        if rul.dim() == 1:
            rul = rul.unsqueeze(0)
        if trajectory_features.dim() == 2:
            trajectory_features = trajectory_features.unsqueeze(0)
        
        # Convert RUL from log-space to linear if requested
        if store_linear_rul:
            rul_linear = torch.expm1(rul.clamp(min=-1.0))
        else:
            rul_linear = rul
        
        B = latent.size(0)
        # Move to CPU to save GPU memory
        latent = latent.detach().cpu()
        soh = soh.detach().cpu()
        rul_linear = rul_linear.detach().cpu()
        trajectory_features = trajectory_features.detach().cpu()
        
        for i in range(B):
            self.memory_keys.append(latent[i])
            
            # Calculate mean SOH for this entry for fast index
            soh_val = soh[i]
            soh_mean = soh_val.mean().item() if soh_val.numel() > 0 else 0.0
            self.soh_values.append(soh_mean)
            
            # Store compact representations
            self.storage_soh.append(soh[i])
            self.storage_rul.append(rul_linear[i])
            # Don't store features to save memory
            self.storage_feats.append(torch.empty(0))
            self.cell_ids.append(cell_id if cell_id is not None else f"cell_{len(self.memory_keys)-1}")
    
    def retrieve(self, query: torch.Tensor, k: int = 5, 
                 soh_constraint: Optional[torch.Tensor] = None,
                 soh_tolerance: float = 0.05) -> Tuple[torch.Tensor, Dict[str, List[torch.Tensor]]]:
        """
        Retrieve k most similar trajectories from memory bank.
        Returns batched tensors/lists.
        """
        if len(self.memory_keys) == 0:
            B = query.size(0)
            dummy_latent = torch.zeros(self.latent_dim, device=query.device)
            return (
                dummy_latent.unsqueeze(0).unsqueeze(0).repeat(B, k, 1),
                {'soh': [], 'rul': [], 'trajectory_features': []}
            )
        
        # Check/Update cache
        if self.cached_memory_stack is None:
            self._update_cache()
            
        memory_stack = self.cached_memory_stack
        soh_values_tensor = self.cached_soh_values
        
        # Ensure tensors are on the correct device
        if memory_stack.device != query.device:
            memory_stack = memory_stack.to(query.device)
            soh_values_tensor = soh_values_tensor.to(query.device)
            self.cached_memory_stack = memory_stack
            self.cached_soh_values = soh_values_tensor
        
        # Compute cosine similarity
        similarity = torch.matmul(query, memory_stack.t())  # (B, M)
        
        # Apply SOH constraint
        if soh_constraint is not None:
            if soh_constraint.dim() > 1:
                soh_constraint = soh_constraint.mean(dim=-1)
            
            # Vectorized filtering
            soh_diff = torch.abs(soh_constraint.unsqueeze(1) - soh_values_tensor.unsqueeze(0))
            
            # Create mask
            valid_mask = soh_diff <= soh_tolerance
            
            # Relax constraint if needed
            valid_counts = valid_mask.sum(dim=1)
            rows_to_relax = valid_counts < k
            
            if rows_to_relax.any():
                relaxed_mask = soh_diff <= (soh_tolerance * 2)
                valid_mask[rows_to_relax] = relaxed_mask[rows_to_relax]
            
            similarity = similarity.masked_fill(~valid_mask, float('-inf'))
        
        # Get top-k indices
        _, topk_indices = torch.topk(similarity, k=min(k, len(self.memory_keys)), dim=1)  # (B, k)
        
        # Retrieve corresponding latents and data
        B = query.size(0)
        topk_indices_cpu = topk_indices.cpu()
        
        retrieved_latents_list = []
        retrieved_soh = [] 
        retrieved_rul = []
        retrieved_feats = []
        
        for b in range(B):
            indices = topk_indices_cpu[b]
            
            # Latents
            latents = memory_stack[indices] # (k, D)
            retrieved_latents_list.append(latents)
            
            # Data
            batch_soh = []
            batch_rul = []
            batch_feats = []
            
            for idx in indices:
                idx_val = idx.item()
                batch_soh.append(self.storage_soh[idx_val])
                batch_rul.append(self.storage_rul[idx_val])
                batch_feats.append(self.storage_feats[idx_val])
                
            retrieved_soh.append(batch_soh)
            retrieved_rul.append(batch_rul)
            retrieved_feats.append(batch_feats)
            
        retrieved_latents = torch.stack(retrieved_latents_list) # (B, k, D)
        
        return retrieved_latents, {
            'soh': retrieved_soh,
            'rul': retrieved_rul,
            'trajectory_features': retrieved_feats
        }
    
    def clear(self):
        """Clear the memory bank."""
        self.memory_keys = []
        self.storage_soh = []
        self.storage_rul = []
        self.storage_feats = []
        self.soh_values = []
        self.cell_ids = []
        self.cached_memory_stack = None
        self.cached_soh_values = None
    
    def size(self) -> int:
        """Return number of trajectories in memory bank."""
        return len(self.memory_keys)


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
        
        # Project retrieved latents to hidden dimension
        self.retrieval_proj = nn.Linear(latent_dim, hidden_dim)
        
        # Cross-attention: current trajectory attends to retrieved trajectories
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network for refinement
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
            # DEBUG MODE: Simple average fusion
            retrieved_proj = self.retrieval_proj(retrieved_latents)  # (B, k, hidden_dim)
            retrieved_mean = retrieved_proj.mean(dim=1, keepdim=True)  # (B, 1, hidden_dim)
            retrieved_mean = retrieved_mean.expand(-1, T, -1)  # (B, T, hidden_dim)
            fused = 0.5 * current_features + 0.5 * retrieved_mean
            return fused
        
        # Normal attention-based fusion
        retrieved_proj = self.retrieval_proj(retrieved_latents)  # (B, k, hidden_dim)
        current_query = current_features  # (B, T, hidden_dim)
        
        fused, attn_weights = self.cross_attention(
            query=current_query,
            key=retrieved_proj,
            value=retrieved_proj
        )
        
        fused = fused + current_features
        fused = self.ffn(fused) + fused
        
        return fused


class RADModel(nn.Module):
    """
    Retrieval Augmented Dynamics (RAD) Model
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
        device: str = 'cpu'
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.retrieval_k = retrieval_k
        self.device = device
        
        # Trajectory encoder
        self.encoder = TrajectoryEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Feature encoder for current trajectory - BIDIRECTIONAL for better context
        self.feature_proj = nn.Linear(input_dim, hidden_dim)
        self.feature_encoder = nn.GRU(
            hidden_dim,
            hidden_dim // 2,  # Half because bidirectional doubles it
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True  # IMPROVEMENT: Bidirectional for better context
        )
        
        # Retrieval attention fusion
        self.fusion = RetrievalAttentionFusion(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Multi-task prediction heads - IMPROVED with deeper networks
        self.head_soh = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  # hidden_dim is now 2x due to bidirectional
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # RUL head takes SOH as input for better correlation (IMPROVEMENT)
        self.head_rul = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),  # +1 for SOH input
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Memory bank
        self.memory_bank = MemoryBank(latent_dim=latent_dim, device=device)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        use_retrieval: bool = True,
        update_memory: bool = False,
        debug_fusion: bool = False,
        target_soh: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with retrieval augmentation.
        
        Args:
            x: (B, T, F) input features
            mask: (B, T) boolean mask (True = valid)
            use_retrieval: Whether to use retrieval augmentation
            update_memory: Whether to update memory bank (for training)
            debug_fusion: If True, use simple averaging instead of attention (for debugging)
            target_soh: (B, T) Optional True SOH targets for "Teacher Forcing" during retrieval
            
        Returns:
            soh_pred: (B, T) SOH predictions
            rul_pred: (B, T) RUL predictions (log-space)
        """
        B, T, F = x.shape
        
        # Encode current trajectory to latent space for retrieval
        current_latent = self.encoder(x, mask)  # (B, latent_dim)
        
        # Encode current trajectory features for prediction
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
            encoded_features, _ = self.feature_encoder(x_proj)  # (B, T, hidden_dim)
        
        # Retrieve similar trajectories from memory bank
        retrieved_data_dict = None
        if use_retrieval and self.memory_bank.size() > 0:
            prelim_soh = self.head_soh(encoded_features).squeeze(-1)  # (B, T)
            
            # TEACHER FORCING: Use True SOH if provided (during training) to guide retrieval
            soh_for_retrieval = target_soh if target_soh is not None else prelim_soh
            
            try:
                retrieved_latents, retrieved_data_dict = self.memory_bank.retrieve(
                    current_latent, 
                    k=self.retrieval_k,
                    soh_constraint=soh_for_retrieval,
                    soh_tolerance=0.05
                )
            except Exception as e:
                print(f"Retrieval failed: {e}")
                retrieved_latents = torch.zeros(B, self.retrieval_k, self.latent_dim, device=x.device)
                retrieved_data_dict = None
            
            if retrieved_data_dict is not None:
                fused_features = self.fusion(
                    current_latent=current_latent,
                    retrieved_latents=retrieved_latents,
                    current_features=encoded_features,
                    debug_mode=debug_fusion
                )
            else:
                fused_features = encoded_features
        else:
            fused_features = encoded_features
        
        # Multi-task predictions
        soh_pred = self.head_soh(fused_features).squeeze(-1)  # (B, T)
        
        # IMPROVEMENT: Feed SOH into RUL head for better correlation
        soh_input = soh_pred.unsqueeze(-1)  # (B, T, 1)
        rul_input = torch.cat([fused_features, soh_input], dim=-1)  # (B, T, hidden_dim + 1)
        rul_pred = self.head_rul(rul_input).squeeze(-1)  # (B, T)
        
        # DEBUG MODE
        if debug_fusion and use_retrieval and self.memory_bank.size() > 0 and retrieved_data_dict is not None:
            # Extract retrieved RUL values
            for b in range(B):
                batch_rul_list = retrieved_data_dict['rul'][b] # List[Tensor]
                
                retrieved_rul_log_values = []
                for rul_val in batch_rul_list:
                    if isinstance(rul_val, torch.Tensor):
                        if rul_val.numel() == 1:
                            retrieved_rul_log_values.append(rul_val.item())
                        elif rul_val.dim() == 1 and rul_val.numel() > 1:
                            retrieved_rul_log_values.append(rul_val.mean().item())
                        else:
                            retrieved_rul_log_values.append(rul_val.flatten().mean().item())
                
                if len(retrieved_rul_log_values) > 0:
                    retrieved_rul_log_mean = np.mean(retrieved_rul_log_values)
                    retrieved_rul_mean_cycles = np.expm1(retrieved_rul_log_mean)
                    
                    rul_pred_cycles = torch.expm1(rul_pred[b])
                    blended_rul_cycles = 0.5 * rul_pred_cycles + 0.5 * retrieved_rul_mean_cycles
                    rul_pred[b] = torch.log1p(blended_rul_cycles.clamp(min=0.0))
        
        return soh_pred, rul_pred
    
    def update_memory_bank(
        self,
        x: torch.Tensor,
        soh: torch.Tensor,
        rul: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cell_ids: Optional[List[str]] = None,
        store_per_cycle: bool = True
    ):
        """
        Update memory bank with new trajectories.
        """
        with torch.no_grad():
            if cell_ids is None:
                cell_ids = [None] * x.size(0)
            
            if store_per_cycle:
                for i in range(x.size(0)):
                    T = x.size(1)
                    m = mask[i] if mask is not None else torch.ones(T, dtype=torch.bool, device=x.device)
                    
                    for t in range(0, T, 2): # STRIDE: 2 for density but save some memory
                        if m[t]:
                            window = min(10, t + 1)
                            start_idx = max(0, t - window + 1)
                            
                            x_context = x[i:i+1, start_idx:t+1, :]
                            m_context = m[start_idx:t+1]
                            
                            latent = self.encoder(x_context, m_context.unsqueeze(0))
                            
                            self.memory_bank.add(
                                latent=latent,
                                soh=soh[i:i+1, t:t+1],
                                rul=rul[i:i+1, t:t+1],
                                trajectory_features=x[i:i+1, t:t+1, :],
                                cell_id=f"{cell_ids[i]}_cycle_{t}" if cell_ids[i] is not None else f"cell_{i}_cycle_{t}",
                                store_linear_rul=True
                            )
                            # DEBUG: Check stored RUL
                            if self.memory_bank.size() % 1000 == 0:
                                stored_val = self.memory_bank.storage_rul[-1]
                                input_val = rul[i, t]
                                print(f"Stored RUL: {stored_val.item():.2f} (from input log: {input_val.item():.2f})")
            else:
                latent = self.encoder(x, mask)
                
                for i in range(x.size(0)):
                    self.memory_bank.add(
                        latent=latent[i:i+1],
                        soh=soh[i:i+1],
                        rul=rul[i:i+1],
                        trajectory_features=x[i:i+1],
                        cell_id=cell_ids[i] if i < len(cell_ids) else None,
                        store_linear_rul=True
                    )
    
    def clear_memory_bank(self):
        """Clear the memory bank."""
        self.memory_bank.clear()
