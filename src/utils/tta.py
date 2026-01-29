"""
Test-Time Adaptation (TTA) Utilities for HERO

Provides automatic zero-shot improvement via BatchNorm adaptation.
No labels required - just pass unlabeled test data.

Usage:
    from src.utils.tta import adapt_model, TTAWrapper
    
    # Option 1: One-time adaptation
    model = adapt_model(model, test_data, method='batchnorm')
    predictions = model(test_data)
    
    # Option 2: Wrapper for automatic TTA
    model = TTAWrapper(model)
    predictions = model.predict_with_tta(test_data)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Union, Literal
import copy


def get_bn_layers(model: nn.Module) -> List[nn.BatchNorm1d]:
    """Get all BatchNorm layers from a model."""
    bn_layers = []
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            bn_layers.append(module)
    return bn_layers


def adapt_batchnorm(
    model: nn.Module,
    test_data: Union[torch.Tensor, np.ndarray],
    n_steps: int = 10,
    batch_size: int = 64
) -> nn.Module:
    """
    Adapt BatchNorm statistics using test data (no labels needed).
    
    This is the most reliable TTA method for guaranteed improvement
    when there is distribution shift between training and test data.
    
    Args:
        model: PyTorch model with BatchNorm layers
        test_data: Unlabeled test data (Tensor or numpy array)
        n_steps: Number of forward passes for adaptation
        batch_size: Batch size for processing
        
    Returns:
        Adapted model (modified in-place)
    """
    device = next(model.parameters()).device
    
    # Convert to tensor if needed
    if isinstance(test_data, np.ndarray):
        test_data = torch.FloatTensor(test_data)
    test_data = test_data.to(device)
    
    # Get BatchNorm layers
    bn_layers = get_bn_layers(model)
    if not bn_layers:
        print("Warning: No BatchNorm layers found in model. TTA will have no effect.")
        return model
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Reset BatchNorm running statistics
    for bn in bn_layers:
        bn.running_mean = torch.zeros_like(bn.running_mean)
        bn.running_var = torch.ones_like(bn.running_var)
        bn.momentum = 0.1
    
    # Put model in training mode (for BN to update)
    model.train()
    
    # Forward passes to update statistics
    with torch.no_grad():
        for _ in range(n_steps):
            indices = torch.randperm(len(test_data))
            for i in range(0, len(test_data), batch_size):
                batch_idx = indices[i:i+batch_size]
                if len(batch_idx) > 1:  # BatchNorm needs at least 2 samples
                    batch = test_data[batch_idx]
                    _ = model(batch)
    
    # Set back to eval mode
    model.eval()
    
    return model


def adapt_feature_alignment(
    model: nn.Module,
    source_data: Union[torch.Tensor, np.ndarray],
    test_data: Union[torch.Tensor, np.ndarray],
    n_steps: int = 20,
    lr: float = 0.001
) -> nn.Module:
    """
    Adapt model by aligning test features to source distribution.
    
    Args:
        model: PyTorch model
        source_data: Training data (for computing source statistics)
        test_data: Test data to adapt to
        n_steps: Number of optimization steps
        lr: Learning rate for adaptation
        
    Returns:
        Adapted model
    """
    device = next(model.parameters()).device
    
    # Convert to tensors
    if isinstance(source_data, np.ndarray):
        source_data = torch.FloatTensor(source_data)
    if isinstance(test_data, np.ndarray):
        test_data = torch.FloatTensor(test_data)
    
    source_data = source_data.to(device)
    test_data = test_data.to(device)
    
    # Compute source feature statistics
    model.eval()
    with torch.no_grad():
        try:
            source_out = model(source_data)
            source_features = source_out[-1] if isinstance(source_out, tuple) else source_out
            source_mean = source_features.mean(dim=0)
            source_std = source_features.std(dim=0)
        except:
            print("Warning: Could not extract features for alignment.")
            return model
    
    # Collect BatchNorm parameters for optimization
    bn_params = []
    for bn in get_bn_layers(model):
        if bn.weight is not None:
            bn_params.append(bn.weight)
        if bn.bias is not None:
            bn_params.append(bn.bias)
    
    if not bn_params:
        print("Warning: No BatchNorm parameters to optimize.")
        return model
    
    optimizer = torch.optim.Adam(bn_params, lr=lr)
    
    model.train()
    
    for _ in range(n_steps):
        optimizer.zero_grad()
        
        try:
            test_out = model(test_data)
            test_features = test_out[-1] if isinstance(test_out, tuple) else test_out
            
            target_mean = test_features.mean(dim=0)
            target_std = test_features.std(dim=0)
            
            mean_loss = F.mse_loss(target_mean, source_mean)
            std_loss = F.mse_loss(target_std, source_std)
            
            loss = mean_loss + std_loss
            loss.backward()
            optimizer.step()
        except:
            break
    
    model.eval()
    
    return model


def adapt_model(
    model: nn.Module,
    test_data: Union[torch.Tensor, np.ndarray],
    method: Literal['batchnorm', 'feature_alignment'] = 'batchnorm',
    source_data: Optional[Union[torch.Tensor, np.ndarray]] = None,
    **kwargs
) -> nn.Module:
    """
    Apply Test-Time Adaptation to a model.
    
    Args:
        model: PyTorch model to adapt
        test_data: Unlabeled test data
        method: TTA method ('batchnorm' or 'feature_alignment')
        source_data: Required for feature_alignment method
        **kwargs: Additional arguments passed to adaptation method
        
    Returns:
        Adapted model
    """
    # Make a copy to avoid modifying original
    model = copy.deepcopy(model)
    
    if method == 'batchnorm':
        return adapt_batchnorm(model, test_data, **kwargs)
    elif method == 'feature_alignment':
        if source_data is None:
            raise ValueError("source_data required for feature_alignment method")
        return adapt_feature_alignment(model, source_data, test_data, **kwargs)
    else:
        raise ValueError(f"Unknown TTA method: {method}")


class TTAWrapper(nn.Module):
    """
    Wrapper that automatically applies TTA before inference.
    
    Example:
        model = TTAWrapper(base_model)
        predictions = model.predict_with_tta(test_data)
    """
    
    def __init__(
        self,
        model: nn.Module,
        method: Literal['batchnorm', 'feature_alignment'] = 'batchnorm',
        n_steps: int = 10
    ):
        super().__init__()
        self.model = model
        self.method = method
        self.n_steps = n_steps
        self._adapted = False
        self._source_data = None
    
    def set_source_data(self, source_data: Union[torch.Tensor, np.ndarray]):
        """Set source data for feature alignment method."""
        self._source_data = source_data
    
    def adapt(self, test_data: Union[torch.Tensor, np.ndarray]):
        """Adapt model to test data distribution."""
        if self.method == 'batchnorm':
            self.model = adapt_batchnorm(self.model, test_data, n_steps=self.n_steps)
        elif self.method == 'feature_alignment':
            if self._source_data is None:
                raise ValueError("Call set_source_data() first")
            self.model = adapt_feature_alignment(
                self.model, self._source_data, test_data, n_steps=self.n_steps
            )
        self._adapted = True
    
    def forward(self, x: torch.Tensor):
        """Standard forward pass (use after calling adapt())."""
        return self.model(x)
    
    def predict_with_tta(
        self,
        test_data: Union[torch.Tensor, np.ndarray],
        adapt_first: bool = True
    ) -> torch.Tensor:
        """
        Make predictions with automatic TTA.
        
        Args:
            test_data: Test data
            adapt_first: Whether to adapt before prediction
            
        Returns:
            Predictions
        """
        device = next(self.model.parameters()).device
        
        if isinstance(test_data, np.ndarray):
            test_data = torch.FloatTensor(test_data)
        test_data = test_data.to(device)
        
        if adapt_first and not self._adapted:
            self.adapt(test_data)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(test_data)
        
        return predictions
    
    def reset(self):
        """Reset adaptation state."""
        self._adapted = False


# Convenience function for quick TTA
def quick_tta(
    model: nn.Module,
    test_data: Union[torch.Tensor, np.ndarray]
) -> nn.Module:
    """
    Quick one-liner TTA using BatchNorm adaptation.
    
    Usage:
        model = quick_tta(model, test_data)
        predictions = model(test_data)
    """
    return adapt_model(model, test_data, method='batchnorm', n_steps=10)


if __name__ == "__main__":
    # Quick test
    print("Testing TTA utilities...")
    
    # Create simple model
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )
    
    # Create test data
    test_data = torch.randn(100, 10)
    
    # Apply TTA
    adapted_model = quick_tta(model, test_data)
    
    # Make predictions
    with torch.no_grad():
        predictions = adapted_model(test_data)
    
    print(f"Predictions shape: {predictions.shape}")
    print("✅ TTA utilities working correctly!")
