"""
Training Script for Causal Degradation Attribution Model

Trains the multi-head attribution model with:
1. Physics-constrained loss (attributions sum to total loss)
2. Physics prior regularization (mechanisms align with known causes)
3. SOH prediction loss

Author: Battery ML Research
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Direct import to avoid __init__.py issues
from src.models.causal_attribution import (
    CausalAttributionModel, 
    DegradationMechanism,
    CausalExplainer,
    AttributionResult
)
from src.data.unified_pipeline import UnifiedDataPipeline, UnifiedBatteryDataset


# Physics-Constrained Loss Function

class CausalAttributionLoss(nn.Module):
    """
    Physics-constrained loss for causal attribution.
    
    Components:
    1. SOH prediction loss
    2. Attribution consistency loss (sum to total loss)
    3. Physics prior regularization
    4. Sparsity regularization (encourage dominant mechanisms)
    """
    
    def __init__(
        self,
        soh_weight: float = 1.0,
        consistency_weight: float = 0.5,
        prior_weight: float = 0.3,
        sparsity_weight: float = 0.1,
    ):
        super().__init__()
        self.soh_weight = soh_weight
        self.consistency_weight = consistency_weight
        self.prior_weight = prior_weight
        self.sparsity_weight = sparsity_weight
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        soh_target: torch.Tensor,
        context: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss.
        
        Args:
            outputs: Model outputs dict
            soh_target: Ground truth SOH
            context: Context vector (for physics prior computation)
        
        Returns:
            Total loss and component losses dict
        """
        losses = {}
        
        # 1. SOH prediction loss
        soh_pred = outputs['soh']
        soh_loss = F.mse_loss(soh_pred, soh_target)
        losses['soh'] = soh_loss.item()
        
        # 2. Attribution consistency loss
        # (Attributions should sum to total loss - already enforced in model)
        # This is a soft regularization for training stability
        total_loss = outputs['total_loss']
        mechanism_sum = sum(
            outputs['attributions'][name] 
            for name in outputs['attributions']
        )
        consistency_loss = F.mse_loss(mechanism_sum, total_loss)
        losses['consistency'] = consistency_loss.item()
        
        # 3. Physics prior regularization
        # Encourage attributions to align with physics priors
        prior_loss = self._physics_prior_loss(outputs, context)
        losses['prior'] = prior_loss.item()
        
        # 4. Sparsity regularization (encourage 1-2 dominant mechanisms)
        # Using entropy-like regularization
        pcts = torch.stack([
            outputs['attributions_pct'][name] 
            for name in outputs['attributions_pct']
        ], dim=-1)
        sparsity_loss = -torch.mean(torch.sum(pcts * torch.log(pcts + 1e-8), dim=-1))
        losses['sparsity'] = sparsity_loss.item()
        
        # Total loss
        total = (
            self.soh_weight * soh_loss +
            self.consistency_weight * consistency_loss +
            self.prior_weight * prior_loss +
            self.sparsity_weight * sparsity_loss
        )
        losses['total'] = total.item()
        
        return total, losses
    
    def _physics_prior_loss(
        self, 
        outputs: Dict[str, torch.Tensor],
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute physics prior alignment loss.
        
        Encourages:
        - High temp → SEI growth, electrolyte decomp
        - Low temp + fast charge → lithium plating
        - High discharge rate → active material loss
        - Low SOC storage → collector corrosion
        """
        loss = torch.tensor(0.0, device=context.device)
        batch_size = context.shape[0]
        
        # Temperature (context[0])
        temp = context[:, 0]
        high_temp_mask = (temp > 0.35).float()
        low_temp_mask = (temp < 0.15).float()
        
        # Charge rate (context[1])
        charge_rate = context[:, 1]
        fast_charge_mask = (charge_rate > 0.8).float()
        
        # Discharge rate (context[2])
        discharge_rate = context[:, 2]
        high_discharge_mask = (discharge_rate > 0.8).float()
        
        # SOC (context[3])
        soc = context[:, 3]
        low_soc_mask = (soc < 0.25).float()
        high_soc_mask = (soc > 0.75).float()
        
        # Mode (context[5])
        storage_mode = context[:, 5]
        
        # Prior 1: High temp should increase SEI and electrolyte
        sei_pct = outputs['attributions_pct'][DegradationMechanism.SEI_GROWTH]
        elec_pct = outputs['attributions_pct'][DegradationMechanism.ELECTROLYTE_DECOMP]
        high_temp_prior = -torch.mean(high_temp_mask * (sei_pct + elec_pct))
        
        # Prior 2: Low temp + fast charge should increase plating
        plating_pct = outputs['attributions_pct'][DegradationMechanism.LITHIUM_PLATING]
        plating_prior = -torch.mean(low_temp_mask * fast_charge_mask * plating_pct)
        
        # Prior 3: High discharge should increase active material loss
        am_pct = outputs['attributions_pct'][DegradationMechanism.ACTIVE_MATERIAL_LOSS]
        am_prior = -torch.mean(high_discharge_mask * am_pct)
        
        # Prior 4: Low SOC storage should increase corrosion
        corr_pct = outputs['attributions_pct'][DegradationMechanism.COLLECTOR_CORROSION]
        corr_prior = -torch.mean(low_soc_mask * storage_mode * corr_pct)
        
        # Prior 5: Storage mode should increase SEI
        storage_sei_prior = -torch.mean(storage_mode * high_soc_mask * sei_pct)
        
        loss = high_temp_prior + plating_prior + am_prior + corr_prior + storage_sei_prior
        
        return loss


# Training Function

def train_causal_model(
    data_root: str = "data",
    output_dir: str = "reports/causal_attribution",
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
) -> Dict:
    """Train the causal attribution model."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("CAUSAL DEGRADATION ATTRIBUTION - TRAINING")
    print("=" * 70)
    
    # Load data using load_datasets method
    print("\n[1/4] Loading data...")
    pipeline = UnifiedDataPipeline(data_root=data_root, use_lithium_features=False)
    pipeline.load_datasets(['nasa', 'calce', 'oxford', 'storage_degradation'])
    
    # Prepare training data
    features_list = []
    contexts_list = []
    sohs_list = []
    
    for sample in pipeline.samples:
        # Skip invalid samples
        if not np.isfinite(sample.soh) or sample.soh < 0.5 or sample.soh > 1.1:
            continue
        if not np.all(np.isfinite(sample.features[:9])):
            continue
            
        features_list.append(sample.features[:9])  # First 9 features
        
        # Determine storage mode
        is_storage = 'storage' in sample.source_dataset.lower()
        
        context = np.array([
            sample.context_vector[0] if len(sample.context_vector) > 0 else 0.25,  # temp
            sample.context_vector[1] if len(sample.context_vector) > 1 else 0.5,   # charge rate
            sample.context_vector[2] if len(sample.context_vector) > 2 else 0.5,   # discharge rate
            sample.context_vector[3] if len(sample.context_vector) > 3 else 0.5,   # soc
            sample.context_vector[4] if len(sample.context_vector) > 4 else 0.0,   # profile
            1.0 if is_storage else 0.0,                                            # mode
        ], dtype=np.float32)
        contexts_list.append(context)
        sohs_list.append(np.clip(sample.soh, 0.5, 1.0))
    
    features = np.stack(features_list)
    contexts = np.stack(contexts_list)
    sohs = np.array(sohs_list, dtype=np.float32)
    
    # Handle NaNs
    features = np.nan_to_num(features, nan=0.0)
    contexts = np.nan_to_num(contexts, nan=0.0)
    
    print(f"  Loaded {len(features)} valid samples")
    print(f"  SOH range: {sohs.min():.2%} - {sohs.max():.2%}")
    
    # Create dataset
    features_t = torch.tensor(features, dtype=torch.float32)
    contexts_t = torch.tensor(contexts, dtype=torch.float32)
    sohs_t = torch.tensor(sohs, dtype=torch.float32)
    
    dataset = TensorDataset(features_t, contexts_t, sohs_t)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}")
    
    # Create model
    print("\n[2/4] Creating model...")
    model = CausalAttributionModel(
        feature_dim=9,
        context_dim=6,
        hidden_dim=128,
        n_mechanisms=5,
    ).to(device)
    
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    criterion = CausalAttributionLoss(
        soh_weight=1.0,
        consistency_weight=0.5,
        prior_weight=0.3,
        sparsity_weight=0.1,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    # Training loop
    print("\n[3/4] Training...")
    history = {'train_loss': [], 'val_loss': [], 'soh_mae': []}
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        
        for feat, ctx, soh in train_loader:
            feat, ctx, soh = feat.to(device), ctx.to(device), soh.to(device)
            
            optimizer.zero_grad()
            outputs = model(feat, ctx)
            loss, loss_components = criterion(outputs, soh, ctx)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        scheduler.step()
        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)
        
        # Validation
        model.eval()
        val_losses = []
        soh_errors = []
        
        with torch.no_grad():
            for feat, ctx, soh in val_loader:
                feat, ctx, soh = feat.to(device), ctx.to(device), soh.to(device)
                outputs = model(feat, ctx)
                loss, _ = criterion(outputs, soh, ctx)
                val_losses.append(loss.item())
                soh_errors.extend(torch.abs(outputs['soh'] - soh).cpu().numpy())
        
        avg_val_loss = np.mean(val_losses)
        soh_mae = np.mean(soh_errors)
        history['val_loss'].append(avg_val_loss)
        history['soh_mae'].append(soh_mae)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), output_path / "causal_model.pt")
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}, SOH MAE={soh_mae:.4f}")
    
    # Load best model
    model.load_state_dict(torch.load(output_path / "causal_model.pt", weights_only=False))
    
    # Final evaluation
    print("\n[4/4] Evaluating...")
    model.eval()
    explainer = CausalExplainer(model)
    
    # Test on various scenarios
    test_scenarios = [
        {
            "name": "High Temp Storage",
            "features": np.array([0.1, 0.2, 0.85, 0.3, 0.5, 0.05, 0.06, 0.04, 0.40], dtype=np.float32),
            "context": np.array([0.40, 0.0, 0.0, 0.85, 1.0, 1.0], dtype=np.float32),
            "expected_primary": "SEI"
        },
        {
            "name": "Fast Charge in Cold",
            "features": np.array([0.12, 0.28, 0.82, 0.32, 0.4, 0.10, 0.10, 0.08, 0.05], dtype=np.float32),
            "context": np.array([0.05, 1.5, 0.5, 0.5, 0.0, 0.0], dtype=np.float32),
            "expected_primary": "Plating"
        },
        {
            "name": "Heavy Cycling",
            "features": np.array([0.15, 0.35, 0.78, 0.38, 0.45, 0.14, 0.14, 0.12, 0.30], dtype=np.float32),
            "context": np.array([0.30, 0.8, 1.5, 0.4, 0.0, 0.0], dtype=np.float32),
            "expected_primary": "Material"
        },
    ]
    
    print("\nScenario Tests:")
    for scenario in test_scenarios:
        result = explainer.explain(scenario['features'], scenario['context'])
        print(f"\n  {scenario['name']}:")
        print(f"    Primary: {result.primary_mechanism}")
        print(f"    SEI: {result.sei_growth:.1f}%, Plating: {result.lithium_plating:.1f}%, AM: {result.active_material_loss:.1f}%")
    
    # Save results
    results = {
        'epochs': epochs,
        'final_train_loss': float(history['train_loss'][-1]),
        'final_val_loss': float(history['val_loss'][-1]),
        'final_soh_mae': float(history['soh_mae'][-1]),
        'best_val_loss': float(best_val_loss),
    }
    
    with open(output_path / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot training curve
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    
    axes[1].plot(history['soh_mae'])
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('SOH MAE')
    
    plt.tight_layout()
    plt.savefig(output_path / "training_curves.png", dpi=150)
    plt.close()
    
    print(f"\n Training complete! Model saved to {output_path}")
    print(f"  Final SOH MAE: {history['soh_mae'][-1]:.4f}")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    
    train_causal_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
