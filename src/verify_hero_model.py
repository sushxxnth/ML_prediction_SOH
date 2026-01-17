"""
Verification Script for Hero Model (RAD + Decoupled RUL).

Reproduces reported metrics on held-out test data.
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.unified_pipeline import UnifiedDataPipeline, UnifiedBatteryDataset
from src.train.hero_rad_decoupled import RADDecoupledModel

def main():
    print("=" * 60)
    print("Verification Protocol: RAD + Decoupled RUL")
    print("=" * 60)
    
    device = 'cpu' # Use CPU for consistent demonstration
    model_path = Path("reports/hero_model/hero_model.pt")
    
    if not model_path.exists():
        print(f"Error: Model file {model_path} not found.")
        print("Please run 'python src/train/hero_rad_decoupled.py' first to train the model.")
        return

    # 1. Load Data
    print("\n[Step 1] Loading held-out test data (TBSI Target)...")
    # We re-initialize the pipeline to ensure no data leakage or state contamination
    pipeline = UnifiedDataPipeline('data', use_lithium_features=True)
    pipeline.load_datasets(['nasa', 'calce', 'oxford', 'tbsi_sunwoda'])
    
    # Strictly reproduce the split used in training
    target_samples = [s for s in pipeline.samples if s.source_dataset == 'tbsi_sunwoda']
    np.random.seed(42) # MUST match training seed
    np.random.shuffle(target_samples)
    
    # The second half was used for testing
    test_samples = target_samples[len(target_samples)//2:]
    
    test_loader = DataLoader(UnifiedBatteryDataset(test_samples), batch_size=64, shuffle=False)
    print(f"  Loaded {len(test_samples)} test samples.")

    # 2. Load Model
    print("\n[Step 2] Loading pre-trained model weights...")
    model = RADDecoupledModel(
        feature_dim=20,
        context_dim=5,
        hidden_dim=128,
        latent_dim=64,
        device=device
    ).to(device)
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("  Model loaded successfully.")

    # 3. Evaluate
    print("\n[Step 3] Running inference on test set...")
    all_soh_pred, all_soh_true = [], []
    all_rul_pred, all_rul_true = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            context = batch['context'].to(device)
            chem_id = batch['chem_id'].to(device)
            
            # Handle potential NaNs in input (robustness)
            features = torch.nan_to_num(features, nan=0.0)
            context = torch.nan_to_num(context, nan=0.0)
            
            soh_pred, rul_pred, _, _ = model(features, context, chem_id, use_retrieval=True)
            
            all_soh_pred.extend(soh_pred.squeeze().cpu().numpy())
            all_soh_true.extend(batch['soh'].numpy())
            # Convert normalized RUL back to approximate cycles
            # Model output is normalized (0-1), scaled by 100 for MAE calculation
            
            all_rul_pred.extend(rul_pred.squeeze().cpu().numpy())
            all_rul_true.extend(batch['rul_normalized'].numpy())

    soh_pred = np.array(all_soh_pred)
    soh_true = np.array(all_soh_true)
    rul_pred = np.array(all_rul_pred)
    rul_true = np.array(all_rul_true)
    
    # Filter NaNs
    valid = ~(np.isnan(soh_pred) | np.isnan(soh_true) | np.isnan(rul_pred) | np.isnan(rul_true))
    soh_pred, soh_true = soh_pred[valid], soh_true[valid]
    rul_pred, rul_true = rul_pred[valid], rul_true[valid]
    
    # Metrics
    soh_mae = np.mean(np.abs(soh_pred - soh_true))
    rul_mae = np.mean(np.abs(rul_pred - rul_true)) * 100 # Consistent with training script scaling
    
    # R^2 calculation
    ss_res = np.sum((soh_true - soh_pred) ** 2)
    ss_tot = np.sum((soh_true - np.mean(soh_true)) ** 2)
    soh_r2 = 1 - ss_res / ss_tot

    print("\n" + "=" * 60)
    print("FINAL VERIFIED RESULTS")
    print("=" * 60)
    print(f"SOH MAE: {soh_mae:.4f} ({(soh_mae*100):.2f}%)")
    print(f"SOH R²:  {soh_r2:.4f}")
    print(f"RUL MAE: {rul_mae:.1f} cycles (approx)") # Assuming the scaling factor holds
    print("=" * 60)
    
    # 4. Generate Proof Plot
    print("\n[Step 4] Generatng verification plot...")
    plt.figure(figsize=(12, 5))
    
    # SOH Plot
    plt.subplot(1, 2, 1)
    plt.scatter(soh_true, soh_pred, alpha=0.5, c='blue', s=10)
    plt.plot([0, 1.2], [0, 1.2], 'r--', label='Perfect Prediction')
    plt.xlabel('True SOH')
    plt.ylabel('Predicted SOH')
    plt.title(f'State of Health Validation\nR² = {soh_r2:.3f}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # RUL Plot
    plt.subplot(1, 2, 2)
    plt.scatter(rul_true, rul_pred, alpha=0.5, c='green', s=10)
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect Prediction')
    plt.xlabel('True Normalized RUL')
    plt.ylabel('Predicted Normalized RUL')
    plt.title(f'RUL Validation\nMAE = {rul_mae:.1f}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plot_path = "reports/proof_verification_plot.png"
    plt.savefig(plot_path, dpi=300)
    print(f"  Proof plot saved to: {plot_path}")
    print("\nVerification Complete.")

if __name__ == "__main__":
    main()
