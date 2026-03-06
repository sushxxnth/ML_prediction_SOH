"""
Fine-tune HERO model on 50% TJU data, test on remaining 50%.

This demonstrates transfer learning capability of HERO model.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.train.hero_rad_decoupled import RADDecoupledModel


class TJUDataset(Dataset):
    """PyTorch Dataset for TJU battery data."""
    
    def __init__(self, samples):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'features': torch.tensor(s['features'], dtype=torch.float32),
            'context': torch.tensor(s['context'], dtype=torch.float32),
            'chem_id': torch.tensor(1, dtype=torch.long),  # NMC=1
            'soh': torch.tensor(s['soh'], dtype=torch.float32),
            'rul_normalized': torch.tensor(s['rul_normalized'], dtype=torch.float32)
        }


def load_tju_data():
    """Load and split TJU dataset."""
    
    data_path = Path("data/new_datasets/RUL-Mamba/data/TJU data/Dataset_3_NCM_NCA_battery_1C.npy")
    data = np.load(data_path, allow_pickle=True).item()
    
    all_samples = []
    
    for cell_name, df in data.items():
        capacity = df['Capacity'].values
        initial_capacity = capacity[0]
        soh = capacity / initial_capacity
        
        feature_cols = [
            'voltage mean', 'voltage std', 'voltage kurtosis', 'voltage skewness',
            'CC Q', 'CC charge time', 'voltage slope', 'voltage entropy',
            'current mean', 'current std', 'current kurtosis', 'current skewness',
            'CV Q', 'CV charge time', 'current slope', 'current entropy'
        ]
        
        for i in range(len(df)):
            features = np.zeros(20, dtype=np.float32)
            for j, col in enumerate(feature_cols):
                if col in df.columns:
                    val = df[col].iloc[i]
                    features[j] = float(val) if not np.isnan(val) else 0.0
            
            # Normalize features
            features = np.clip(features / (np.abs(features).max() + 1e-8), -1, 1)
            
            # Calculate RUL
            eol_threshold = 0.8
            rul = 0
            for future_i in range(i, len(df)):
                if soh[future_i] < eol_threshold:
                    rul = future_i - i
                    break
            else:
                rul = len(df) - i
            
            rul_normalized = min(rul / 1000.0, 1.0)
            
            context = np.array([
                25.0 / 60.0, 1.0 / 3.0, 1.0 / 4.0, 0.5, 0.0
            ], dtype=np.float32)
            
            all_samples.append({
                'cell_id': cell_name,
                'cycle': i,
                'features': features,
                'context': context,
                'soh': float(soh[i]),
                'rul': rul,
                'rul_normalized': rul_normalized
            })
    
    # Shuffle and split 50/50
    np.random.seed(42)
    indices = np.random.permutation(len(all_samples))
    split = len(all_samples) // 2
    
    train_samples = [all_samples[i] for i in indices[:split]]
    test_samples = [all_samples[i] for i in indices[split:]]
    
    return train_samples, test_samples


def finetune_hero_on_tju():
    """Fine-tune HERO on 50% TJU data."""
    
    # Load pre-trained HERO model
    model_path = Path("reports/hero_model/hero_model.pt")
    
    print("=" * 60)
    print("FINE-TUNING HERO ON TJU (50%)")
    print("=" * 60)
    
    model = RADDecoupledModel(
        feature_dim=20,
        context_dim=5,
        hidden_dim=128,
        latent_dim=64,
        n_chemistries=5,
        device='cpu'
    )
    
    model.load_state_dict(torch.load(model_path, weights_only=False, map_location='cpu'))
    print(f" Loaded pre-trained model from {model_path}")
    
    # Load data
    train_samples, test_samples = load_tju_data()
    print(f"\nData split: {len(train_samples)} train, {len(test_samples)} test")
    
    train_loader = DataLoader(TJUDataset(train_samples), batch_size=64, shuffle=True)
    test_loader = DataLoader(TJUDataset(test_samples), batch_size=64)
    
    # Freeze feature encoder (transfer learning)
    for name, param in model.named_parameters():
        if 'feature_encoder' in name:
            param.requires_grad = False
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable:,} trainable / {total:,} total")
    
    # Fine-tuning
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4, weight_decay=1e-5
    )
    
    epochs = 30
    print(f"\n{'Epoch':<8} {'Train Loss':<12} {'SOH MAE':<12} {'RUL MAE':<12}")
    print("-" * 44)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        
        for batch in train_loader:
            features = batch['features']
            context = batch['context']
            chem_id = batch['chem_id']
            soh_true = batch['soh']
            rul_true = batch['rul_normalized']
            
            features = torch.nan_to_num(features, nan=0.0)
            context = torch.nan_to_num(context, nan=0.0)
            
            soh_pred, rul_pred, _, _ = model(features, context, chem_id, use_retrieval=False)
            
            soh_loss = F.mse_loss(soh_pred.squeeze(), soh_true)
            rul_loss = F.mse_loss(rul_pred.squeeze(), rul_true)
            loss = soh_loss + 0.5 * rul_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        # Evaluate
        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            all_soh_pred, all_soh_true = [], []
            all_rul_pred, all_rul_true = [], []
            
            with torch.no_grad():
                for batch in test_loader:
                    features = torch.nan_to_num(batch['features'], nan=0.0)
                    context = torch.nan_to_num(batch['context'], nan=0.0)
                    
                    soh_pred, rul_pred, _, _ = model(features, context, batch['chem_id'])
                    
                    all_soh_pred.extend(soh_pred.squeeze().numpy())
                    all_soh_true.extend(batch['soh'].numpy())
                    all_rul_pred.extend(rul_pred.squeeze().numpy())
                    all_rul_true.extend(batch['rul_normalized'].numpy())
            
            soh_mae = np.mean(np.abs(np.array(all_soh_pred) - np.array(all_soh_true)))
            rul_mae = np.mean(np.abs(np.array(all_rul_pred) - np.array(all_rul_true))) * 1000
            
            print(f"{epoch+1:<8} {total_loss/n_batches:<12.4f} {soh_mae:<12.4f} {rul_mae:<12.1f}")
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL RESULTS AFTER FINE-TUNING")
    print("=" * 60)
    
    model.eval()
    all_soh_pred, all_soh_true = [], []
    all_rul_pred, all_rul_true = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            features = torch.nan_to_num(batch['features'], nan=0.0)
            context = torch.nan_to_num(batch['context'], nan=0.0)
            
            soh_pred, rul_pred, _, _ = model(features, context, batch['chem_id'])
            
            all_soh_pred.extend(soh_pred.squeeze().numpy())
            all_soh_true.extend(batch['soh'].numpy())
            all_rul_pred.extend(rul_pred.squeeze().numpy())
            all_rul_true.extend(batch['rul_normalized'].numpy())
    
    soh_pred = np.array(all_soh_pred)
    soh_true = np.array(all_soh_true)
    rul_pred = np.array(all_rul_pred)
    rul_true = np.array(all_rul_true)
    
    soh_mae = np.mean(np.abs(soh_pred - soh_true))
    soh_rmse = np.sqrt(np.mean((soh_pred - soh_true) ** 2))
    
    ss_res = np.sum((soh_true - soh_pred) ** 2)
    ss_tot = np.sum((soh_true - np.mean(soh_true)) ** 2)
    soh_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    rul_mae = np.mean(np.abs(rul_pred - rul_true)) * 1000
    
    print(f"\n SOH Prediction:")
    print(f"   MAE:  {soh_mae:.4f} ({soh_mae*100:.2f}%)")
    print(f"   RMSE: {soh_rmse:.4f}")
    print(f"   R²:   {soh_r2:.4f}")
    
    print(f"\n RUL Prediction:")
    print(f"   MAE:  {rul_mae:.1f} cycles")
    
    # Comparison
    print("\n" + "=" * 60)
    print("IMPROVEMENT COMPARISON")
    print("=" * 60)
    print(f"\n{'Metric':<20} {'Zero-Shot':<20} {'After Fine-Tune':<20} {'Improvement':<15}")
    print("-" * 75)
    
    zeroshot_soh = 0.1713
    zeroshot_rul = 857.4
    
    soh_improvement = (zeroshot_soh - soh_mae) / zeroshot_soh * 100
    rul_improvement = (zeroshot_rul - rul_mae) / zeroshot_rul * 100
    
    print(f"{'SOH MAE':<20} {zeroshot_soh*100:.2f}%{'':<14} {soh_mae*100:.2f}%{'':<14} {soh_improvement:+.1f}%")
    print(f"{'RUL MAE (cycles)':<20} {zeroshot_rul:.1f}{'':<14} {rul_mae:.1f}{'':<14} {rul_improvement:+.1f}%")
    print(f"{'SOH R²':<20} {'-3.61':<20} {soh_r2:.4f}")
    
    # Save results
    results = {
        'before_finetuning': {
            'soh_mae': 0.1713,
            'rul_mae_cycles': 857.4,
            'soh_r2': -3.61
        },
        'after_finetuning': {
            'soh_mae': float(soh_mae),
            'soh_rmse': float(soh_rmse),
            'soh_r2': float(soh_r2),
            'rul_mae_cycles': float(rul_mae)
        },
        'improvement': {
            'soh_mae_reduction': f"{soh_improvement:.1f}%",
            'rul_mae_reduction': f"{rul_improvement:.1f}%"
        },
        'train_samples': len(train_samples),
        'test_samples': len(test_samples),
        'epochs': epochs
    }
    
    output_path = Path("reports/tju_finetuned_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save fine-tuned model
    model_output = Path("reports/hero_model/hero_model_tju_finetuned.pt")
    torch.save(model.state_dict(), model_output)
    
    print(f"\n Results saved to {output_path}")
    print(f" Fine-tuned model saved to {model_output}")
    
    return results


if __name__ == '__main__':
    finetune_hero_on_tju()
