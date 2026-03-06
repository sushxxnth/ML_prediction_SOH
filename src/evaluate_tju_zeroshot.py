"""
Zero-shot evaluation of HERO model on TJU dataset.

This script tests how well the HERO model (trained on NASA, CALCE, Oxford) 
generalizes to the TJU dataset without any fine-tuning.
"""

import numpy as np
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.train.hero_rad_decoupled import RADDecoupledModel


def load_tju_data():
    """Load TJU dataset and prepare for HERO evaluation."""
    
    data_path = Path("data/new_datasets/RUL-Mamba/data/TJU data/Dataset_3_NCM_NCA_battery_1C.npy")
    
    if not data_path.exists():
        raise FileNotFoundError(f"TJU data not found at {data_path}")
    
    data = np.load(data_path, allow_pickle=True).item()
    
    print("=" * 60)
    print("TJU DATASET LOADED")
    print("=" * 60)
    print(f"Cells: {list(data.keys())}")
    
    samples = []
    
    for cell_name, df in data.items():
        print(f"\n{cell_name}: {len(df)} cycles")
        
        # Get capacity as SOH (normalize by first cycle)
        capacity = df['Capacity'].values
        initial_capacity = capacity[0]
        soh = capacity / initial_capacity  # Normalized SOH
        
        # Extract features (16 available, pad to 20)
        feature_cols = [
            'voltage mean', 'voltage std', 'voltage kurtosis', 'voltage skewness',
            'CC Q', 'CC charge time', 'voltage slope', 'voltage entropy',
            'current mean', 'current std', 'current kurtosis', 'current skewness',
            'CV Q', 'CV charge time', 'current slope', 'current entropy'
        ]
        
        for i in range(len(df)):
            # Get cycle features
            features = np.zeros(20, dtype=np.float32)
            for j, col in enumerate(feature_cols):
                if col in df.columns:
                    val = df[col].iloc[i]
                    features[j] = float(val) if not np.isnan(val) else 0.0
            
            # Normalize features to 0-1 range approximately
            features = np.clip(features / (np.abs(features).max() + 1e-8), -1, 1)
            
            # Calculate RUL (remaining cycles until 80% SOH)
            eol_threshold = 0.8
            rul = 0
            for future_i in range(i, len(df)):
                if soh[future_i] < eol_threshold:
                    rul = future_i - i
                    break
            else:
                rul = len(df) - i  # Not yet reached EOL
            
            rul_normalized = min(rul / 1000.0, 1.0)  # Normalize to 0-1
            
            # Context: [temp, charge_rate, discharge_rate, soc, profile]
            # TJU is at 25°C, 1C charge, assume 1C discharge
            context = np.array([
                25.0 / 60.0,  # Temperature normalized
                1.0 / 3.0,    # Charge rate normalized (1C)
                1.0 / 4.0,    # Discharge rate normalized (1C)
                0.5,          # Mid SOC
                0.0           # Constant current profile
            ], dtype=np.float32)
            
            samples.append({
                'cell_id': cell_name,
                'cycle': i,
                'features': features,
                'context': context,
                'soh': soh[i],
                'rul': rul,
                'rul_normalized': rul_normalized
            })
    
    print(f"\nTotal samples: {len(samples)}")
    return samples


def evaluate_hero_on_tju():
    """Run zero-shot evaluation of HERO on TJU."""
    
    # Load HERO model
    model_path = Path("reports/hero_model/hero_model.pt")
    
    if not model_path.exists():
        print(f"ERROR: HERO model not found at {model_path}")
        print("Please run 'python src/train/hero_rad_decoupled.py' first")
        return
    
    print("\n" + "=" * 60)
    print("LOADING HERO MODEL")
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
    model.eval()
    print(f" Model loaded from {model_path}")
    
    # Load TJU data
    samples = load_tju_data()
    
    # Chemistry ID for NCM/NCA (closest to NMC = 1)
    chem_id = 1  # NMC
    
    # Run evaluation
    print("\n" + "=" * 60)
    print("ZERO-SHOT EVALUATION")
    print("=" * 60)
    
    all_soh_pred = []
    all_soh_true = []
    all_rul_pred = []
    all_rul_true = []
    
    with torch.no_grad():
        for sample in samples:
            features = torch.tensor(sample['features'], dtype=torch.float32).unsqueeze(0)
            context = torch.tensor(sample['context'], dtype=torch.float32).unsqueeze(0)
            chem = torch.tensor([chem_id], dtype=torch.long)
            
            # Handle NaN
            features = torch.nan_to_num(features, nan=0.0)
            context = torch.nan_to_num(context, nan=0.0)
            
            # Forward pass
            soh_pred, rul_pred, _, _ = model(features, context, chem, use_retrieval=True)
            
            all_soh_pred.append(float(soh_pred[0]))
            all_soh_true.append(sample['soh'])
            all_rul_pred.append(float(rul_pred[0]))
            all_rul_true.append(sample['rul_normalized'])
    
    # Calculate metrics
    soh_pred = np.array(all_soh_pred)
    soh_true = np.array(all_soh_true)
    rul_pred = np.array(all_rul_pred)
    rul_true = np.array(all_rul_true)
    
    # SOH metrics
    soh_mae = np.mean(np.abs(soh_pred - soh_true))
    soh_rmse = np.sqrt(np.mean((soh_pred - soh_true) ** 2))
    
    # R² for SOH
    ss_res = np.sum((soh_true - soh_pred) ** 2)
    ss_tot = np.sum((soh_true - np.mean(soh_true)) ** 2)
    soh_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    # RUL metrics (convert back to cycles)
    rul_pred_cycles = rul_pred * 1000
    rul_true_cycles = rul_true * 1000
    rul_mae = np.mean(np.abs(rul_pred_cycles - rul_true_cycles))
    
    print("\n" + "=" * 60)
    print("ZERO-SHOT RESULTS ON TJU (NCM/NCA)")
    print("=" * 60)
    
    print(f"\n SOH Prediction:")
    print(f"   MAE:  {soh_mae:.4f} ({soh_mae*100:.2f}%)")
    print(f"   RMSE: {soh_rmse:.4f}")
    print(f"   R²:   {soh_r2:.4f}")
    
    print(f"\n RUL Prediction:")
    print(f"   MAE:  {rul_mae:.1f} cycles")
    
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"\n{'Metric':<20} {'TJU (Zero-Shot)':<20} {'Original Test Set':<20}")
    print("-" * 60)
    print(f"{'SOH MAE':<20} {soh_mae:.4f} ({soh_mae*100:.2f}%){'':<5} 0.013 (1.3%)")
    print(f"{'SOH R²':<20} {soh_r2:.4f}{'':<14} 0.97")
    print(f"{'RUL MAE (cycles)':<20} {rul_mae:.1f}{'':<14} 7.3")
    
    # Per-cell breakdown
    print("\n" + "=" * 60)
    print("PER-CELL BREAKDOWN")
    print("=" * 60)
    
    for cell_name in ['CY25_1', 'CY25_2', 'CY25_3']:
        cell_mask = [s['cell_id'] == cell_name for s in samples]
        cell_soh_pred = soh_pred[cell_mask]
        cell_soh_true = soh_true[cell_mask]
        
        cell_mae = np.mean(np.abs(cell_soh_pred - cell_soh_true))
        print(f"  {cell_name}: SOH MAE = {cell_mae:.4f} ({cell_mae*100:.2f}%)")
    
    # Save results
    results = {
        'dataset': 'TJU (NCM/NCA)',
        'n_samples': len(samples),
        'soh_mae': float(soh_mae),
        'soh_rmse': float(soh_rmse),
        'soh_r2': float(soh_r2),
        'rul_mae_cycles': float(rul_mae),
        'zero_shot': True
    }
    
    import json
    output_path = Path("reports/tju_zeroshot_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n Results saved to {output_path}")
    
    return results


if __name__ == '__main__':
    evaluate_hero_on_tju()
