"""
Uncertainty Quantification for HERO Battery Predictions.

Implements Bootstrap Confidence Intervals to quantify prediction uncertainty.
This is essential for top-tier journal submissions and real-world BMS deployment.
"""

import numpy as np
import torch
import json
from pathlib import Path
import sys
from sklearn.metrics import mean_absolute_error, r2_score
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.train.hero_rad_decoupled import RADDecoupledModel


def load_test_data():
    """Load TJU test data for uncertainty analysis."""
    
    tju_path = Path("data/new_datasets/RUL-Mamba/data/TJU data/Dataset_3_NCM_NCA_battery_1C.npy")
    data = np.load(tju_path, allow_pickle=True).item()
    
    samples = []
    
    for cell_name, df in data.items():
        capacity = df['Capacity'].values
        initial_capacity = capacity[0]
        cell_soh = capacity / initial_capacity
        
        feature_cols = [
            'voltage mean', 'voltage std', 'voltage kurtosis', 'voltage skewness',
            'CC Q', 'CC charge time', 'voltage slope', 'voltage entropy',
            'current mean', 'current std', 'current kurtosis', 'current skewness',
            'CV Q', 'CV charge time', 'current slope', 'current entropy'
        ]
        
        for i in range(0, len(df), 5):
            features = np.zeros(20, dtype=np.float32)
            for j, col in enumerate(feature_cols):
                if col in df.columns:
                    val = df[col].iloc[i]
                    features[j] = float(val) if not np.isnan(val) else 0.0
            
            features = np.clip(features / (np.abs(features).max() + 1e-8), -1, 1)
            
            eol_threshold = 0.8
            rul = 0
            for future_i in range(i, len(df)):
                if cell_soh[future_i] < eol_threshold:
                    rul = future_i - i
                    break
            else:
                rul = len(df) - i
            
            rul_normalized = min(rul / 1000.0, 1.0)
            
            context = np.array([25.0/60.0, 1.0/3.0, 1.0/4.0, 0.5, 0.0], dtype=np.float32)
            
            samples.append({
                'features': features,
                'context': context,
                'soh': float(cell_soh[i]),
                'rul': rul_normalized
            })
    
    return samples


def bootstrap_confidence_intervals(predictions, targets, n_bootstrap=1000, confidence=0.95):
    """
    Calculate bootstrap confidence intervals for metrics.
    
    Args:
        predictions: Model predictions
        targets: Ground truth values
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default 95%)
    
    Returns:
        Dict with mean, std, lower, upper bounds
    """
    n_samples = len(predictions)
    
    # Store metrics from each bootstrap sample
    mae_samples = []
    r2_samples = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        pred_sample = predictions[indices]
        target_sample = targets[indices]
        
        mae = mean_absolute_error(target_sample, pred_sample)
        
        # Handle edge case for R2
        if np.var(target_sample) > 1e-10:
            r2 = r2_score(target_sample, pred_sample)
        else:
            r2 = np.nan
        
        mae_samples.append(mae)
        if not np.isnan(r2):
            r2_samples.append(r2)
    
    # Calculate confidence intervals
    alpha = 1 - confidence
    
    mae_mean = np.mean(mae_samples)
    mae_std = np.std(mae_samples)
    mae_lower = np.percentile(mae_samples, alpha/2 * 100)
    mae_upper = np.percentile(mae_samples, (1 - alpha/2) * 100)
    
    r2_mean = np.mean(r2_samples)
    r2_std = np.std(r2_samples)
    r2_lower = np.percentile(r2_samples, alpha/2 * 100)
    r2_upper = np.percentile(r2_samples, (1 - alpha/2) * 100)
    
    return {
        'mae': {
            'mean': float(mae_mean),
            'std': float(mae_std),
            'lower': float(mae_lower),
            'upper': float(mae_upper)
        },
        'r2': {
            'mean': float(r2_mean),
            'std': float(r2_std),
            'lower': float(r2_lower),
            'upper': float(r2_upper)
        }
    }


def run_uncertainty_quantification():
    """Run uncertainty quantification for HERO predictions."""
    
    print("=" * 70)
    print("UNCERTAINTY QUANTIFICATION FOR HERO PREDICTIONS")
    print("Method: Bootstrap Confidence Intervals (1000 samples, 95% CI)")
    print("=" * 70)
    
    # Load model
    model_path = Path("reports/hero_model/hero_model_tju_finetuned.pt")
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return
    
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
    print(" Loaded HERO model")
    
    # Load test data
    samples = load_test_data()
    print(f" Loaded {len(samples)} test samples")
    
    # Split 50% for held-out test
    np.random.seed(42)
    indices = np.random.permutation(len(samples))
    test_indices = indices[len(indices)//2:]
    test_samples = [samples[i] for i in test_indices]
    
    print(f" Using {len(test_samples)} held-out samples for UQ")
    
    # Get predictions
    all_soh_pred, all_soh_true = [], []
    all_rul_pred, all_rul_true = [], []
    
    with torch.no_grad():
        for sample in test_samples:
            features = torch.tensor(sample['features'], dtype=torch.float32).unsqueeze(0)
            context = torch.tensor(sample['context'], dtype=torch.float32).unsqueeze(0)
            chem_id = torch.tensor([1], dtype=torch.long)
            
            features = torch.nan_to_num(features, nan=0.0)
            context = torch.nan_to_num(context, nan=0.0)
            
            soh_pred, rul_pred, _, _ = model(features, context, chem_id, use_retrieval=False)
            
            all_soh_pred.append(float(soh_pred[0]))
            all_soh_true.append(sample['soh'])
            all_rul_pred.append(float(rul_pred[0]))
            all_rul_true.append(sample['rul'])
    
    soh_pred = np.array(all_soh_pred)
    soh_true = np.array(all_soh_true)
    rul_pred = np.array(all_rul_pred)
    rul_true = np.array(all_rul_true)
    
    # Calculate bootstrap CIs for SOH
    print("\nCalculating bootstrap confidence intervals...")
    print("(This may take a moment)")
    
    soh_uq = bootstrap_confidence_intervals(soh_pred, soh_true, n_bootstrap=1000)
    rul_uq = bootstrap_confidence_intervals(rul_pred, rul_true, n_bootstrap=1000)
    
    # Results
    print("\n" + "=" * 70)
    print("SOH PREDICTION UNCERTAINTY")
    print("=" * 70)
    
    print(f"\n{'Metric':<15} {'Value':<15} {'95% CI':}")
    print("-" * 50)
    print(f"{'MAE':<15} {soh_uq['mae']['mean']*100:.2f}%{'':<8} [{soh_uq['mae']['lower']*100:.2f}%, {soh_uq['mae']['upper']*100:.2f}%]")
    print(f"{'R²':<15} {soh_uq['r2']['mean']:.4f}{'':<8} [{soh_uq['r2']['lower']:.4f}, {soh_uq['r2']['upper']:.4f}]")
    
    print("\n" + "=" * 70)
    print("RUL PREDICTION UNCERTAINTY")
    print("=" * 70)
    
    rul_mae_mean = rul_uq['mae']['mean'] * 1000
    rul_mae_lower = rul_uq['mae']['lower'] * 1000
    rul_mae_upper = rul_uq['mae']['upper'] * 1000
    
    print(f"\n{'Metric':<15} {'Value':<15} {'95% CI':}")
    print("-" * 50)
    print(f"{'MAE':<15} {rul_mae_mean:.1f} cycles{'':<4} [{rul_mae_lower:.1f}, {rul_mae_upper:.1f}] cycles")
    print(f"{'R²':<15} {rul_uq['r2']['mean']:.4f}{'':<8} [{rul_uq['r2']['lower']:.4f}, {rul_uq['r2']['upper']:.4f}]")
    
    # Summary for paper
    print("\n" + "=" * 70)
    print("PAPER-READY RESULTS (Copy-Paste)")
    print("=" * 70)
    
    print(f"""
SOH Prediction:
  MAE = {soh_uq['mae']['mean']*100:.2f}% (95% CI: [{soh_uq['mae']['lower']*100:.2f}%, {soh_uq['mae']['upper']*100:.2f}%])
  R² = {soh_uq['r2']['mean']:.3f} (95% CI: [{soh_uq['r2']['lower']:.3f}, {soh_uq['r2']['upper']:.3f}])

RUL Prediction:
  MAE = {rul_mae_mean:.1f} cycles (95% CI: [{rul_mae_lower:.1f}, {rul_mae_upper:.1f}] cycles)
  R² = {rul_uq['r2']['mean']:.3f} (95% CI: [{rul_uq['r2']['lower']:.3f}, {rul_uq['r2']['upper']:.3f}])
""")
    
    # Per-prediction uncertainty (Monte Carlo style with small noise)
    print("=" * 70)
    print("PER-PREDICTION UNCERTAINTY (Sample)")
    print("=" * 70)
    
    # Show a few sample predictions with individual uncertainty
    print(f"\n{'Sample':<10} {'True SOH':<12} {'Pred SOH':<12} {'Error':<12} {'Uncertainty':<12}")
    print("-" * 58)
    
    residuals = np.abs(soh_pred - soh_true)
    std_residual = np.std(residuals)
    
    for i in range(min(5, len(test_samples))):
        true = soh_true[i]
        pred = soh_pred[i]
        error = abs(pred - true)
        # Estimate per-prediction uncertainty based on residual distribution
        uncertainty = 1.96 * std_residual  # 95% CI based on std
        
        print(f"{i+1:<10} {true*100:.2f}%{'':<5} {pred*100:.2f}%{'':<5} {error*100:.2f}%{'':<5} ±{uncertainty*100:.2f}%")
    
    # Save results
    results = {
        'method': 'Bootstrap Confidence Intervals',
        'n_bootstrap': 1000,
        'confidence_level': 0.95,
        'n_test_samples': len(test_samples),
        'soh': {
            'mae_mean': float(soh_uq['mae']['mean']),
            'mae_lower': float(soh_uq['mae']['lower']),
            'mae_upper': float(soh_uq['mae']['upper']),
            'r2_mean': float(soh_uq['r2']['mean']),
            'r2_lower': float(soh_uq['r2']['lower']),
            'r2_upper': float(soh_uq['r2']['upper'])
        },
        'rul': {
            'mae_mean_cycles': float(rul_mae_mean),
            'mae_lower_cycles': float(rul_mae_lower),
            'mae_upper_cycles': float(rul_mae_upper),
            'r2_mean': float(rul_uq['r2']['mean']),
            'r2_lower': float(rul_uq['r2']['lower']),
            'r2_upper': float(rul_uq['r2']['upper'])
        }
    }
    
    output_path = Path("reports/uncertainty_quantification_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n Results saved to {output_path}")
    
    return results


if __name__ == '__main__':
    run_uncertainty_quantification()
