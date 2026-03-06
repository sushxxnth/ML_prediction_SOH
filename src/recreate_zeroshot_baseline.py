"""
Zero-Shot Baseline Comparison - Recreated Original Script

This script recreates the original zero-shot evaluation that generated
reports/zeroshot_baseline_comparison.json with HERO RUL MAE = 44.0 cycles.

The structure matches sota_baseline_comparison.py but evaluates all models
in a zero-shot setting (trained on LCO, tested on Panasonic NCA).

"""

import os

# Avoid OpenMP shared memory and runtime conflicts in constrained environments
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("KMP_USE_SHM", "0")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

import sys
import random
import json
import tempfile
import subprocess
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


RUL_SCALE_CYCLES = 1000.0  # Paper-aligned RUL scaling


def get_torch():
    import torch
    import torch.nn as nn
    return torch, nn


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    torch, _ = get_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def standardize_fit_transform(X: np.ndarray):
    """Standardize features using train statistics."""
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    X_scaled = (X - mean) / std
    return X_scaled, mean, std


def standardize_transform(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    """Apply train-set standardization to new data."""
    return (X - mean) / std


def mean_absolute_error_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def run_random_forest_subprocess(
    X_train: np.ndarray,
    soh_train: np.ndarray,
    rul_norm_train: np.ndarray,
    X_test: np.ndarray,
    soh_test: np.ndarray,
    rul_cycles_test: np.ndarray
) -> dict:
    """
    Run Random Forest baseline in a separate process to avoid OpenMP
    conflicts between PyTorch and scikit-learn.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / "rf_data.npz"
        out_path = Path(tmpdir) / "rf_out.json"
        np.savez_compressed(
            data_path,
            X_train=X_train,
            soh_train=soh_train,
            rul_norm_train=rul_norm_train,
            X_test=X_test,
            soh_test=soh_test,
            rul_cycles_test=rul_cycles_test,
            rul_scale=np.array([RUL_SCALE_CYCLES], dtype=np.float32),
        )

        env = os.environ.copy()
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        env.setdefault("NUMEXPR_NUM_THREADS", "1")
        env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
        env.setdefault("KMP_USE_SHM", "0")
        env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        env.setdefault("MKL_THREADING_LAYER", "GNU")

        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--rf-only",
            str(data_path),
            "--rf-out",
            str(out_path),
        ]
        subprocess.run(cmd, check=True, env=env)

        if not out_path.exists():
            raise RuntimeError("Random Forest subprocess did not produce output.")
        with open(out_path, "r") as f:
            return json.load(f)


def run_rf_only(data_path: Path, out_path: Path) -> None:
    """Random Forest baseline runner (sklearn-only)."""
    from sklearn.ensemble import RandomForestRegressor

    data = np.load(data_path, allow_pickle=True)
    X_train = data["X_train"]
    soh_train = data["soh_train"]
    rul_norm_train = data["rul_norm_train"]
    X_test = data["X_test"]
    soh_test = data["soh_test"]
    rul_cycles_test = data["rul_cycles_test"]
    rul_scale = float(data["rul_scale"][0])

    X_train_scaled, mean, std = standardize_fit_transform(X_train)
    X_test_scaled = standardize_transform(X_test, mean, std)

    rf_soh = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=1)
    rf_rul = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=1)

    rf_soh.fit(X_train_scaled, soh_train)
    rf_rul.fit(X_train_scaled, rul_norm_train)

    soh_pred = rf_soh.predict(X_test_scaled)
    rul_pred_norm = rf_rul.predict(X_test_scaled)

    soh_mae = mean_absolute_error_np(soh_test, soh_pred) * 100.0
    soh_r2 = r2_score_np(soh_test, soh_pred)

    rul_pred_cycles = rul_pred_norm * rul_scale
    rul_mae = float(np.mean(np.abs(rul_pred_cycles - rul_cycles_test)))

    with open(out_path, "w") as f:
        json.dump(
            {
                "soh_mae": float(soh_mae),
                "soh_r2": float(soh_r2),
                "rul_mae": float(rul_mae),
            },
            f,
            indent=2,
        )


def build_augmented_feature_vector(cycle, cell, lithium_cache) -> np.ndarray:
    """Create 20D feature vector (9 base + 11 lithium) to match HERO training."""
    from src.data.lithium_inventory_integration import augment_cycle_with_lithium_features
    try:
        features = augment_cycle_with_lithium_features(cycle, cell, lithium_cache)
    except Exception:
        base = cycle.to_feature_vector()
        li = np.zeros(11, dtype=np.float32)
        features = np.concatenate([base, li])
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def load_zeroshot_samples(
    max_train_samples: int = 4000,
    max_test_samples: int = 500,
    force_refresh: bool = False
):
    """
    Load unified samples for zero-shot evaluation.
    Train: LCO (NASA, CALCE, Oxford)
    Test: Panasonic NCA (panasonic_18650pf)
    """
    data_root = Path("data")
    # Optional Panasonic cache refresh
    panasonic_cache = data_root / "unified_cache" / "panasonic_18650pf" / "panasonic_18650pf_processed.json"
    if force_refresh and panasonic_cache.exists():
        try:
            panasonic_cache.unlink()
        except Exception:
            pass
    
    from src.data.unified_pipeline import UnifiedDataPipeline
    pipeline = UnifiedDataPipeline(str(data_root), use_lithium_features=True)
    pipeline.load_datasets(['nasa', 'calce', 'oxford', 'panasonic_18650pf'])
    
    source_samples = [
        s for s in pipeline.samples
        if s.source_dataset in ['nasa', 'calce', 'oxford']
    ]
    target_samples = [
        s for s in pipeline.samples
        if s.source_dataset in ['panasonic_18650pf', 'panasonic']
    ]
    
    # Subsample to match paper scale (optional)
    rng_train = np.random.default_rng(42)
    if max_train_samples and len(source_samples) > max_train_samples:
        idx = rng_train.choice(len(source_samples), size=max_train_samples, replace=False)
        source_samples = [source_samples[i] for i in idx]
    
    rng_test = np.random.default_rng(123)
    if max_test_samples and len(target_samples) > max_test_samples:
        idx = rng_test.choice(len(target_samples), size=max_test_samples, replace=False)
        target_samples = [target_samples[i] for i in idx]
    
    return source_samples, target_samples


def samples_to_arrays(samples):
    """Convert UnifiedSample list to arrays for baseline training/eval."""
    X_list, soh_list, rul_norm_list, rul_cycles_list, eol_list = [], [], [], [], []
    for s in samples:
        if s.features is None:
            continue
        features = np.nan_to_num(s.features, nan=0.0, posinf=0.0, neginf=0.0)
        if s.soh is None or not np.isfinite(s.soh):
            continue
        if s.rul is None or not np.isfinite(s.rul):
            continue
        if s.rul_normalized is None or not np.isfinite(s.rul_normalized):
            continue
        if s.eol_cycle is None or not np.isfinite(s.eol_cycle):
            continue
        X_list.append(features.astype(np.float32))
        soh_list.append(float(np.clip(s.soh, 0.0, 1.2)))
        # Paper-aligned normalization: RUL / 1000 cycles
        rul_cycles = float(max(s.rul, 0.0))
        rul_norm = float(np.clip(rul_cycles / RUL_SCALE_CYCLES, 0.0, 1.0))
        rul_norm_list.append(rul_norm)
        rul_cycles_list.append(rul_cycles)
        eol_list.append(float(RUL_SCALE_CYCLES))
    
    return (
        np.array(X_list, dtype=np.float32),
        np.array(soh_list, dtype=np.float32),
        np.array(rul_norm_list, dtype=np.float32),
        np.array(rul_cycles_list, dtype=np.float32),
        np.array(eol_list, dtype=np.float32),
    )


def evaluate_hero_zeroshot(
    train_samples,
    test_samples,
    pretrain_epochs: int = 100,
    device: str = "cpu"
):
    """Train HERO on LCO and evaluate zero-shot on Panasonic NCA."""
    torch, _ = get_torch()
    from src.train.hero_rad_decoupled import (
        RADDecoupledModel, train_combined_model
    )

    class ZeroShotDataset(torch.utils.data.Dataset):
        """Dataset wrapper with paper-aligned RUL normalization."""
        def __init__(self, samples):
            self.samples = samples
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            s = self.samples[idx]
            features = np.nan_to_num(s.features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            context = np.nan_to_num(s.context_vector, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            chem_id = int(s.chem_id)
            soh = float(np.clip(s.soh, 0.0, 1.2))
            rul_cycles = float(max(s.rul, 0.0))
            rul_norm = float(np.clip(rul_cycles / RUL_SCALE_CYCLES, 0.0, 1.0))
            return {
                "features": torch.tensor(features, dtype=torch.float32),
                "context": torch.tensor(context, dtype=torch.float32),
                "chem_id": torch.tensor(chem_id, dtype=torch.long),
                "soh": torch.tensor(soh, dtype=torch.float32),
                "rul_normalized": torch.tensor(rul_norm, dtype=torch.float32),
                "rul": torch.tensor(rul_cycles, dtype=torch.float32),
                "eol_cycle": torch.tensor(RUL_SCALE_CYCLES, dtype=torch.float32),
            }
    print("\n" + "=" * 50)
    print("Training HERO (LCO) + Evaluating Zero-Shot (Panasonic NCA)")
    print("=" * 50)
    
    # Split train/val by samples (shuffle)
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(train_samples))
    split = int(0.85 * len(indices))
    train_idx, val_idx = indices[:split], indices[split:]
    
    train_set = [train_samples[i] for i in train_idx]
    val_set = [train_samples[i] for i in val_idx]
    
    train_loader = torch.utils.data.DataLoader(
        ZeroShotDataset(train_set), batch_size=64, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        ZeroShotDataset(val_set), batch_size=64, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        ZeroShotDataset(test_samples), batch_size=64, shuffle=False
    )
    
    model = RADDecoupledModel(
        feature_dim=20,
        context_dim=5,
        hidden_dim=128,
        latent_dim=64,
        n_chemistries=5,
        device=device
    ).to(device)
    
    train_combined_model(train_loader, val_loader, model, device=device, epochs=pretrain_epochs)
    
    # Evaluate on Panasonic test
    model.eval()
    all_soh_pred, all_soh_true = [], []
    all_rul_pred, all_rul_true = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            context = batch['context'].to(device)
            chem_id = batch['chem_id'].to(device)
            
            features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            context = torch.nan_to_num(context, nan=0.0)
            
            soh_pred, rul_pred, _, _ = model(features, context, chem_id)
            
            rul_true_cycles = batch['rul'].numpy()
            rul_pred_cycles = (rul_pred.squeeze().cpu().numpy()) * RUL_SCALE_CYCLES
            
            all_soh_pred.extend(soh_pred.squeeze().cpu().numpy())
            all_soh_true.extend(batch['soh'].numpy())
            all_rul_pred.extend(rul_pred_cycles)
            all_rul_true.extend(rul_true_cycles)
    
    soh_pred = np.array(all_soh_pred)
    soh_true = np.array(all_soh_true)
    rul_pred_cycles = np.array(all_rul_pred)
    rul_true_cycles = np.array(all_rul_true)
    
    valid = ~(np.isnan(soh_pred) | np.isnan(soh_true) | np.isnan(rul_pred_cycles) | np.isnan(rul_true_cycles))
    soh_pred, soh_true = soh_pred[valid], soh_true[valid]
    rul_pred_cycles, rul_true_cycles = rul_pred_cycles[valid], rul_true_cycles[valid]
    
    soh_mae = np.mean(np.abs(soh_pred - soh_true)) * 100.0
    ss_res = np.sum((soh_true - soh_pred) ** 2)
    ss_tot = np.sum((soh_true - np.mean(soh_true)) ** 2)
    soh_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    rul_mae = np.mean(np.abs(rul_pred_cycles - rul_true_cycles))
    
    print(f"  SOH MAE:  {soh_mae:.2f}%")
    print(f"  SOH R²:   {soh_r2:.4f}")
    print(f"  RUL MAE:  {rul_mae:.1f} cycles")
    
    return {
        "soh_mae": float(soh_mae),
        "soh_r2": float(soh_r2),
        "rul_mae": float(rul_mae)
    }


def train_baseline_model(model, X_train, soh_train, rul_norm_train, epochs=200, lr=0.001):
    """Train baseline on normalized targets."""
    torch, nn = get_torch()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        soh_pred, rul_pred = model(X_train)
        loss = criterion(soh_pred, soh_train) + criterion(rul_pred, rul_norm_train)
        loss.backward()
        optimizer.step()
    return model


def evaluate_baseline_zeroshot(
    model,
    X_train,
    soh_train,
    rul_norm_train,
    X_test,
    soh_test,
    rul_norm_test,
    rul_cycles_test,
    eol_cycles_test,
    name,
    epochs=200
):
    """Train baseline on LCO and evaluate zero-shot on Panasonic NCA."""
    torch, _ = get_torch()
    print(f"\nTraining + Evaluating {name} (Zero-Shot LCO → NCA)")
    
    # Standardize features (fit on train, apply to test) for baselines
    X_train_scaled, mean, std = standardize_fit_transform(X_train)
    X_test_scaled = standardize_transform(X_test, mean, std)
    
    # Tensors
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    soh_train_t = torch.tensor(soh_train, dtype=torch.float32)
    rul_norm_train_t = torch.tensor(rul_norm_train, dtype=torch.float32)
    
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
    soh_test_t = torch.tensor(soh_test, dtype=torch.float32)
    
    # Train
    model = train_baseline_model(model, X_train_t, soh_train_t, rul_norm_train_t, epochs=epochs)
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        soh_pred, rul_pred = model(X_test_t)
    
    soh_pred = soh_pred.numpy()
    rul_pred = rul_pred.numpy()
    
    soh_mae = mean_absolute_error_np(soh_test, soh_pred) * 100.0
    soh_r2 = r2_score_np(soh_test, soh_pred)
    
    rul_pred_cycles = rul_pred * RUL_SCALE_CYCLES
    rul_mae = np.mean(np.abs(rul_pred_cycles - rul_cycles_test))
    
    return {
        "soh_mae": float(soh_mae),
        "soh_r2": float(soh_r2),
        "rul_mae": float(rul_mae),
    }


def run_zeroshot_baseline_comparison():
    """Main function to recreate the zero-shot baseline comparison."""
    from src.sota_baseline_comparison import (
        LSTMBaseline, GRUBaseline, CNNLSTMBaseline,
        TransformerBaseline, MLPBaseline
    )
    
    print("=" * 70)
    print("ZERO-SHOT BASELINE COMPARISON (LCO → Panasonic NCA)")
    print("=" * 70)
    
    # Reproducibility
    set_seed(42)
    
    # Load unified samples
    print("\nLoading unified samples for zero-shot...")
    train_samples, test_samples = load_zeroshot_samples(max_train_samples=4000, force_refresh=False)
    
    if len(train_samples) == 0 or len(test_samples) == 0:
        print("ERROR: Could not load zero-shot samples.")
        return
    
    # Convert to arrays for baselines
    X_train, soh_train, rul_norm_train, rul_cycles_train, eol_train = samples_to_arrays(train_samples)
    X_test, soh_test, rul_norm_test, rul_cycles_test, eol_test = samples_to_arrays(test_samples)
    
    print(f"Loaded {len(X_train)} LCO samples for training")
    print(f"Loaded {len(X_test)} Panasonic samples for testing")
    
    results = {}
    
    # Train + evaluate HERO from scratch on LCO
    hero_results = evaluate_hero_zeroshot(
        train_samples,
        test_samples,
        pretrain_epochs=100,
        device="cpu"
    )
    results["HERO"] = hero_results
    
    # Train + evaluate baselines
    baseline_models = {
        "LSTM": LSTMBaseline(X_train.shape[1]),
        "GRU": GRUBaseline(X_train.shape[1]),
        "CNN-LSTM": CNNLSTMBaseline(X_train.shape[1]),
        "Transformer": TransformerBaseline(X_train.shape[1]),
        "MLP": MLPBaseline(X_train.shape[1]),
    }
    
    for name, model in baseline_models.items():
        results[name] = evaluate_baseline_zeroshot(
            model,
            X_train,
            soh_train,
            rul_norm_train,
            X_test,
            soh_test,
            rul_norm_test,
            rul_cycles_test,
            eol_test,
            name,
            epochs=200
        )
        print(f"  {name} RUL MAE: {results[name]['rul_mae']:.1f} cycles")
    
    # Random Forest (run in isolated subprocess to avoid OpenMP conflicts)
    print("\nTraining + Evaluating Random Forest (Zero-Shot LCO → NCA)")
    rf_metrics = run_random_forest_subprocess(
        X_train,
        soh_train,
        rul_norm_train,
        X_test,
        soh_test,
        rul_cycles_test
    )
    results["Random Forest"] = rf_metrics
    print(f"  Random Forest RUL MAE: {results['Random Forest']['rul_mae']:.1f} cycles")
    
    # Summary
    print("\n" + "=" * 70)
    print("ZERO-SHOT RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<20} {'SOH MAE':<12} {'SOH R²':<10} {'RUL MAE':<15}")
    print("-" * 57)
    
    for name, metrics in results.items():
        soh_pct = f"{metrics['soh_mae']:.2f}%" if 'soh_mae' in metrics else "N/A"
        r2 = f"{metrics['soh_r2']:.4f}" if 'soh_r2' in metrics else "N/A"
        rul = f"{metrics['rul_mae']:.1f} cycles" if 'rul_mae' in metrics else "N/A"
        marker = " ★" if name == 'HERO' else ""
        print(f"{name:<20} {soh_pct:<12} {r2:<10} {rul:<15}{marker}")
    
    # Save updated results
    output_path = Path("reports/zeroshot_baseline_comparison_reproduced.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n Results saved to {output_path}")
    print("\nNOTE: All baseline results were re-trained on LCO and evaluated zero-shot on Panasonic NCA.")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-shot baseline comparison.")
    parser.add_argument("--rf-only", dest="rf_only", type=str, default=None,
                        help="Run Random Forest only using provided npz data path.")
    parser.add_argument("--rf-out", dest="rf_out", type=str, default=None,
                        help="Output path for RF metrics JSON.")
    args = parser.parse_args()

    if args.rf_only:
        if not args.rf_out:
            raise SystemExit("--rf-out is required when using --rf-only")
        run_rf_only(Path(args.rf_only), Path(args.rf_out))
    else:
        run_zeroshot_baseline_comparison()
