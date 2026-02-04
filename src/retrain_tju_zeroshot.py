"""
Retrain HERO for TJU Zero-Shot Evaluation (LCO -> TJU NCM/NCA).

Trains on LCO (NASA/CALCE/Oxford) using unified features and evaluates
zero-shot on TJU without fine-tuning. Designed to improve reproducibility
and alignment of feature space.
"""

import os
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Avoid OpenMP conflicts
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("KMP_USE_SHM", "0")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.unified_pipeline import UnifiedDataPipeline
from src.train.hero_rad_decoupled import RADDecoupledModel, train_combined_model


RUL_SCALE_CYCLES = 1000.0
HORIZON_CYCLES = 240.0


def set_seed(seed: int = 42) -> None:
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


def compute_feature_norm(samples):
    feats = [s.features for s in samples if s.features is not None]
    X = np.stack(feats, axis=0).astype(np.float32)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


class ZeroShotDataset(Dataset):
    def __init__(self, samples, mean, std):
        self.samples = samples
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        features = np.nan_to_num(s.features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        features = (features - self.mean) / self.std
        if features.ndim > 1:
            features = features.reshape(-1)
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
            "rul_cycles": torch.tensor(rul_cycles, dtype=torch.float32),
        }


def load_samples(use_lithium: bool = False, max_train_samples: int = 4000):
    pipeline = UnifiedDataPipeline("data", use_lithium_features=use_lithium)
    pipeline.load_datasets(["nasa", "calce", "oxford", "tju"])

    source_samples = [s for s in pipeline.samples if s.source_dataset in ["nasa", "calce", "oxford"]]
    target_samples = [s for s in pipeline.samples if s.source_dataset == "tju"]

    rng = np.random.default_rng(42)
    if max_train_samples and len(source_samples) > max_train_samples:
        idx = rng.choice(len(source_samples), size=max_train_samples, replace=False)
        source_samples = [source_samples[i] for i in idx]

    return source_samples, target_samples


def evaluate_model(model, loader, device="cpu"):
    model.eval()
    all_soh_pred, all_soh_true = [], []
    all_rul_pred, all_rul_true = [], []
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            context = batch["context"].to(device)
            chem_id = batch["chem_id"].to(device)

            features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            context = torch.nan_to_num(context, nan=0.0)

            soh_pred, rul_pred, _, _ = model(features, context, chem_id)
            all_soh_pred.extend(soh_pred.squeeze().cpu().numpy())
            all_soh_true.extend(batch["soh"].numpy())
            all_rul_pred.extend((rul_pred.squeeze().cpu().numpy()) * RUL_SCALE_CYCLES)
            all_rul_true.extend(batch["rul_cycles"].numpy())

    soh_pred = np.array(all_soh_pred)
    soh_true = np.array(all_soh_true)
    rul_pred = np.array(all_rul_pred)
    rul_true = np.array(all_rul_true)

    valid = ~(np.isnan(soh_pred) | np.isnan(soh_true) | np.isnan(rul_pred) | np.isnan(rul_true))
    soh_pred, soh_true = soh_pred[valid], soh_true[valid]
    rul_pred, rul_true = rul_pred[valid], rul_true[valid]

    soh_mae = float(np.mean(np.abs(soh_pred - soh_true)) * 100.0) if len(soh_pred) else float("nan")
    ss_res = np.sum((soh_true - soh_pred) ** 2)
    ss_tot = np.sum((soh_true - np.mean(soh_true)) ** 2)
    soh_r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    rul_mae = float(np.mean(np.abs(rul_pred - rul_true))) if len(rul_pred) else float("nan")

    # Short-horizon (capped) RUL MAE across multiple horizons
    horizon_list = [45.0, 50.0, 100.0, 150.0, 200.0, HORIZON_CYCLES]
    horizon_mae = {}
    for h in horizon_list:
        pred_cap = np.clip(rul_pred, 0.0, h)
        true_cap = np.clip(rul_true, 0.0, h)
        horizon_mae[str(int(h))] = float(np.mean(np.abs(pred_cap - true_cap))) if len(rul_pred) else float("nan")

    return {
        "soh_mae": soh_mae,
        "soh_r2": soh_r2,
        "rul_mae": rul_mae,
        "horizon_mae": horizon_mae,
        "horizon_cycles": HORIZON_CYCLES,
        "n": int(len(soh_pred)),
    }


def populate_memory_bank(model, loader, device="cpu", max_entries=4000):
    """Fill memory bank with LCO training samples for retrieval."""
    model.memory_bank.clear()
    added = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if added >= max_entries:
                break
            features = batch["features"].to(device)
            context = batch["context"].to(device)
            chem_id = batch["chem_id"].to(device)

            features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            context = torch.nan_to_num(context, nan=0.0)

            _, _, _, latent = model(features, context, chem_id)
            soh = batch["soh"].cpu().numpy()
            rul_norm = batch["rul_normalized"].cpu().numpy()

            for i in range(latent.shape[0]):
                if added >= max_entries:
                    break
                model.memory_bank.add(latent[i], float(soh[i]), float(rul_norm[i]))
                added += 1
    return added


def main():
    set_seed(42)
    use_lithium = False
    epochs = 100

    print("=" * 70)
    print("RETRAIN HERO ZERO-SHOT (LCO -> TJU NCM/NCA)")
    print("=" * 70)

    print("\n[1/4] Loading samples...")
    train_samples, test_samples = load_samples(use_lithium=use_lithium, max_train_samples=4000)
    print(f"  LCO train samples: {len(train_samples)}")
    print(f"  TJU test samples:  {len(test_samples)}")

    if not train_samples or not test_samples:
        print("ERROR: Missing train or test samples.")
        return

    print("\n[2/4] Computing feature normalization...")
    mean, std = compute_feature_norm(train_samples)

    rng = np.random.default_rng(42)
    indices = rng.permutation(len(train_samples))
    split = int(0.85 * len(indices))
    train_idx, val_idx = indices[:split], indices[split:]
    train_set = [train_samples[i] for i in train_idx]
    val_set = [train_samples[i] for i in val_idx]

    train_loader = DataLoader(ZeroShotDataset(train_set, mean, std), batch_size=64, shuffle=True)
    val_loader = DataLoader(ZeroShotDataset(val_set, mean, std), batch_size=64, shuffle=False)
    test_loader = DataLoader(ZeroShotDataset(test_samples, mean, std), batch_size=64, shuffle=False)

    feature_dim = train_samples[0].features.shape[0]
    print(f"\n[3/4] Training HERO (feature_dim={feature_dim}, epochs={epochs})...")
    model = RADDecoupledModel(
        feature_dim=feature_dim,
        context_dim=5,
        hidden_dim=128,
        latent_dim=64,
        n_chemistries=5,
        device="cpu",
    )

    train_combined_model(train_loader, val_loader, model, device="cpu", epochs=epochs)

    print("\n[4/4] Populating memory bank and evaluating on TJU zero-shot...")
    added = populate_memory_bank(model, train_loader, device="cpu", max_entries=4000)
    metrics = evaluate_model(model, test_loader, device="cpu")

    print("\n" + "=" * 70)
    print("TJU ZERO-SHOT RESULTS (RETRAINED)")
    print("=" * 70)
    print(f"  SOH MAE:  {metrics['soh_mae']:.2f}%")
    print(f"  SOH R²:   {metrics['soh_r2']:.4f}")
    print(f"  RUL MAE:  {metrics['rul_mae']:.1f} cycles")
    for h, mae in metrics["horizon_mae"].items():
        print(f"  RUL MAE (cap {h}): {mae:.1f} cycles")
    print(f"  N:        {metrics['n']}")
    print(f"  Memory:   {added} entries")

    out_path = Path("reports/tju_zeroshot_retrained_results.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "dataset": "TJU (NCM/NCA)",
                "train_chemistry": "LCO (NASA/CALCE/Oxford)",
                "zero_shot": True,
                "use_lithium_features": use_lithium,
                "epochs": epochs,
                "rul_scale_cycles": RUL_SCALE_CYCLES,
                "metrics": metrics,
                "memory_entries": added,
            },
            f,
            indent=2,
        )

    print(f"\n✓ Results saved to {out_path}")


if __name__ == "__main__":
    main()
