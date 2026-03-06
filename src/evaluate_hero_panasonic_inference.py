"""
Panasonic Zero-Shot Inference Using Existing HERO Checkpoint

Runs inference only (no retraining) to produce Panasonic NCA results
using the existing HERO model checkpoint. Builds a memory bank from LCO
samples for retrieval, then evaluates on Panasonic 18650PF.
"""

import os
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Avoid OpenMP shared memory/runtime conflicts
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("KMP_USE_SHM", "0")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.train.hero_rad_decoupled import RADDecoupledModel
from src.data.unified_pipeline import UnifiedDataPipeline


RUL_SCALE_CYCLES = 1000.0  # Paper-aligned RUL scaling


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


class ZeroShotDataset(Dataset):
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
            "rul_cycles": torch.tensor(rul_cycles, dtype=torch.float32),
        }


def load_samples(max_train_samples: int = 4000, max_test_samples: int = 500):
    pipeline = UnifiedDataPipeline("data", use_lithium_features=True)
    pipeline.load_datasets(["nasa", "calce", "oxford", "panasonic_18650pf"])

    source_samples = [s for s in pipeline.samples if s.source_dataset in ["nasa", "calce", "oxford"]]
    target_samples = [s for s in pipeline.samples if s.source_dataset in ["panasonic_18650pf", "panasonic"]]

    rng_train = np.random.default_rng(42)
    if max_train_samples and len(source_samples) > max_train_samples:
        idx = rng_train.choice(len(source_samples), size=max_train_samples, replace=False)
        source_samples = [source_samples[i] for i in idx]

    rng_test = np.random.default_rng(123)
    if max_test_samples and len(target_samples) > max_test_samples:
        idx = rng_test.choice(len(target_samples), size=max_test_samples, replace=False)
        target_samples = [target_samples[i] for i in idx]

    return source_samples, target_samples


def build_memory_bank(model, samples, device="cpu", batch_size=128, max_entries=4000):
    loader = DataLoader(ZeroShotDataset(samples), batch_size=batch_size, shuffle=False)
    added = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if added >= max_entries:
                break
            features = batch["features"].to(device)
            context = batch["context"].to(device)
            chem_id = batch["chem_id"].to(device)
            soh = batch["soh"].cpu().numpy()
            rul_norm = batch["rul_normalized"].cpu().numpy()

            features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            context = torch.nan_to_num(context, nan=0.0)

            _, _, _, latent = model(features, context, chem_id)

            for i in range(latent.shape[0]):
                if added >= max_entries:
                    break
                model.memory_bank.add(latent[i], float(soh[i]), float(rul_norm[i]))
                added += 1

    return added


def evaluate_panasonic(model, samples, device="cpu", batch_size=128):
    loader = DataLoader(ZeroShotDataset(samples), batch_size=batch_size, shuffle=False)

    all_soh_pred, all_soh_true = [], []
    all_rul_pred, all_rul_true = [], []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            context = batch["context"].to(device)
            chem_id = batch["chem_id"].to(device)

            features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            context = torch.nan_to_num(context, nan=0.0)

            soh_pred, rul_pred, _, _ = model(features, context, chem_id)

            all_soh_pred.extend(soh_pred.squeeze().cpu().numpy())
            all_soh_true.extend(batch["soh"].cpu().numpy())

            rul_pred_cycles = rul_pred.squeeze().cpu().numpy() * RUL_SCALE_CYCLES
            all_rul_pred.extend(rul_pred_cycles)
            all_rul_true.extend(batch["rul_cycles"].cpu().numpy())

    soh_pred = np.array(all_soh_pred)
    soh_true = np.array(all_soh_true)
    rul_pred = np.array(all_rul_pred)
    rul_true = np.array(all_rul_true)

    valid = ~(np.isnan(soh_pred) | np.isnan(soh_true) | np.isnan(rul_pred) | np.isnan(rul_true))
    soh_pred, soh_true = soh_pred[valid], soh_true[valid]
    rul_pred, rul_true = rul_pred[valid], rul_true[valid]

    soh_mae = float(np.mean(np.abs(soh_pred - soh_true)) * 100.0)
    ss_res = np.sum((soh_true - soh_pred) ** 2)
    ss_tot = np.sum((soh_true - np.mean(soh_true)) ** 2)
    soh_r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    rul_mae = float(np.mean(np.abs(rul_pred - rul_true)))

    return {"soh_mae": soh_mae, "soh_r2": soh_r2, "rul_mae": rul_mae, "n": int(len(soh_pred))}


def main():
    set_seed(42)
    print("=" * 70)
    print("HERO ZERO-SHOT INFERENCE (Panasonic NCA, Inference-Only)")
    print("=" * 70)

    checkpoint = Path("reports/hero_model/hero_model.pt")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    print("\n[1/3] Loading data...")
    train_samples, test_samples = load_samples(max_train_samples=4000, max_test_samples=500)
    print(f"  LCO samples: {len(train_samples)}")
    print(f"  Panasonic samples: {len(test_samples)}")

    print("\n[2/3] Loading HERO checkpoint...")
    model = RADDecoupledModel(
        feature_dim=20,
        context_dim=5,
        hidden_dim=128,
        latent_dim=64,
        n_chemistries=5,
        device="cpu",
    )
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=False)

    print("\n[3/3] Building memory bank from LCO and evaluating Panasonic...")
    added = build_memory_bank(model, train_samples, device="cpu", max_entries=4000)
    print(f"  Memory entries added: {added}")

    metrics = evaluate_panasonic(model, test_samples, device="cpu")

    print("\n" + "=" * 70)
    print("PANASONIC ZERO-SHOT RESULTS (Inference-Only)")
    print("=" * 70)
    print(f"  SOH MAE:  {metrics['soh_mae']:.2f}%")
    print(f"  SOH R²:   {metrics['soh_r2']:.4f}")
    print(f"  RUL MAE:  {metrics['rul_mae']:.1f} cycles")
    print(f"  N:        {metrics['n']}")

    out_path = Path("reports/panasonic_zeroshot_inference_results.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "dataset": "Panasonic 18650PF (NCA)",
                "train_chemistry": "LCO (NASA/CALCE/Oxford)",
                "inference_only": True,
                "rul_scale_cycles": RUL_SCALE_CYCLES,
                "memory_entries": added,
                "metrics": metrics,
            },
            f,
            indent=2,
        )
    print(f"\n Saved to {out_path}")


if __name__ == "__main__":
    main()
