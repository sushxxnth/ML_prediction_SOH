"""
Phase 2 of the zero-shot table rebuild: HERO + memory-bank injection.

Retrains HERO exactly as run_zeroshot_table_rebuild.py (same seed/protocol,
LCO source, LCO-only bank), saves the checkpoint, then evaluates:
  1. HERO zero-shot (sanity anchor — must match phase 1: SOH 9.12%, RUL 182.6)
  2. HERO + bank injection: the SAME labeled adaptation cell given to the
     few-shot fine-tuned baselines is added to the memory bank (no gradient
     updates, no retraining), then TJU eval cells are re-evaluated.

This is HERO's native adaptation mechanism and the apples-to-apples
counterpart of few-shot fine-tuning for the R2.6 comparison.

Run:  PYTHONPATH=. arch -arm64 python3 run_zeroshot_bank_injection.py
Output: reports/zeroshot_bank_injection.json, reports/hero_zeroshot_rebuild.pt
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.train.hero_rad_decoupled import RADDecoupledModel, train_combined_model
from src.retrain_tju_zeroshot import ZeroShotDataset, populate_memory_bank
from run_zeroshot_table_rebuild import (
    RUL_SCALE_CYCLES, SEED, HERO_EPOCHS, set_seed, load_split, metrics_from,
)

CKPT = Path("reports/hero_zeroshot_rebuild.pt")


def eval_hero(model, loader):
    model.eval()
    soh_p, soh_t, rul_p, rul_t = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            f = torch.nan_to_num(batch["features"], nan=0.0, posinf=1.0, neginf=-1.0)
            c = torch.nan_to_num(batch["context"], nan=0.0)
            soh_pred, rul_pred, _, _ = model(f, c, batch["chem_id"])
            soh_p.extend(soh_pred.squeeze().numpy())
            soh_t.extend(batch["soh"].numpy())
            rul_p.extend(rul_pred.squeeze().numpy() * RUL_SCALE_CYCLES)
            rul_t.extend(batch["rul_cycles"].numpy())
    return metrics_from(np.array(soh_p), np.array(soh_t), np.array(rul_p), np.array(rul_t))


def inject_cell(model, loader):
    """Add the adaptation cell's trajectories to the memory bank (no training)."""
    added = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            f = torch.nan_to_num(batch["features"], nan=0.0, posinf=1.0, neginf=-1.0)
            c = torch.nan_to_num(batch["context"], nan=0.0)
            _, _, _, latent = model(f, c, batch["chem_id"])
            soh = batch["soh"].numpy()
            rul_norm = batch["rul_normalized"].numpy()
            for i in range(latent.shape[0]):
                model.memory_bank.add(latent[i], float(soh[i]), float(rul_norm[i]))
                added += 1
    return added


def main():
    set_seed(SEED)
    source, adapt, evals, adapt_cell, eval_cells = load_split()

    feats = np.stack([np.nan_to_num(s.features, nan=0.0).reshape(-1) for s in source]).astype(np.float32)
    mean, std = feats.mean(0), feats.std(0)
    std = np.where(std < 1e-6, 1.0, std)

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(source))
    split = int(0.85 * len(idx))
    train_set = [source[i] for i in idx[:split]]
    val_set = [source[i] for i in idx[split:]]

    train_loader = DataLoader(ZeroShotDataset(train_set, mean, std), batch_size=64, shuffle=True)
    val_loader = DataLoader(ZeroShotDataset(val_set, mean, std), batch_size=64, shuffle=False)
    adapt_loader = DataLoader(ZeroShotDataset(adapt, mean, std), batch_size=64, shuffle=False)
    eval_loader = DataLoader(ZeroShotDataset(evals, mean, std), batch_size=64, shuffle=False)

    feature_dim = int(feats.shape[1])
    model = RADDecoupledModel(feature_dim=feature_dim, context_dim=5, hidden_dim=128,
                              latent_dim=64, n_chemistries=5, device="cpu")

    if CKPT.exists():
        print(f"Loading cached checkpoint {CKPT}")
        model.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=False))
    else:
        print(f"Training HERO (feature_dim={feature_dim}, epochs={HERO_EPOCHS})...")
        set_seed(SEED)
        train_combined_model(train_loader, val_loader, model, device="cpu", epochs=HERO_EPOCHS)
        torch.save(model.state_dict(), CKPT)
        print(f"Saved checkpoint to {CKPT}")

    added = populate_memory_bank(model, train_loader, device="cpu", max_entries=4000)
    print(f"LCO-only bank: {added} entries")

    zero_shot = eval_hero(model, eval_loader)
    print("HERO (zero-shot):", zero_shot)

    injected = inject_cell(model, adapt_loader)
    print(f"Injected {injected} adapt-cell entries "
          f"({injected / (added + injected) * 100:.1f}% of bank)")
    with_injection = eval_hero(model, eval_loader)
    print("HERO (+bank injection):", with_injection)

    out = Path("reports/zeroshot_bank_injection.json")
    with open(out, "w") as f:
        json.dump({
            "protocol": {
                "source": "LCO (NASA/CALCE/Oxford), 4000 samples, 20D features",
                "target": "TJU NCM/NCA; adapt cell injected into bank, eval on remaining cells",
                "adapt_cell": str(adapt_cell),
                "eval_cells": [str(c) for c in eval_cells],
                "bank_lco_entries": added,
                "bank_injected_entries": injected,
                "seed": SEED,
            },
            "HERO (zero-shot)": zero_shot,
            "HERO (+bank injection)": with_injection,
        }, f, indent=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
