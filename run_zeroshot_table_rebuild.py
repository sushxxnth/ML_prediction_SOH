"""
Honest rebuild of the zero-shot chemistry-transfer table (tab:zeroshot).

Background: the paper's table values (HERO 44.0 RUL / 0.74% SOH, LSTM 98.3,
etc.) exist only as a static JSON with no reproducing script in the repo;
reconstruction attempts do not come close. This script regenerates the entire
table from scratch under ONE clean, fully reproducible protocol:

  Source:   LCO (NASA, CALCE, Oxford), 4,000 samples, 20D unified features
            (9 base + 11 chemistry-invariant lithium-inventory features).
  Target:   TJU NCM/NCA (Dataset_3, 1C). The adaptation cell (first cell by
            sorted id) supplies labeled data for few-shot fine-tuning and
            unlabeled data for CORAL/MMD; ALL methods are evaluated on the
            remaining cells only. No TJU sample enters HERO's memory bank.
  Metrics:  SOH MAE (%), SOH R^2, uncapped RUL MAE (cycles, scale 1000).

Methods: HERO (retrained on LCO, LCO-only memory bank), LSTM, GRU, CNN-LSTM,
Transformer, MLP, Random Forest; plus CORAL / MMD (unlabeled adaptation) and
few-shot fine-tuning (labeled adaptation cell) variants.

Run:  PYTHONPATH=. arch -arm64 python3 run_zeroshot_table_rebuild.py
Output: reports/zeroshot_table_rebuild.json
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.unified_pipeline import UnifiedDataPipeline
from src.train.hero_rad_decoupled import RADDecoupledModel, train_combined_model
from src.recreate_zeroshot_baseline import (
    standardize_fit_transform,
    standardize_transform,
    mean_absolute_error_np,
    r2_score_np,
    run_random_forest_subprocess,
)
from src.retrain_tju_zeroshot import ZeroShotDataset, populate_memory_bank
from run_domain_adaptation_baselines import (
    coral_transform,
    MMDRegressor,
    train_mmd,
    train_model,
)

RUL_SCALE_CYCLES = 1000.0
SEED = 42
HERO_EPOCHS = int(os.environ.get("REBUILD_HERO_EPOCHS", 100))
BASELINE_EPOCHS = int(os.environ.get("REBUILD_BASELINE_EPOCHS", 200))
FT_EPOCHS = int(os.environ.get("REBUILD_FT_EPOCHS", 50))
FT_LR = 1e-4
MMD_LAMBDA = 0.1  # 1.0 diverged in the Panasonic pilot run


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def load_split():
    pipeline = UnifiedDataPipeline("data", use_lithium_features=True)
    pipeline.load_datasets(["nasa", "calce", "oxford", "tju"])

    source = [s for s in pipeline.samples if s.source_dataset in ("nasa", "calce", "oxford")]
    target = [s for s in pipeline.samples if s.source_dataset == "tju"]

    rng = np.random.default_rng(SEED)
    if len(source) > 4000:
        idx = rng.choice(len(source), size=4000, replace=False)
        source = [source[i] for i in idx]

    cells = sorted({s.cell_id for s in target})
    adapt_cell, eval_cells = cells[0], cells[1:]
    adapt = [s for s in target if s.cell_id == adapt_cell]
    evals = [s for s in target if s.cell_id != adapt_cell]
    print(f"TJU cells: {cells}")
    print(f"adapt cell: {adapt_cell} ({len(adapt)} samples); "
          f"eval cells: {eval_cells} ({len(evals)} samples)")
    return source, adapt, evals, adapt_cell, eval_cells


def to_arrays(samples):
    X, soh, rul_norm, rul_cycles = [], [], [], []
    for s in samples:
        if s.features is None or s.soh is None or s.rul is None:
            continue
        f = np.nan_to_num(s.features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        if not (np.isfinite(s.soh) and np.isfinite(s.rul)):
            continue
        X.append(f.reshape(-1))
        soh.append(float(np.clip(s.soh, 0.0, 1.2)))
        rc = float(max(s.rul, 0.0))
        rul_cycles.append(rc)
        rul_norm.append(float(np.clip(rc / RUL_SCALE_CYCLES, 0.0, 1.0)))
    return (np.array(X, np.float32), np.array(soh, np.float32),
            np.array(rul_norm, np.float32), np.array(rul_cycles, np.float32))


def metrics_from(soh_pred, soh_true, rul_pred_cycles, rul_true_cycles):
    return {
        "soh_mae": float(mean_absolute_error_np(soh_true, soh_pred) * 100.0),
        "soh_r2": float(r2_score_np(soh_true, soh_pred)),
        "rul_mae": float(np.mean(np.abs(rul_pred_cycles - rul_true_cycles))),
    }


def eval_torch(model, X_eval, soh_eval, rul_cycles_eval):
    model.eval()
    with torch.no_grad():
        soh_pred, rul_pred = model(X_eval)
    return metrics_from(soh_pred.numpy(), soh_eval,
                        rul_pred.numpy() * RUL_SCALE_CYCLES, rul_cycles_eval)


def run_hero(source, evals):
    """Retrain HERO on LCO with an LCO-only memory bank; evaluate on TJU eval cells."""
    set_seed()
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
    eval_loader = DataLoader(ZeroShotDataset(evals, mean, std), batch_size=64, shuffle=False)

    feature_dim = int(feats.shape[1])
    model = RADDecoupledModel(feature_dim=feature_dim, context_dim=5, hidden_dim=128,
                              latent_dim=64, n_chemistries=5, device="cpu")
    print(f"Training HERO (feature_dim={feature_dim}, epochs={HERO_EPOCHS})...")
    train_combined_model(train_loader, val_loader, model, device="cpu", epochs=HERO_EPOCHS)

    added = populate_memory_bank(model, train_loader, device="cpu", max_entries=4000)
    print(f"Memory bank populated with {added} LCO entries (0 TJU).")

    model.eval()
    soh_p, soh_t, rul_p, rul_t = [], [], [], []
    with torch.no_grad():
        for batch in eval_loader:
            f = torch.nan_to_num(batch["features"], nan=0.0, posinf=1.0, neginf=-1.0)
            c = torch.nan_to_num(batch["context"], nan=0.0)
            soh_pred, rul_pred, _, _ = model(f, c, batch["chem_id"])
            soh_p.extend(soh_pred.squeeze().numpy())
            soh_t.extend(batch["soh"].numpy())
            rul_p.extend(rul_pred.squeeze().numpy() * RUL_SCALE_CYCLES)
            rul_t.extend(batch["rul_cycles"].numpy())
    return metrics_from(np.array(soh_p), np.array(soh_t), np.array(rul_p), np.array(rul_t))


def main():
    from src.sota_baseline_comparison import (
        LSTMBaseline, GRUBaseline, CNNLSTMBaseline, TransformerBaseline, MLPBaseline
    )

    set_seed()
    source, adapt, evals, adapt_cell, eval_cells = load_split()

    X_src, soh_src, rul_norm_src, _ = to_arrays(source)
    X_ad, soh_ad, rul_norm_ad, _ = to_arrays(adapt)
    X_ev, soh_ev, _, rul_cycles_ev = to_arrays(evals)
    print(f"arrays: source={len(X_src)} adapt={len(X_ad)} eval={len(X_ev)}")

    X_src_s, mean, std = standardize_fit_transform(X_src)
    X_ad_s = standardize_transform(X_ad, mean, std)
    X_ev_s = standardize_transform(X_ev, mean, std)

    t = lambda a: torch.tensor(a, dtype=torch.float32)
    results = {"protocol": {
        "source": "LCO (NASA, CALCE, Oxford), 4000 samples, 20D lithium-augmented features",
        "target": "TJU NCM/NCA (Dataset_3, 1C)",
        "adapt_cell": str(adapt_cell),
        "eval_cells": [str(c) for c in eval_cells],
        "n_eval": int(len(X_ev)),
        "rul": "uncapped MAE in cycles (scale 1000, EOL 80%)",
        "fewshot": f"labeled adaptation cell ({len(X_ad)} samples), FT {FT_EPOCHS} epochs lr {FT_LR}",
        "mmd_lambda": MMD_LAMBDA,
        "seed": SEED,
    }}

    # ---- HERO ----
    results["HERO (zero-shot)"] = run_hero(source, evals)
    print("HERO:", results["HERO (zero-shot)"])

    # ---- Standard baselines ----
    factories = {
        "LSTM": lambda: LSTMBaseline(X_src.shape[1]),
        "GRU": lambda: GRUBaseline(X_src.shape[1]),
        "CNN-LSTM": lambda: CNNLSTMBaseline(X_src.shape[1]),
        "Transformer": lambda: TransformerBaseline(X_src.shape[1]),
        "MLP": lambda: MLPBaseline(X_src.shape[1]),
    }
    for name, factory in factories.items():
        set_seed()
        m = train_model(factory(), t(X_src_s), t(soh_src), t(rul_norm_src),
                        epochs=BASELINE_EPOCHS)
        results[f"{name} (zero-shot)"] = eval_torch(m, t(X_ev_s), soh_ev, rul_cycles_ev)
        print(name, results[f"{name} (zero-shot)"])

    # ---- Random Forest ----
    rf = run_random_forest_subprocess(X_src, soh_src, rul_norm_src,
                                      X_ev, soh_ev, rul_cycles_ev)
    results["Random Forest (zero-shot)"] = rf
    print("RF:", rf)

    # ---- Domain adaptation: CORAL (unlabeled adapt cell) ----
    for name in ("LSTM", "Transformer", "MLP"):
        set_seed()
        X_src_coral = coral_transform(X_src_s, X_ad_s).astype(np.float32)
        m = train_model(factories[name](), t(X_src_coral), t(soh_src), t(rul_norm_src),
                        epochs=BASELINE_EPOCHS)
        results[f"{name} + CORAL"] = eval_torch(m, t(X_ev_s), soh_ev, rul_cycles_ev)
        print(name, "+CORAL", results[f"{name} + CORAL"])

    # ---- Domain adaptation: MMD (unlabeled adapt cell) ----
    set_seed()
    mmd = MMDRegressor(X_src.shape[1])
    train_mmd(mmd, t(X_src_s), t(soh_src), t(rul_norm_src), t(X_ad_s), lam=MMD_LAMBDA)
    mmd.eval_mode()
    with torch.no_grad():
        soh_pred, rul_pred = mmd(t(X_ev_s))
    results["MLP + MMD"] = metrics_from(soh_pred.numpy(), soh_ev,
                                        rul_pred.numpy() * RUL_SCALE_CYCLES, rul_cycles_ev)
    print("MLP +MMD", results["MLP + MMD"])

    # ---- Few-shot fine-tuning (labeled adapt cell) ----
    for name in ("LSTM", "Transformer", "MLP"):
        set_seed()
        m = train_model(factories[name](), t(X_src_s), t(soh_src), t(rul_norm_src),
                        epochs=BASELINE_EPOCHS)
        m = train_model(m, t(X_ad_s), t(soh_ad), t(rul_norm_ad),
                        epochs=FT_EPOCHS, lr=FT_LR)
        results[f"{name} + few-shot FT"] = eval_torch(m, t(X_ev_s), soh_ev, rul_cycles_ev)
        print(name, "+FT", results[f"{name} + few-shot FT"])

    out = Path("reports/zeroshot_table_rebuild.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out}\n")

    print(f"{'Method':<28}{'SOH MAE':>10}{'SOH R2':>10}{'RUL MAE':>12}")
    for k, v in results.items():
        if k == "protocol":
            continue
        print(f"{k:<28}{v['soh_mae']:>9.2f}%{v['soh_r2']:>10.3f}{v['rul_mae']:>10.1f} cy")


if __name__ == "__main__":
    main()
