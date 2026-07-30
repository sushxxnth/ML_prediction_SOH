"""
HERO zero-shot evaluation on a single TJU eval cell (CY25_2), for the
paper's third end-to-end case study (Illustration 3).

This reuses run_zeroshot_table_rebuild.py's protocol byte-for-byte (same
LCO source sample, same seed, same epoch count, same LCO-only memory bank)
so the Stage-1 numbers here are directly comparable to Table 1's published
zero-shot row -- the only difference is that Table 1 aggregates metrics over
both eval cells (CY25_2 + CY25_3, 1,841 samples combined) while this script
evaluates CY25_2 alone, in cycle order, so a per-cycle trajectory can be
plotted (Table 1 only ever reports the aggregate).

TJU cells, sorted: CY25_1 (adapt cell, held out entirely here as it is in
Table 1 -- unused), CY25_2, CY25_3 (eval cells). No TJU sample of any cell
enters training or the memory bank.

Run: PYTHONPATH=. arch -arm64 python3 scripts/run_tju_zeroshot_case.py
Output: reports/hero_model/tju_cy25_2_zeroshot_results.json
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.unified_pipeline import UnifiedDataPipeline
from src.train.hero_rad_decoupled import RADDecoupledModel, train_combined_model
from src.retrain_tju_zeroshot import ZeroShotDataset, populate_memory_bank
from run_zeroshot_table_rebuild import RUL_SCALE_CYCLES, SEED, HERO_EPOCHS, set_seed

EVAL_CELL = "CY25_2"
OUT = ROOT / "reports" / "hero_model" / "tju_cy25_2_zeroshot_results.json"


def main():
    set_seed()
    pipeline = UnifiedDataPipeline(str(ROOT / "data"), use_lithium_features=True)
    pipeline.load_datasets(["nasa", "calce", "oxford", "tju"])

    source = [s for s in pipeline.samples if s.source_dataset in ("nasa", "calce", "oxford")]
    target = [s for s in pipeline.samples if s.source_dataset == "tju"]
    cells = sorted({s.cell_id for s in target})
    print(f"TJU cells: {cells} (adapt cell {cells[0]!r} excluded from this run entirely)")

    eval_samples = sorted([s for s in target if s.cell_id == EVAL_CELL],
                          key=lambda s: s.cycle_idx)
    if not eval_samples:
        raise SystemExit(f"{EVAL_CELL} not found among TJU samples")

    rng = np.random.default_rng(SEED)
    if len(source) > 4000:
        idx = rng.choice(len(source), size=4000, replace=False)
        source = [source[i] for i in idx]
    print(f"LCO source pool: {len(source)} samples; "
          f"eval cell {EVAL_CELL}: {len(eval_samples)} samples "
          f"(cycles {eval_samples[0].cycle_idx}-{eval_samples[-1].cycle_idx})")

    feats = np.stack([np.nan_to_num(s.features, nan=0.0).reshape(-1)
                      for s in source]).astype(np.float32)
    mean, std = feats.mean(0), feats.std(0)
    std = np.where(std < 1e-6, 1.0, std)

    idx = rng.permutation(len(source))
    split = int(0.85 * len(idx))
    train_set = [source[i] for i in idx[:split]]
    val_set = [source[i] for i in idx[split:]]

    train_loader = DataLoader(ZeroShotDataset(train_set, mean, std), batch_size=64, shuffle=True)
    val_loader = DataLoader(ZeroShotDataset(val_set, mean, std), batch_size=64, shuffle=False)
    eval_loader = DataLoader(ZeroShotDataset(eval_samples, mean, std), batch_size=64, shuffle=False)

    feature_dim = int(feats.shape[1])
    model = RADDecoupledModel(feature_dim=feature_dim, context_dim=5, hidden_dim=128,
                              latent_dim=64, n_chemistries=5, device="cpu")
    print(f"Training HERO on LCO (feature_dim={feature_dim}, epochs={HERO_EPOCHS}, "
          f"seed={SEED}, matching Table 1's protocol)...")
    train_combined_model(train_loader, val_loader, model, device="cpu", epochs=HERO_EPOCHS)

    added = populate_memory_bank(model, train_loader, device="cpu", max_entries=4000)
    print(f"Memory bank populated with {added} LCO entries (0 TJU).")

    model.eval()
    soh_p, soh_t, rul_p, rul_t, cyc = [], [], [], [], []
    with torch.no_grad():
        for batch in eval_loader:
            f = torch.nan_to_num(batch["features"], nan=0.0, posinf=1.0, neginf=-1.0)
            c = torch.nan_to_num(batch["context"], nan=0.0)
            soh_pred, rul_pred, _, _ = model(f, c, batch["chem_id"])
            soh_p.extend(soh_pred.squeeze(-1).numpy())
            soh_t.extend(batch["soh"].numpy())
            rul_p.extend(rul_pred.squeeze(-1).numpy() * RUL_SCALE_CYCLES)
            rul_t.extend(batch["rul_cycles"].numpy())
    for s in eval_samples:
        cyc.append(int(s.cycle_idx))

    soh_p, soh_t = np.array(soh_p), np.array(soh_t)
    rul_p, rul_t = np.array(rul_p), np.array(rul_t)

    mae = float(np.mean(np.abs(soh_p - soh_t)) * 100)
    ss_res = float(np.sum((soh_t - soh_p) ** 2))
    ss_tot = float(np.sum((soh_t - soh_t.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rul_mae = float(np.mean(np.abs(rul_p - rul_t)))

    print(f"\n{EVAL_CELL} zero-shot (LCO -> TJU, no target-chemistry sample in "
          f"training or memory bank):")
    print(f"  SOH MAE={mae:.2f}%  R2={r2:.3f}  RUL MAE={rul_mae:.1f} cycles")

    summary = {
        "eval_cell": f"TJU_{EVAL_CELL}",
        "protocol": ("Zero-shot: LCO (NASA+CALCE+Oxford) source, TJU adapt cell "
                     f"{cells[0]!r} excluded, evaluated on {EVAL_CELL} only. "
                     "Same source sampling, seed, and epoch count as "
                     "run_zeroshot_table_rebuild.py (Table 1)."),
        "seed": SEED,
        "epochs": HERO_EPOCHS,
        "n_source": len(source),
        "n_eval": len(eval_samples),
        "soh_mae_pct": mae,
        "soh_r2": r2,
        "rul_mae_cycles": rul_mae,
        "memory_bank_size": added,
        "per_cycle": {
            "cycle_idx": cyc,
            "soh_true": soh_t.tolist(),
            "soh_pred": soh_p.tolist(),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"\nWritten to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
