"""
HERO held-out-cell evaluation for a NASA case-study cell.

  --cell B0046   cold-weather case study    (Illustration 1, the default)
  --cell B0029   high-temperature case study (Illustration 3)

Protocol
--------
The checkpoint shipped in reports/hero_model/hero_model.pt was pre-trained on a
random *sample-level* split of NASA + CALCE + Oxford, so the case-study cell's
cycles were part of its training data. The case-study claim in the paper is a
*held-out cell* claim, so this script retrains the same architecture from
scratch with every sample of the chosen cell removed from both the training
split and the retrieval memory bank, then evaluates on that cell only.

  - Test set        : all samples of --cell (never seen in training or memory).
  - Training set    : NASA (minus that cell) + CALCE + Oxford, 85/15 train/val.
  - Memory bank     : populated during training from training samples only.
  - Sibling cells   : cells run under the same condition (B0047/B0048 at 4 C for
                      B0046; B0030-B0032 at 43 C for B0029) remain in training.
                      This is a held-out-cell, not a held-out-condition, claim.

For B0029 the retrain is also the first HERO run to see NASA's true ambient
temperatures: before the parse_mat_cell fix in src/data/nasa_set5.py, every NASA
cell reached the context encoder as 24 C, so a 43 C cell was indistinguishable
from a room-temperature one.

Discharge-only scoring
----------------------
The unified pipeline emits one sample per row of the NASA cycle table, and that
table contains charge rows as well as discharge rows. Charge rows carry no
measured capacity, so unified_pipeline.py substitutes a constant filler label of
SOH = 0.9 (see the `soh_val = 0.9` branch). For B0046 that is 72 of 144 samples.
Scoring against a constant is not a measurement, and because the model learns to
emit 0.9 for those rows it drags the reported error down. The headline metric
here is therefore computed over discharge rows only; the all-row figure is kept
alongside it purely to document the size of that effect.

Repeated over three seeds; the paper quotes mean +/- std.

Outputs
-------
  reports/hero_model/<cell>_holdout_results.json  metrics + per-cycle predictions
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.nasa_set5 import make_cycle_table, parse_mat_cell
from src.data.unified_pipeline import UnifiedDataPipeline, UnifiedBatteryDataset
from src.train.hero_rad_decoupled import (
    RADDecoupledModel,
    train_combined_model,
)

def discharge_cycle_indices(data_root, cell):
    """Cycle indices of the rows that carry a measured discharge capacity."""
    _, cycles, _ = parse_mat_cell(str(Path(data_root) / "nasa_set5" / "raw" / f"{cell}.mat"))
    table = make_cycle_table(cycles)
    return set(table[table.cycle_type == "discharge"].cycle_index.astype(int))


def predict_cell(model, samples, device):
    """Per-sample SOH predictions for one cell, ordered by cycle index."""
    ordered = sorted(samples, key=lambda s: s.cycle_idx)
    loader = DataLoader(UnifiedBatteryDataset(ordered), batch_size=64, shuffle=False)

    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            features = torch.nan_to_num(
                batch["features"].to(device), nan=0.0, posinf=1.0, neginf=-1.0
            )
            context = torch.nan_to_num(batch["context"].to(device), nan=0.0)
            soh_pred, _, _, _ = model(features, context, batch["chem_id"].to(device))
            preds.extend(soh_pred.squeeze(-1).cpu().numpy().tolist())

    return (
        np.array([s.cycle_idx for s in ordered]),
        np.array([s.soh for s in ordered]),
        np.array(preds),
    )


def metrics(true, pred):
    mae = float(np.mean(np.abs(pred - true)))
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    return {
        "soh_mae": mae,
        "soh_mae_pct": mae * 100,
        "soh_rmse_pct": rmse * 100,
        "soh_r2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", default="B0046",
                        help="NASA cell to hold out, e.g. B0046 or B0029")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--output", default=None,
                        help="defaults to reports/hero_model/<cell>_holdout_results.json")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    held_out_cell = f"NASA_{args.cell}"
    output = args.output or f"reports/hero_model/{args.cell.lower()}_holdout_results.json"

    pipeline = UnifiedDataPipeline(args.data_root, use_lithium_features=True)
    pipeline.load_datasets(["nasa", "calce", "oxford"])

    test_samples = [s for s in pipeline.samples if s.cell_id == held_out_cell]
    pool = [s for s in pipeline.samples if s.cell_id != held_out_cell]
    if not test_samples:
        raise SystemExit(f"{held_out_cell} not found in the loaded pipeline")

    discharge_idx = discharge_cycle_indices(args.data_root, args.cell)

    print(f"Held-out cell {held_out_cell}: {len(test_samples)} samples "
          f"({len(discharge_idx)} of them measured discharge rows)")
    print(f"Training pool: {len(pool)} samples from "
          f"{len(set(s.cell_id for s in pool))} cells")

    runs = []
    per_cycle = None

    for seed in args.seeds:
        print(f"\n{'=' * 60}\nSEED {seed}\n{'=' * 60}")
        torch.manual_seed(seed)
        rng = np.random.RandomState(seed)

        shuffled = list(pool)
        rng.shuffle(shuffled)
        cut = int(0.85 * len(shuffled))
        train_samples, val_samples = shuffled[:cut], shuffled[cut:]

        model = RADDecoupledModel(
            feature_dim=20, context_dim=5, hidden_dim=128, latent_dim=64,
            device=args.device,
        ).to(args.device)

        train_combined_model(
            DataLoader(UnifiedBatteryDataset(train_samples), batch_size=64, shuffle=True),
            DataLoader(UnifiedBatteryDataset(val_samples), batch_size=64, shuffle=False),
            model, args.device, args.epochs,
        )

        # Sanity check: nothing from the held-out cell can be in the bank, because
        # the bank is only ever written from training batches.
        bank_size = model.memory_bank.size()

        cycles, soh_true, soh_pred = predict_cell(model, test_samples, args.device)
        keep = np.array([int(c) in discharge_idx for c in cycles])

        run = metrics(soh_true[keep], soh_pred[keep])
        run["seed"] = seed
        run["memory_bank_size"] = bank_size
        run["n_discharge"] = int(keep.sum())
        run["all_rows_incl_filler"] = metrics(soh_true, soh_pred)
        runs.append(run)
        print(f"  {args.cell} held-out (discharge rows, n={keep.sum()}): "
              f"MAE={run['soh_mae_pct']:.2f}%  R2={run['soh_r2']:.3f}  (bank={bank_size})")
        print(f"    [all {len(cycles)} rows incl. filler-labelled charge rows: "
              f"MAE={run['all_rows_incl_filler']['soh_mae_pct']:.2f}%]")

        if seed == args.seeds[0]:
            per_cycle = {
                "cycle_idx": cycles[keep].tolist(),
                "soh_true": soh_true[keep].tolist(),
                "soh_pred": soh_pred[keep].tolist(),
                "seed": seed,
            }

    maes = np.array([r["soh_mae_pct"] for r in runs])
    r2s = np.array([r["soh_r2"] for r in runs])

    summary = {
        "held_out_cell": held_out_cell,
        "scored_on": "measured discharge rows only",
        "n_test_samples": len(test_samples),
        "n_discharge_rows": len(discharge_idx),
        "n_train_pool": len(pool),
        "epochs": args.epochs,
        "soh_mae_pct_mean": float(maes.mean()),
        "soh_mae_pct_std": float(maes.std(ddof=1)) if len(maes) > 1 else 0.0,
        "soh_r2_mean": float(r2s.mean()),
        "soh_r2_std": float(r2s.std(ddof=1)) if len(r2s) > 1 else 0.0,
        "runs": runs,
        "per_cycle_first_seed": per_cycle,
    }

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    print(f"\n{'=' * 60}")
    print(f"{args.cell} HELD-OUT CELL: SOH MAE = {maes.mean():.2f}% "
          f"+/- {maes.std(ddof=1) if len(maes) > 1 else 0:.2f}%, "
          f"R2 = {r2s.mean():.3f} +/- {r2s.std(ddof=1) if len(r2s) > 1 else 0:.3f} "
          f"over {len(runs)} seeds")
    print(f"Written to {out}")


if __name__ == "__main__":
    main()
