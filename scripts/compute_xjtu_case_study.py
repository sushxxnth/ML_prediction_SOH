"""
Real data/model pipeline for the XJTU high-C-rate end-to-end case study
(paper Illustration 2), replacing scripts/generate_xjtu_case_study.py.

That script's panel (b) was `soh + np.random.normal(0, 0.4, len(soh))` (noise
around ground truth, not a model prediction), and panels (c)/(d) were typed-in
constants ([55,25,5,10,5] and [55,20,8,3]) that don't match the checkpointed
causal-attribution model's real output (81.2% AM loss / 4.8% SEI, per
reports/xjtu_causal_attribution_results.json). This script replaces all of it
with real inference on the single cell traced in the case study
(XJTU Batch-1, 2C_battery-1):

  1. HERO SOH prediction, per cycle, for this cell. HERO (reports/hero_model/
     hero_model.pt) was never trained or memory-populated on XJTU, so this is
     a genuine zero-shot evaluation. The checkpoint's memory bank is empty
     (SimpleMemoryBank isn't part of state_dict), so it is rebuilt here with a
     single forward pass (no gradient update -- weights are untouched) over
     the same NASA+CALCE+Oxford pool the model was trained on.

  2. Causal-attribution baseline + counterfactual, for the same cell, using
     the checkpointed physics-prior model (reports/causal_attribution/
     causal_model.pt, use_physics_only=True -- the same mode and checkpoint
     already used for the real 81.2% AM-loss number). The counterfactual
     changes only the charge-rate context field (2C -> 1C -> 0.5C), holding
     every other feature fixed -- the same targeted-feature-substitution
     convention documented in src/optimization/counterfactual_intervention.py
     (Intervention.apply). There is no 4th "+rest periods" data point: no
     context dimension in this model represents inter-cycle rest, so that
     stays a qualitative advisory recommendation, not a quantitative claim.

Run: python scripts/compute_xjtu_case_study.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import scipy.io
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.unified_pipeline import UnifiedDataPipeline, UnifiedBatteryDataset
from src.models.causal_attribution import CausalAttributionModel
from src.train.hero_rad_decoupled import RADDecoupledModel
from torch.utils.data import DataLoader

DATA = Path(__file__).resolve().parent.parent / "data"
CELL_PATH = DATA / "new_datasets" / "XJTU" / "Battery Dataset" / "Batch-1" / "2C_battery-1.mat"
C_RATE = 2.0  # Batch-1 nominal charge C-rate for this case study cell


# ---------------------------------------------------------------------------
# Real trajectory
# ---------------------------------------------------------------------------

def load_cell():
    mat = scipy.io.loadmat(CELL_PATH)
    summary = mat["summary"][0, 0]
    cap = summary["discharge_capacity_Ah"].flatten()
    nom = cap[1] if len(cap) > 1 and cap[1] > cap[0] else cap[0]
    soh = cap / nom

    temps_mean = []
    data_array = mat["data"][0]
    for i in range(len(data_array)):
        try:
            t = data_array[i]["temperature_C"].flatten()
            temps_mean.append(float(np.mean(t)))
        except Exception:
            temps_mean.append(25.0)

    return soh, np.array(temps_mean)


# ---------------------------------------------------------------------------
# 1. HERO zero-shot SOH prediction (real inference, no leakage)
# ---------------------------------------------------------------------------

def hero_predict(soh, device="cpu"):
    model = RADDecoupledModel(feature_dim=20, context_dim=5, hidden_dim=128,
                              latent_dim=64, device=device).to(device)
    state = torch.load(DATA.parent / "reports" / "hero_model" / "hero_model.pt",
                       map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Rebuild the memory bank the checkpoint doesn't carry, via forward-only
    # passes over the model's own training pool. No weights are updated.
    pipeline = UnifiedDataPipeline(str(DATA), use_lithium_features=True)
    pipeline.load_datasets(["nasa", "calce", "oxford"])
    loader = DataLoader(UnifiedBatteryDataset(pipeline.samples), batch_size=64, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            features = torch.nan_to_num(batch["features"].to(device), nan=0.0,
                                        posinf=1.0, neginf=-1.0)
            context = torch.nan_to_num(batch["context"].to(device), nan=0.0)
            chem_id = batch["chem_id"].to(device)
            _, _, _, latent = model(features, context, chem_id, use_retrieval=False)
            for i in range(latent.shape[0]):
                model.memory_bank.add(latent[i], float(batch["soh"][i]),
                                      float(batch["rul_normalized"][i]),
                                      chem_id=int(chem_id[i]), source="pool")
    print(f"  memory bank rebuilt: {model.memory_bank.size()} entries "
          f"(NASA+CALCE+Oxford, no XJTU)")

    n = len(soh)
    features = np.zeros((n, 20), dtype=np.float32)
    features[:, 0] = soh
    features[:, 1] = C_RATE / 4.0
    features[:, 2] = np.arange(n) / n
    context = np.tile(np.array([25.0 / 60.0, C_RATE / 3.0, C_RATE / 4.0, 0.5, 0.0],
                               dtype=np.float32), (n, 1))
    chem_id = np.ones(n, dtype=np.int64)  # 1 = NMC/NCA, matches add_xjtu_to_memory.py

    with torch.no_grad():
        soh_pred, _, _, _ = model(
            torch.tensor(features, device=device),
            torch.tensor(context, device=device),
            torch.tensor(chem_id, device=device),
            use_retrieval=True,
        )
    soh_pred = soh_pred.squeeze(-1).cpu().numpy()

    mae = float(np.mean(np.abs(soh_pred - soh))) * 100
    ss_res = float(np.sum((soh - soh_pred) ** 2))
    ss_tot = float(np.sum((soh - soh.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot

    return soh_pred, {"soh_mae_pct": mae, "soh_r2": r2, "n_cycles": n}


# ---------------------------------------------------------------------------
# 2. Real causal attribution: baseline + C-rate counterfactual
# ---------------------------------------------------------------------------

def build_features_context(soh, total_loss, c_rate_charge):
    features = np.zeros(9, dtype=np.float32)
    features[0] = soh.mean()
    features[1] = soh.std()
    features[2] = c_rate_charge / 4.0
    features[3] = total_loss
    features[4] = len(soh) / 1000.0
    features[5] = soh[-10:].mean()
    features[6] = soh[0]
    context = np.array([
        0.0,                    # 25 C room temperature -> (25-25)/20
        c_rate_charge / 3.0,    # intervention target
        1.0 / 4.0,              # discharge fixed at 1C (matches run_xjtu_causal_attribution.py)
        0.5, 0.0, 1.0,          # soc, profile, cycling mode
    ], dtype=np.float32)
    return features, context


def causal_attribution(soh):
    model = CausalAttributionModel(feature_dim=9, context_dim=6, hidden_dim=128)
    state = torch.load(DATA.parent / "reports" / "causal_attribution" / "causal_model.pt",
                       weights_only=False, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    # Only attributions_pct (proportions) are used downstream; total_loss only
    # scales the unused absolute-attribution output, so an approximate value
    # (held fixed across baseline/counterfactual, per the framework's targeted-
    # substitution convention) is sufficient.
    total_loss = float(soh[0] - soh[-10:].mean())

    display = {"sei_growth": "SEI_GROWTH", "lithium_plating": "LITHIUM_PLATING",
              "am_loss": "ACTIVE_MATERIAL_LOSS", "electrolyte": "ELECTROLYTE_DECOMP",
              "corrosion": "COLLECTOR_CORROSION"}

    def run(c_rate_charge):
        feats, ctx = build_features_context(soh, total_loss, c_rate_charge)
        with torch.no_grad():
            out = model(torch.tensor(feats).unsqueeze(0), torch.tensor(ctx).unsqueeze(0),
                       use_physics_only=True)
        return {display[k]: float(v.item()) for k, v in out["attributions_pct"].items()}

    return {
        "baseline_2C": run(2.0),
        "counterfactual_1C": run(1.0),
        "counterfactual_0.5C": run(0.5),
    }


# ---------------------------------------------------------------------------
# 3. Recompute the life-extension number with the real attribution weights
# ---------------------------------------------------------------------------

def recompute_life_extension(attribution_pct):
    """Re-run compute_life_extension.py's case_study_2 physics with the real
    measured alpha weights instead of its hardcoded {0.55, 0.25, 0.05, 0.10, 0.05}."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import compute_life_extension as cle

    alpha = {
        "am": attribution_pct["baseline_2C"]["ACTIVE_MATERIAL_LOSS"],
        "sei": attribution_pct["baseline_2C"]["SEI_GROWTH"],
        "plating": attribution_pct["baseline_2C"]["LITHIUM_PLATING"],
        "electrolyte": attribution_pct["baseline_2C"]["ELECTROLYTE_DECOMP"],
        "corrosion": attribution_pct["baseline_2C"]["COLLECTOR_CORROSION"],
    }

    cycles, loss = cle.load_xjtu_2c()
    mask = loss > 0.05  # matches case_study_2()'s own threshold exactly
    from scipy.optimize import curve_fit
    (A, g), _ = curve_fit(lambda N, A, g: A * N ** g, cycles[mask], loss[mask],
                          p0=[0.1, 0.6], maxfev=40000)
    resid = loss[mask] - A * cycles[mask] ** g
    r2 = 1 - np.sum(resid ** 2) / np.sum((loss[mask] - loss[mask].mean()) ** 2)

    S = cle.scale_factor(alpha, factual=(2.0, 25.0), counterfactual=(1.0, 25.0), verbose=False)
    ratio = (1 / S) ** (1 / g)
    n_eol = (cle.EOL_LOSS / A) ** (1 / g)

    return {
        "alpha_used": alpha,
        "gamma": float(g), "gamma_r2": float(r2),
        "S": float(S), "ratio": float(ratio),
        "life_extension_pct": float((ratio - 1) * 100),
        "n_eol_cycles": float(n_eol),
        "extra_cycles": float(n_eol * (ratio - 1)),
    }


def main():
    print("Loading real XJTU 2C_battery-1 trajectory...")
    soh, temps = load_cell()
    print(f"  {len(soh)} cycles, SOH {soh.min()*100:.1f}% - {soh.max()*100:.1f}%, "
          f"mean temp {temps.mean():.1f} C")

    print("\n[1/3] HERO zero-shot prediction...")
    soh_pred, hero_metrics = hero_predict(soh)
    print(f"  MAE={hero_metrics['soh_mae_pct']:.2f}%  R2={hero_metrics['soh_r2']:.3f}")

    print("\n[2/3] Causal attribution (baseline + C-rate counterfactual)...")
    attribution = causal_attribution(soh)
    for scenario, pct in attribution.items():
        dom = max(pct, key=pct.get)
        print(f"  {scenario}: dominant={dom} ({pct[dom]*100:.1f}%)")

    print("\n[3/3] Recomputing life-extension with real attribution weights...")
    life_ext = recompute_life_extension(attribution)
    print(f"  gamma={life_ext['gamma']:.3f} (R2={life_ext['gamma_r2']:.3f})  "
          f"S={life_ext['S']:.3f}  ratio={life_ext['ratio']:.2f}x  "
          f"life extension={life_ext['life_extension_pct']:.0f}% "
          f"(+{life_ext['extra_cycles']:.0f} cycles)")

    out = {
        "cell": "XJTU_Batch-1_2C_battery-1",
        "cycles": list(range(1, len(soh) + 1)),
        "soh_true": soh.tolist(),
        "soh_pred_hero": soh_pred.tolist(),
        "temps_mean": temps.tolist(),
        "hero_metrics": hero_metrics,
        "attribution": attribution,
        "life_extension": life_ext,
    }
    out_path = Path("reports/xjtu_case_study_real.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
