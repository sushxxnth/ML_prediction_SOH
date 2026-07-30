"""
Real data/model pipeline for the TJU cross-chemistry end-to-end case study
(paper Illustration 3), the counterpart to the cold-weather NASA case
(Illustration 1, plating) and the high-C-rate XJTU case (Illustration 2,
mechanical stress).

Cell: TJU CY25_2 -- NCM/NCA chemistry (every other case study in this paper
uses LCO), 25 C ambient, 1C/1C constant-current cycling, 904 measured cycles,
31.7% capacity loss, crossing 20% loss at a measured cycle (~492) well inside
the record. This is one of the two strict zero-shot evaluation cells already
defined by Table 1's protocol (the adaptation cell, CY25_1, is excluded here
exactly as it is there); this script does not introduce a new protocol, it
traces one of the two existing eval cells through all four pipeline stages.

Two NASA cells were tried first (B0029/B0031 at 43 C, B0053/B0054/B0055 at
4C/2A) and rejected: hot cells saturate the SOH head's clamp at every seed,
and the 4C/2A cold cells are seed-unstable (R^2 ranges from +0.66 to -1.79
across identical reruns) or largely duplicate Illustration 1's plating story.
TJU's zero-shot failure, by contrast, is not a new problem to diagnose: it is
the same failure mode Table 1 already reports and explains (no target-
chemistry sample in training or memory bank), and Stages 2-4 tell a materially
different physics story (thermally-modulated active-material loss, not
plating) on a chemistry no other illustration touches.

Provenance of every number written here:

  1. Trajectory -- measured discharge capacities, loaded directly from the raw
     TJU npy (data/new_datasets/RUL-Mamba/data/TJU data/
     Dataset_3_NCM_NCA_battery_1C.npy), the same source src/data/tju_loader.py
     parses into the cached dataset. Asserted against that cache's final
     capacity to catch any divergence.

  2. Stage 1 -- HERO zero-shot metrics, read from the JSON written by
     scripts/run_tju_zeroshot_case.py (LCO-trained, LCO-only memory bank, same
     seed/epoch protocol as Table 1's published zero-shot row, evaluated in
     cycle order on this cell alone). Nothing is re-inferred here.

  3. Stages 2/3 -- the checkpointed physics-prior attribution model
     (reports/causal_attribution/causal_model.pt, use_physics_only=True), same
     checkpoint and mode as Illustrations 1 and 2. Four scenarios differing
     only in the context fields being intervened on (ambient temperature,
     discharge rate); every other feature held fixed, per the targeted-
     feature-substitution convention in
     src/optimization/counterfactual_intervention.py.

  4. Stage 4 -- power-law fit on this cell's own loss curve (record already
     crosses 20% loss, so the EOL baseline is measured, not extrapolated) and
     the counterfactual scale factor from compute_life_extension.scale_factor,
     using the measured attribution weights from step 3. A lever is only
     quantified when no single mechanism's rate ratio exceeds RATIO_GUARD; see
     scripts/compute_nasa_case_study.py for why that guard exists (a near-zero
     attribution meeting a near-zero baseline rate can dominate the sum).

Run: python scripts/compute_tju_case_study.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.models.causal_attribution import CausalAttributionModel
import compute_life_extension as cle

CELL = "CY25_2"
AMBIENT_C = 25.0
C_CHARGE = 1.0
C_DISCHARGE = 1.0
WARM_TARGET_C = 35.0
COOL_TARGET_C = 15.0
RATE_TARGET = 0.5

RATIO_GUARD = 10.0
HERO_JSON = ROOT / "reports" / "hero_model" / "tju_cy25_2_zeroshot_results.json"
NPY_PATH = (ROOT / "data" / "new_datasets" / "RUL-Mamba" / "data" / "TJU data"
           / "Dataset_3_NCM_NCA_battery_1C.npy")
CACHE_JSON = ROOT / "data" / "unified_cache" / "tju" / "tju_processed.json"

DISPLAY = {"sei_growth": "SEI_GROWTH", "lithium_plating": "LITHIUM_PLATING",
           "am_loss": "ACTIVE_MATERIAL_LOSS", "electrolyte": "ELECTROLYTE_DECOMP",
           "corrosion": "COLLECTOR_CORROSION"}
ALPHA_KEYS = {"am": "ACTIVE_MATERIAL_LOSS", "sei": "SEI_GROWTH",
              "plating": "LITHIUM_PLATING", "electrolyte": "ELECTROLYTE_DECOMP",
              "corrosion": "COLLECTOR_CORROSION"}


# ---------------------------------------------------------------------------
# 1. Real trajectory
# ---------------------------------------------------------------------------

def load_cell():
    data = np.load(NPY_PATH, allow_pickle=True).item()
    df = data[CELL].sort_values("Cycle")
    caps = df["Capacity"].to_numpy(dtype=float)
    cycles = df["Cycle"].to_numpy(dtype=int)

    cached = json.loads(CACHE_JSON.read_text())["cells"][CELL]
    assert abs(caps[-1] - cached["final_capacity"]) < 1e-6, \
        "trajectory diverged from the cached TJU dataset"

    return cycles, caps


# ---------------------------------------------------------------------------
# 2. Stage 1: HERO zero-shot metrics (read, not recomputed)
# ---------------------------------------------------------------------------

def hero_zeroshot():
    if not HERO_JSON.exists():
        raise SystemExit(f"{HERO_JSON} not found -- run:\n"
                         f"  python scripts/run_tju_zeroshot_case.py")
    d = json.loads(HERO_JSON.read_text())
    if d["eval_cell"] != f"TJU_{CELL}":
        raise SystemExit(f"{HERO_JSON} is for {d['eval_cell']}, not {CELL}")
    return d


# ---------------------------------------------------------------------------
# 3. Stages 2/3: attribution under the factual condition and three interventions
# ---------------------------------------------------------------------------

def build_features_context(soh, total_loss, temp_c, c_rate_discharge):
    """Same 9-feature / 6-context convention as the XJTU and NASA case-study
    scripts. Charge rate is unchanged across all scenarios, so features[2] and
    context[1] stay fixed; only ambient temperature (context[0]) and discharge
    rate (context[2]) are intervened on."""
    features = np.zeros(9, dtype=np.float32)
    features[0] = soh.mean()
    features[1] = soh.std()
    features[2] = C_CHARGE / 4.0
    features[3] = total_loss
    features[4] = len(soh) / 1000.0
    features[5] = soh[-10:].mean()
    features[6] = soh[0]
    context = np.array([
        (temp_c - 25.0) / 20.0,
        C_CHARGE / 3.0,
        c_rate_discharge / 4.0,
        0.5, 0.0, 1.0,
    ], dtype=np.float32)
    return features, context


def causal_attribution(soh):
    model = CausalAttributionModel(feature_dim=9, context_dim=6, hidden_dim=128)
    model.load_state_dict(torch.load(
        ROOT / "reports" / "causal_attribution" / "causal_model.pt",
        weights_only=False, map_location="cpu"))
    model.eval()

    total_loss = float(soh[0] - soh[-10:].mean())

    def run(temp_c, c_rate_discharge):
        feats, ctx = build_features_context(soh, total_loss, temp_c, c_rate_discharge)
        with torch.no_grad():
            out = model(torch.tensor(feats).unsqueeze(0), torch.tensor(ctx).unsqueeze(0),
                        use_physics_only=True)
        return {DISPLAY[k]: float(v.item()) for k, v in out["attributions_pct"].items()}

    scenarios = [
        ("factual", f"Measured\n({AMBIENT_C:g} °C, {C_DISCHARGE:g}C)", AMBIENT_C, C_DISCHARGE),
        ("warm", f"Warm to {WARM_TARGET_C:g} °C\n(rate held)", WARM_TARGET_C, C_DISCHARGE),
        ("cool", f"Cool to {COOL_TARGET_C:g} °C\n(rate held)", COOL_TARGET_C, C_DISCHARGE),
        ("slow", f"Discharge {C_DISCHARGE:g}C → {RATE_TARGET:g}C\n(temp held)", AMBIENT_C, RATE_TARGET),
    ]
    return [{"key": k, "label": lab, "temp_c": t, "c_rate_discharge": d,
             "attributions_pct": run(t, d)} for k, lab, t, d in scenarios]


# ---------------------------------------------------------------------------
# 4. Stage 4: life extension from this cell's own fade exponent
# ---------------------------------------------------------------------------

def lever_scale_factor(alpha, factual, counterfactual):
    ratios = {m: float(cle.RATES[m](*counterfactual) / cle.RATES[m](*factual))
              for m in alpha}
    S = float(sum(alpha[m] * ratios[m] for m in alpha))
    return S, ratios


def life_extension(caps, scenarios):
    loss = (1 - caps / caps[0]) * 100
    cycles = np.arange(1, len(caps) + 1)
    mask = loss > 0.05
    (A, g), _ = curve_fit(lambda N, A, g: A * N ** g, cycles[mask], loss[mask],
                          p0=[0.1, 0.6], maxfev=40000)
    resid = loss[mask] - A * cycles[mask] ** g
    r2 = 1 - np.sum(resid ** 2) / np.sum((loss[mask] - loss[mask].mean()) ** 2)

    # Measured EOL crossing (interpolated against a running max, robust to
    # small non-monotonic dips), matching compute_life_extension.cycles_to_eol.
    run_max = np.maximum.accumulate(loss)
    i = int(np.argmax(run_max >= cle.EOL_LOSS))
    measured_eol = (float(np.interp(cle.EOL_LOSS, [run_max[i - 1], run_max[i]],
                                    [cycles[i - 1], cycles[i]]))
                    if i > 0 else float(cycles[0]))
    fit_eol = (cle.EOL_LOSS / A) ** (1 / g)

    base = next(s for s in scenarios if s["key"] == "factual")["attributions_pct"]
    alpha = {k: base[v] for k, v in ALPHA_KEYS.items()}
    factual = (C_DISCHARGE, AMBIENT_C)

    levers = {}
    for key, cf in (("warm", (C_DISCHARGE, WARM_TARGET_C)),
                    ("cool", (C_DISCHARGE, COOL_TARGET_C)),
                    ("rate", (RATE_TARGET, AMBIENT_C))):
        S, ratios = lever_scale_factor(alpha, factual, cf)
        worst = max(ratios, key=lambda m: ratios[m])
        usable = ratios[worst] <= RATIO_GUARD and S > 0
        entry = {"counterfactual": {"c_rate_discharge": cf[0], "temp_c": cf[1]},
                 "S": S, "rate_ratios": ratios,
                 "largest_ratio_mechanism": worst, "largest_ratio": ratios[worst],
                 "quantified": bool(usable)}
        if usable:
            ratio = (1 / S) ** (1 / g)
            entry.update({
                "ratio": float(ratio),
                "life_extension_pct": float((ratio - 1) * 100),
                "extra_cycles": float(measured_eol * (ratio - 1)),
                "gamma_sensitivity_pct": {f"{gg:.2f}": float(((1 / S) ** (1 / gg) - 1) * 100)
                                          for gg in (0.6, 0.8, float(g), 1.0, 1.2)},
            })
        else:
            entry["excluded_because"] = (
                f"{worst} rate ratio is {ratios[worst]:.0f}x, above the {RATIO_GUARD:g}x "
                "guard; at its attribution floor that single term would dominate S")
        levers[key] = entry

    return {
        "alpha_used": alpha,
        "gamma": float(g), "gamma_r2": float(r2), "fit_A": float(A),
        "n_fit_points": int(mask.sum()),
        "observed_cycles": int(len(caps)),
        "observed_loss_pct": float(loss[-1]),
        "measured_eol_cycles": measured_eol,
        "fit_eol_cycles": float(fit_eol),
        "ratio_guard": RATIO_GUARD,
        "levers": levers,
    }


def main():
    print(f"Loading real TJU {CELL} trajectory...")
    cycles_raw, caps = load_cell()
    soh = caps / caps[0]
    print(f"  {len(caps)} cycles (raw index {cycles_raw[0]}-{cycles_raw[-1]}), "
          f"SOH {soh.max()*100:.1f}% - {soh.min()*100:.1f}%, ambient {AMBIENT_C:g} C (fixed)")

    print("\n[1/3] HERO zero-shot metrics...")
    hero = hero_zeroshot()
    print(f"  MAE={hero['soh_mae_pct']:.2f}%  R2={hero['soh_r2']:.3f}  "
          f"(seed={hero['seed']}, {hero['n_eval']} eval cycles)")

    print("\n[2/3] Causal attribution (factual + three interventions)...")
    scenarios = causal_attribution(soh)
    for s in scenarios:
        pct = s["attributions_pct"]
        dom = max(pct, key=pct.get)
        print(f"  {s['key']:<8} {dom:<22} {pct[dom]*100:5.1f}%   "
              f"[AM {pct['ACTIVE_MATERIAL_LOSS']*100:5.1f}% | "
              f"SEI {pct['SEI_GROWTH']*100:5.1f}%]")

    print("\n[3/3] Life extension from this cell's fade exponent...")
    life = life_extension(caps, scenarios)
    print(f"  gamma={life['gamma']:.3f} (R2={life['gamma_r2']:.3f}), "
          f"measured EOL cycle={life['measured_eol_cycles']:.0f} "
          f"(fit predicts {life['fit_eol_cycles']:.0f}; record={life['observed_cycles']} cycles)")
    for key, lev in life["levers"].items():
        if lev["quantified"]:
            print(f"  {key:<6} S={lev['S']:.3f} -> {lev['ratio']:.2f}x "
                  f"({lev['life_extension_pct']:.0f}%, +{lev['extra_cycles']:.0f} cycles)")
        else:
            print(f"  {key:<6} S={lev['S']:.2f} -- not quantified "
                  f"({lev['largest_ratio_mechanism']} ratio {lev['largest_ratio']:.0f}x)")

    out = {
        "cell": f"TJU_{CELL}",
        "chemistry": "NCM/NCA",
        "ambient_c": AMBIENT_C,
        "protocol": f"Constant current, {C_CHARGE:g}C charge / {C_DISCHARGE:g}C discharge",
        "cycles": cycles_raw.tolist(),
        "capacity_ah": caps.tolist(),
        "soh_true": soh.tolist(),
        "hero_zeroshot": hero,
        "attribution_scenarios": scenarios,
        "life_extension": life,
    }
    out_path = ROOT / "reports" / "tju_cy25_2_case_study_real.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWritten to {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
