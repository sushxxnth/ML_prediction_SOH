"""Illustration 1 (cold-weather plating, NASA B0046) on real inference.

Companion to compute_xjtu_case_study.py and compute_tju_case_study.py, and
written to the same convention, so that all three end-to-end illustrations in
the paper are produced by the same pipeline rather than by hand.

What this replaces
------------------
Illustration 1's Stage 2/3 percentages (70% plating; 10%/5%/0% under the three
counterfactual scenarios) were carried over from an early figure and from the
synthetic scenario 1 of validate_counterfactual_optimization.py -- a 0 C,
1.5C, 2.0 Ah condition point that no NASA cell occupies. B0046 is a real 4 C
cell charged at 1.5 A (~0.9C), so those numbers described neither the cell nor
the released models. Everything below is read out of the checkpointed model.

Sources, all real
-----------------
  1. Trajectory     -- discharge capacities from data/nasa_set5/raw/B0046.mat,
                       via compute_life_extension.load_nasa (same loader the
                       measured group comparison uses).
  2. Stage 1        -- reports/hero_model/b0046_holdout_results.json, produced
                       by scripts/run_hero_cell_holdout.py --cell B0046.
                       Held-out cell, in-chemistry: unlike Illustrations 2/3
                       this is a regime where HERO's prediction succeeds.
  3. Stages 2/3     -- reports/causal_attribution/causal_model.pt with
                       use_physics_only=True, the same checkpoint and mode as
                       Illustrations 2 and 3.
  4. Stage 4        -- life extension. Unlike Illustrations 2 and 3 this one is
                       MEASURED: B0046/47/48 (4 C) against B0005/06/07/18
                       (24 C) under an identical protocol. The counterfactual
                       module's scale factor S is then reported as a
                       cross-check against that measurement, using this cell's
                       own attribution weights rather than a literature split.

Run:  python scripts/compute_nasa_case_study.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.models.causal_attribution import CausalAttributionModel
import compute_life_extension as cle

CELL = "B0046"
AMBIENT_C = 4.0
C_CHARGE = 0.9          # 1.5 A on a ~1.7 Ah cell
C_DISCHARGE = 1.0       # 2 A
WARM_TACTICAL_C = 15.0  # the advisory's preheat target
WARM_MEASURED_C = 24.0  # the ambient of the measured comparison group
CHARGE_HALVED = 0.45    # the current lever, kept to expose its null result

HERO_JSON = ROOT / "reports" / "hero_model" / "b0046_holdout_results.json"

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
    caps, amb = cle.load_nasa(CELL)
    assert amb == int(AMBIENT_C), (
        f"{CELL} ambient reads {amb} C, expected {AMBIENT_C:g} C -- the "
        f"nasa_set5 parser regressed")
    return caps, amb


# ---------------------------------------------------------------------------
# 2. Stage 1: HERO held-out-cell metrics (read, not recomputed)
# ---------------------------------------------------------------------------

def hero_holdout():
    if not HERO_JSON.exists():
        raise SystemExit(f"{HERO_JSON} not found -- run:\n"
                         f"  python scripts/run_hero_cell_holdout.py --cell {CELL}")
    d = json.loads(HERO_JSON.read_text())
    if d["held_out_cell"] != f"NASA_{CELL}":
        raise SystemExit(f"{HERO_JSON} is for {d['held_out_cell']}, not {CELL}")
    return d


# ---------------------------------------------------------------------------
# 3. Stages 2/3: attribution under the factual condition and the interventions
# ---------------------------------------------------------------------------

def build_features_context(soh, total_loss, temp_c, c_rate_charge, c_rate_discharge):
    """Same 9-feature / 6-context convention as the XJTU and TJU scripts."""
    features = np.zeros(9, dtype=np.float32)
    features[0] = soh.mean()
    features[1] = soh.std()
    features[2] = c_rate_charge / 4.0
    features[3] = total_loss
    features[4] = len(soh) / 1000.0
    features[5] = soh[-10:].mean()
    features[6] = soh[0]
    context = np.array([
        (temp_c - 25.0) / 20.0,
        c_rate_charge / 3.0,
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

    def run(temp_c, c_charge):
        feats, ctx = build_features_context(soh, total_loss, temp_c, c_charge,
                                            C_DISCHARGE)
        with torch.no_grad():
            out = model(torch.tensor(feats).unsqueeze(0),
                        torch.tensor(ctx).unsqueeze(0), use_physics_only=True)
        return {DISPLAY[k]: float(v.item()) for k, v in out["attributions_pct"].items()}

    scenarios = [
        ("factual", f"Measured\n({AMBIENT_C:g} °C, {C_CHARGE:g}C charge)",
         AMBIENT_C, C_CHARGE),
        ("warm_tactical", f"Preheat to {WARM_TACTICAL_C:g} °C\n(current held)",
         WARM_TACTICAL_C, C_CHARGE),
        ("warm_measured", f"Warm to {WARM_MEASURED_C:g} °C\n(measured group)",
         WARM_MEASURED_C, C_CHARGE),
        ("charge_halved", f"Charge {C_CHARGE:g}C → {CHARGE_HALVED:g}C\n(temp held)",
         AMBIENT_C, CHARGE_HALVED),
    ]
    return [{"key": k, "label": lab, "temp_c": t, "c_rate_charge": c,
             "attributions_pct": run(t, c)} for k, lab, t, c in scenarios]


def charge_lever_is_inert(scenarios, tol=1e-3):
    """Does halving the charge current move the attribution at all?

    It does not, and the paper says so: the plating prior in
    src/models/causal_attribution.py is gated on temperature and cycling mode
    with no C-rate term, so a charge-rate intervention cannot change the
    plating attribution in physics-only mode. Detected here rather than
    asserted, so that the claim tracks the code if the prior is ever changed.
    """
    base = next(s for s in scenarios if s["key"] == "factual")["attributions_pct"]
    cf = next(s for s in scenarios if s["key"] == "charge_halved")["attributions_pct"]
    deltas = {m: abs(cf[m] - base[m]) for m in base}
    return max(deltas.values()) < tol, deltas


# ---------------------------------------------------------------------------
# 4. Stage 4: measured life extension, plus the model cross-check
# ---------------------------------------------------------------------------

def measured_life_extension():
    """The controlled 4 C vs 24 C comparison already present in NASA."""
    from scipy import stats

    groups = {}
    per_cell = {}
    for name in cle.COLD_CELLS + cle.WARM_CELLS:
        caps, amb = cle.load_nasa(name)
        n = cle.cycles_to_eol(caps)
        groups.setdefault(amb, []).append(n)
        per_cell[name] = {"ambient_c": amb, "cycles_to_eol": n}

    cold, warm = groups[4], groups[24]
    r_mean = float(np.mean(warm) / np.mean(cold))
    r_med = float(np.median(warm) / np.median(cold))
    u, p_u = stats.mannwhitneyu(warm, cold, alternative="greater")
    t, p_t = stats.ttest_ind(warm, cold, equal_var=False)

    return {
        "per_cell": per_cell,
        "cold_group_cycles": cold, "warm_group_cycles": warm,
        "cold_mean": float(np.mean(cold)), "warm_mean": float(np.mean(warm)),
        "ratio_mean": r_mean, "ratio_median": r_med,
        "life_extension_pct": (r_mean - 1) * 100,
        "extra_cycles": float(np.mean(warm) - np.mean(cold)),
        "mannwhitney_u": float(u), "mannwhitney_p": float(p_u),
        "welch_t": float(t), "welch_p": float(p_t),
        "ranges_disjoint": bool(max(cold) < min(warm)),
    }


def model_cross_check(scenarios, measured):
    """S from this cell's own attribution weights, against the measured ratio."""
    base = next(s for s in scenarios if s["key"] == "factual")["attributions_pct"]
    alpha = {k: base[v] for k, v in ALPHA_KEYS.items()}

    S = cle.scale_factor(alpha, factual=(C_CHARGE, AMBIENT_C),
                         counterfactual=(C_CHARGE, WARM_MEASURED_C), verbose=False)
    ratios = {m: float(cle.RATES[m](C_CHARGE, WARM_MEASURED_C)
                       / cle.RATES[m](C_CHARGE, AMBIENT_C)) for m in alpha}
    predicted = 1.0 / S
    err = abs(predicted - measured["ratio_mean"]) / measured["ratio_mean"] * 100

    return {
        "alpha_used": alpha,
        "alpha_source": "checkpointed attribution model, this cell, factual condition",
        "S": float(S), "rate_ratios": ratios,
        "predicted_ratio_gamma1": float(predicted),
        "measured_ratio": measured["ratio_mean"],
        "error_pct": float(err),
        "gamma_sensitivity": {f"{g:.2f}": float((1 / S) ** (1 / g))
                              for g in (0.52, 0.75, 1.0, 1.25)},
    }


def main():
    print(f"Loading real NASA {CELL} trajectory...")
    caps, amb = load_cell()
    soh = caps / caps[0]
    print(f"  {len(caps)} discharge cycles, SOH {soh[0]*100:.1f}% -> "
          f"{soh[-1]*100:.1f}%, ambient {amb} C (measured, all cycles)")

    print("\n[1/4] HERO held-out-cell metrics...")
    hero = hero_holdout()
    print(f"  MAE={hero['soh_mae_pct_mean']:.2f}% +/- {hero['soh_mae_pct_std']:.2f}  "
          f"R2={hero['soh_r2_mean']:.3f} +/- {hero['soh_r2_std']:.3f}  "
          f"({hero['n_discharge_rows']} discharge rows)")

    print("\n[2/4] Causal attribution (factual + three interventions)...")
    scenarios = causal_attribution(soh)
    for s in scenarios:
        pct = s["attributions_pct"]
        dom = max(pct, key=pct.get)
        print(f"  {s['key']:<14} {dom:<22} {pct[dom]*100:5.1f}%   "
              f"[plating {pct['LITHIUM_PLATING']*100:5.1f}% | "
              f"AM {pct['ACTIVE_MATERIAL_LOSS']*100:5.1f}%]")

    inert, deltas = charge_lever_is_inert(scenarios)
    print(f"\n  charge-rate lever inert: {inert} "
          f"(max attribution shift {max(deltas.values())*100:.4f} pp)")

    print("\n[3/4] Measured life extension (4 C vs 24 C groups)...")
    measured = measured_life_extension()
    print(f"  cold {measured['cold_group_cycles']} (mean {measured['cold_mean']:.0f})  "
          f"vs warm {measured['warm_group_cycles']} (mean {measured['warm_mean']:.0f})")
    print(f"  MEASURED: {measured['life_extension_pct']:.0f}% "
          f"({measured['ratio_mean']:.2f}x, +{measured['extra_cycles']:.0f} cycles); "
          f"U={measured['mannwhitney_u']:.0f} p={measured['mannwhitney_p']:.3f}, "
          f"Welch t={measured['welch_t']:.2f} p={measured['welch_p']:.3f}")

    print("\n[4/4] Counterfactual-module cross-check against that measurement...")
    cross = model_cross_check(scenarios, measured)
    print(f"  alpha (model-derived): " + ", ".join(
        f"{k}={v*100:.1f}%" for k, v in cross["alpha_used"].items()))
    print(f"  S={cross['S']:.4f} -> predicted {cross['predicted_ratio_gamma1']:.2f}x "
          f"vs measured {cross['measured_ratio']:.2f}x "
          f"({cross['error_pct']:.1f}% error)")

    out = {
        "cell": f"NASA_{CELL}",
        "chemistry": "LCO",
        "ambient_c": AMBIENT_C,
        "protocol": "1.5 A CC-CV charge (~0.9C) / 2 A discharge (~1C)",
        "capacity_ah": caps.tolist(),
        "soh_true": soh.tolist(),
        "hero_holdout": hero,
        "attribution_scenarios": scenarios,
        "charge_lever_inert": inert,
        "charge_lever_max_shift_pp": float(max(deltas.values()) * 100),
        "measured_life_extension": measured,
        "model_cross_check": cross,
    }
    out_path = ROOT / "reports" / "nasa_b0046_case_study_real.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWritten to {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
