"""
Life-extension computation for the two end-to-end case studies.

Replaces the previously hardcoded "+120 cycles / 31%" and "22% ROI" figures.

Case study 1 (cold-weather) is grounded in a controlled comparison that already
exists inside the NASA dataset: B0046/47/48 are cycled entirely at 4 C ambient,
while B0005/06/07/18 are cycled at 24 C under an identical 1.5 A CC-CV charge
and 2 A discharge. Cycles to 80% SOH are therefore directly measurable in both
groups, and the life extension from avoiding cold charging needs no
extrapolation model at all. The counterfactual module's prediction is then
checked against that measured ratio.

Case study 2 (high C-rate) has no matched comparison group, so it uses the
physics-based extrapolation described below.

Extrapolation model (case study 2 only)
---------------------------------------
    L(N; c) = A(c) * N^gamma,      A(c) = sum_m alpha_m * r_m(c)

Under an intervention c -> c', with gamma and the mechanism decomposition held
fixed to first order,

    S = sum_m alpha_m * r_m(c') / r_m(c)
    N'_EOL / N_EOL = (1 / S)^(1 / gamma)

Per-mechanism rate laws r_m are taken verbatim from the model's physics modules
(src/models/pinn_physics_module.py) and the counterfactual sensitivity table
(src/optimization/counterfactual_intervention.py).

Run:  python scripts/compute_life_extension.py
"""

import numpy as np
import scipy.io
from pathlib import Path
from scipy import stats
from scipy.optimize import curve_fit

R_GAS = 8.314
T_REF = 298.15       # PHYSICS.T_reference
T_CRIT = 278.15      # LithiumPlatingEquation.T_critical (5 C)
T_SCALE = 10.0       # LithiumPlatingEquation.T_scale
E_A_SEI = 35_000.0   # J/mol, Pure-PINN fitted value
BETA_AM = 1.48       # Pure-PINN fitted C-rate stress exponent
ALPHA_PLATING = 0.5  # mid-point of the literature range [0.3, 0.7]
EOL_LOSS = 20.0      # % capacity loss defining end of life (80% SOH)

DATA = Path(__file__).resolve().parent.parent / "data"

COLD_CELLS = ["B0046", "B0047", "B0048"]              # 4 C, all cycles
WARM_CELLS = ["B0005", "B0006", "B0007", "B0018"]     # 24 C, all cycles

# B0046's attribution under its factual condition, read out of
# reports/causal_attribution/causal_model.pt (use_physics_only=True) by
# scripts/compute_nasa_case_study.py. Kept here as a literal so this script
# stays standalone; rerun that script if the checkpoint changes.
B0046_ALPHA = {"plating": 0.75075, "sei": 0.10525, "am": 0.05132,
               "electrolyte": 0.04634, "corrosion": 0.04634}

# Likewise for the XJTU 2C cell, from scripts/compute_xjtu_case_study.py. This
# script previously carried a separate {am .55, sei .25, ...} literature split
# here, so the two released scripts disagreed about the same published number
# (27% vs 41% life extension). They now share one source of truth.
XJTU_2C_ALPHA = {"am": 0.8122636675834656, "sei": 0.048272617161273956,
                 "plating": 0.04648788273334503,
                 "electrolyte": 0.04648788273334503,
                 "corrosion": 0.04648788273334503}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def rate_plating(c_rate, temp_c):
    T = temp_c + 273.15
    cold = _sigmoid((T_CRIT - T) / 5.0)
    enhance = min(np.exp((T_REF - T) / T_SCALE), 10.0)
    return cold * enhance * max(c_rate, 0.1) ** ALPHA_PLATING


def _arrhenius(temp_c):
    return np.exp(-E_A_SEI / (2.0 * R_GAS * (temp_c + 273.15)))


def rate_sei(c_rate, temp_c):
    return _arrhenius(temp_c) * max(c_rate, 0.1) ** 0.3


def rate_am(c_rate, temp_c):
    return max(c_rate, 0.1) ** BETA_AM


def rate_electrolyte(c_rate, temp_c):
    return _arrhenius(temp_c) * max(c_rate, 0.1) ** 0.1


def rate_corrosion(c_rate, temp_c):
    return _arrhenius(temp_c) * max(c_rate, 0.1) ** 0.2


RATES = {"plating": rate_plating, "sei": rate_sei, "am": rate_am,
         "electrolyte": rate_electrolyte, "corrosion": rate_corrosion}


def scale_factor(alpha, factual, counterfactual, verbose=True):
    """S = sum_m alpha_m * r_m(c') / r_m(c)."""
    S = 0.0
    for m, a in alpha.items():
        if a == 0:
            continue
        rho = RATES[m](*counterfactual) / RATES[m](*factual)
        S += a * rho
        if verbose:
            print(f"    rho_{m:<12}= {rho:8.4f}   alpha*rho = {a * rho:.4f}")
    if verbose:
        print(f"    {'S (total)':<17}= {S:8.4f}")
    return S


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_nasa(name, skip_formation=True):
    """Discharge capacities for a NASA cell, with ambient temperature.

    The first recorded discharge sits well above the subsequent ones (a
    pre-conditioning measurement, not degradation); it is dropped so that the
    reference capacity is the first steady-state cycle.
    """
    mat = scipy.io.loadmat(DATA / "nasa_set5" / "raw" / f"{name}.mat",
                           simplify_cells=True)
    cyc = mat[name]["cycle"]
    disch = [c for c in cyc
             if c["type"] == "discharge" and "Capacity" in c["data"]]
    amb = int(np.ravel(disch[0]["ambient_temperature"])[0])
    caps = np.array([float(np.ravel(c["data"]["Capacity"])[0]) for c in disch])
    caps = caps[caps > 0.1]                   # drop aborted measurement cycles
    if skip_formation:
        caps = caps[1:]
    return caps, amb


def cycles_to_eol(caps, thresh=EOL_LOSS):
    """First cycle at which capacity loss reaches `thresh` %, interpolated.

    A running maximum makes the crossing robust to the capacity-recovery bumps
    that follow rest periods in the NASA cells.
    """
    loss = (1 - caps / caps[0]) * 100
    run = np.maximum.accumulate(loss)
    N = np.arange(1, len(caps) + 1)
    i = int(np.argmax(run >= thresh))
    if run[i] < thresh:
        return None
    if i == 0:
        return float(N[0])
    return float(np.interp(thresh, [run[i - 1], run[i]], [N[i - 1], N[i]]))


# --------------------------------------------------------------------------
# Case study 1: measured, plus model cross-check
# --------------------------------------------------------------------------

def case_study_1():
    print("=" * 74)
    print("Case study 1 - cold-weather charging (NASA, measured comparison)")
    print("=" * 74)
    print("  Identical 1.5 A CC-CV charge / 2 A discharge in both groups;")
    print("  the only controlled difference is ambient temperature.\n")

    groups = {}
    print(f"  {'cell':<8}{'ambient':>9}{'cycles to 80% SOH':>21}")
    for name in COLD_CELLS + WARM_CELLS:
        caps, amb = load_nasa(name)
        n = cycles_to_eol(caps)
        groups.setdefault(amb, []).append(n)
        print(f"  {name:<8}{amb:>7} C{n:>21.0f}")

    cold, warm = groups[4], groups[24]
    r_mean = np.mean(warm) / np.mean(cold)
    r_med = np.median(warm) / np.median(cold)
    u, p_u = stats.mannwhitneyu(warm, cold, alternative="greater")
    t, p_t = stats.ttest_ind(warm, cold, equal_var=False)

    print(f"\n   4 C (n={len(cold)}): mean {np.mean(cold):.0f}, "
          f"median {np.median(cold):.0f}, range {min(cold):.0f}-{max(cold):.0f}")
    print(f"  24 C (n={len(warm)}): mean {np.mean(warm):.0f}, "
          f"median {np.median(warm):.0f}, range {min(warm):.0f}-{max(warm):.0f}")
    print(f"\n  MEASURED life extension: {(r_mean - 1) * 100:.0f}% "
          f"({r_mean:.2f}x by means, {r_med:.2f}x by medians), "
          f"+{np.mean(warm) - np.mean(cold):.0f} cycles")
    print(f"  Mann-Whitney U={u:.0f}, p={p_u:.3f};  Welch t={t:.2f}, p={p_t:.3f}")
    print(f"  group ranges disjoint: {max(cold) < min(warm)}")

    print("\n  Counterfactual-module cross-check (4 C -> 24 C at fixed current):")
    # B0046's own attribution, read out of the checkpointed model rather than
    # set from literature. Regenerate with scripts/compute_nasa_case_study.py,
    # which writes the same weights to reports/nasa_b0046_case_study_real.json.
    # The previous hardcoded {plating .70, sei .20, am .10} split was a
    # literature figure, not this cell's; it also omitted the electrolyte and
    # corrosion terms entirely, which the model does assign (4.6% each).
    S = scale_factor(B0046_ALPHA, factual=(0.9, 4.0), counterfactual=(0.9, 24.0))
    print("    predicted life ratio by fatigue exponent:")
    for g in (0.52, 0.75, 1.0, 1.25):
        print(f"      gamma={g:<5} -> {(1 / S) ** (1 / g):.2f}x")
    err = abs((1 / S) - r_mean) / r_mean * 100
    print(f"    at gamma=1.0: {1 / S:.2f}x vs measured {r_mean:.2f}x "
          f"({err:.0f}% error)")
    return r_mean


# --------------------------------------------------------------------------
# Case study 2: physics-based extrapolation (no matched comparison group)
# --------------------------------------------------------------------------

def load_xjtu_2c():
    p = (DATA / "new_datasets" / "XJTU" / "Battery Dataset" / "Batch-1"
         / "2C_battery-1.mat")
    cap = scipy.io.loadmat(p)["summary"][0, 0]["discharge_capacity_Ah"].flatten()
    return np.arange(1, len(cap) + 1), (1 - cap / cap[0]) * 100


def case_study_2():
    print("\n" + "=" * 74)
    print("Case study 2 - XJTU 2C NCM, high C-rate mechanical stress")
    print("=" * 74)
    cycles, loss = load_xjtu_2c()

    mask = loss > 0.05
    (A, g), _ = curve_fit(lambda N, A, g: A * N ** g,
                          cycles[mask], loss[mask], p0=[0.1, 0.6], maxfev=40000)
    resid = loss[mask] - A * cycles[mask] ** g
    r2 = 1 - np.sum(resid ** 2) / np.sum((loss[mask] - loss[mask].mean()) ** 2)
    print(f"  observed: {len(cycles)} cycles, final loss {loss[-1]:.1f}%")
    print(f"  power-law fit L(N) = {A:.4g} * N^{g:.3f}  (R^2 = {r2:.4f})\n")

    S = scale_factor(XJTU_2C_ALPHA, factual=(2.0, 25.0),
                     counterfactual=(1.0, 25.0))
    ratio = (1 / S) ** (1 / g)
    n_eol = (EOL_LOSS / A) ** (1 / g)
    print(f"\n  cycles to {EOL_LOSS:.0f}% loss: {n_eol:.0f} -> {n_eol * ratio:.0f}")
    print(f"  life extension: {(ratio - 1) * 100:.0f}% "
          f"(+{n_eol * (ratio - 1):.0f} cycles), factor {ratio:.2f}x")
    print("\n  sensitivity to gamma:")
    for gg in (0.52, g, 1.0, 1.5):
        rr = (1 / S) ** (1 / gg)
        print(f"    gamma={gg:.3f} -> {(rr - 1) * 100:>6.0f}%  ({rr:.2f}x)")


if __name__ == "__main__":
    case_study_1()
    case_study_2()
