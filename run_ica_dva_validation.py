"""
ICA/DVA degradation-mode validation study (R3.1 / editor "physics not clearly presented").

Independent electrochemical validation of the PINN causal-attribution model on the
LLI-vs-LAM axis, using incremental capacity analysis (ICA, dQ/dV) and differential
voltage analysis (DVA, dV/dQ) computed from RAW cycling curves that the attribution
model never sees (it only receives context vectors: T, C-rates, SOC, mode).

Method
------
For each cell we fit every aged constant-current (CC) curve as an affine
transformation of the fresh reference curve of the SAME cell at the SAME rate:

    V_aged(Q) = V_ref(a*Q + b) + c

The three parameters separate the three canonical single-rate degradation
signatures (Dubarry et al., J. Power Sources 2012; Birkl et al., 2017):

    a > 1  : proportional compression of the whole curve along Q
             -> active-material loss (LAM) signature
    b > 0  : rigid translation of features toward lower cell capacity
             (electrode slippage) -> lithium-inventory loss (LLI) signature
    c      : rate-independent voltage offset at fixed current
             -> pure ohmic/polarization growth (impedance)

The end-of-CC capacity fade then decomposes analytically. With s_end the slope
dV/dQ of the reference at its CC endpoint Q_r:

    dQ_cc = Q_r*(1 - 1/a)   [LAM term]
          + b/a             [LLI term]
          + (c/s_end)/a     [polarization term]

We report each cell's LAM/LLI/polarization shares of CC-capacity fade, at the
end of life and at matched 10%/20% discharge-capacity fade, plus classic ICA
peak metrics (peak position shift, height and area ratios) as secondary
features.

Known limitation (stated in the paper): at a single C-rate, kinetic (diffusion)
limitation can also compress the accessible window and inflate the apparent-LAM
term. The validation therefore rests on RELATIVE contrasts across temperature
groups and C-rates, not on absolute mode fractions.

Datasets and model predictions being tested
-------------------------------------------
NASA (charge CC 1.5 A, identical protocol across groups; LCO-type 18650):
    24 degC  B0005/6/7/18   model attribution: Active-material loss  (LAM axis)
    43 degC  B0029/30/31/32 model attribution: SEI growth            (LLI axis)
     4 degC  B0045/46/47/48 model attribution: Lithium plating       (LLI axis)

XJTU (NCM, discharge CC 2 A = 1C shared across batches):
    Batch-1 2C charge   model: AM loss 49.2% dominant
    Batch-3 2.5C charge model: AM loss 62.7% dominant
    Batch-2 3C charge   model: AM loss 69.2% dominant
    -> tests both the LAM-dominance claim and the ordinal trend with C-rate.

Outputs
-------
    reports/ica_dva_validation/ica_dva_results.json
    reports/ica_dva_validation/curves_cache.npz   (per-cell curve samples for figure)

Run:  arch -arm64 python3 run_ica_dva_validation.py
"""

import json
import os
import glob
import numpy as np
import scipy.io as sio
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

OUT_DIR = "reports/ica_dva_validation"
NASA_DIR = "data/nasa_set5/raw"
XJTU_DIR = "data/new_datasets/XJTU/Battery Dataset"

NASA_GROUPS = {
    "NASA_24C": {"cells": ["B0005", "B0006", "B0007", "B0018"], "temp": 24,
                 "model_attribution": "ACTIVE_MATERIAL_LOSS", "model_axis": "LAM"},
    "NASA_43C": {"cells": ["B0029", "B0030", "B0031", "B0032"], "temp": 43,
                 "model_attribution": "SEI_GROWTH", "model_axis": "LLI"},
    "NASA_4C":  {"cells": ["B0045", "B0046", "B0047", "B0048"], "temp": 4,
                 "model_attribution": "LITHIUM_PLATING", "model_axis": "LLI"},
}

# from reports/xjtu_causal_attribution_results.json (mean AM-loss attribution share)
XJTU_GROUPS = {
    "XJTU_2C":   {"dir": "Batch-1", "pattern": "2C_battery-*.mat",   "crate": 2.0,
                  "model_am_share": 0.812},
    "XJTU_2p5C": {"dir": "Batch-3", "pattern": "R2.5_battery-*.mat", "crate": 2.5,
                  "model_am_share": 0.853},
    "XJTU_3C":   {"dir": "Batch-2", "pattern": "3C_battery-*.mat",   "crate": 3.0,
                  "model_am_share": 0.858},
}


# ---------------------------------------------------------------- extraction

def _integrate_q(I, t_sec):
    return np.concatenate([[0.0], np.cumsum(0.5 * (I[1:] + I[:-1]) * np.diff(t_sec))]) / 3600.0


def _resample(Q, V, n=400, win=41):
    """Resample V(Q) onto a uniform Q grid and smooth (noise floor reduction)."""
    if len(Q) < 60:
        return Q, V
    qi = np.linspace(Q[0], Q[-1], n)
    vi = np.interp(qi, Q, V)
    return qi, savgol_filter(vi, win, 3)


def nasa_charge_cc(cyc, imin=1.35, vmax=4.19, vmin=3.2, min_pts=80):
    """CC portion (1.5 A) of a NASA charge cycle -> (Q [Ah], V [V]) or None."""
    d = cyc["data"]
    V = np.asarray(d["Voltage_measured"], float)
    I = np.asarray(d["Current_measured"], float)
    t = np.asarray(d["Time"], float)
    mask = (I > imin) & (V < vmax) & (V > vmin)
    if mask.sum() < min_pts:
        return None
    V, I, t = V[mask], I[mask], t[mask]
    Q = _integrate_q(I, t)
    keep = np.concatenate([[True], np.diff(V) > -0.02])
    V, Q = V[keep], Q[keep]
    if len(V) < min_pts:
        return None
    return _resample(Q, savgol_filter(V, min(21, len(V) // 2 * 2 - 1), 3))


def xjtu_discharge_cc(cyc, min_pts=300):
    """1C CC discharge portion of an XJTU cycle record -> (Q [Ah], V [V]) or None."""
    V = np.asarray(cyc["voltage_V"], float)
    I = np.asarray(cyc["current_A"], float)
    t = np.asarray(cyc["relative_time_min"], float) * 60.0
    mask = I < -1.9
    if mask.sum() < min_pts:
        return None
    V, I, t = V[mask], -I[mask], t[mask]
    Q = _integrate_q(I, t)
    keep = np.concatenate([[True], np.diff(V) < 0.02])  # discharge: V decreasing
    V, Q = V[keep], Q[keep]
    if len(V) < min_pts:
        return None
    return _resample(Q, savgol_filter(V, min(51, len(V) // 2 * 2 - 1), 3))


# ---------------------------------------------------------------- affine fit

def fit_affine(Qr, Vr, Qa, Va, charge=True, x0=None):
    """Fit V_aged(Q) = V_ref(a*Q + b) + c. Returns (a, b, c, rms).

    Physical bounds: a >= 1 (active material cannot grow), and polarization
    can only increase with age (c >= 0 on charge, c <= 0 on discharge).
    Multi-start over a, optionally warm-started from the previous cycle's
    solution (continuation along the ageing trajectory).
    """
    f = interp1d(Qr, Vr, bounds_error=False, fill_value="extrapolate")

    def resid(p):
        a, b, c = p
        return f(a * Qa + b) + c - Va

    if charge:
        lo, hi = [0.98, -0.05, -0.02], [4.0, 1.0, 0.4]
    else:
        lo, hi = [0.98, -0.05, -0.4], [4.0, 1.0, 0.02]

    starts = [[a0, 0.0, 0.0] for a0 in (1.0, 1.3, 1.8, 2.5)]
    if x0 is not None:
        starts.insert(0, list(np.clip(x0, lo, hi)))
    best = None
    for s in starts:
        r = least_squares(resid, s, bounds=(lo, hi))
        rms = float(np.sqrt(np.mean(r.fun ** 2)))
        if best is None or rms < best[1]:
            best = (r.x, rms)
    (a, b, c), rms = best
    return a, b, c, rms


def fit_single_mode(Qr, Vr, Qa, Va, charge=True):
    """Model comparison: pure-LAM transform (compression only, b=0) vs
    pure-LLI transform (rigid shift only, a=1), each with a free polarization
    offset. Returns rms of both -> which single degradation mode explains the
    aged curve better. Robust to the a/b trade-off of the full affine fit."""
    f = interp1d(Qr, Vr, bounds_error=False, fill_value="extrapolate")
    c_lo, c_hi = (-0.02, 0.4) if charge else (-0.4, 0.02)

    r_lam = least_squares(lambda p: f(p[0] * Qa) + p[1] - Va,
                          [1.1, 0.0], bounds=([1.0, c_lo], [4.0, c_hi]))
    r_lli = least_squares(lambda p: f(Qa + p[0]) + p[1] - Va,
                          [0.05, 0.0], bounds=([-0.05, c_lo], [1.0, c_hi]))
    return (float(np.sqrt(np.mean(r_lam.fun ** 2))),
            float(np.sqrt(np.mean(r_lli.fun ** 2))))


def fade_decomposition(a, b, c, Qr, Vr):
    """Split end-of-CC capacity fade into LAM / LLI / polarization terms (Ah)."""
    q_end = Qr[-1]
    n = max(5, len(Qr) // 20)
    s_end = abs((Vr[-1] - Vr[-n]) / (Qr[-1] - Qr[-n] + 1e-12))
    lam = q_end * (1.0 - 1.0 / a)
    lli = b / a
    pol = (abs(c) / (s_end + 1e-12)) / a
    return lam, lli, pol


def ica_peak_metrics(Q, V, vgrid, charge=True):
    """Main dQ/dV peak position/height/area on a fixed V grid."""
    order = np.argsort(V)
    Vs, Qs = V[order], Q[order]
    Vu, iu = np.unique(Vs, return_index=True)
    if len(Vu) < 50:
        return None
    Qi = np.interp(vgrid, Vu, Qs[iu])
    Qi = savgol_filter(Qi, 51, 3)
    dqdv = np.gradient(Qi, vgrid)
    if not charge:
        dqdv = np.abs(dqdv)
    pk = int(np.argmax(dqdv))
    vp = vgrid[pk]
    w = (vgrid > vp - 0.15) & (vgrid < vp + 0.15)
    area = float(np.trapezoid(dqdv[w], vgrid[w]))
    return {"peak_v": float(vp), "peak_h": float(dqdv[pk]), "peak_area": area,
            "dqdv": dqdv}


# ---------------------------------------------------------------- per cell

def analyse_cell(curves, cap_traj, vgrid, charge=True, sample_every=1,
                 rms_max=0.015):
    """curves: list of (cycle_index, Q, V). cap_traj: capacity per cycle_index.

    Returns trajectory of affine parameters + fade shares, ICA metrics for
    reference and final curve, and matched-fade snapshots.
    """
    if len(curves) < 4:
        return None
    # reference: the highest-CC-capacity curve among the first 10 (cycle 0 can
    # be irregular; a low-capacity glitch must not become the reference)
    ref_pool = curves[:10]
    ref_idx = int(np.argmax([Q[-1] for _, Q, _ in ref_pool]))
    _, Qr, Vr = ref_pool[ref_idx]
    ref_ica = ica_peak_metrics(Qr, Vr, vgrid, charge)
    traj = []
    prev = None
    for k, Qa, Va in curves[ref_idx + 1::sample_every]:
        a, b, c, rms = fit_affine(Qr, Vr, Qa, Va, charge=charge, x0=prev)
        if rms > rms_max:  # bad fit -> skip
            continue
        prev = (a, b, c)
        lam, lli, pol = fade_decomposition(a, b, c, Qr, Vr)
        tot = lam + lli + pol
        traj.append({
            "cycle": int(k), "a": a, "b": b, "c": c, "rms_mV": rms * 1000,
            "q_cc": float(Qa[-1]),
            "fade_lam_Ah": lam, "fade_lli_Ah": lli, "fade_pol_Ah": pol,
            "share_lam": lam / tot if tot > 1e-4 else None,
            "share_lli": lli / tot if tot > 1e-4 else None,
            "share_pol": pol / tot if tot > 1e-4 else None,
            "soh": float(cap_traj.get(int(k), np.nan)),
        })
    if not traj:
        return None
    kf, Qf, Vf = curves[-1]
    fin_ica = ica_peak_metrics(Qf, Vf, vgrid, charge)
    rms_lam, rms_lli = fit_single_mode(Qr, Vr, Qf, Vf, charge=charge)
    final_soh = float(cap_traj.get(int(kf), np.nan))

    # matched-fade snapshots at 10% / 20% SOH fade
    snaps = {}
    sohs = np.array([t["soh"] for t in traj])
    for target in (0.90, 0.80):
        ok = np.where(sohs <= target)[0]
        if len(ok):
            snaps[f"at_{int((1-target)*100)}pct_fade"] = traj[int(ok[0])]

    out = {
        "n_fits": len(traj),
        "ref_q_cc": float(Qr[-1]),
        "final": traj[-1],
        "matched_fade": snaps,
        "single_mode_rms_mV": {"pure_LAM": rms_lam * 1000, "pure_LLI": rms_lli * 1000,
                               "better": "LAM" if rms_lam < rms_lli else "LLI",
                               "at_soh": final_soh},
        "trajectory": traj,
    }
    if ref_ica and fin_ica:
        out["ica_peak"] = {
            "dV_peak_mV": (fin_ica["peak_v"] - ref_ica["peak_v"]) * 1000,
            "height_ratio": fin_ica["peak_h"] / ref_ica["peak_h"],
            "area_ratio": fin_ica["peak_area"] / ref_ica["peak_area"],
        }
        out["_curves"] = {  # for the figure; stripped from JSON
            "vgrid": vgrid, "dqdv_ref": ref_ica["dqdv"], "dqdv_final": fin_ica["dqdv"],
            "Qr": Qr, "Vr": Vr, "Qf": Qf, "Vf": Vf,
        }
    return out


# ---------------------------------------------------------------- runners

def run_nasa():
    vgrid = np.linspace(3.5, 4.19, 400)
    results = {}
    for gname, g in NASA_GROUPS.items():
        results[gname] = {"meta": {k: v for k, v in g.items()}, "cells": {}}
        for cell in g["cells"]:
            path = os.path.join(NASA_DIR, f"{cell}.mat")
            if not os.path.exists(path):
                print(f"  {cell}: missing"); continue
            m = sio.loadmat(path, simplify_cells=True)
            cy = m[cell]["cycle"]
            # some files (4 degC campaign) interleave cycles at 24 degC and at
            # the target temperature; reference and aged curves must share the
            # same ambient, otherwise thermal polarization masquerades as ageing
            cy = [c for c in cy if c.get("ambient_temperature") == g["temp"]]
            # capacity trajectory keyed by preceding-charge index
            cap = {}
            charges = []
            last_dis_cap = np.nan
            init_cap = None
            for i, c in enumerate(cy):
                if c["type"] == "discharge" and "Capacity" in c["data"]:
                    q = float(c["data"]["Capacity"])
                    if q > 0.5:
                        last_dis_cap = q
                        if init_cap is None:
                            init_cap = q
                if c["type"] == "charge":
                    r = nasa_charge_cc(c)
                    if r is not None:
                        charges.append((i, *r))
                        cap[i] = last_dis_cap
            if init_cap:
                cap = {k: v / init_cap for k, v in cap.items()}
            res = analyse_cell(charges, cap, vgrid, charge=True)
            if res is None:
                print(f"  {cell}: insufficient data"); continue
            results[gname]["cells"][cell] = res
            _print_cell(cell, gname, res)
    return results


def run_xjtu():
    vgrid = np.linspace(2.7, 4.10, 400)
    results = {}
    for gname, g in XJTU_GROUPS.items():
        results[gname] = {"meta": {k: v for k, v in g.items()}, "cells": {}}
        files = sorted(glob.glob(os.path.join(XJTU_DIR, g["dir"], g["pattern"])))
        for path in files:
            cell = os.path.basename(path)[:-4]
            m = sio.loadmat(path, simplify_cells=True)
            data = m["data"]
            summ = m["summary"]
            dis_cap = np.asarray(summ["discharge_capacity_Ah"], float)
            curves = []
            cap = {}
            init = None
            for i, c in enumerate(data):
                desc = str(c.get("description", ""))
                # batch-3 rotates discharge rates (0.5C..5C); keep only the 1C
                # discharge cycles so curves are comparable across all batches
                if "test capacity" in desc or "and 1C discharge" not in desc:
                    continue
                r = xjtu_discharge_cc(c)
                if r is None:
                    continue
                curves.append((i, *r))
                if i < len(dis_cap):
                    if init is None and dis_cap[i] > 0.5:
                        init = dis_cap[i]
                    cap[i] = dis_cap[i] / init if init else np.nan
            res = analyse_cell(curves, cap, vgrid, charge=False,
                               sample_every=max(1, len(curves) // 60),
                               rms_max=0.030)
            if res is None:
                print(f"  {cell}: insufficient data"); continue
            results[gname]["cells"][cell] = res
            _print_cell(cell, gname, res)
    return results


def _print_cell(cell, gname, res):
    f = res["final"]
    shares = (f"{f['share_lam']:.2f}/{f['share_lli']:.2f}/{f['share_pol']:.2f}"
              if f["share_lam"] is not None else "n/a (no fade)")
    print(f"  {cell} ({gname}): a={f['a']:.3f} b={f['b']*1000:.0f}mAh "
          f"c={f['c']*1000:.0f}mV shares LAM/LLI/POL = {shares} "
          f"(rms {f['rms_mV']:.1f} mV, n={res['n_fits']})")


def summarise(results):
    """Group-level verdicts.

    NASA (truncated charge-CC segments): the affine decomposition is stable
    (rms 2-6 mV) -> mean LAM/LLI/POL fade shares.
    XJTU (full 1C discharge curves anchored at top-of-charge): the a/b
    trade-off makes the three-way shares unstable, but the two single-mode
    transforms are well-posed -> pure-LAM vs pure-LLI model-comparison votes.
    (The single-mode comparison is NOT valid for NASA: a short aged charge
    segment can slide anywhere along the reference, which degenerately favors
    the shift model.)
    """
    summary = {}
    for gname, g in results.items():
        if "model_am_share" in g["meta"]:  # XJTU group -> single-mode votes
            votes, pairs, sohs = [], [], []
            for cell, r in g["cells"].items():
                sm = r.get("single_mode_rms_mV")
                if not sm:
                    continue
                votes.append(sm["better"])
                pairs.append((sm["pure_LAM"], sm["pure_LLI"]))
                sohs.append(sm.get("at_soh"))
            if not votes:
                continue
            arr = np.array(pairs)
            summary[gname] = {
                "n_cells": len(votes),
                "lam_better": int(sum(v == "LAM" for v in votes)),
                "lli_better": int(sum(v == "LLI" for v in votes)),
                "median_rms_pure_LAM_mV": float(np.median(arr[:, 0])),
                "median_rms_pure_LLI_mV": float(np.median(arr[:, 1])),
                "mean_final_soh": float(np.nanmean(np.array(sohs, float))),
                "ica_dominant_axis": "LAM" if sum(v == "LAM" for v in votes) > len(votes) / 2 else "LLI",
            }
            continue
        rows_final, rows_m = [], []
        for cell, r in g["cells"].items():
            f = r["final"]
            if f["share_lam"] is not None:
                rows_final.append((f["share_lam"], f["share_lli"], f["share_pol"]))
            mf = r["matched_fade"].get("at_20pct_fade") or r["matched_fade"].get("at_10pct_fade")
            if mf and mf["share_lam"] is not None:
                rows_m.append((mf["share_lam"], mf["share_lli"], mf["share_pol"]))
        if not rows_final:
            continue
        arr = np.array(rows_final)
        entry = {
            "n_cells": len(rows_final),
            "mean_share_lam": float(arr[:, 0].mean()),
            "mean_share_lli": float(arr[:, 1].mean()),
            "mean_share_pol": float(arr[:, 2].mean()),
            "std_share_lam": float(arr[:, 0].std()),
            "ica_dominant_axis": "LAM" if arr[:, 0].mean() > arr[:, 1].mean() else "LLI",
        }
        if rows_m:
            am = np.array(rows_m)
            entry["matched_fade_mean_share_lam"] = float(am[:, 0].mean())
            entry["matched_fade_mean_share_lli"] = float(am[:, 1].mean())
            entry["matched_fade_mean_share_pol"] = float(am[:, 2].mean())
        summary[gname] = entry
    return summary


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=== NASA (charge CC 1.5 A, identical across temperature groups) ===")
    nasa = run_nasa()
    print("\n=== XJTU (discharge CC 1C, identical across C-rate batches) ===")
    xjtu = run_xjtu()

    results = {**nasa, **xjtu}
    summary = summarise(results)

    print("\n=== GROUP SUMMARY ===")
    for gname, s in summary.items():
        meta = results[gname]["meta"]
        model = meta.get("model_axis") or f"AM {meta.get('model_am_share', 0):.0%}"
        if "mean_share_lam" in s:
            detail = (f"fade shares LAM {s['mean_share_lam']:.2f} | "
                      f"LLI {s['mean_share_lli']:.2f} | POL {s['mean_share_pol']:.2f}")
        else:
            detail = (f"pure-LAM better in {s['lam_better']}/{s['n_cells']} cells "
                      f"(median rms {s['median_rms_pure_LAM_mV']:.0f} vs "
                      f"{s['median_rms_pure_LLI_mV']:.0f} mV, mean final SOH "
                      f"{s['mean_final_soh']:.2f})")
        print(f"{gname}: {detail}  -> ICA says {s['ica_dominant_axis']}, "
              f"model says {model} (n={s['n_cells']})")

    # save curves separately (npz), strip from JSON
    curves = {}
    for gname, g in results.items():
        for cell, r in g["cells"].items():
            cur = r.pop("_curves", None)
            if cur is not None:
                for key, arr in cur.items():
                    curves[f"{gname}__{cell}__{key}"] = np.asarray(arr)
    np.savez_compressed(os.path.join(OUT_DIR, "curves_cache.npz"), **curves)

    with open(os.path.join(OUT_DIR, "ica_dva_results.json"), "w") as fh:
        json.dump({"summary": summary, "groups": results}, fh, indent=1)
    print(f"\nSaved {OUT_DIR}/ica_dva_results.json and curves_cache.npz")


if __name__ == "__main__":
    main()
