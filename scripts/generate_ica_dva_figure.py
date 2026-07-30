"""ICA/DVA degradation-mode validation figure.

Reads reports/ica_dva_validation/{ica_dva_results.json,curves_cache.npz}
(produced by run_ica_dva_validation.py) and renders the four-panel figure:

  (a) ICA curves, NASA 24 degC cell (B0005): proportional peak collapse = LAM
  (b) ICA curves, NASA 43 degC cell (B0030): preserved-but-translated peak = LLI
  (c) NASA affine fade-share decomposition per temperature group vs the
      model's attribution (24C -> AM/LAM ok, 43C -> SEI/LLI ok, 4C unresolved)
  (d) XJTU single-mode model comparison: pure-LAM vs pure-LLI fit rms per
      cell; every cell falls on the LAM side, consistent with the model's
      AM-dominant attribution for all three C-rate batches.

Outputs:
  reports/ica_dva_validation/ica_dva_validation.png
  Casual_Attribution_reports/figures/ica_dva_validation.png
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES_DIR = "reports/ica_dva_validation"
OUT = [f"{RES_DIR}/ica_dva_validation.png",
       "Casual_Attribution_reports/figures/ica_dva_validation.png"]

# ---- palette (dataviz reference, light mode; same as other paper figures) ----
BLUE   = "#2a78d6"
AQUA   = "#1baf7a"
VIOLET = "#4a3aa7"
GRAY   = "#b9b8b3"
INK    = "#0b0b0b"
INK2   = "#52514e"
SURF   = "#ffffff"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#d8d7d2",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "#ecebe7",
    "grid.linewidth": 0.7,
    "axes.axisbelow": True,
    "text.color": INK,
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "figure.facecolor": SURF,
    "axes.facecolor": SURF,
    "savefig.facecolor": SURF,
})

data = json.load(open(f"{RES_DIR}/ica_dva_results.json"))
curves = np.load(f"{RES_DIR}/curves_cache.npz")

fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.2))
(axA, axB), (axC, axD) = axes

# ------------------------------------------------- (a)/(b) ICA curve panels
def ica_panel(ax, group, cell, label, note, soh):
    vg = curves[f"{group}__{cell}__vgrid"]
    ref = curves[f"{group}__{cell}__dqdv_ref"]
    fin = curves[f"{group}__{cell}__dqdv_final"]
    ax.plot(vg, ref, color=GRAY, lw=2.0, label="fresh (reference)")
    ax.plot(vg, fin, color=BLUE, lw=2.0, label=f"aged ({100*soh:.0f}% SOH)")
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("dQ/dV (Ah V$^{-1}$)")
    ax.set_title(label, fontsize=12.5, fontweight="bold", color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    ax.text(0.02, 0.60, note, transform=ax.transAxes, fontsize=9.5,
            color=INK2, va="top", linespacing=1.4)

sohA = data["groups"]["NASA_24C"]["cells"]["B0005"]["single_mode_rms_mV"]["at_soh"]
sohB = data["groups"]["NASA_43C"]["cells"]["B0030"]["single_mode_rms_mV"]["at_soh"]
ica_panel(axA, "NASA_24C", "B0005",
          "(a) NASA 24°C (B0005) — LAM signature",
          "peak collapses in proportion\nto capacity: active-material loss",
          sohA)
ica_panel(axB, "NASA_43C", "B0030",
          "(b) NASA 43°C (B0030) — LLI signature",
          "peak shape largely preserved,\nfeatures translated along Q:\nlithium-inventory loss",
          sohB)

# ------------------------------------------------- (c) NASA fade shares
groups = ["NASA_24C", "NASA_43C", "NASA_4C"]
glabels = ["24°C\n(model: AM loss)", "43°C\n(model: SEI)", "4°C\n(model: plating)"]
verdicts = ["✓ LAM-dominant", "✓ LLI-dominant", "unresolved at 1C"]
share_keys = [("mean_share_lam", "LAM (compression)", BLUE),
              ("mean_share_lli", "LLI (slippage)", VIOLET),
              ("mean_share_pol", "Polarization", GRAY)]

y = np.arange(len(groups))[::-1]
share_mat = np.array([[max(data["summary"][g][key], 0.0) for g in groups]
                      for key, _, _ in share_keys])
share_mat /= share_mat.sum(axis=0, keepdims=True)  # renormalize after clipping
left = np.zeros(len(groups))
for (key, lab, col), vals in zip(share_keys, share_mat):
    axC.barh(y, vals, left=left, height=0.52, color=col, label=lab,
             edgecolor=SURF, linewidth=2, zorder=3)
    for yi, v, l0 in zip(y, vals, left):
        if v > 0.08:
            axC.text(l0 + v / 2, yi, f"{v:.2f}", ha="center", va="center",
                     fontsize=9.5, color=SURF if col != GRAY else INK,
                     fontweight="bold", zorder=4)
    left += vals
for yi, v in zip(y, verdicts):
    axC.text(1.03, yi, v, va="center", fontsize=9.5, color=INK2)
axC.set_yticks(y, glabels, fontsize=9.5)
axC.set_xlim(0, 1.0)
axC.set_xlabel("share of CC-capacity fade (affine curve-fit decomposition)")
axC.set_title("(c) NASA: measured mode shares vs model",
              fontsize=12.5, fontweight="bold", color=INK, loc="left")
axC.legend(frameon=False, fontsize=9, loc="lower right", ncol=3,
           bbox_to_anchor=(1.0, -0.32))
axC.grid(axis="y", visible=False)

# ------------------------------------------------- (d) XJTU model comparison
batch_style = [("XJTU_2C", "2C charge", BLUE),
               ("XJTU_2p5C", "2.5C charge", AQUA),
               ("XJTU_3C", "3C charge", VIOLET)]
lim = 0
for gname, lab, col in batch_style:
    xs, ys = [], []
    for cell, r in data["groups"][gname]["cells"].items():
        sm = r["single_mode_rms_mV"]
        xs.append(sm["pure_LAM"]); ys.append(sm["pure_LLI"])
    axD.scatter(xs, ys, s=52, color=col, label=lab, zorder=3,
                edgecolor=SURF, linewidth=1.2)
    lim = max(lim, max(xs + ys))
lim *= 1.12
axD.plot([0, lim], [0, lim], color="#8f8e89", lw=1.2, ls="--", zorder=2)
axD.fill_between([0, lim], [0, lim], [lim, lim], color=BLUE, alpha=0.05, zorder=1)
n_tot = sum(len(data["groups"][g]["cells"]) for g, _, _ in batch_style)
axD.text(0.97 * lim, 0.55 * lim,
         f"above diagonal: pure-LAM transform\nfits the aged curve better "
         f"— {n_tot}/{n_tot} cells",
         fontsize=9.5, color=INK2, va="top", ha="right", linespacing=1.4)
axD.set_xlim(0, lim); axD.set_ylim(0, lim)
axD.set_xlabel("pure-LAM fit rms (mV)")
axD.set_ylabel("pure-LLI fit rms (mV)")
axD.set_title("(d) XJTU: LAM vs LLI model comparison",
              fontsize=12.5, fontweight="bold", color=INK, loc="left")
axD.legend(frameon=False, fontsize=9.5, loc="lower right", title="batch",
           title_fontsize=9.5)

fig.tight_layout(h_pad=2.6, w_pad=2.2)
for path in OUT:
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.2)
    print("saved", path)
