import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

SUMMARY_PATH = os.environ.get("NASA_SET5_SUMMARY", "data/nasa_set5/summary.csv")
OUT_DIR = Path(os.environ.get("EDA_OUT", "reports/nasa_set5"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Reading {SUMMARY_PATH}")
summary = pd.read_csv(SUMMARY_PATH)
print(summary.head())

# Basic stats
stats = summary.groupby('cell_id').agg(
    n_cycles=("cycle_index", "max"),
    ir_start=("IR", lambda s: np.nanmedian(s.iloc[:5])),
    ir_end=("IR", lambda s: np.nanmedian(s.tail(5))),
    eol_cycle=("EoL", lambda s: int(np.argmax(s.values)) if s.any() else -1),
)
stats_path = OUT_DIR / "stats.csv"
stats.to_csv(stats_path)
print(f"Saved stats to {stats_path}")

# Plot IR and SOH_R trajectories per cell (small multiples)
cell_ids = sorted(summary['cell_id'].unique())

for metric in ["IR", "SOH_R"]:
    n = len(cell_ids)
    cols = 6
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*2), sharex=True)
    axes = axes.ravel()
    for i, cid in enumerate(cell_ids):
        ax = axes[i]
        dfc = summary[summary['cell_id'] == cid]
        ax.plot(dfc['cycle_index'], dfc[metric], lw=1)
        # mark EoL
        if dfc['EoL'].any():
            eol_idx = dfc.loc[dfc['EoL']].index[0]
            eol_cyc = int(dfc.loc[eol_idx, 'cycle_index'])
            ax.axvline(eol_cyc, color='r', ls='--', lw=0.8)
        ax.set_title(cid, fontsize=8)
    for j in range(i+1, rows*cols):
        axes[j].axis('off')
    fig.suptitle(f"{metric} trajectories by cell")
    fig.tight_layout(rect=[0,0,1,0.96])
    outp = OUT_DIR / f"{metric.lower()}_trajectories.png"
    fig.savefig(outp, dpi=150)
    plt.close(fig)
    print(f"Saved {outp}")

# EoL distribution
fig, ax = plt.subplots(figsize=(6,4))
eol_by_cell = summary.groupby('cell_id')['EoL'].apply(lambda s: int(np.argmax(s.values)) if s.any() else -1)
valid = [v for v in eol_by_cell.values if v >= 0]
ax.hist(valid, bins=20)
ax.set_xlabel('EoL cycle index')
ax.set_ylabel('Count')
ax.set_title('Distribution of EoL cycles (IR threshold)')
outp = OUT_DIR / "eol_distribution.png"
fig.tight_layout()
fig.savefig(outp, dpi=150)
plt.close(fig)
print(f"Saved {outp}")

print("EDA complete.")
