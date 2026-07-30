"""Regenerate the PATT domain-classification validation figure directly from
reports/patt_classifier/patt_results.json (the artifact saved alongside
patt_best.pt). Replaces the previous figure whose confusion-matrix counts and
accuracy labels did not match the recorded results.

Artifact values (May 2026 run):
  confusion_matrix [[388, 0], [1, 352]]  rows/cols ordered [storage=0, cycling=1]
  accuracy 99.87%, precision (cycling) 100%, recall (cycling) 99.72%, F1 99.86%
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = "/Users/sushanth.c/physics_informed_model/ML_prediction_SOH/reports/patt_classifier/patt_results.json"
OUT = "/Users/sushanth.c/physics_informed_model/ML_prediction_SOH/Casual_Attribution_reports"

with open(RES) as f:
    r = json.load(f)
cm = np.array(r["test_metrics"]["confusion_matrix"])  # rows true [storage, cycling]
acc = r["test_metrics"]["accuracy"]
prec = r["test_metrics"]["precision"]
rec = r["test_metrics"]["recall"]
f1 = r["test_metrics"]["f1_score"]

n_storage, n_cycling = cm[0].sum(), cm[1].sum()
storage_rec = cm[0, 0] / n_storage
cycling_rec = cm[1, 1] / n_cycling

BLUE = "#2a78d6"; AQUA = "#1baf7a"; RED = "#e34948"; INK = "#0b0b0b"; INK2 = "#52514e"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.edgecolor": "#d8d7d2",
    "axes.linewidth": 0.8, "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

fig, axes = plt.subplots(2, 2, figsize=(10.5, 9))
fig.suptitle("PATT Domain Classification (Cycling vs Storage), Held-Out Test Set",
             fontsize=13.5, fontweight="bold")

# (a) confusion matrix
axA = axes[0, 0]
im = axA.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
for i in range(2):
    for j in range(2):
        axA.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=17,
                 fontweight="bold", color="white" if cm[i, j] > cm.max()/2 else INK)
axA.set_xticks([0, 1]); axA.set_yticks([0, 1])
axA.set_xticklabels(["Storage", "Cycling"]); axA.set_yticklabels(["Storage", "Cycling"])
axA.set_xlabel("Predicted class"); axA.set_ylabel("Actual class")
axA.set_title("a  Confusion Matrix", loc="left", fontweight="bold")

# (b) per-class recall + overall accuracy
axB = axes[0, 1]
vals = [acc * 100, cycling_rec * 100, storage_rec * 100]
labels = ["Overall", "Cycling", "Storage"]
bars = axB.bar(labels, vals, color=[BLUE, AQUA, AQUA], width=0.55, zorder=3)
for b, v in zip(bars, vals):
    axB.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.1f}%",
             ha="center", va="bottom", fontsize=10.5, fontweight="bold")
axB.set_ylim(99.0, 100.35); axB.set_ylabel("Accuracy / recall (%)")
axB.grid(axis="y", color="#ecebe7", zorder=0)
axB.set_title("b  Classification Accuracy (y-axis from 99%)", loc="left", fontweight="bold")

# (c) test-set composition
axC = axes[1, 0]
axC.pie([n_cycling, n_storage],
        labels=[f"Cycling\n{n_cycling} windows\n({100*n_cycling/(n_cycling+n_storage):.1f}%)",
                f"Storage\n{n_storage} windows\n({100*n_storage/(n_cycling+n_storage):.1f}%)"],
        colors=[AQUA, BLUE], autopct=None, startangle=90,
        wedgeprops=dict(width=0.42, edgecolor="white"))
axC.set_title("c  Test-Set Composition", loc="left", fontweight="bold")

# (d) metrics vs MLP baseline
axD = axes[1, 1]
mvals = [acc * 100, prec * 100, rec * 100, f1 * 100]
mlabels = ["Accuracy", "Precision", "Recall", "F1"]
bars = axD.bar(mlabels, mvals, color=BLUE, width=0.55, zorder=3)
for b, v in zip(bars, mvals):
    axD.text(b.get_x() + b.get_width()/2, v + 0.15, f"{v:.1f}%",
             ha="center", va="bottom", fontsize=10.5, fontweight="bold")
axD.axhline(92.2, color=RED, ls="--", lw=1.5, zorder=4)
axD.text(3.4, 92.6, "MLP baseline (92.2%)", color=RED, fontsize=9.5, ha="right")
axD.set_ylim(88, 101.6); axD.set_ylabel("Score (%), cycling = positive class")
axD.grid(axis="y", color="#ecebe7", zorder=0)
axD.set_title("d  Performance Metrics", loc="left", fontweight="bold")

fig.tight_layout(rect=[0, 0, 1, 0.95])
for path in (f"{OUT}/patt_domain_classification.png",
             f"{OUT}/figures/patt_domain_classification.png"):
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.2)
plt.close(fig)
print(f"written; overall={acc*100:.2f}% cycling_recall={cycling_rec*100:.2f}% "
      f"storage_recall={storage_rec*100:.2f}%")
