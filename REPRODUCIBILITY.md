# Reproducibility Guide

This document maps **every quantitative claim in the manuscript** to the exact
script that produces it and the released artifact that stores it. Nothing is
hardcoded: numbers are either recomputed live by running the released model
checkpoints, or read from per-experiment result files that are themselves
produced by the seeded scripts listed below.

> **One-command check:** `python3 reproduce_paper_results.py` verifies all 27
> reported numbers against this repository and prints, for each, the paper value,
> the recomputed value, and PASS/FAIL. This supersedes the older
> `REPRODUCE_PAPER_CLAIMS.py`, which targeted the pre-revision submission and is
> no longer maintained.

---

## Step 0 — Environment

```bash
# Python 3.9+ (arm64 and x86_64 both fine, but keep torch/numpy on the SAME arch)
pip install torch numpy pandas matplotlib scikit-learn scipy

# Pre-trained checkpoints (~1 MB) into reports/
python3 download_weights.py
```

The two **live** checks (causal attribution, counterfactual optimizer) need
`torch`. The other 25 checks only read released JSON artifacts and need no GPU.

---

## Step 1 — Verify everything at once

```bash
python3 reproduce_paper_results.py          # run all checks
python3 reproduce_paper_results.py --list   # list checks + their source scripts
```

Expected: **`PASS 27  FAIL 0`** on a machine where `torch` imports cleanly.
(On a machine where `torch` cannot load, the single live-only check — the 96.0%
attribution accuracy — is reported `SKIP` with the command to run it in a torch
environment; the counterfactual check falls back to its released artifact.)

---

## Step 2 — Claim → script → artifact map

Sections refer to the main manuscript. "Live" = runs a model checkpoint; the
rest recompute from the named released artifact under `reports/`.

### §3.1 Prediction, early warning, zero-shot transfer

| Claim (paper value) | Script | Artifact |
|---|---|---|
| Zero-shot HERO SOH MAE **9.1%**, RUL MAE **182.6 cyc** (Table 2) | `run_zeroshot_table_rebuild.py` | `reports/zeroshot_table_rebuild.json` |
| Best domain adaptation **5.6%** (Transformer+CORAL) | `run_zeroshot_table_rebuild.py` | same |
| Memory-bank injection leaves zero-shot **unchanged** | `run_zeroshot_bank_injection.py` | `reports/zeroshot_bank_injection.json` |
| Early warning **13/14** knee-points, recall **92.9%**, F1 **74.3%**, lead **121 cyc** | `run_early_warning_reconstruction.py` | `reports/early_warning_reconstruction.json` |
| Threshold-sweep F1 stays in **71–79%** band | `run_threshold_sensitivity.py` | `reports/early_warning_threshold_sensitivity.json` |

### §3.2 Physics-informed causal attribution

| Claim (paper value) | Script | Artifact |
|---|---|---|
| Attribution accuracy **96.0%** (72/75) | **[LIVE]** `VERIFY_96_ACCURACY.py` | released checkpoint `reports/pinn_causal/*.pt` |
| Data-driven baseline **93.3%** (in-distribution) | `train_datadriven_baseline.py` | `reports/causal_attribution/unified_validation/datadriven_baseline_results.json` |
| LODO cross-val: hybrid **89.3%** (67/75), data-driven **92.0%** (69/75) | `run_physics_value_experiments.py`, `train_datadriven_baseline.py` | `.../unified_validation_report.json`, `.../datadriven_baseline_results.json` |
| Storage-plating violations: data-driven **37.1%** → hybrid **0%** | `run_physics_value_experiments.py` | `.../physics_value_experiments.json` |
| Counterfactual avg reduction **34.6 pp** | **[LIVE]** `validate_counterfactual_optimization.py` | `reports/counterfactual_validation_results.json` |
| Matched-pair natural experiment: **3/3** direction, **27.2%** rate error | `validate_counterfactual_ground_truth.py` | `reports/counterfactual_ground_truth_validation.json` |
| XJTU high-C-rate: AM-loss dominant **25/25** (81.2 / 85.3 / 85.8%) | `src/run_xjtu_causal_attribution.py` | `reports/xjtu_causal_attribution_results.json` |
| EIS ρ = **0.65**, p = **0.023** (interfacial R vs storage T) | `run_eis_validation.py` | `reports/eis_attribution_validation.json` |
| ICA/DVA degradation-mode cross-check (25/25 LAM on XJTU; NASA 24 °C LAM / 43 °C LLI) | `run_ica_dva_validation.py` | `reports/ica_dva_validation/` |

### §3.3 Advisory / PATT

| Claim (paper value) | Script | Artifact |
|---|---|---|
| PATT held-out window accuracy **99.9%** | `train_patt_classifier.py` | `reports/patt_classifier/patt_results.json` |
| PATT cell-level split **99.6% ± 0.6%** (leakage control) | `train_patt_cell_split.py` | `reports/patt_cell_split/patt_cell_split_results.json` |

### §4 End-to-end illustrations

| Claim (paper value) | Script | Artifact |
|---|---|---|
| **Ill. 1** NASA B0046: plating **75.1%**, measured life ext. **147%** (p=0.029), CF ratio **2.60×** | `scripts/compute_nasa_case_study.py`, `scripts/compute_life_extension.py` | `reports/nasa_b0046_case_study_real.json` |
| **Ill. 2** XJTU: zero-shot R² **−0.44**, AM-loss **81.2%**, life ext. 41% | `scripts/compute_xjtu_case_study.py` | `reports/xjtu_case_study_real.json` |
| **Ill. 3** TJU: zero-shot R² **−0.67**, AM-loss **78.7%**, life ext. 154% | `scripts/compute_tju_case_study.py` | `reports/tju_cy25_2_case_study_real.json` |

### Figures

| Figure | Generator |
|---|---|
| Fig. 1 architecture overview | `scripts/generate_fig1_and_hero_performance.py` |
| Fig. 2 data-flow overview | `scripts/generate_dataflow_figure.py` |
| Fig. 3 physics map | `scripts/generate_physics_map_figure.py` |
| Fig. 8 counterfactual validation | `scripts/plot_counterfactual_validation.py` |
| Fig. 11 NASA case study | `scripts/plot_nasa_case_study.py` |
| Supp. ICA/DVA | `scripts/generate_ica_dva_figure.py` |
| Supp. PATT validation | `scripts/generate_patt_validation_figure.py` |

---

## Dataset notes

Two **distinct, non-overlapping** dataset groups are used:

| Purpose | Datasets | Size |
|---|---|---|
| HERO memory bank (prediction) | NASA, CALCE, Oxford, TJU, XJTU | 76 cells / 3,979 trajectories |
| Attribution benchmark (75 scenarios) | NASA, Panasonic EV, Nature MATR, Randomized 40 °C, HUST LFP | 75 scenarios |
| Storage / calendar aging (PATT, EIS) | Stanford Calendar Aging | 60 cells (+94-cell EIS corpus) |

All datasets are public; download links are in `README.md`.

---

## What the physics priors do and do not buy (stated honestly)

The ablations above show the explicit priors add a **modest 2.7 pp** in-distribution
accuracy margin and **no** detectable margin under leave-one-dataset-out. Their
demonstrated value is instead (i) **guaranteed physical consistency** — 37.1% of
storage-mode attributions are physically impossible ("plating without current")
for the data-driven variant vs **0%** for the physics-gated hybrid, on a
4,000-condition sweep; and (ii) **auditable, human-readable decision rules**. The
paper's claims are framed accordingly.

---

## Contact

- **Hariprasad Kodamana** — hkodamana@iitd.ac.in
- **Manojkumar Ramteke** — mcramteke@chemical.iitd.ac.in
