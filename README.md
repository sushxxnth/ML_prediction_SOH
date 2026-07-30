# Battery Health Management with Physics-Informed Causal Attribution

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sushxxnth/ML_prediction_SOH/blob/main/Reproduce_Paper_Results.ipynb)

> **Paper**: "Analyzing Degradation and Extending Life of Electric Vehicle Batteries using Physics-Aware Transformers"

> **Reproduce every reported number in ~5 minutes:** click **Open in Colab** above, or run `python3 reproduce_paper_results.py` locally. See [Reproducing Paper Results](#reproducing-paper-results).


---

## Overview

This repository contains the official implementation of the Physics-Informed Battery Health Management framework. The proposed system extends beyond traditional Remaining Useful Life (RUL) prediction by providing mechanism-specific causal attribution and counterfactual optimization interventions. 

The framework addresses a critical limitation in current Battery Management Systems (BMS): the inability to attribute capacity fade to specific underlying electrochemical degradation mechanisms (e.g., Solid Electrolyte Interphase (SEI) growth, lithium plating, active material loss) using only non-invasive operating data.

---

## Technical Contributions

The framework comprises four integrated modules:

### 1. Predictive Engine: Hybrid Estimation via Retrieval Optimization (HERO)
- A retrieval-augmented SOH/RUL prediction architecture utilizing cross-attention over a memory bank of 3,979 degradation trajectories.
- Attains a strong **in-distribution** held-out SOH MAE of 0.74% (R² = 0.990). Under a strict zero-shot cross-chemistry protocol we report the calibrated finding that transfer is difficult for **all** methods evaluated (HERO included), with unsupervised feature alignment (CORAL) giving the largest gains — see Table 2 of the paper.

### 2. Diagnostic Engine: Hybrid Physics-Informed Neural Network (PINN)
- A multi-head causal attribution network bounded by electrochemical priors (Arrhenius kinetics, Tafel equations).
- Achieves 96.0% accuracy in isolating the dominant degradation mechanism from macroscopic voltage/current/temperature time-series data.

### 3. Proactive Monitoring: Early Warning Engine
- Detects the onset of nonlinear capacity fade and knee-point acceleration.
- Detects 13 of 14 genuine knee-point failures (92.9% recall, 74.3% F1) on the 34 NASA cells, with an average lead time of 121 cycles prior to end-of-life.

### 4. Prescriptive Advisory: Counterfactual Optimizer
- Simulates mechanism trajectories under hypothetical operating conditions using a differentiable physics proxy.
- Recommends mathematically optimal, actionable interventions (e.g., specific current reductions, thermal adjustments) to explicitly mitigate the dominant degradation mechanism.

---

## Benchmark Results

| Metric | Result | Significance |
|--------|--------|--------------|
| **Causal Attribution Accuracy** | 96.0% in-distribution (72/75); 89.3% under leave-one-dataset-out | Verifiable mechanism diagnosis across 5 chemistry and condition groups. |
| **Physical consistency** | 0% storage-plating violations (vs 37.1% for a physics-free variant) | The physics gate makes impossible attributions impossible for *any* input. |
| **SOH Prediction (HERO)** | 0.74% MAE, R² = 0.990 (in-distribution) | Robust trajectory forecasting within training chemistries. |
| **Early Warning** | 92.9% recall, 121-cycle lead time | Proactive intervention prior to knee-point failure. |
| **Domain Classification (PATT)** | 99.9% (held-out window); 99.6% (cell-level split) | Accurately distinguishes storage (calendar) vs. cycling aging. |
| **Counterfactual Optimizer** | 34.6 pp avg reduction in dominant mechanism | Prescribes interventions that measurably cut the dominant degradation mode. |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/sushxxnth/ML_prediction_SOH.git
cd ML_prediction_SOH

# Install required dependencies
pip install torch numpy pandas matplotlib scikit-learn scipy openpyxl
```

---

## Reproducing Paper Results

The repository includes pre-trained weights and validation scripts to reproduce all quantitative claims in the manuscript. Checkpoints and result artifacts are hosted as a [GitHub Release (v1.1.0)](https://github.com/sushxxnth/ML_prediction_SOH/releases/tag/v1.1.0) to keep the repository lightweight.

### Fastest path — Google Colab (no local setup)

Click the **[Open in Colab](https://colab.research.google.com/github/sushxxnth/ML_prediction_SOH/blob/main/Reproduce_Paper_Results.ipynb)** badge (or open `Reproduce_Paper_Results.ipynb`) and **Runtime → Run all**. A free CPU runtime is enough; the whole notebook takes about 5 minutes and prints a table of all 27 reported numbers with `PASS`/`FAIL`. A fully green run shows `PASS 27  FAIL 0`. No GPU, no dataset download, and no manual setup are required — the notebook clones the repo, installs dependencies, downloads the released artifacts, and runs the verifier.

### Setup (one-time, after cloning locally)

```bash
# Download pre-trained weights and result files (~1.3 MB)
python3 download_weights.py
```

This installs into `reports/` (gitignored, local only): the Hybrid PINN, PATT, HERO and causal-model checkpoints, plus every per-experiment result artifact the verifier reads.

### Where each result comes from (repository map)

| Paper claim | Run this | §  |
|---|---|---|
| **All 27 numbers at once** | `reproduce_paper_results.py` | — |
| Causal attribution 96.0% (live) | `VERIFY_96_ACCURACY.py` | 3.2 |
| Data-driven baseline 93.3%, LODO 89.3%, storage-plating 37.1%→0 | `run_physics_value_experiments.py`, `train_datadriven_baseline.py` | 3.2 |
| Counterfactual 34.6 pp; matched-pair 3/3, 27.2% | `validate_counterfactual_optimization.py`, `validate_counterfactual_ground_truth.py` | 3.2 |
| Zero-shot table (Table 2) + CORAL/MMD/few-shot | `run_zeroshot_table_rebuild.py`, `run_domain_adaptation_baselines.py` | 3.1 |
| Early warning 92.9% recall / 121-cyc lead; threshold sweep | `run_early_warning_reconstruction.py`, `run_threshold_sensitivity.py` | 3.1 |
| EIS ρ=0.65; ICA/DVA degradation-mode cross-check | `run_eis_validation.py`, `run_ica_dva_validation.py` | 3.2 |
| PATT 99.9% / 99.6% cell-level | `train_patt_classifier.py`, `train_patt_cell_split.py` | 3.3 |
| End-to-end case studies (NASA / XJTU / TJU) | `scripts/compute_{nasa,xjtu,tju}_case_study.py` | 4 |
| Figures 1-3, 8, 11 | `scripts/generate_*` / `scripts/plot_*` | — |

`reproduce_paper_results.py` needs **no dataset download** — it reads the released artifacts and runs the two model checks live. The individual `run_*.py` / `train_*.py` / `validate_*` scripts in the table above instead **regenerate** those artifacts from the raw battery datasets (~2.7 GB, see [Training Your Own Models](#training-your-own-models)), so run them only if you want to rebuild an artifact end-to-end.

The complete claim → script → artifact map is in **[REPRODUCIBILITY.md](REPRODUCIBILITY.md)**.

### Verify all paper claims

```bash
python3 reproduce_paper_results.py          # 27 checks, paper-vs-computed, PASS/FAIL
python3 reproduce_paper_results.py --list   # list each check and its source script
```

Each check prints the value stated in the paper, the value recomputed from this
repository, and PASS/FAIL. Two checks run the model live (need `torch`); the rest
read released, seeded result artifacts. See **[REPRODUCIBILITY.md](REPRODUCIBILITY.md)**
for the full claim → script → artifact map.

### Google Colab Quick Start

```python
!git clone https://github.com/sushxxnth/ML_prediction_SOH.git
%cd ML_prediction_SOH
!pip install torch numpy pandas matplotlib scikit-learn scipy openpyxl
!python3 download_weights.py
!python3 reproduce_paper_results.py
```

### Specific Verifications

**1. Causal Attribution Accuracy (96.0%, runs the checkpoint)**
```bash
python3 VERIFY_96_ACCURACY.py
```

**2. Counterfactual Optimizer (34.6 pp) and matched-pair ground truth**
```bash
python3 validate_counterfactual_optimization.py
python3 validate_counterfactual_ground_truth.py
```

**3. Early warning, EIS, ICA/DVA, zero-shot table**
```bash
python3 run_early_warning_reconstruction.py
python3 run_eis_validation.py
python3 run_ica_dva_validation.py
python3 run_zeroshot_table_rebuild.py
```

---

## Datasets

The models were trained and validated on a comprehensive aggregation of publicly available datasets encompassing four lithium-ion chemistries (LCO, NCM, NCA, LFP) and diverse operating conditions (-40°C to 50°C, 0.5C to 8C rates):

1. **[NASA Ames Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)** (34 cells)
2. **[CALCE Battery Research Group](https://calce.umd.edu/battery-data)** (18 cells)
3. **[Oxford Battery Degradation Dataset](https://batteryintelligence.web.ox.ac.uk/data-and-code)** (8 cells)
4. **[TJU (Tongji University)](https://zenodo.org/records/6405084)** (40 cells)
5. **[XJTU Battery Dataset](https://github.com/Ruifeng-Tan/BatteryLife)** (26 cells)
6. **[Stanford Calendar Aging Dataset](https://web.stanford.edu/group/chuehgroup/datasets.html)** (60 cells)

---

## Repository Structure

```
ML_prediction_SOH/
├── src/
│   ├── models/
│   │   ├── pinn_causal_attribution.py     # Diagnostic Engine
│   │   ├── rad_model.py                   # HERO Predictive Engine
│   │   └── physics_aware_transformer.py   # PATT Domain Classifier
│   ├── optimization/
│   │   └── counterfactual_intervention.py # Prescriptive Advisory
│   ├── advisory/
│   │   └── warning_engine.py              # Early Warning System
│   └── data/                              # Dataloaders and pipelines
├── scripts/                               # Plotting and figure generation utilities
├── figures/                               # Generated paper visualizations
├── reports/                               # Pre-trained model weights and JSON results
│   ├── pinn_causal/
│   ├── hero_model/
│   └── patt_classifier/
├── reproduce_paper_results.py             # Main reproducibility script (27 checks) for reviewers
├── REPRODUCIBILITY.md                     # Claim → script → artifact map
└── VERIFY_96_ACCURACY.py                  # PINN evaluation script (live 96% check)
```

---

## The Data

The framework is trained and validated on two complementary sets of publicly available datasets encompassing four lithium-ion chemistries (LCO, NCM, NCA, LFP) and diverse operating conditions (-40°C to 50°C, 0.5C to 8C rates).

**HERO Memory Bank** (retrieval-augmented prediction — 3,979 trajectories, 76 cells):

| Dataset | What It Is | Cells | Why It Matters |
|---------|------------|-------|----------------|
| **NASA Ames** | Various temps & chemistries | 34 | The gold standard for battery research |
| **CALCE** | Maryland's battery tests | 18 | Real manufacturers, real conditions |
| **Oxford** | High-precision tracking | 8 | Extremely clean, controlled data |
| **TJU** | Tongji University | 40 | Cross-chemistry transfer testing |
| **XJTU** | High C-rate stress | 26 | Aggressive driving scenarios |

**Attribution & Advisory Validation** (five chemistry and condition groups — 75 benchmark scenarios):

| Group | Conditions | Scenarios |
|-------|------------|----------|
| NASA Ames | 4–43°C, 0.5–2C | 15 |
| Panasonic EV | US06, HWFET, LA92, UDDS drive cycles | 15 |
| Nature MATR | 1C–8C fast charging | 15 |
| Randomized | 40°C high-temperature stress | 15 |
| HUST LFP | Various LFP cycling protocols | 15 |

> **Note:** The two evaluation suites share no test overlap, ensuring unbiased cross-evaluation.

---

## The Models (Technical)

### HERO: Hybrid Estimation via Retrieval Optimization
- **What**: Retrieval-augmented SOH/RUL prediction with cross-attention
- **Performance**: in-distribution SOH MAE 0.74% (R² = 0.990). Zero-shot cross-chemistry transfer remains hard for all methods evaluated (see Table 2); the memory bank alone does not confer chemistry adaptation.
- **Weights**: `reports/hero_model/hero_model.pt`

### Hybrid PINN: Physics-Informed Neural Network with Expert Priors
- **What**: 5-head network that attributes capacity loss to specific mechanisms
- **Performance**: 96.0% in-distribution (72/75), 89.3% under leave-one-dataset-out. The explicit priors add a modest 2.7 pp over an identical data-driven network (18.7 pp over the boundary-aware variant); their principal value is **guaranteed physical consistency** (0% vs 37.1% impossible storage-plating attributions) and auditable rules, not raw accuracy.
- **Weights**: `reports/pinn_causal/pinn_causal_retrained.pt`

### PATT: Physics-Aware Temporal Transformer
- **What**: Classifies whether the battery is being used or stored
- **Performance**: 99.9% held-out accuracy (99.6% ± 0.6% under a leakage-controlled cell-level split), 99.7% cycling recall
- **Weights**: `reports/patt_classifier/patt_model.pt`

---

## Training Your Own Models

Training requires the raw datasets (~2.7 GB). Download them first:

**📥 [Download all datasets (Google Drive)](https://drive.google.com/file/d/1FMSJ8T4dIHcE_WFxYvfjc6Qr1zJF2Mei/view?usp=sharing)**

```bash
# Unzip into repo root (creates data/ folder)
unzip ML_SOH_datasets.zip

# Train causal attribution
python3 src/train/train_causal.py --epochs 100

# Train domain classifier
python3 train_patt_classifier.py --epochs 50

# Train HERO
python3 src/train/hero_rad_decoupled.py --pretrain_epochs 100 --finetune_epochs 30
```

See **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** for full details including Colab instructions and expected results.

---

## Authors


**Sushanth Chandrashekar**  
Computer Science & Engineering, Bangalore University  

**Sarina Uke**  
Energy Science & Engineering, IIT Delhi  

**Hariprasad Kodamana** (Corresponding)  
Chemical Engineering & AI, IIT Delhi  
📧 hkodamana@iitd.ac.in

**Manojkumar Ramteke** (Corresponding)  
Chemical Engineering, AI & IT, IIT Delhi  
📧 mcramteke@chemical.iitd.ac.in


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions regarding the methodology or codebase, please contact the corresponding authors:
- **Hariprasad Kodamana**: hkodamana@iitd.ac.in
- **Manojkumar Ramteke**: mcramteke@chemical.iitd.ac.in
