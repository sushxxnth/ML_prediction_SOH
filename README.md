# Battery Health Management with Physics-Informed Causal Attribution

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)

> **Paper**: "Analyzing Degradation and Extending Life of Electric Vehicle Batteries using Physics-Aware Transformers"  


---

## Overview

This repository contains the official implementation of the Physics-Informed Battery Health Management framework. The proposed system extends beyond traditional Remaining Useful Life (RUL) prediction by providing mechanism-specific causal attribution and counterfactual optimization interventions. 

The framework addresses a critical limitation in current Battery Management Systems (BMS): the inability to attribute capacity fade to specific underlying electrochemical degradation mechanisms (e.g., Solid Electrolyte Interphase (SEI) growth, lithium plating, active material loss) using only non-invasive operating data.

---

## Technical Contributions

The framework comprises four integrated modules:

### 1. Predictive Engine: Hybrid Estimation via Retrieval Optimization (HERO)
- A retrieval-augmented RUL prediction architecture utilizing cross-attention over a memory bank of 3,979 degradation trajectories.
- Demonstrates zero-shot generalization to unseen battery chemistries, achieving a 55% reduction in prediction error compared to standard sequential baselines.

### 2. Diagnostic Engine: Hybrid Physics-Informed Neural Network (PINN)
- A multi-head causal attribution network bounded by electrochemical priors (Arrhenius kinetics, Tafel equations).
- Achieves 96.0% accuracy in isolating the dominant degradation mechanism from macroscopic voltage/current/temperature time-series data.

### 3. Proactive Monitoring: Early Warning Engine
- Detects the onset of nonlinear capacity fade and knee-point acceleration.
- Achieves an 88.9% F1 score in predictive failure detection, providing an average lead time of 99 cycles prior to end-of-life.

### 4. Prescriptive Advisory: Counterfactual Optimizer
- Simulates mechanism trajectories under hypothetical operating conditions using a differentiable physics proxy.
- Recommends mathematically optimal, actionable interventions (e.g., specific current reductions, thermal adjustments) to explicitly mitigate the dominant degradation mechanism.

---

## Benchmark Results

| Metric | Result | Significance |
|--------|--------|--------------|
| **Causal Attribution Accuracy** | 96.0% (95% CI: 90.7–100.0%) | Verifiable mechanism diagnosis across 5 chemistry and condition groups. |
| **Zero-Shot RUL (MAE)** | 44.0 cycles | 55% error reduction on unseen NCA chemistry vs. LSTM baselines. |
| **Early Warning Lead Time** | 99 cycles | Enables proactive intervention prior to failure onset. |
| **SOH Prediction (HERO)** | 99.0% R² | Robust trajectory forecasting across chemistries. |
| **Domain Classification (PATT)** | 99.2% | Accurately distinguishes storage (calendar) vs. cycling aging. |

---

## Installation

```bash
# Clone the repository
# Install required dependencies
pip install torch numpy pandas matplotlib scikit-learn scipy
```

---

## Reproducing Paper Results

The repository includes pre-trained weights and validation scripts to reproduce all quantitative claims in the manuscript. Model weights are hosted as a [GitHub Release](https://github.com/sushxxnth/ML_prediction_SOH/releases/tag/v1.0.0) to keep the repository lightweight.

### Setup (one-time, after cloning)

```bash
# Install dependencies
pip install torch numpy pandas matplotlib scikit-learn scipy

# Download pre-trained weights and result files (~1 MB)
python3 download_weights.py
```

This installs the following into `reports/` (gitignored, local only):
- `pinn_causal_retrained.pt` — Hybrid PINN (96.0% causal accuracy)
- `patt_best.pt` — PATT domain classifier (99.2% accuracy)
- `hero_model.pt` — HERO prediction model (SOH R²=99%, RUL MAE=44 cycles)
- Verification result JSON files

### Verify all paper claims

```bash
python3 REPRODUCE_PAPER_CLAIMS.py
```

### Specific Verifications

**1. Causal Attribution Accuracy (96.0%)**
```bash
python3 VERIFY_96_ACCURACY.py
```

**2. Zero-Shot Prediction (HERO, 44-cycle MAE)**
```bash
python3 verify_hero_zeroshot.py
```

**3. Counterfactual Ground-Truth Validation**
```bash
python3 validate_counterfactual_ground_truth.py
```

---

## Datasets

The models were trained and validated on a comprehensive aggregation of publicly available datasets encompassing four lithium-ion chemistries (LCO, NCM, NCA, LFP) and diverse operating conditions (-40°C to 50°C, 0.5C to 8C rates):

1. **NASA Ames Prognostics Data Repository** (34 cells)
2. **CALCE Battery Research Group** (18 cells)
3. **Oxford Battery Degradation Dataset** (8 cells)
4. **TJU (Tongji University)** (40 cells)
5. **XJTU Battery Dataset** (26 cells)
6. **Stanford Calendar Aging Dataset** (60 cells)

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
├── REPRODUCE_PAPER_CLAIMS.py              # Main reproducibility script for reviewers
└── VERIFY_96_ACCURACY.py                  # PINN evaluation script
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
- **What**: Retrieval-augmented RUL prediction with cross-attention
- **Performance**: 99% R² on SOH, 55% better than baselines on new chemistries
- **Weights**: `reports/hero_model/hero_model.pt`

### Hybrid PINN: Physics-Informed Neural Network with Expert Priors
- **What**: 5-head network that attributes capacity loss to specific mechanisms
- **Performance**: 96% accuracy (bootstrap 95% CI: 90.7–100.0%; expert priors add **18.7 percentage points** over boundary-aware baseline!)
- **Weights**: `reports/pinn_causal/pinn_causal_retrained.pt`

### PATT: Physics-Aware Temporal Transformer
- **What**: Classifies whether the battery is being used or stored
- **Performance**: 99.2% accuracy, 100% recall on cycling (never misses active use)
- **Weights**: `reports/patt_classifier/patt_model.pt`

---

## Training Your Own Models

If you want to retrain from scratch (not needed for verification):

```bash
# Train causal attribution
python3 src/train/train_causal.py

# Train domain classifier
python3 train_patt_classifier.py

# Train HERO
python3 src/train/hero_rad_decoupled.py

# Full evaluation suite
python3 test_unified_validation.py
```

**Note**: Training requires the full datasets. See **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** for detailed instructions including dataset download links, expected directory structure, and training times.

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
