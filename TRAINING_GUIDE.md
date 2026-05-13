# Training from Scratch — Complete Guide

## Overview

There are **3 models** in this repository. You can either:

1. **Verify paper claims** using pre-trained weights (fast, ~2 min)
2. **Retrain from scratch** using the raw datasets (~1–2 hours)

Both paths require downloading files first. Follow the steps below.

---

## Quick Start (Google Colab / Any Linux Machine)

Copy and paste this entire block into a Colab cell or terminal:

```python
# Step 1: Clone the repository
!git clone https://github.com/sushxxnth/ML_prediction_SOH.git
%cd ML_prediction_SOH

# Step 2: Install dependencies
!pip install torch numpy pandas matplotlib scikit-learn scipy openpyxl

# Step 3: Download datasets (2.7 GB zip from Google Drive)
!pip install gdown
!gdown 1FMSJ8T4dIHcE_WFxYvfjc6Qr1zJF2Mei
!unzip -q ML_SOH_datasets.zip
!rm ML_SOH_datasets.zip

# Step 4: Download pre-trained weights (needed for verification)
!python3 download_weights.py

# Step 5: Verify all paper claims using pre-trained weights
!python3 REPRODUCE_PAPER_CLAIMS.py
```

**Expected output:** `7/7 claims verified`

---

## To Retrain Models from Scratch

After completing Steps 1–3 above (clone, install, download data), run the training scripts:

```python
# Train Model 1: Causal Attribution PINN (~10–30 min on CPU)
!python3 src/train/train_causal.py --epochs 100

# Train Model 2: HERO RUL Prediction (~30–60 min on CPU)
!python3 src/train/hero_rad_decoupled.py --pretrain_epochs 100 --finetune_epochs 30

# Train Model 3: PATT Domain Classifier (~5–15 min on CPU)
!python3 train_patt_classifier.py --epochs 50
```

### What to Expect from Retraining

| Model | Paper Result | What You'll Likely Get |
|-------|-------------|----------------------|
| Causal PINN | 96.0% | 90–96% (depends on random seed) |
| HERO RUL MAE | 44 cycles | 40–55 cycles |
| PATT | 99.2% | 97–99%+ |

> **Note:** Exact numbers vary slightly across runs due to random initialization. The paper results were the best of several training runs.

---

## Dataset Details

All 6 datasets are bundled in the zip file above. For reference, here are the original sources:

| Dataset | Cells | Original Source |
|---------|-------|----------------|
| NASA Ames | 34 | https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/ |
| CALCE | 18 | https://calce.umd.edu/battery-data |
| Oxford | 8 | https://batteryintelligence.web.ox.ac.uk/data-and-code |
| TBSI Sunwoda | — | https://github.com/terencetaothucb/TBSI-Sunwoda-Battery-Dataset |
| Stanford Calendar Aging | 259 | https://web.stanford.edu/group/chuehgroup/datasets.html |
| XJTU | 26 | https://github.com/Ruifeng-Tan/BatteryLife |
| TJU | 40 | https://zenodo.org/records/6405084 |

### Expected Directory Structure

After unzipping, the `data/` folder should look like:

```
data/
├── nasa_set5/raw/       (34 .mat files: B0005.mat, B0006.mat, ...)
├── calce/               (CS2_data/, CX2_data/)
├── tbsi_sunwoda/        (TBSI-Sunwoda-Battery-Dataset-main/Labels.xlsx)
├── stanford_calendar/   (259 JSON files + stanford_sampled_diagnostic.csv)
├── new_datasets/
│   ├── XJTU/Battery Dataset/  (Batch-1/, Batch-2/, Batch-3/)
│   └── RUL-Mamba/data/TJU data/
└── unified_cache/       (pre-processed JSON caches)
```

---

## Key Architecture Decisions

- All models share the `UnifiedDataPipeline` (`src/data/unified_pipeline.py`) which handles loading, feature extraction, and context encoding
- HERO uses domain-adversarial training + memory bank retrieval — training is the most complex
- PATT falls back to synthetic data generation if real data is not found, but accuracy will be lower
