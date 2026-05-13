# Training from Scratch — Complete Guide

## Overview

Retraining all models end-to-end. There are **3 models** to train, each with its own script. The main blocker is **downloading the raw datasets** (not included in the git repo due to size).

---

## Step 1: Download Datasets

All datasets are publicly available. Create a `data/` directory and populate it:

```
ML_prediction_SOH/
└── data/
    ├── nasa_set5/raw/           ← NASA Ames (34 cells, LCO)
    ├── calce/                   ← CALCE (18 cells, various)
    ├── oxford/                  ← Oxford (8 cells)
    ├── tbsi_sunwoda/            ← TBSI Sunwoda (fast charging)
    ├── stanford_calendar/       ← Stanford Calendar Aging (storage/PATT)
    │   └── stanford_sampled_diagnostic.csv
    └── new_datasets/
        ├── XJTU/Battery Dataset/ ← XJTU (26 cells)
        └── RUL-Mamba/data/TJU data/ ← TJU (40 cells)
```

### Download Links

| Dataset | Download URL | Used By | Notes |
|---------|-------------|---------|-------|
| NASA Ames (34 cells) | https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/ | All 3 models | Download "Battery Data Set", extract `.mat` files into `data/nasa_set5/raw/` |
| CALCE (18 cells) | https://calce.umd.edu/battery-data | Causal, HERO | Download CS2 and CX2 datasets, place in `data/calce/` |
| Oxford (8 cells) | https://batteryintelligence.web.ox.ac.uk/data-and-code | Causal, HERO | Also on Kaggle: search "Oxford Battery Degradation Dataset 1" |
| TBSI Sunwoda | https://github.com/terencetaothucb/TBSI-Sunwoda-Battery-Dataset | HERO | `git clone` the repo into `data/tbsi_sunwoda/` |
| Stanford Calendar Aging (60 cells) | https://web.stanford.edu/group/chuehgroup/datasets.html | PATT | See also OSF: https://osf.io/ (search "decade calendar aging"). Place CSV in `data/stanford_calendar/` |
| XJTU (26 cells) | https://github.com/Ruifeng-Tan/BatteryLife | PATT | Clone and place battery data in `data/new_datasets/XJTU/Battery Dataset/` |
| TJU (40 cells) | https://zenodo.org/records/6405084 | Zero-shot eval | Download NCM+NCA dataset, place in `data/new_datasets/RUL-Mamba/data/TJU data/` |

> [!IMPORTANT]
> **NASA** is the most critical dataset — all 3 models use it. Start there. The pipeline will generate synthetic fallback data for missing datasets, but results will be worse.
>
> **If NASA's official page is down**, search Kaggle for "NASA battery dataset B0005 B0006 B0007" to find community mirrors of the same `.mat` files.

---

## Step 2: Train the 3 Models

### Model 1: Causal Attribution PINN (96% accuracy)

```bash
python3 src/train/train_causal.py --epochs 100
```

- **Loads**: NASA, CALCE, Oxford, Stanford storage data
- **Outputs**: `reports/causal_attribution/causal_model.pt`
- **Time**: ~10–30 min on CPU

### Model 2: HERO Prediction (RUL MAE 44 cycles)

```bash
python3 src/train/hero_rad_decoupled.py --pretrain_epochs 100 --finetune_epochs 30
```

- **Loads**: NASA, CALCE, Oxford (source) → TBSI Sunwoda (target for fine-tuning)
- **Outputs**: `reports/hero_model/hero_model.pt`
- **Time**: ~30–60 min on CPU

### Model 3: PATT Domain Classifier (99.2% accuracy)

```bash
python3 train_patt_classifier.py --epochs 50
```

- **Loads**: NASA, CALCE, Oxford, XJTU (cycling) + Stanford Calendar (storage)
- **Outputs**: `reports/patt_classifier/patt_best.pt`
- **Time**: ~5–15 min on CPU

---

## Step 3: Verify Trained Models

After training, run the verification suite:

```bash
python3 REPRODUCE_PAPER_CLAIMS.py
```

Or individual checks:

```bash
python3 VERIFY_96_ACCURACY.py  # Causal attribution test
python3 verify_hero_zeroshot.py  # HERO zero-shot test
python3 verify_patt_performance.py  # PATT test
```

---

## What to Expect

| Model | Paper Result | What You'll Likely Get |
|-------|-------------|----------------------|
| Causal PINN | 96.0% | 90–96% (depends on random seed) |
| HERO RUL MAE | 44 cycles | 40–55 cycles |
| PATT | 99.2% | 97–99%+ |

> [!NOTE]
> Exact numbers will vary slightly across runs due to random initialization. The paper results were the best of several training runs.

---

## Quick Start on Google Colab

If the one wants to avoid local setup, this Colab cell does everything:

```python
!git clone https://github.com/sushxxnth/ML_prediction_SOH.git
%cd ML_prediction_SOH
!mkdir -p data

# Download NASA dataset (example — PI needs to upload other datasets)
# Upload datasets to data/ folder using Colab file browser

# Train all 3 models
!python3 src/train/train_causal.py --epochs 100
!python3 src/train/hero_rad_decoupled.py --pretrain_epochs 100
!python3 train_patt_classifier.py --epochs 50

# Verify
!python3 REPRODUCE_PAPER_CLAIMS.py
```

---

## Key Architecture Decisions

- All models share the `UnifiedDataPipeline` (`src/data/unified_pipeline.py`) which handles loading, feature extraction, and context encoding
- Pipeline will **auto-generate synthetic fallback data** if a dataset is missing, but accuracy will be lower
- HERO uses domain-adversarial training + memory bank retrieval — training is the most complex
- PATT falls back to synthetic data generation if Stanford Calendar CSV is not found
