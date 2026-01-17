# Physics-Informed Battery Health Management System

A machine learning framework for battery State of Health (SOH) and Remaining Useful Life (RUL) prediction with causal mechanism attribution and actionable user recommendations.

## Key Features

| Component | Description | Performance |
|-----------|-------------|-------------|
| **HERO** | Retrieval-augmented prediction engine | 88% RUL error reduction |
| **Causal Attribution** | Physics-constrained mechanism identification | 90.7% accuracy |
| **Advisory System** | Context-aware recommendations | 2.7 recommendations/scenario |
| **Early Warning** | Slope-based degradation detection | 96% recall, 99 cycles lead time |

## Results

- **SOH MAE:** 0.69% (95% CI: [0.63%, 0.75%])
- **RUL MAE:** 16.2 cycles (95% CI: [14.5, 17.9])
- **Zero-shot:** Works on unseen chemistries without retraining
- **Datasets:** NASA, CALCE, Oxford, TJU, XJTU (76 cells, 5 chemistries)

## Project Structure

```
├── Casual_Attribution_reports/   # Paper and figures
│   ├── cce_paper.tex            # Main LaTeX paper
│   └── references.bib           # Bibliography (48 citations)
├── src/
│   ├── models/
│   │   └── causal_attribution.py    # Causal mechanism model
│   ├── train/
│   │   └── hero_rad_decoupled.py    # HERO prediction engine
│   ├── advisory/
│   │   ├── suggestion_generator.py  # Recommendation engine
│   │   └── warning_engine.py        # Early warning system
│   └── context/
│       └── extended_context.py      # Context encoding
├── data/                         # Datasets
├── models/                       # Trained models (.pt)
└── reports/                      # Validation results
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run XJTU causal attribution validation
python3 src/run_xjtu_causal_attribution.py

# Run SOTA baseline comparison
python3 src/sota_baseline_comparison.py

# Run uncertainty quantification
python3 src/uncertainty_quantification.py
```

## Paper

**Title:** Extending Electric Vehicle Battery Life via Mechanism-Specific Causal Awareness

**Authors:** Sushanth Chandrashekar, Sarina Uke, Manojkumar Ramteke

**Target:** Computers and Chemical Engineering

## Author

Sushanth Chandrashekar  
University Visvesvaraya College of Engineering, Bangalore University
