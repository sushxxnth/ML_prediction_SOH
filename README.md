# Battery Health Prediction with Causal Attribution

Code repository for the manuscript submitted to *Computers and Chemical Engineering*.

## What This Does

Electric vehicles spend 90%+ of their time parked, yet most battery management systems focus on cycling degradation. This codebase implements a framework that:

- Predicts remaining useful life across different battery chemistries (88% error reduction vs baselines)
- Identifies *which* degradation mechanism is causing capacity loss (92% accuracy)
- Generates actionable recommendations based on the diagnosis

## Repository Structure

```
├── src/
│   ├── models/           # Neural network architectures
│   │   ├── pinn_causal_attribution.py   # Hybrid PINN (92% attribution accuracy)
│   │   ├── rad_model.py                 # HERO prediction engine
│   │   └── physics_aware_transformer.py # PATT domain classifier
│   ├── advisory/         # Recommendation generation
│   │   └── warning_engine.py            # Early warning system
│   └── data/             # Dataset loaders
├── reports/              # Trained models and results
│   ├── pinn_causal/      # Weights for 92% model
│   ├── patt_classifier/  # Domain classification model
│   └── fleet_rad/        # HERO memory bank
├── Casual_Attribution_reports/   # Paper figures and LaTeX
└── data/                 # Processed datasets
```

## Reproducing Key Results

### Causal Attribution (92% Accuracy)

The hybrid PINN model achieves 92-96% accuracy on 75 benchmark scenarios:

```bash
python3 VERIFY_92_ACCURACY.py
```

**Expected output:**
```
RESULTS
======================================================================
  NASA        : 14/15 ( 93.3%)
  Panasonic   : 15/15 (100.0%)
  Nature      : 15/15 (100.0%)
  Randomized  : 14/15 ( 93.3%)
  HUST        : 14/15 ( 93.3%)
----------------------------------------------------------------------
  Overall     : 72/75 ( 96.0%)

✓ SUCCESS: 92% accuracy threshold achieved!
```

### Other Results

Additional training and validation scripts are available in the repository:
- `train_pinn_correct.py` - Retrain the PINN model
- `train_patt_classifier.py` - Train the domain classifier
- `test_unified_validation.py` - Full validation suite

## Dependencies

```bash
pip install torch numpy pandas matplotlib scikit-learn
```

Tested on Python 3.8+ with PyTorch 1.9+.

## Datasets

The memory bank includes trajectories from:
- NASA Ames Prognostics Center
- CALCE (University of Maryland)
- Oxford Battery Degradation Dataset
- TJU (Tongji University)
- XJTU (Xi'an Jiaotong University)

Raw data should be placed in `data/` following the structure in `src/data/`.

## Model Weights

Pre-trained weights are included in `reports/`:
- `pinn_causal/pinn_causal_retrained.pt` - Hybrid PINN (92% accuracy)
- `patt_classifier/patt_model.pt` - Domain classifier
- `fleet_rad/memory_bank.pt` - HERO memory bank

## Paper

**Extending Electric Vehicle Battery Life via Mechanism-Specific Causal Awareness**

Sushanth Chandrashekar¹, Sarina Uke², Manojkumar Ramteke²*

¹ University Visvesvaraya College of Engineering, Bangalore University  
² Indian Institute of Technology Delhi

*Corresponding author: mcramteke@chemical.iitd.ac.in

## License

MIT License. See [LICENSE](LICENSE) for details.
