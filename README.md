# Battery Health Management with Physics-Informed Causal Attribution

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)

> **Paper**: "Extending Electric Vehicle Battery Life via Mechanism-Specific Causal Awareness"  
> **Journal**: Computers and Chemical Engineering (Submitted)

---

## Why This Matters

Here's the thing about batteries: most systems can tell you *when* they'll fail, but not *why*. If your EV battery is degrading faster than expected, is it because you're charging in the cold? Leaving it at 100% for too long? Driving aggressively? 

This framework answers that question. It doesn't just predict remaining life—it diagnoses the specific electrochemical mechanism causing the problem and suggests how to fix it.

Think of it as moving from "Check Engine Light" to "Your catalytic converter is failing due to lean fuel mixture - here's how to address it."

---

## What We Built

We created a complete battery health management system with four key capabilities:

### 1. Predictive (HERO Model)
- Predicts how much life your battery has left
- Works across different battery chemistries without retraining (zero-shot)
- **55% more accurate** than standard deep learning methods on new battery types
- *"I've never seen this exact battery before, but based on 3,979 similar cases, here's what's going to happen..."*

### 2. � Diagnostic (Hybrid PINN)
- Identifies *which* degradation mechanism is the culprit
- 92% accurate at pinpointing whether it's lithium plating, SEI growth, active material loss, etc.
- Combines physics equations with expert knowledge
- *"Your capacity drop isn't normal aging—it's 70% lithium plating from cold-weather charging."*

### 3. ⚡ Proactive (Early Warning)
- Detects problems **99 cycles before failure** (that's 2-3 months for a typical EV)
- 88.9% F1 score—catches 96% of failures with few false alarms
- *"Your degradation just accelerated. We need to investigate."*

### 4. 💡 Actionable (Advisory System)
- Generates specific, physics-based recommendations
- "Reduce charge rate to 1.5A" beats "charge slower" every time
- Tested on real NASA and XJTU battery data
- *"Here's exactly what to change, and here's how much it will help."*

---

## Quick Results

| What We Measured | Result | What It Means |
|------------------|--------|---------------|
| Causal Diagnosis | 92% accuracy | 9 out of 10 times, we correctly identify why your battery is degrading |
| Zero-Shot Learning | 55% error reduction | Works on battery types we've never seen before |
| Early Warning | 99 cycles ahead | 2-3 months notice before your battery hits end-of-life |
| HERO Prediction | 99% R² | Near-perfect SOH predictions across chemistries |
| Domain Classification | 99.6% accuracy | Almost never confuses cycling with storage |

---

## Try It Yourself

### Installation (2 minutes)

```bash
# Clone and navigate
git clone https://github.com/yourusername/battery-health-causal.git
cd battery-health-causal/ML_prediction_SOH

# Install (just standard PyTorch + numpy stack)
pip install torch numpy pandas matplotlib scikit-learn

# Test installation
python3 VERIFY_92_ACCURACY.py
```

###Reproducing Paper Results

#### Verify 92% Causal Accuracy

```bash
python3 VERIFY_92_ACCURACY.py
```

This runs our trained model on 75 test scenarios across 5 datasets. You should see:
```
══════════════════════════════════════════════════════════════════════
RESULTS
══════════════════════════════════════════════════════════════════════
  NASA        : 14/15 ( 93.3%)
  Panasonic   : 15/15 (100.0%)
  Nature      : 15/15 (100.0%)
  Randomized  : 14/15 ( 93.3%)
  HUST        : 14/15 ( 93.3%)
──────────────────────────────────────────────────────────────────────
  Overall     : 69/75 ( 92.0%)
```

#### Verify 44 Cycle Zero-Shot Accuracy

```bash
python3 verify_hero_zeroshot.py
```

This shows HERO achieving 44.0 cycle RUL MAE on NCA chemistry it's never seen before (55% better than LSTM baseline).

#### All Claims at Once

```bash
python3 verify_all_paper_claims.py
```

Validates every number in the paper. Takes ~30 seconds.

---

## How It Works (Non-Technical)

**Problem**: Battery management systems track capacity (SOH) but not *why* it's dropping.

**Our Solution**: Three neural networks working together:

1. **HERO** looks through a library of 3,979 battery life stories and finds similar cases. "Your battery looks like these 5—here's what happened to them."

2. **Hybrid PINN** applies physics equations to figure out the mechanism. "Based on your temperature, charging rate, and SOH, this is lithium plating."

3. **PATT** knows whether you're driving or parked, because recommendations differ drastically. "You're in storage mode—optimize for calendar aging, not cycling."

Then the system combines everything: *"In 99 cycles you'll hit 80% capacity. The main cause is SEI growth (60%) and plating (25%). Park at 50% SOC instead of 90% to cut SEI growth by 40%."*

---

## Project Structure

```
ML_prediction_SOH/
├── verify_hero_zeroshot.py        #  Reproduces 44 MAE result  
├── VERIFY_92_ACCURACY.py           #  Reproduces 92% causal accuracy
├── verify_all_paper_claims.py     #  Validates all paper claims
├── src/
│   ├── models/
│   │   ├── pinn_causal_attribution.py   # Physics-informed diagnosis (92%)
│   │   ├── rad_model.py                  # HERO prediction engine
│   │   └── physics_aware_transformer.py  # PATT classifier (99.6%)
│   ├── optimization/
│   │   └── counterfactual_intervention.py  # "What if" simulator
│   ├── advisory/
│   │   └── warning_engine.py             # Early warning system
│   └── data/
│       └── unified_pipeline.py           # Data loaders
└── reports/                         # Pre-trained weights & results
    ├── pinn_causal/                # Causal model weights
    ├── hero_model/                 # HERO weights
    ├── patt_classifier/            # Domain classifier
    └── zeroshot_baseline_comparison.json  # HERO 44 MAE results
```

---

## The Data

We didn't just test on one dataset. We used 5 independent sources with different chemistries, temperatures, and driving patterns:

| Dataset | What It Is | Cells | Why It Matters |
|---------|------------|-------|----------------|
| **NASA Ames** | Various temps & chemistries | 34 | The gold standard for battery research |
| **CALCE** | Maryland's battery tests | 18 | Real manufacturers, real conditions |
| **Oxford** | High-precision tracking | 8 | Extremely clean, controlled data |
| **TJU** | Tongji University | 40 | Zero-shot transfer testing |
| **XJTU** | High C-rate stress | 26 | Aggressive driving scenarios |

**Total**: 3,979 degradation trajectories from 76 cells. If it works here, it works in the real world.

---

## The Models (Technical)

### HERO: Hybrid Estimation via Retrieval Optimization
- **What**: Retrieval-augmented RUL prediction with cross-attention
- **Why it's cool**: First battery model with retrieval—learns from every battery it's ever seen
- **Performance**: 99% R² on SOH, 55% better than baselines on new chemistries
- **Weights**: `reports/fleet_rad/rad_finetuned_best.pt`

### Hybrid PINN: Physics-Informed Neural Network with Expert Priors
- **What**: 5-head network that attributes capacity loss to specific mechanisms
- **Why it's cool**: Combines physics equations with domain expert knowledge
- **Performance**: 92% accuracy (expert priors add 14.7 percentage points!)
- **Weights**: `reports/pinn_causal/pinn_causal_retrained.pt`

### PATT: Physics-Aware Temporal Transformer
- **What**: Classifies whether the battery is being used or stored
- **Why it's cool**: Learns Arrhenius kinetics and diffusion—physics embeddings!
- **Performance**: 99.6% accuracy, 100% recall on cycling (never misses active use)
- **Weights**: `reports/patt_classifier/patt_model.pt`

---

## Training Your Own Models

If you want to retrain from scratch (not needed for verification):

```bash
# Train causal attribution
python3 src/train/train_causal.py

# Train domain classifier
python3 src/train/train_patt_classifier.py

# Train HERO
python3 src/train/hero_rad_decoupled.py

# Full evaluation suite
python3 test_unified_validation.py
```

**Note**: Training requires the full datasets. Results in the paper used the pre-trained weights included in `reports/`.

---

## Authors

This is a collaboration between Bangalore University and IIT Delhi:

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

---

## License

MIT License. Use it, modify it, ships products with it—we just ask for attribution.

---

## Questions?

- **"Does this work on my specific battery?"** Probably! We tested on 5 different chemistries. If it's a lithium-ion battery, the physics is the same.
- **"Can I deploy this in production?"** The models are production-ready. You'd need to integrate with your BMS, but the prediction and diagnosis logic is solid.
- **"What if I want to add a new degradation mechanism?"** Add it to the PINN architecture in `src/models/pinn_causal_attribution.py`. The framework is extensible.

Reach out to the corresponding authors for collaborations or questions!

---

**⭐ If you find this useful, star the repo so others can find it too!**
