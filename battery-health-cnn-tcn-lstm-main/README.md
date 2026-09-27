<div align="center">

# 🔋 BATTERY HEALTH PREDICTION

### CNN–TCN–LSTM with Attention for Lithium-Ion SOH Estimation

<sub>Deep learning · NASA Battery Aging Dataset · Cross-Battery Generalization Study</sub>

</div>

<br>

---

<br>

## 📌 Overview

<sub>

A deep learning framework for estimating **State of Health (SOH)** in Lithium-Ion batteries. The central question isn't just *"can we predict SOH?"* — it's *"can a model trained on some batteries generalize to a battery it has never seen?"* That turns out to be the hard part, and this README reports the results honestly, including where the architecture falls short.

</sub>

<br>

| | |
|:--|:--|
| 🧠 **Architecture** | CNN → TCN → LSTM → Attention |
| 🔋 **Dataset** | NASA Li-Ion Battery Aging Dataset |
| 📈 **Features** | dQ/dV, dV/dQ, dI/dV |
| 🔄 **Preprocessing** | Voltage interpolation + Savitzky–Golay smoothing |
| 🎯 **Validation** | Leave-One-Battery-Out cross-validation |
| 🧪 **Comparisons** | LSTM baseline + component ablation study |
| 👀 **Interpretability** | Attention-weight visualization per cycle |

<br>

---

<br>

## 🧩 Pipeline

<br>

```
NASA Battery Aging Data
        │
        ▼
   Data Loading
        │
        ▼
Signal Preprocessing
   ├── Smoothing (Savitzky–Golay)
   ├── Voltage-domain interpolation
   └── Numerical derivatives
        │
        ▼
Electrochemical Features
   ├── dQ/dV
   ├── dV/dQ
   └── dI/dV
        │
        ▼
Sequence Construction (20 consecutive cycles)
        │
        ▼
   CNN → TCN → LSTM → Attention
        │
        ▼
     SOH Prediction
        │
        ▼
 Evaluation & Analysis
```

<br>

---

<br>

## 🔋 Dataset

<sub>

Four NASA battery cells are used, each serving as a cross-validation fold.

</sub>

<br>

| Battery | Role |
|:--:|:--:|
| B0005 | CV fold |
| B0006 | CV fold |
| B0007 | CV fold |
| B0018 | CV fold |

<br>

**Processed tensor shapes**

```
X : (samples, 20, 3, 300)   # 20 cycles × 3 channels × 300 voltage points
y : (samples,)              # SOH labels
```

<br>

| Feature | Description |
|:--|:--|
| `dQ/dV` | Differential capacity w.r.t. voltage |
| `dV/dQ` | Differential voltage w.r.t. capacity |
| `dI/dV` | Differential current w.r.t. voltage |

<br>

---

<br>

## ⚙️ Preprocessing

<sub>

**1. Signal Smoothing** — Savitzky–Golay filtering reduces noise while preserving curve shape.

**2. Voltage-Domain Interpolation** — cycles are resampled onto a common grid (2.5 V → 4.2 V, 300 points) so cycles with different raw sample counts become comparable.

**3. Electrochemical Feature Extraction** — numerical derivatives yield dQ/dV, dV/dQ, and dI/dV.

**4. Temporal Sequence Construction** — 20 consecutive cycles are grouped into one input of shape `20 × 3 × 300`.

</sub>

<br>

---

<br>

## 🧠 Model Architecture

<br>

```
Input
  │
  ▼
CNN                  → local patterns per cycle
  │
  ▼
TCN                  → dilated convolutions (1 → 2 → 4 → 8), multi-scale temporal patterns
  │
  ▼
LSTM                 → sequential dependencies across cycles
  │
  ▼
Additive Attention   → learns which cycles matter, gives interpretable weights
  │
  ▼
Fully Connected Regressor
  │
  ▼
SOH
```

<br>

---

<br>

## 🏋️ Training Configuration

<br>

| Parameter | Value |
|:--|:--:|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss | MSE |
| Batch Size | 64 |
| Max Epochs | 200 |
| Early Stopping Patience | 20 |
| Gradient Clipping | 5.0 |
| Device | CUDA if available, else CPU |

<br>

---

<br>

## 🧪 Results

<br>

### Leave-One-Battery-Out Cross-Validation

<sub>Each experiment trains on three batteries and tests on the fourth, fully unseen, battery.</sub>

<br>

**Full Hybrid Model — CNN–TCN–LSTM–Attention**

| Battery | MAE | RMSE | R² |
|:--:|:--:|:--:|:--:|
| B0005 | 0.097381 | 0.125371 | −0.737823 |
| B0006 | 0.094473 | 0.109599 | −0.089520 |
| B0007 | 0.061850 | 0.079056 | −0.041903 |
| B0018 | 0.092557 | 0.101428 | −1.315784 |
| **Average** | **0.086565** | **0.103863** | **−0.546257** |

<sub>

A negative average R² means the model performs worse than simply predicting the mean SOH for unseen batteries — a real result, not a bug, and it shows the architecture doesn't yet generalize across cells with different degradation behavior.

</sub>

<br>

### LSTM Baseline

<sub>A simpler single-branch LSTM, trained under the identical protocol.</sub>

<br>

| Battery | MAE | RMSE | R² |
|:--:|:--:|:--:|:--:|
| B0005 | 0.080447 | 0.099260 | −0.089329 |
| B0006 | 0.125011 | 0.149667 | −1.031768 |
| B0007 | 0.045598 | 0.058007 | 0.439062 |
| B0018 | 0.082472 | 0.100507 | −1.273923 |
| **Average** | **0.083382** | **0.101860** | **−0.488990** |

<sub>The plain LSTM slightly edges out the full hybrid model on every averaged metric.</sub>

<br>

### Ablation Study

| Model | MAE | RMSE | R² |
|:--|:--:|:--:|:--:|
| CNN Only | 0.076197 | 0.091754 | −0.157223 |
| 🏆 **CNN-LSTM** | **0.054995** | **0.068928** | **0.258955** |
| CNN-TCN-LSTM | 0.083947 | 0.095680 | −0.399187 |
| Full Model (+Attention) | 0.080828 | 0.097948 | −0.519467 |

<sub>

**Best performer: CNN-LSTM**, by a clear margin on every metric. Adding TCN and Attention made results worse under this setup, not better.

</sub>

<br>

---

<br>

## 🔑 Key Findings

<sub>

- **More complexity ≠ better generalization.** The simplest working combination (CNN-LSTM) outperformed the full CNN-TCN-LSTM-Attention stack across the board.
- **Cross-battery generalization is the real bottleneck.** Performance is inconsistent across cells, with several folds producing negative R².
- **Reported as a finding, not hidden.** The value here is in showing *where* the architecture breaks down — which points directly at next steps rather than at scaling the model further.

</sub>

<br>

---

<br>

## 📊 Visualizations

<sub>Generated automatically and saved to `results/plots/`</sub>

<br>

| Category | Plots |
|:--|:--|
| **Prediction Analysis** | Actual vs. Predicted SOH · Prediction Error · Battery-wise Predictions · CV Predictions |
| **Training Analysis** | Train/Val Loss History · CV Error · CV R² |
| **Model Analysis** | Attention Weight Heatmaps · Ablation Comparison · Model Comparison |

<br>

---

<br>

## 📁 Repository Structure

<br>

```
battery-health-cnn-tcn-lstm/
│
├── data/
│   ├── raw/NASA/
│   └── processed/
│
├── results/
│   ├── experiments/
│   ├── final_summary/
│   └── plots/
│
├── scripts/
│   ├── build_dataset.py
│   ├── build_sequences.py
│   ├── train.py
│   ├── evaluate.py
│   ├── cross_validate.py
│   ├── baseline_lstm.py
│   ├── ablation_study.py
│   ├── model_comparison.py
│   ├── attention_plot.py
│   ├── battery_prediction_plots.py
│   ├── cv_prediction_plots.py
│   ├── generate_plots.py
│   └── final_summary.py
│
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── features/
│   ├── models/
│   ├── training/
│   └── visualization/
│
├── requirements.txt
└── README.md
```

<br>

---

<br>

## 🚀 Installation

<sub>

**Clone the repository**

</sub>

```bash
git clone https://github.com/GitankRana/battery-health-cnn-tcn-lstm.git
cd battery-health-cnn-tcn-lstm
```

<sub>

**Create a virtual environment** (Windows PowerShell)

</sub>

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

<sub>

**Install dependencies**

</sub>

```bash
pip install -r requirements.txt
```

<br>

---

<br>

## ▶️ Usage

```bash
# 1. Build the dataset
python scripts/build_dataset.py
python scripts/build_sequences.py

# 2. Train the hybrid model
python scripts/train.py

# 3. Evaluate
python scripts/evaluate.py

# 4. Leave-one-battery-out cross-validation
python scripts/cross_validate.py

# 5. LSTM baseline
python scripts/baseline_lstm.py

# 6. Ablation study
python scripts/ablation_study.py

# 7. Generate all visualizations
python scripts/generate_plots.py
python scripts/attention_plot.py
python scripts/battery_prediction_plots.py
python scripts/cv_prediction_plots.py
python scripts/model_comparison.py
```

<br>

---

<br>

## 🔮 Future Work

<sub>

- Per-battery / global normalization to reduce distribution shift between cells
- Systematic hyperparameter optimization (currently fixed, not tuned)
- Alternative or additional electrochemical features
- Sequence-length experiments — is 20 cycles the right window?
- Stronger regularization (dropout, weight decay)
- Battery-specific degradation curve analysis
- Larger, more diverse validation sets across chemistries
- Additional baselines (Transformers, GRU, pure TCN)

</sub>

<br>

---

<br>

## 📚 Reference

<sub>

**Dataset:** [NASA Prognostics Center of Excellence — Battery Data Set](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

**Focus:** Deep-learning-based Lithium-Ion battery State of Health estimation, with emphasis on honest evaluation of cross-battery generalization.

</sub>

<br>

---

<br>

<div align="center">

## 👨‍💻 Author

**Gitank Rana**
<sub>Electrical Engineering · Netaji Subhas University of Technology (NSUT)</sub>

[GitHub](https://github.com/GitankRana)

<br>

⭐ **If this is useful for your own battery prognostics work, open an issue or discussion.**

</div>
