# BMS-EMS AI Health Integration Project

**Research Question:** How does uncertainty in AI-estimated battery State of Health (SOH) propagate through a health-aware Energy Management Strategy (EMS) and affect vehicle energy consumption, battery degradation, motor degradation, and driving performance?

---

## Repository Structure

This is a unified monorepo containing both the Battery Management System (BMS) and Energy Management Strategy (EMS) research projects:

```
non2/
├── bms_gtr/              # BMS AI for battery SOH estimation
│   ├── src/              # CNN-TCN-LSTM-Attention model
│   ├── scripts/          # Training and evaluation scripts
│   ├── data/             # Processed battery data
│   ├── results/          # Trained model checkpoints
│   ├── requirements.txt
│   └── README.md
│
├── ems_yash/             # EMS for dual-motor BEV powertrain
│   ├── src/              # Powertrain models, BMS interface, agents
│   ├── data/             # BMS SOH trace, motor efficiency maps, driving cycles
│   ├── results/          # Phase 3 experimental results
│   ├── docs/             # Phase 3 documentation
│   ├── requirements.txt
│   └── README.md
│
├── README.md             # This file
├── .gitignore
└── setup.ps1             # Optional Windows setup script
```

---

## Overview

### BMS Project (bms_gtr/)

The BMS project implements a deep learning model for battery State of Health (SOH) estimation:

**Architecture:**
- CNN → TCN → LSTM → Multi-Head Attention
- Input: Voltage, current, temperature time series
- Output: Battery SOH estimate (0-1 range)

**Training Data:**
- NASA battery aging dataset
- Multiple battery types under various operating conditions

**Performance:**
- Test MAE: 0.0195
- Test RMSE: 0.0244

**Key Files:**
- `src/model.py` - Neural network architecture
- `scripts/train.py` - Training script
- `scripts/test.py` - Evaluation script
- `results/best_model.pt` - Trained model checkpoint (620 KB)

### EMS Project (ems_yash/)

The EMS project implements a health-aware energy management strategy for a dual-motor Battery Electric Vehicle (BEV):

**Architecture:**
- Dual-motor powertrain (front/rear axles)
- Battery equivalent circuit model
- Motor efficiency maps (digitized from Wu et al. 2024)
- Health degradation models (battery and motors)
- BMS-EMS health interface (Phase 3)

**Key Components:**
- `src/environment/` - Powertrain simulation
- `src/agent/` - Health-aware deterministic policy
- `src/bms_interface/` - BMS-EMS integration (Phase 3)
- `data/bms_soh_trace.npz` - Offline BMS predictions (8 KB)

---

## Phase 3: BMS-EMS Integration

Phase 3 integrates AI-estimated battery SOH from the BMS with the EMS controller to study the impact of SOH estimation errors on vehicle behavior.

### Phase 3A: Interface Layer

**Implemented:**
- `BMSHealthInterface` - Routes either true physical SOH or AI-estimated SOH to EMS
- `SOHSource` enum - TRUE_PHYSICAL vs BMS_ESTIMATED modes
- Health state tracking at each simulation timestep
- 6-dimensional RL state space: [velocity, wheel_torque, SOC, battery_SOH, motor1_SOH, motor2_SOH]

**Architecture:**
```
Physical Degradation Model (ground truth)
        ↓
BMSHealthInterface
        ↓
    TRUE_PHYSICAL mode → uses true physical SOH
    BMS_ESTIMATED mode → uses AI-estimated SOH
        ↓
EMS Controller
```

**Validation:**
- ✅ Interface returns identical SOH to physical model in TRUE_PHYSICAL mode
- ✅ Physical SOH evolution remains independent of EMS-visible SOH
- ✅ Motor SOH always uses physical degradation (no AI estimation)
- ✅ RL environment dimensions preserved

### Phase 3B: AI SOH Estimation Integration

**Implemented:**
- `SOHTraceAdapter` - Provides time-aligned AI SOH estimates from offline BMS predictions
- `HealthAwareDeterministicPolicy` - Rule-based EMS that responds to battery SOH
- Offline BMS trace: 556 samples over 11,100 simulated hours
- TRUE_PHYSICAL vs BMS_ESTIMATED experimental comparison

**Key Results:**
- **SOH Estimation Error:** MAE = 4.67%, RMSE = 4.67%
- **EMS Decision Impact:** +2.8% change in torque allocation (σ_tor)
- **Battery Degradation:** Negligible difference (2.4×10⁻⁸)
- **Motor Degradation:** ±1-2% reflecting torque allocation shift
- **Performance Impact:** -0.075% reward decrease

**Key Findings:**
1. AI-estimated battery SOH successfully reaches EMS through the interface
2. 4.67% SOH estimation error causes measurable 2.8% EMS decision change
3. Physical battery SOH evolution remains independent (correct architecture)
4. System-level impact is modest - EMS functions with imperfect SOH information

**Motor Efficiency Map Fix:**
- Issue: Interpolation failures at operating points outside digitized contour data
- Solution: Nearest-neighbor extrapolation fallback for boundary points
- Result: All typical driving scenarios now execute without failures

---

## Important Limitations

### Current State

1. **No Trained DDPG Controller:** The original Wu et al. (2024) paper used a trained DDPG-GRU-Self-Attention controller. No trained checkpoint is currently available. Phase 3B therefore uses a deterministic health-aware policy as a controlled experimental policy.

2. **Offline BMS Trace:** The BMS SOH estimates are pre-computed offline (sample-and-hold mechanism) rather than real-time neural network inference during simulation. This is sufficient to validate the BMS-EMS integration architecture.

3. **Short Episodes:** Phase 3B experiments use 100-step episodes (100 seconds) which show minimal battery/motor degradation. Longer episodes would amplify the observed effects.

4. **Steady-State Driving:** The driving cycle uses simplified vehicle dynamics (target_velocity = current_velocity) for reproducibility.

5. **Motor Efficiency Extrapolation:** The motor efficiency maps use nearest-neighbor extrapolation at operating boundaries beyond the digitized data region.

### Relationship to Base Paper

**Base Paper:** Wu et al. (2024), "Energy Management Strategy for Battery Electric Vehicles Considering Battery Degradation Using DDPG with GRU and Self-Attention"

**What This Project Does:**
- Validates the **novel Phase 3 contribution**: Integration of BMS AI battery health estimation with EMS controller
- The base paper assumes perfect SOH knowledge
- Phase 3 extends this by adding realistic BMS estimation error and demonstrating its impact

**What This Project Does NOT Do:**
- Reproduce the trained DDPG controller (no checkpoint available)
- Match the paper's quantitative performance metrics
- Claim equivalence to original experimental results

**Scientific Contribution:** This is an **architectural validation** of the BMS-EMS integration layer, demonstrating that SOH estimation accuracy matters for EMS decision quality.

---

## Installation

### Prerequisites

- Python 3.10+ (tested with Python 3.14)
- Windows 10/11 (Linux/Mac may work but not tested)
- ~2 GB disk space

### Option 1: Separate Environments (Recommended for Compatibility)

If the BMS and EMS have conflicting dependencies, use separate virtual environments:

**BMS Environment:**
```powershell
cd bms_gtr
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**EMS Environment:**
```powershell
cd ems_yash
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Option 2: Unified Environment

If dependencies are compatible (verify first):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r bms_gtr/requirements.txt
pip install -r ems_yash/requirements.txt
```

---

## Usage

### BMS: Train and Evaluate

**Training:**
```powershell
cd bms_gtr
python scripts/train.py
```

**Evaluation:**
```powershell
python scripts/test.py
```

**Inference:**
```python
from src.model import BatterySOHModel
import torch

model = BatterySOHModel()
model.load_state_dict(torch.load('results/best_model.pt'))
model.eval()

# Use model for inference
```

### EMS: Run Baseline and Phase 3 Experiments

**Phase 3A: Interface Validation**
```powershell
cd ems_yash
python test_phase3a_rl_env.py
```

Expected output: All tests pass, confirming BMS interface integration.

**Phase 3B: Integration Validation**
```powershell
python validate_phase3b_integration.py
```

Expected output: Validates TRUE_PHYSICAL and BMS_ESTIMATED modes work correctly.

**Phase 3B: Full Experiment**
```powershell
python run_phase3b_experiment.py
```

Expected output: Complete TRUE vs AI comparison showing SOH estimation impact on EMS decisions.

**Motor Efficiency Map Diagnostic**
```powershell
python diagnose_motor_map_issue.py
```

Expected output: Validates motor efficiency interpolation across operating range.

---

## Results

Results are saved in:
- `ems_yash/results/` - CSV files with trajectory data
- `ems_yash/docs/` - Phase 3 documentation
- `ems_yash/phase3b_experiment_output.txt` - Latest Phase 3B results
- `ems_yash/PHASE3B_COMPLETION_SUMMARY.md` - Phase 3B summary

**Phase 3B Key Results:**

| Metric | TRUE Mode | AI Mode | Difference |
|--------|-----------|---------|------------|
| Mean EMS SOH | 1.0000 | 0.9533 | -4.67% |
| Mean σ_tor | 0.5000 | 0.5140 | +2.80% |
| Battery degradation | 0.000014342 | 0.000014365 | +0.016% |
| Total reward | 9528.557 | 9527.838 | -0.075% |

---

## Testing

**BMS Tests:**
```powershell
cd bms_gtr
python scripts/test.py
```

**EMS Tests:**
```powershell
cd ems_yash

# Phase 3A tests
python test_phase3a_rl_env.py

# Phase 3B integration tests  
python validate_phase3b_integration.py

# Motor efficiency map tests
python diagnose_motor_map_issue.py
```

---

## Project History

**Phase 1:** BMS neural network development (CNN-TCN-LSTM-Attention)  
**Phase 2:** EMS powertrain simulation and health-aware control  
**Phase 3A:** BMS-EMS interface layer  
**Phase 3B:** AI SOH estimation integration and experimental validation  

**Current Status:** Phase 3B complete and validated. Motor efficiency map issue resolved. Both TRUE_PHYSICAL and BMS_ESTIMATED experiments execute successfully.

---

## Citation

If you use this code, please cite the base paper:

```
Wu et al. (2024), "Energy Management Strategy for Battery Electric Vehicles 
Considering Battery Degradation Using DDPG with GRU and Self-Attention"
```

And acknowledge this Phase 3 integration work.

---

## License

Research project. Check with project owners before redistribution.

---

## Contact

For questions about:
- **BMS:** Check `bms_gtr/README.md`
- **EMS:** Check `ems_yash/README.md`
- **Phase 3 Integration:** See `ems_yash/docs/` and `PHASE3B_COMPLETION_SUMMARY.md`

---

## Reproducibility

**Random Seed:** 42 (where applicable)  
**Deterministic:** Yes (Phase 3B experiments are deterministic)  
**Expected Runtime:** 
- Phase 3A tests: ~5 seconds
- Phase 3B validation: ~5 seconds  
- Phase 3B experiment: ~10 seconds

**Verification:**
All Phase 3 results should match the reported values within numerical precision.

---

**Repository Version:** 1.0  
**Last Updated:** 2026-09-25  
**Status:** Phase 3B complete and validated
