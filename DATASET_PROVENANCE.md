# Dataset Provenance and Processing Record

## 1. Purpose

This document records the sources, processing procedures, conventions,
and reconstruction methodology used to prepare the driving-cycle datasets
for the reinforcement-learning energy-management study.

All datasets are represented using the canonical format:

    time_s,velocity_kmh,slope_rad

All standard cycles use a 1 Hz sampling interval.

---

## 2. WLTC Class 3b

Source:
JRC / WLTP cycle data repository.

Source file:
V_class3b.txt

Processing:
- Parsed the official velocity trace.
- Retained the original 1 Hz sampling.
- Converted velocity to km/h representation.
- Added slope_rad = 0.

Output:

data/raw/WLTC_Class3b_raw.csv

Samples:
1801

---

## 3. NEDC

Source:
JRC / WLTP cycle data repository.

Source file:
V_nedc.txt

Processing:
- Parsed the NEDC velocity trace.
- Retained the 1 Hz active trace.
- Added slope_rad = 0.

Output:

data/raw/NEDC_raw.csv

Samples:
1180

Note:
For the mixed training cycle, a 20-second zero-speed interval is inserted
between WLTC and the active NEDC trace according to the adopted project
convention.

---

## 4. Mixed Training Cycle

Composition:

WLTC Class 3b + 20-second idle interval + NEDC

Output:

data/processed/Mixed_Training_Cycle.csv

Samples:

3001

Sampling:

1 Hz

---

## 5. UDDS

Source:
EPA dynamometer driving schedule data.

Processing:
- Parsed the time/speed trace.
- Converted mph to km/h.
- Retained 1 Hz sampling.
- Added slope_rad = 0.

Output:

data/raw/UDDS_raw.csv

Samples:

1370

---

## 6. HWFET

Source:
EPA Highway Fuel Economy Test cycle data.

Processing:
- Parsed the time/speed trace.
- Converted mph to km/h.
- Retained 1 Hz sampling.
- Added slope_rad = 0.

Output:

data/raw/HWFET_raw.csv

Samples:

766

---

## 7. CLTC-P

Source:
China Light-Duty Vehicle Test Cycle (CLTC-P).

Machine-readable source:
MOTIVES-LAB BEV energy-consumption estimator cycle dataset.

Processing:
- Parsed the 1800-point CLTC-P velocity trace.
- Converted the source time indexing to project sample indexing.
- Retained 1 Hz sampling.
- Added slope_rad = 0.

Output:

data/raw/CLTC_P_raw.csv

Samples:

1800

---

## 8. FTP75

Source:
EPA Federal Test Procedure driving cycle.

Processing:
- Parsed the time/speed trace.
- Converted mph to km/h.
- Retained 1 Hz sampling.
- Added slope_rad = 0.

Output:

data/raw/FTP75_raw.csv

Samples:

1875

---

## 9. Mixed Test Cycle

Composition:

UDDS + HWFET + CLTC-P + FTP75

Processing:
- Concatenated the four standard cycles.
- Removed duplicate boundary samples where required.
- Reassigned global time sequentially.
- Maintained 1 Hz sampling.

Output:

data/processed/Mixed_Test_Cycle.csv

Samples:

5808

---

## 10. Actual Driving Cycle

### Original provenance

The actual test cycle corresponds to Fig. 4(c) of:

Wu et al. (2024),
"Health-awareness energy management strategy for battery electric
vehicles based on self-attention deep reinforcement learning."

The paper states that the traffic cycle was constructed from real data
and derived from Ref. [40].

### Reconstruction

The original numerical time-series velocity data were not available
for direct use.

Therefore, the published Fig. 4(c) curve was digitized from the
published figure.

Processing pipeline:

Published Fig. 4(c)
        |
        v
Image-based curve extraction
        |
        v
Pixel coordinates converted to plot coordinates
        |
        v
Time-sorted velocity points
        |
        v
1 Hz linear interpolation
        |
        v
Actual_Driving_Cycle_raw.csv

### Classification

This dataset is explicitly classified as:

**Figure-derived reconstruction**

It must not be represented as the original raw sensor dataset.

### Endpoint treatment

The extracted curve produced small non-zero endpoint values due to
pixel-level extraction uncertainty.

The published curve visually begins and ends at approximately zero
velocity.

Therefore:

- t = 0 s was set to 0 km/h.
- final sample was set to 0 km/h.

The uncorrected 1 Hz extraction is retained as:

data/raw/Actual_Driving_Cycle_extracted_1Hz.csv

Final reconstructed dataset:

data/raw/Actual_Driving_Cycle_raw.csv

Samples:

3450

Sampling:

1 Hz

---

## 11. Data Integrity

The following automated checks were performed:

- canonical column structure
- expected sample counts
- absence of missing values
- numeric data types
- strictly increasing time
- 1-second sampling interval
- non-negative velocity
- finite velocity values
- slope consistency
- distance calculation

All datasets passed the Phase 1 integrity audit.

---

## 12. Reproducibility Statement

Standard driving cycles are retained separately from processed mixed
cycles.

Raw source data are not overwritten during processing.

The actual driving cycle is explicitly identified as a reconstruction
from the published figure rather than an original raw time-series log.

Any future publication or thesis using this dataset should preserve
this provenance statement.