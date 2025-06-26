# IIB-Project-Machine-Learning-Models-to-Predict-Glucose-Levels-from-Chronic-Vagus-Nerve-Recordings

# Autonomic–Glucose Modelling Project

This repository contains code for Structural Equation Modelling (SEM) and Hidden Markov Models (HMM) applied to chronic vagus‐nerve recordings and blood glucose data in anesthetized rats.

## Directory Structure

```text
project_root/
├── dataframe_9/
│ ├── plots/
│ │ └── … signal & feature plots (.png)
│ ├── integrated_dataframe_9.pkl
│ ├── raw_plots.py
│ └── SEM.py
│
├── dataframe_10/ ← identical structure to dataframe_9, but using integrated_dataframe_10.pkl
│ ├── plots/
│ ├── integrated_dataframe_10.pkl
│ ├── raw_plots.py
│ └── SEM.py
│
├── dataframe_14/ ← same structure
│ ├── plots/
│ ├── integrated_dataframe_14.pkl
│ ├── raw_plots.py
│ └── SEM.py
│
├── dataframe_16/ ← same structure
│ ├── plots/
│ ├── integrated_dataframe_16.pkl
│ ├── raw_plots.py
│ └── SEM.py
│
├── dataframe_11/
│ ├── plots/ ← same signal‐plot folder as above
│ ├── integrated_dataframe_11.pkl
│ ├── raw_plots.py
│ ├── SEM.py
│ └── HMM/ ← HMM experiment code for dataset 11
│
├── dataframe_19/
│ ├── plots/
│ ├── integrated_dataframe_19.pkl
│ ├── raw_plots.py
│ ├── SEM.py
│ └── HMM/ ← HMM experiment code for dataset 19
│
├── dataframe_9+10/
│ └── HMM/ ← combined HMM code across 9 & 10
│
└── dataframe_combined_GL_decreasing_case/
└── SEM_combined.py ← SEM code on pooled GL‐decreasing data
```

---

## Common Files

### `integrated_dataframe_*.pkl`
- **Input:** Pickled pandas DataFrame with columns `glucose`, `filtered` (VNA), `HR`, `BR`.
- **Usage:** Loaded by both `raw_plots.py` and `SEM.py`.

### `raw_plots.py`
1. Loads the `.pkl` DataFrame.
2. Trims to valid glucose windows.
3. Interpolates glucose to full resolution.
4. Plots raw time series of VNA, GL, HR, BR.
5. Clips VNA to physiological bounds and re‐plots.
- **Output:** PNGs for visual QC.

### `SEM.py`
1. Loads DataFrame and thresholds VNA.
2. Extracts nine non‐overlapping‐window features:
   - **Amplitude:** MAV, RMS, STD, MAX  
   - **Frequency:** MF, ZCR, SSC  
   - **Shape:** Kurtosis, Skewness  
3. Interpolates, downsamples (mean/median), median‐filters, normalizes.
4. Defines & fits a two‐latent‐factor SEM (PNS & SNS) in `semopy`.
5. Prints parameter estimates, fit indices (CFI, GFI, AIC).
6. Exports SEM path‐diagram PNG.
- **Output:** Feature time‐series plots, SEM results & diagram.

---

## HMM Experiments

1. Loads the normalized DataFrame.
2. Defines observation vectors. 
3. Tunes memory order `k` (1–200 windows):
   - Trains Gaussian‐emission HMM (2 states).
   - Decodes via Viterbi.
   - Computes classification accuracy.
4. Plots accuracy vs. `k` and state overlays on GL/HR/BR.
- **Output:** Accuracy curves, overlay figures, summary of best orders & scores.

---

Author: Jingtong Chen
Date: June 2025




