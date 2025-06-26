import pickle
import semopy
from semopy import Model
from scipy.signal import medfilt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import kurtosis, skew
from hmmlearn.hmm import GaussianHMM

# File path
pkl_file_path = r'C:\Users\Jingtong Chen\Desktop\IIB\4th year project\code\dataframe_16\integrated_dataframe_16.pkl'

# Load data
with open(pkl_file_path, 'rb') as file:
    data = pickle.load(file)

# Extract raw arrays
gl = np.array(data['glucose'], dtype=float)       # Glucose levels
vna = np.array(data['filtered'], dtype=float)     # Vagus nerve activity
hr = np.array(data['HR'], dtype=float)            # Heart rate
br = np.array(data['BR'], dtype=float)            # Breathing rate

# time conversion: each sample is 0.1 ms → 1e-4 s
def make_time_axis(n_samples):
    return np.arange(n_samples) * 1e-4  # seconds

# --- 1. Find valid window based on glucose only ---
valid = ~np.isnan(gl)

# GL plot
x_valid = np.nonzero(valid)[0] * 1e-4  # convert indices to seconds
y_valid = gl[valid]
plt.figure(figsize=(9, 3))
plt.scatter(x_valid, y_valid, s=20, c='green', marker='o', alpha=1)
plt.title("Glucose Level (GL)")
plt.ylabel("Concentration (mg/dL)")
plt.xlabel("Time (s)")
plt.grid(which='both', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

first, last = np.argmax(valid), len(gl) - 1 - np.argmax(valid[::-1])
# Trim all series to this window
gl, vna, hr, br = [arr[first:last+1] for arr in (gl, vna, hr, br)]

# --- 2. Interpolate glucose at full resolution ---
gl_series = pd.Series(gl)
gl_series = gl_series.interpolate(method='linear', limit_direction='both')
gl = gl_series.values

# prepare time axes (post-trim)
t = make_time_axis(len(vna))

# VNA plot
plt.figure(figsize=(9, 3))
plt.plot(t, vna, color='#1f77b4', linewidth=1.5)
plt.title("Vagus Nerve Activity (VNA)")
plt.ylabel("Amplitude (μV)")
plt.xlabel("Time (s)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 3))
plt.plot(t, gl, color='green', linewidth=1.5)
plt.title("Glucose Level (GL)")
plt.ylabel("Concentration (mg/dL)")
plt.xlabel("Time (s)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 3))
plt.plot(t, hr, color='red', linewidth=1.5)
plt.title("Heart Rate (HR)")
plt.ylabel("Beats per Minute (bpm)")
plt.xlabel("Time (s)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 3))
plt.plot(t, br, color='blue', linewidth=1.5)
plt.title("Breathing Rate (BR)")
plt.ylabel("Bursts per Minute")
plt.xlabel("Time (s)")
plt.tight_layout()
plt.show()

# --- Thresholded VNA ---
threshold_min, threshold_max = -60, 60
vna_thresholded = np.clip(vna, threshold_min, threshold_max)

plt.figure(figsize=(9, 3))
plt.plot(t, vna_thresholded, color='#1f77b4', linewidth=1.5)
plt.title("Vagus Nerve Activity (VNA)")
plt.ylabel("Amplitude (μV)")
plt.xlabel("Time (s)")
plt.tight_layout()
plt.show()
