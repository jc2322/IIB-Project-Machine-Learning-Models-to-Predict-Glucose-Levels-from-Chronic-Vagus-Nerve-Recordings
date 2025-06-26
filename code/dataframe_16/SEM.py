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

# Function to downsample a signal using non-overlapping windows by computing the median
def downsample_median(signal, window_size):
    signal = np.array(signal)
    trimmed_length = len(signal) - (len(signal) % window_size)
    trimmed_signal = signal[:trimmed_length]
    reshaped_signal = trimmed_signal.reshape(-1, window_size)
    return np.median(reshaped_signal, axis=1)

# Function to downsample a signal using non-overlapping windows by computing the mean
def downsample_mean(signal, window_size):
    signal = np.array(signal)
    trimmed_length = len(signal) - (len(signal) % window_size)
    trimmed_signal = signal[:trimmed_length]
    reshaped_signal = trimmed_signal.reshape(-1, window_size)
    return reshaped_signal.mean(axis=1)

# Function to compute feature extraction metrics with non-overlapping windows
def compute_feature(signal, window_size, feature_type):
    signal = np.array(signal)
    trimmed_length = len(signal) - (len(signal) % window_size)
    trimmed_signal = signal[:trimmed_length]
    reshaped_signal = trimmed_signal.reshape(-1, window_size)
    if feature_type == 'MF':
        fs = 10000  # Sampling frequency
        return [np.sum(f * Pxx) / np.sum(Pxx) for window in reshaped_signal
                for f, Pxx in [welch(window, fs=fs, nperseg=window_size)]]
    elif feature_type == 'RMS':
        return np.sqrt(np.mean(reshaped_signal**2, axis=1))
    elif feature_type == 'STD':
        return np.std(reshaped_signal, axis=1)
    elif feature_type == 'MAX':
        return np.max(reshaped_signal, axis=1)
    elif feature_type == 'MAV':
        return np.mean(np.abs(reshaped_signal), axis=1)
    elif feature_type == 'ZCR':
        # Counts the number of times the signal crosses the zero amplitude axis within a window.
        # A higher firing rate may result in more frequent zero crossings due to increased signal oscillations.
        return np.sum(np.diff(np.sign(reshaped_signal), axis=1) != 0, axis=1) / window_size
    elif feature_type == 'SSC':
        # SSC counts the number of times the slope of the signal changes sign within a window, 
        # which corresponds to the number of times the signal changes direction. 
        # This feature can capture the oscillatory nature of neural signals and is indicative of the rate of spike occurrences.
        return np.sum((np.diff(np.sign(np.diff(reshaped_signal, axis=1))) != 0), axis=1)
    elif feature_type == 'WA':
        return np.sum(np.abs(np.diff(reshaped_signal, axis=1)) > np.std(reshaped_signal, axis=1, keepdims=True), axis=1)
    elif feature_type == 'Kurtosis':
        # Kurtosis measures the "tailedness" of the signal's amplitude distribution. 
        # A higher kurtosis value indicates a higher presence of outliers, which can correspond to spike events in neural signals.
        return np.apply_along_axis(kurtosis, 1, reshaped_signal)
    elif feature_type == 'Skewness':
        return np.apply_along_axis(skew, 1, reshaped_signal)
    else:
        raise ValueError("Invalid feature type")

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

# --- 1. Find valid window based on glucose only ---
valid = ~np.isnan(gl)
first, last = np.argmax(valid), len(gl) - 1 - np.argmax(valid[::-1])
# Trim all series to this window
gl, vna, hr, br = [arr[first:last+1] for arr in (gl, vna, hr, br)]

# --- 2. Threshold VNA and Compute Features ---
window_size = 30000
threshold_min, threshold_max = -60, 60
vna_thresholded = np.clip(vna, threshold_min, threshold_max)
mf_feature = compute_feature(vna_thresholded, window_size, 'MF')
rms_feature = compute_feature(vna_thresholded, window_size, 'RMS')
std_feature = compute_feature(vna_thresholded, window_size, 'STD')
max_feature = compute_feature(vna_thresholded, window_size, 'MAX')
mav_feature = compute_feature(vna_thresholded, window_size, 'MAV')
zcr_feature = compute_feature(vna_thresholded, window_size, 'ZCR')
ssc_feature = compute_feature(vna_thresholded, window_size, 'SSC')
kurtosis_feature = compute_feature(vna_thresholded, window_size, 'Kurtosis')
skewness_feature = compute_feature(vna_thresholded, window_size, 'Skewness')

# --- 3. Interpolate glucose at full resolution ---
gl_series = pd.Series(gl)
gl_series = gl_series.interpolate(method='linear', limit_direction='both')
gl = gl_series.values

# --- 4. Downsample each series ---
gl_ds  = downsample_mean(gl, window_size)
hr_ds  = downsample_median(hr, window_size)
br_ds  = downsample_median(br, window_size)

# --- 5. Smooth after downsampling ---
apply_med = lambda x: medfilt(x, kernel_size=21)
for arr in (mf_feature, rms_feature, std_feature, max_feature, mav_feature, zcr_feature, ssc_feature, kurtosis_feature, skewness_feature, gl_ds, hr_ds, br_ds):
    arr[:] = apply_med(arr)

# --- 6. Build and normalize DataFrame ---
combined_df = pd.DataFrame({
    'GL':  gl_ds,
    'HR':  hr_ds,
    'BR':  br_ds,
    'MF': mf_feature,
    'RMS': rms_feature,
    'STD': std_feature,
    'MAX': max_feature,
    'MAV': mav_feature,
    'ZCR': zcr_feature,
    'SSC': ssc_feature,
    'Kurtosis': kurtosis_feature,
    'Skewness': skewness_feature
})


normalized_df = (combined_df - combined_df.min()) / (combined_df.max() - combined_df.min())

print(len(normalized_df))
# --- Check for NaNs in the combined DataFrame ---
nan_columns = combined_df.columns[combined_df.isnull().any()]
if not nan_columns.empty:
    print(f"\nNaNs found in columns of combined_df: {list(nan_columns)}")
    for col in nan_columns:
        print(f"  → {col}: NaNs at indices {combined_df[combined_df[col].isnull()].index.tolist()}")

# --- 7. Assert No NaNs in the Final DataFrame ---
assert not normalized_df.isnull().values.any(), "NaN values found in the normalized dataframe"


'''
for column in normalized_df.columns:
    print(f"Length of {column}: {len(normalized_df[column])}")

# Extract the normalized values for GL, HR, and BR
gl_normalized = normalized_df['GL']
hr_normalized = normalized_df['HR']
br_normalized = normalized_df['BR']


# Number of windows
n_w = len(gl_normalized)
# Generate an array of times in seconds
t_secs = np.arange(n_w) * 3  

# GL plot
plt.figure(figsize=(9, 3))
plt.plot(t_secs, gl_normalized, color='green', linewidth=1.5)
plt.title("Glucose Level (GL)")
plt.ylabel("Min–Max Normalised")
plt.xlabel("Time (s)")
plt.tight_layout()

# HR plot
plt.figure(figsize=(9, 3))
plt.plot(t_secs, hr_normalized, color='red', linewidth=1.5)
plt.title("Heart Rate (HR)")
plt.ylabel("Min–Max Normalised")
plt.xlabel("Time (s)")
plt.tight_layout()

# BR plot
plt.figure(figsize=(9, 3))
plt.plot(t_secs, br_normalized, color='blue', linewidth=1.5)
plt.title("Breathing Rate (BR)")
plt.ylabel("Min–Max Normalised")
plt.xlabel("Time (s)")
plt.tight_layout()

plt.show()


# --- feature plots, now vs. seconds ---
fig, axes = plt.subplots(3, 3, figsize=(12, 9))
features_to_plot = {
    'Mean Absolute Value (MAV)': normalized_df['MAV'],
    'Mean Frequency (MF)': normalized_df['MF'],
    'Root Mean Square (RMS)': normalized_df['RMS'],
    'Standard Deviation (STD)': normalized_df['STD'],
    'Maximum (MAX)': normalized_df['MAX'],
    'Zero-Crossing Rate (ZCR)': normalized_df['ZCR'],
    'Slope Sign Changes (SSC)': normalized_df['SSC'],
    'Kurtosis': normalized_df['Kurtosis'],
    'Skewness': normalized_df['Skewness']
}

for ax, (title, feature) in zip(axes.flatten(), features_to_plot.items()):
    ax.plot(t_secs, feature, linewidth=0.8)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Min–Max Normalised', fontsize=10)
    ax.grid(alpha=0.3)

plt.tight_layout(h_pad=2)
plt.show()

'''
# SEM analysis
sem_desc = """
# Measurement Model
PNS =~ SSC + GL + HR + BR
SNS =~ HR + BR + GL

# Structural Model
PNS ~ SNS
"""
model = Model(sem_desc)
res_opt = model.fit(normalized_df)
estimates = model.inspect()
stats = semopy.calc_stats(model)

# Print SEM results
print(res_opt)
print(estimates)
print(stats.T)

# Plot SEM
output_path = "C:\\Users\\Jingtong Chen\\Desktop\\IIB\\4th year project\\code\\dataframe_16\\dataframe16_SSC_w=1.png"
g = semopy.semplot(model, output_path)
plt.show()
