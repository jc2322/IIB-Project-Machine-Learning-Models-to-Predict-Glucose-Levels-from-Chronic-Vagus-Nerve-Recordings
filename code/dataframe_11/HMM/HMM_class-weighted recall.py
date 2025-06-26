import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from hmmlearn.hmm import GaussianHMM
from scipy.signal import welch
from scipy.stats import kurtosis, skew
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from sklearn.metrics import confusion_matrix

# --- Utility Functions ---

def downsample_mean(signal, window_size):
    signal = np.array(signal)
    trimmed_length = len(signal) - (len(signal) % window_size)
    trimmed_signal = signal[:trimmed_length]
    reshaped_signal = trimmed_signal.reshape(-1, window_size)
    return reshaped_signal.mean(axis=1)

def downsample_median(signal, window_size):
    signal = np.array(signal)
    trimmed_length = len(signal) - (len(signal) % window_size)
    trimmed_signal = signal[:trimmed_length]
    reshaped_signal = trimmed_signal.reshape(-1, window_size)
    return np.median(reshaped_signal, axis=1)

def compute_feature(signal, window_size, feature_type):
    signal = np.array(signal)
    trimmed_length = len(signal) - (len(signal) % window_size)
    reshaped = signal[:trimmed_length].reshape(-1, window_size)
    if feature_type == 'MAV':
        return np.mean(np.abs(reshaped), axis=1)
    elif feature_type == 'ZCR':
        return np.sum(np.diff(np.sign(reshaped), axis=1) != 0, axis=1) / window_size
    else:
        raise ValueError(f"Unknown feature: {feature_type}")

def augment_observations_order(X, order):
    """
    Given an observation array X of shape (n_samples, d),
    returns an augmented observation array of shape (n_samples - order, (order+1)*d)
    where each row is the concatenation of [X[t-order], ..., X[t]].
    """
    n_samples, d = X.shape
    if order == 0:
        return X
    return np.hstack([X[i:n_samples - order + i] for i in range(order+1)])

# -------------------------------
# Load and Process Data from dataframe 11
# -------------------------------
pkl_file_path = r'C:\Users\Jingtong Chen\Desktop\IIB\4th year project\code\dataframe_11\integrated_dataframe_11.pkl'
with open(pkl_file_path, 'rb') as file:
    data = pickle.load(file)

# Extract raw arrays
gl = np.array(data['glucose'], dtype=float)       # Glucose levels
vna = np.array(data['filtered'], dtype=float)     # Vagus nerve activity
hr = np.array(data['HR'], dtype=float)            # Heart rate
br = np.array(data['BR'], dtype=float)            # Breathing rate

# [600000, 7200008, 19200021, 24000026, 31200034, 36990041, 40800045, 47400052, 48000053, 50400056, 52200058, 57600064, 62990070]
# Set outliers indices to NaN, gl[36990041], gl[40800045]
for idx in [36990041, 40800045]:
    gl[idx] = np.nan

# --- 1. Find valid window based on glucose only ---
valid = ~np.isnan(gl)
first, last = np.argmax(valid), len(gl) - 1 - np.argmax(valid[::-1])
# Trim all series to this window
gl, vna, hr, br = [arr[first:last+1] for arr in (gl, vna, hr, br)]

# --- 2. Threshold VNA and Compute Features ---
window_size = 30000
threshold_min, threshold_max = -30, 30
vna = np.clip(vna, threshold_min, threshold_max)
mav = compute_feature(vna, window_size, 'MAV')
zcr = compute_feature(vna, window_size, 'ZCR')

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
for arr in (mav, zcr, gl_ds, hr_ds, br_ds):
    arr[:] = apply_med(arr)

# --- 6. Build and normalize DataFrame ---
combined_df = pd.DataFrame({
    'MAV': mav,
    'ZCR': zcr,
    'GL':  gl_ds,
    'HR':  hr_ds,
    'BR':  br_ds
})
norm_df = (combined_df - combined_df.min()) / (combined_df.max() - combined_df.min())

# --- 7. Assert No NaNs in the Final DataFrame ---
# NaNs in BR from 2045 to 2071, total 2079 datapoints
# Trim norm_df to include only rows from index 0 to 2044
norm_df = norm_df.iloc[:2045].reset_index(drop=True)
assert not norm_df.isnull().values.any(), "NaN values found in the normalized dataframe"

# --- HMM Training & Plotting Routine ---
def train_and_plot_hmm(df, features, order, n_states=2):
    """
    Train an order-HMM on given features and plot hidden states over GL, HR, BR
    with time in seconds on the x-axis.
    """
    X = df[features].values
    X_aug = augment_observations_order(X, order)

    model = GaussianHMM(n_components=n_states, covariance_type='diag', n_iter=300, random_state=42)
    model.fit(X_aug)
    states = model.predict(X_aug)

    # Build time vectors (seconds)
    n_total = len(df)
    time_full   = np.arange(n_total) * 3       # 3 seconds per window for the lines
    idx         = np.arange(order, order + len(states))
    time_states = idx * 3                      # 3 seconds per window for the scatter

    # Plot GL, HR, BR with state coloring
    plt.figure(figsize=(12, 6))
    for col, pos in zip(['GL','HR','BR'], [1,2,3]):
        plt.subplot(3,1,pos)
        plt.plot(time_full, df[col].values, color='#9467bd', label=col)
        colors = ['#ff7f0e', '#17becf']
        custom_cmap = ListedColormap(colors)
        plt.scatter(time_states, df[col].values[idx], c=states, cmap=custom_cmap, s=10)
        plt.ylabel(col)
        if pos == 1:
            legend_elements = [Patch(facecolor=colors[i], label=f'State {i}') for i in range(len(colors))]
            plt.legend(handles=legend_elements, title='Hidden States')
            plt.title(f"Order-{order} HMM (features={features}) hidden states over signals")
    plt.xlabel('Time (s)')
    plt.tight_layout()
    plt.show()

# 8. Generate true labels based on the trimmed, downsampled dataframe
#    We need labels aligned with norm_df's rows
# Compute the split window index using the original `first` and `window_size`
split_pt = 47400052
split_win = (split_pt - first) // window_size
# Number of windows in final norm_df
df_windows = len(norm_df)
# True labels: 1=increasing before split_win, 0=decreasing after or at split_win
true_labels = np.array([1 if w < split_win else 0 for w in range(df_windows)])

# --- 9. Compute class-weighted recall for orders 1 through 200 ---
orders = list(range(1, 201))
scores = []
alpha = 5.0   # weight on recall_0 (decreasing class)
labels_full = true_labels

for order in orders:
    X_aug = augment_observations_order(norm_df[['MAV','ZCR']].values, order)
    labels = labels_full[order:]
    model = GaussianHMM(n_components=2, covariance_type='diag',
                        n_iter=300, random_state=42)
    pred = model.fit(X_aug).predict(X_aug)

    best_score = -np.inf
    # try both mappings of HMM states → {0,1}
    for pm in (pred, 1 - pred):
        tn, fp, fn, tp = confusion_matrix(labels, pm, labels=[0,1]).ravel()
        recall0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        recall1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # class-weighted recall
        score = (alpha * recall0 + recall1) / (alpha + 1)
        best_score = max(best_score, score)

    scores.append(best_score)

best_idx   = int(np.argmax(scores))
best_order = orders[best_idx]
best_score = scores[best_idx]
print(f"Best order (class-weighted recall): {best_order} → {best_score*100:.2f}%")

# --- 10. Plot class-weighted recall vs. order ---
plt.figure(figsize=(10,4))
plt.plot(orders, np.array(scores)*100, label=f'α={alpha}')
plt.scatter([best_order], [best_score*100], color='red',
            label=f'Best (order={best_order})')
plt.xlabel('HMM Order')
plt.ylabel('Class-Weighted Recall (%)')
plt.title('Dataframe 11: HMM Class-Weighted Recall vs. Model Order')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Example plots
for ord in [best_order]:
    train_and_plot_hmm(norm_df, ['MAV','ZCR'], ord)
