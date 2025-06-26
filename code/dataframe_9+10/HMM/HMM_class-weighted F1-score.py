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
from sklearn.metrics import fbeta_score

import os
os.environ['OMP_NUM_THREADS'] = '5'

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
    # build [X[t-order], ..., X[t]] for each t
    augmented = np.hstack([X[i:n_samples - order + i] for i in range(order+1)])
    return augmented

# -------------------------------
# Paths for DataFrames 9 (GL ↑) and 10 (GL ↓)
# -------------------------------
pkl_paths = [
    r'C:\Users\Jingtong Chen\Desktop\IIB\4th year project\code\dataframe_9\integrated_dataframe_9.pkl',
    r'C:\Users\Jingtong Chen\Desktop\IIB\4th year project\code\dataframe_10\integrated_dataframe_10.pkl',
]

window_size = 30000
threshold_min, threshold_max = -70, 70

# Containers
filtered_list, glucose_list, hr_list, br_list = [], [], [], []
window_counts = []

# Load, record window counts, and collect signals
for path in pkl_paths:
    with open(path, 'rb') as f:
        data = pickle.load(f)
    filt = np.array(data['filtered'], dtype=float)
    # count windows for this segment
    count = (len(filt) - (len(filt) % window_size)) // window_size
    window_counts.append(count)
    filtered_list.append(filt)
    glucose_list.append(np.array(data['glucose'], dtype=float))
    hr_list.append(np.array(data['HR'], dtype=float))
    br_list.append(np.array(data['BR'], dtype=float))

# Concatenate signals
vna = np.concatenate(filtered_list)
gl = np.concatenate(glucose_list)
hr = np.concatenate(hr_list)
br = np.concatenate(br_list)

# --- 1. Find valid window based on glucose only ---
valid = ~np.isnan(gl)
first, last = np.argmax(valid), len(gl) - 1 - np.argmax(valid[::-1])
# Trim all series to this window
gl, vna, hr, br = [arr[first:last+1] for arr in (gl, vna, hr, br)]

# --- 2. Threshold VNA and Compute Features ---
window_size = 30000
threshold_min, threshold_max = -40, 40
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
combined_df['BR'] = combined_df['BR'].interpolate()
norm_df = (combined_df - combined_df.min()) / (combined_df.max() - combined_df.min())

# --- 7. Assert No NaNs in the Final DataFrame ---
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

# 8. Generate true labels based on the trimmed, downsampled dataframe, split at 10 min after df10
# length of df9 is 18600682 so 18600682+6000000=24600682
split_pt = 24600682
split_win = (split_pt - first) // window_size
# Number of windows in final norm_df
df_windows = len(norm_df)
# True labels: 1=increasing before split_win, 0=decreasing after or at split_win
true_labels = np.array([1 if w < split_win else 0 for w in range(df_windows)])

# --- 9. Compute class-weighted F1 for orders 1..200 ---
orders = list(range(1, 201))
combined_f1 = []
alpha = 5.0     # how many times more you care about F1 of class 0 vs class 1
beta  = 1.0     # beta=1 → F1

for order in orders:
    X_aug = augment_observations_order(norm_df[['MAV','ZCR']].values, order)
    labels = true_labels[order:]
    model  = GaussianHMM(n_components=2, covariance_type='diag',
                         n_iter=300, random_state=42)
    pred   = model.fit(X_aug).predict(X_aug)

    best_score = -np.inf
    # try both ways of mapping HMM states → {0,1}
    for pm in (pred, 1 - pred):
        # compute per-class Fβ
        # average=None → returns array [Fβ_class0, Fβ_class1]
        f_scores = fbeta_score(labels, pm,
                               beta=beta,
                               average=None,
                               labels=[0,1])

        f0, f1 = f_scores  # f0=Fβ for class 0, f1=Fβ for class 1
        combined = (alpha * f0 + f1) / (alpha + 1)
        best_score = max(best_score, combined)

    combined_f1.append(best_score)

# find the best order
best_idx   = int(np.argmax(combined_f1))
best_order = orders[best_idx]
best_value = combined_f1[best_idx]
print(f"Best order (class-weighted F1): {best_order} → {best_value*100:.2f}%")

# --- 10. Plot the results ---
plt.figure(figsize=(10,4))
plt.plot(orders, np.array(combined_f1)*100, label=f'α={alpha}')
plt.scatter([best_order], [best_value*100], color='red',
            label=f'Best (order={best_order})')
plt.xlabel('HMM Order')
plt.ylabel('Class-Weighted F1 (%)')
plt.title('Dataframe 9+10: Class-Weighted F1 vs. Model Order')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Example plots
for ord in [best_order]:
    train_and_plot_hmm(norm_df, ['MAV','ZCR'], ord)
