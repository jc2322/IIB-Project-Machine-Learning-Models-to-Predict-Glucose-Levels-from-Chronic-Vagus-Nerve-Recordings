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
# Load and Process Data from dataframe 19
# -------------------------------
pkl_file_path = r'C:\Users\Jingtong Chen\Desktop\IIB\4th year project\code\dataframe_19\integrated_dataframe_19.pkl'
with open(pkl_file_path, 'rb') as file:
    data = pickle.load(file)

# Extract raw arrays
gl = np.array(data['glucose'], dtype=float)       # Glucose levels
vna = np.array(data['filtered'], dtype=float)     # Vagus nerve activity
hr = np.array(data['HR'], dtype=float)            # Heart rate
br = np.array(data['BR'], dtype=float)            # Breathing rate
'''
print(np.where(~np.isnan(gl))[0])
indices = [0, 3000003, 6000007, 10200011, 12000013, 15000017, 18000020, 21000023,
           24000027, 27000030, 30000033, 33000037, 36000040, 40800045, 45000050, 48000053,
           51000057, 54000060, 57000063, 60000067, 63600071, 67200075, 69000077]
gl_values_at_indices = gl[indices]
print(gl_values_at_indices)
[163.8 172.8 167.4 183.6 185.4 216.  216.  205.2 203.4 226.8 241.2 271.8
 244.8 279.  270.  266.4 261.  259.2 223.2 219.6 221.4 203.4 194.4]
decreasing: [15000017, 24000027]
decreasing: [36000040, 69000077]
'''
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
norm_df = (combined_df - combined_df.min()) / (combined_df.max() - combined_df.min())

# --- 7. Assert No NaNs in the Final DataFrame ---
assert not norm_df.isnull().values.any(), "NaN values found in the normalized dataframe"

# --- HMM Training & Plotting Routine ---
def train_and_plot_hmm(df, features, order, n_states=2):
    """
    Train an order-HMM on given features and plot hidden states over GL, HR, BR.
    df        : normalized DataFrame with columns 'GL','HR','BR' and feature cols
    features  : list of feature column names (e.g. ['MAV','ZCR'])
    order     : int, the HMM order
    n_states  : int, number of hidden states
    """
    X = df[features].values
    X_aug = augment_observations_order(X, order)

    model = GaussianHMM(n_components=n_states, covariance_type='diag', n_iter=300, random_state=42)
    model.fit(X_aug)
    states = model.predict(X_aug)

    # Align index for plotting
    idx = np.arange(order, order + len(states))

    # Plot GL, HR, BR with state coloring
    plt.figure(figsize=(12, 6))
    for col, pos in zip(['GL','HR','BR'], [1,2,3]):
        plt.subplot(3,1,pos)
        plt.plot(df[col].values, color='#9467bd', label=col)
        colors = ['#ff7f0e', '#17becf']
        custom_cmap = ListedColormap(colors) 
        plt.scatter(idx, df[col].values[idx], c=states, cmap=custom_cmap, s=10)
        plt.ylabel(col)
        if pos == 1:
            legend_elements = [Patch(facecolor=colors[i], label=f'State {i}') for i in range(len(colors))]
            plt.legend(handles=legend_elements, title='Hidden States')
            plt.title(f"Order-{order} HMM (features={features}) hidden states over signals")
    plt.xlabel('Window Index')
    plt.tight_layout()
    plt.show()

# 8. Generate true labels using multiple decreasing intervals
# Define intervals (in original sample indices) marked as decreasing
decr_intervals = [(15000017, 24000027), (36000040, 69000077)]
# Compute window-based labels for norm_df
n_windows = len(norm_df)
true_labels = []
for w in range(n_windows):
    # compute original sample range for this window
    samp_start = first + w * window_size
    samp_end   = samp_start + window_size - 1
    # check if this window overlaps any decreasing interval
    is_decr = any((samp_start <= end and samp_end >= start) for start, end in decr_intervals)
    true_labels.append(0 if is_decr else 1)
true_labels = np.array(true_labels)

# 9. Evaluate accuracy for orders 1 through 100
orders = list(range(1, 201))
accuracies = []
X_full = norm_df[['MAV','ZCR','HR','BR']].values
for order in orders:
    X_aug = augment_observations_order(X_full, order)
    model = GaussianHMM(n_components=2, covariance_type='diag', n_iter=300, random_state=42)
    pred_states = model.fit(X_aug).predict(X_aug)
    labels = true_labels[order:]
    # Allow best state-label mapping
    acc1 = (pred_states == labels).mean()
    acc2 = ((1 - pred_states) == labels).mean()
    accuracies.append(max(acc1, acc2))

# Print best order and accuracy
best_idx = int(np.argmax(accuracies))
best_order = orders[best_idx]
best_acc = accuracies[best_idx]
print(f"Best order: {best_order} with accuracy {best_acc*100:.2f}%")

# 10. Plot accuracy vs order
plt.figure(figsize=(10, 4))
plt.plot(orders, np.array(accuracies)*100)
plt.xlabel('HMM Order')
plt.ylabel('Classification Accuracy (%)')
plt.title('Dataframe 19: HMM Accuracy vs. Model Order')
# Annotate best point
plt.scatter([best_order], [best_acc*100], color='red')
plt.grid(True)
plt.tight_layout()
plt.show()

# Example plots
for ord in [best_order]:
    train_and_plot_hmm(norm_df, ['MAV','ZCR','HR','BR'], ord)
