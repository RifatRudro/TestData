import matplotlib
matplotlib.use("Agg")  # headless backend, no display available
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 1. Define your original class names
class_names = ["fresh", "ripe", "overripe", "rotten", "background"]
background_idx = class_names.index("background")

# --- Synthetic demo data (no y_true/y_pred existed in this repo) ---
rng = np.random.default_rng(42)
n_samples = 500
y_true = rng.integers(0, len(class_names), size=n_samples)
# make predictions mostly correct, with some noise, to get a realistic-looking matrix
y_pred = y_true.copy()
noise_mask = rng.random(n_samples) < 0.25
y_pred[noise_mask] = rng.integers(0, len(class_names), size=noise_mask.sum())

# --- METHOD A: If you have your raw `y_true` and `y_pred` arrays/lists ---
# Filter out any instances where either the true or predicted label is 'background'
mask = (np.array(y_true) != background_idx) & (np.array(y_pred) != background_idx)
y_true_filtered = np.array(y_true)[mask]
y_pred_filtered = np.array(y_pred)[mask]

# Create filtered class names list
filtered_class_names = [c for c in class_names if c != "background"]

# Compute and normalize the confusion matrix
cm = confusion_matrix(y_true_filtered, y_pred_filtered)
cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)
# Handle potential division by zero for classes with zero samples
cm_normalized = np.nan_to_num(cm_normalized)

# --- PLOTTING THE CONFUSION MATRIX ---
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm_normalized,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=filtered_class_names,
    yticklabels=filtered_class_names,
    vmin=0.0,
    vmax=1.0,
)

plt.title("Confusion Matrix Normalized")
plt.xlabel("True")
plt.ylabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("Saved confusion_matrix.png")
