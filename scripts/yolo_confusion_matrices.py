import matplotlib
matplotlib.use("Agg")  # headless backend, no display available
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Define class names
class_names = ["Fresh", "Ripe", "Overripe", "Rotten"]

# Data extracted from the tables (Per-class accuracy/recall representation)
models_data = {
    "YOLO11s (Overall Acc: 0.937)": {
        "matrix": np.array(
            [[0.927, 0.0, 0.0, 0.0],  # Fresh
             [0.0, 0.853, 0.0, 0.0],  # Ripe
             [0.0, 0.0, 0.997, 0.0],  # Overripe
             [0.0, 0.0, 0.0, 0.992]]  # Rotten
        ),
        "precision": [0.8672, 0.9823, 1.0000, 0.9840],
        "recall": [0.9971, 0.7525, 0.9944, 1.0000],
    },
    "YOLO11m (Overall Acc: 0.928)": {
        "matrix": np.array(
            [[0.864, 0.0, 0.0, 0.0],
             [0.0, 0.821, 0.0, 0.0],
             [0.0, 0.0, 0.997, 0.0],
             [0.0, 0.0, 0.0, 0.989]],
        ),
        "precision": [0.7612, 0.9771, 1.0000, 0.9808],
        "recall": [0.9971, 0.7220, 0.9944, 0.9967],
    },
    "YOLO11l (Overall Acc: 0.993)": {
        "matrix": np.array(
            [[0.987, 0.0, 0.0, 0.0],
             [0.0, 0.985, 0.0, 0.0],
             [0.0, 0.0, 0.997, 0.0],
             [0.0, 0.0, 0.0, 0.987]],
        ),
        "precision": [0.9825, 0.9767, 1.0000, 0.9807],
        "recall": [0.9912, 0.9932, 0.9944, 0.9935],
    },
}

# Create a figure with 3 subplots for the models
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (model_name, data) in zip(axes, models_data.items()):
  sns.heatmap(
      data["matrix"],
      annot=True,
      fmt=".3f",
      cmap="Blues",
      xticklabels=class_names,
      yticklabels=class_names,
      vmin=0.0,
      vmax=1.0,
      cbar=True,
      ax=ax,
  )
  ax.set_title(model_name, fontsize=12, fontweight="bold")
  ax.set_xlabel("Predicted Class")
  ax.set_ylabel("True Class")

plt.tight_layout()
plt.savefig("yolo_confusion_matrices.png", dpi=150)
print("Saved yolo_confusion_matrices.png")
