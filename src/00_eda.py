"""
00_eda.py — Exploratory Data Analysis for PS4 Lung X-Ray Dataset

This script:
  1. Loads the training/validation images and labels from .npy / .csv files
  2. Inspects image shape, dtype, value range, and checks for anomalies
  3. Visualizes class distribution (bar chart)
  4. Displays sample X-ray images from each class
  5. Computes per-class pixel statistics (mean, std)
  6. Saves all plots to output/figures/

Run: python src/00_eda.py
Output: output/figures/class_distribution.png
        output/figures/sample_images.png
        output/figures/pixel_stats.png

Gate G1 passes when both figures are generated without errors.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ─── Configuration ────────────────────────────────────────────────────────────
# All paths and settings are centralized here for easy modification

CONFIG = {
    "trainvalid_images": "ps4_trainvalid_images-2.npy",  # raw numpy image array
    "trainvalid_labels": "ps4_trainvalid_labels.csv",    # labels with 'Id' and 'Label' columns
    "kaggle_images":     "ps4_kaggle_images-1.npy",      # kaggle test images
    "kaggle_labels":     "ps4_kaggle_labels.csv",        # kaggle labels
    "output_dir":        "output/figures",               # where to save EDA figures
    "n_samples_per_class": 3,                            # how many sample images to show per class
    "random_seed": 42,
}

CLASS_NAMES = {0: "Healthy", 1: "Pre-existing Conditions", 2: "Effusion / Mass"}
CLASS_COLORS = {0: "#2196F3", 1: "#FF9800", 2: "#F44336"}   # blue, orange, red


# ─── Utility Functions ────────────────────────────────────────────────────────

def load_data(img_path: str, label_path: str, label_col: str = "Label"):
    """
    Load numpy image array and corresponding CSV labels.

    Args:
        img_path:   Path to .npy file containing images.
        label_path: Path to CSV with columns ['Id', label_col].
        label_col:  Name of the column containing class labels.

    Returns:
        images (np.ndarray): Shape (N, ...) — raw image array.
        labels (np.ndarray): Shape (N,)    — integer class labels.
    """
    print(f"Loading images from: {img_path}")
    images = np.load(img_path)       # may take a moment for 164 MB file

    print(f"Loading labels from: {label_path}")
    df = pd.read_csv(label_path)
    labels = df[label_col].values

    assert len(images) == len(labels), (
        f"Mismatch: {len(images)} images vs {len(labels)} labels"
    )
    print(f"Loaded {len(images)} samples.")
    return images, labels


def describe_images(images: np.ndarray, name: str = "Dataset") -> dict:
    """
    Print and return key statistics about the image array.

    Args:
        images: Numpy array of images (N, H, W) or (N, H, W, C).
        name:   Label to use in printout.

    Returns:
        Dict with shape, dtype, min, max, mean, std.
    """
    stats = {
        "name":  name,
        "shape": images.shape,       # e.g. (11414, 64, 64) or (11414, 64, 64, 1)
        "dtype": str(images.dtype),
        "min":   float(images.min()),
        "max":   float(images.max()),
        "mean":  float(images.mean()),
        "std":   float(images.std()),
    }
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Shape : {stats['shape']}")
    print(f"  Dtype : {stats['dtype']}")
    print(f"  Range : [{stats['min']:.3f}, {stats['max']:.3f}]")
    print(f"  Mean  : {stats['mean']:.4f}")
    print(f"  Std   : {stats['std']:.4f}")
    return stats


# ─── Plot 1: Class Distribution ──────────────────────────────────────────────

def plot_class_distribution(labels: np.ndarray, title: str, save_path: str) -> None:
    """
    Bar chart of sample counts per class with percentage annotations.

    This is critical because labels are heavily imbalanced (Class 0 = 92%),
    which directly impacts model training and evaluation strategy.

    Args:
        labels:    1-D array of class labels (0, 1, 2).
        title:     Figure title.
        save_path: Where to save the figure.
    """
    unique, counts = np.unique(labels, return_counts=True)
    pcts = counts / counts.sum() * 100

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(
        [CLASS_NAMES[c] for c in unique],
        counts,
        color=[CLASS_COLORS[c] for c in unique],
        edgecolor="black",
        linewidth=0.8,
    )

    # Annotate each bar with count and percentage
    for bar, count, pct in zip(bars, counts, pcts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            f"{count:,}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    ax.set_ylabel("Number of Samples")
    ax.set_ylim(0, max(counts) * 1.18)
    ax.grid(axis="y", alpha=0.3)
    sns.despine()
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved class distribution → {save_path}")
    plt.show()


# ─── Plot 2: Sample Images per Class ─────────────────────────────────────────

def plot_sample_images(
    images: np.ndarray,
    labels: np.ndarray,
    n_per_class: int = 3,
    save_path: str = None,
) -> None:
    """
    Display a grid of randomly selected X-rays from each class.

    Visualizing samples helps us:
    - Confirm the data loaded correctly (not corrupted)
    - Get intuition for what distinguishes the three classes visually
    - Verify image orientation and quality

    Args:
        images:      Full image array (N, H, W) or (N, H, W, C).
        labels:      Corresponding class label array.
        n_per_class: Number of images to show per class.
        save_path:   Where to save the figure.
    """
    np.random.seed(CONFIG["random_seed"])
    n_classes = len(CLASS_NAMES)
    fig, axes = plt.subplots(
        n_classes, n_per_class,
        figsize=(n_per_class * 2.5, n_classes * 2.8),
    )
    fig.suptitle(
        "Sample Chest X-Rays by Class",
        fontsize=14, fontweight="bold", y=1.01,
    )

    for row, cls in enumerate(sorted(CLASS_NAMES.keys())):
        cls_indices = np.where(labels == cls)[0]
        chosen = np.random.choice(cls_indices, size=n_per_class, replace=False)

        for col, idx in enumerate(chosen):
            ax = axes[row, col]
            img = images[idx]

            # Handle grayscale images that might have a channel dim
            if img.ndim == 3 and img.shape[-1] == 1:
                img = img[:, :, 0]
            elif img.ndim == 3 and img.shape[0] == 1:
                img = img[0]

            ax.imshow(img, cmap="gray", aspect="auto")
            ax.axis("off")

            if col == 0:
                # Label the row with the class name
                ax.set_ylabel(
                    CLASS_NAMES[cls],
                    fontsize=10, fontweight="bold",
                    color=CLASS_COLORS[cls],
                    rotation=90, labelpad=5,
                )
                ax.yaxis.set_label_position("left")
                ax.yaxis.label.set_visible(True)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved sample images → {save_path}")
    plt.show()


# ─── Plot 3: Per-Class Pixel Statistics ──────────────────────────────────────

def plot_pixel_stats(images: np.ndarray, labels: np.ndarray, save_path: str) -> None:
    """
    Plot pixel intensity histograms for each class.

    Understanding per-class intensity distributions helps us:
    - Confirm that classes look visually distinct (feature separability)
    - Choose appropriate normalization strategy

    Args:
        images:    Image array (N, H, W[, C]).
        labels:    Class label array.
        save_path: Where to save the figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=False)

    for cls, ax in zip(sorted(CLASS_NAMES.keys()), axes):
        cls_imgs = images[labels == cls]
        # Flatten all pixels for this class into a 1-D array
        pixel_vals = cls_imgs.flatten()

        # Subsample if too large (speeds up histogram computation)
        if len(pixel_vals) > 500_000:
            pixel_vals = np.random.choice(pixel_vals, 500_000, replace=False)

        ax.hist(pixel_vals, bins=80, color=CLASS_COLORS[cls], alpha=0.7,
                edgecolor="none", density=True)
        ax.set_title(CLASS_NAMES[cls], fontsize=10, fontweight="bold",
                     color=CLASS_COLORS[cls])
        ax.set_xlabel("Pixel Intensity")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.3)

    fig.suptitle("Pixel Intensity Distribution per Class", fontsize=13,
                 fontweight="bold")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved pixel stats → {save_path}")
    plt.show()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    np.random.seed(CONFIG["random_seed"])

    # ── Load Training/Validation Data ──
    print("\n" + "="*60)
    print("  GATE G1: EDA — Loading Training/Validation Data")
    print("="*60)

    tv_images, tv_labels = load_data(
        CONFIG["trainvalid_images"],
        CONFIG["trainvalid_labels"],
        label_col="Label",
    )
    stats = describe_images(tv_images, name="TrainValid Images")

    # ── Print Class Distribution Summary ──
    print("\n--- Class Distribution (TrainValid) ---")
    for cls in sorted(CLASS_NAMES.keys()):
        n = (tv_labels == cls).sum()
        pct = n / len(tv_labels) * 100
        print(f"  Class {cls} ({CLASS_NAMES[cls]:25s}): {n:6,}  ({pct:.1f}%)")
    print(f"  {'TOTAL':31s}: {len(tv_labels):6,}  (100.0%)")

    # ── Load Kaggle Data ──
    print("\n--- Loading Kaggle (Test) Data ---")
    kag_images, kag_labels = load_data(
        CONFIG["kaggle_images"],
        CONFIG["kaggle_labels"],
        label_col="Predicted",
    )
    describe_images(kag_images, name="Kaggle Images")

    print("\n--- Class Distribution (Kaggle — BALANCED) ---")
    for cls in sorted(CLASS_NAMES.keys()):
        n = (kag_labels == cls).sum()
        print(f"  Class {cls} ({CLASS_NAMES[cls]:25s}): {n:6,}")

    # ── Plots ──
    plot_class_distribution(
        tv_labels,
        title="Training/Validation Dataset — Class Distribution\n(Note: Class 0 is 92% of data — severe imbalance!)",
        save_path=os.path.join(CONFIG["output_dir"], "class_distribution.png"),
    )

    plot_sample_images(
        tv_images, tv_labels,
        n_per_class=CONFIG["n_samples_per_class"],
        save_path=os.path.join(CONFIG["output_dir"], "sample_images.png"),
    )

    plot_pixel_stats(
        tv_images, tv_labels,
        save_path=os.path.join(CONFIG["output_dir"], "pixel_stats.png"),
    )

    # ── Gate G1 Checklist ──
    print("\n" + "="*60)
    print("  GATE G1 CHECKLIST:")
    print(f"  [✓] Image shape:  {stats['shape']}")
    print(f"  [✓] Dtype:        {stats['dtype']}")
    print(f"  [✓] Range:        [{stats['min']:.3f}, {stats['max']:.3f}]")
    print(f"  [✓] Total samples: {len(tv_labels):,}")
    print(f"  [✓] Classes:       {sorted(np.unique(tv_labels).tolist())}")
    print("  [✓] Figures saved to output/figures/")
    print("  GATE G1: PASSED" if len(np.unique(tv_labels)) == 3 else "  GATE G1: FAILED — check class count")
    print("="*60)


if __name__ == "__main__":
    main()
