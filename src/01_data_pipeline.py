"""
01_data_pipeline.py — Data Loading, Splitting, and DataLoader Construction

Environment: Python 3.13 | PyTorch 2.9.0+cu128 | 8× NVIDIA H200 NVL (139 GB each)
Data spec:   images (11414, 60, 60, 1) float32 [0,1] | labels 0/1/2

Key design decisions:
  1. STRATIFIED split — preserves class ratios (92/6/2%) across train/val/test.
  2. CLASS WEIGHTS — inverse-frequency weighting for CrossEntropyLoss.
  3. WeightedRandomSampler — balanced mini-batches expose model to rare classes.
  4. Augmentation only during training; val/test use clean images.
  5. H200-optimized: batch_size=512 (tiny 60x60 images, 139 GB VRAM).

Run: python src/01_data_pipeline.py  (sanity check → Gate G2)
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split

# ─── Configuration ────────────────────────────────────────────────────────────

CONFIG = {
    # Paths (relative to hw4/ root)
    "trainvalid_images": "ps4_trainvalid_images-2.npy",
    "trainvalid_labels": "ps4_trainvalid_labels.csv",
    "kaggle_images":     "ps4_kaggle_images-1.npy",
    "kaggle_labels":     "ps4_kaggle_labels.csv",

    # Split: 70% train / 15% val / 15% test (stratified)
    "val_size":  0.15,
    "test_size": 0.15,

    # H200-optimized: 60×60 images fit thousands per batch in 139 GB VRAM
    "batch_size":  512,
    "num_workers": 4,

    # Native image size — no resize needed for custom CNNs
    # Use 224 for pretrained ResNet (override via data_cfg)
    "img_size": 60,

    "use_weighted_sampler": True,
    "random_seed": 42,
}

CLASS_NAMES = {0: "Healthy", 1: "Pre-existing", 2: "Effusion/Mass"}


# ─── Dataset ──────────────────────────────────────────────────────────────────

class XRayDataset(Dataset):
    """
    PyTorch Dataset wrapping a numpy array of 60×60 chest X-ray images.

    Images arrive as (N, 60, 60, 1) float32 in [0,1] — already normalized.
    We convert to PyTorch channel-first (C, H, W) format and optionally
    apply torchvision transforms for augmentation or resizing.

    Args:
        images:     numpy array (N, H, W, 1) or (N, H, W), float32.
        labels:     numpy array (N,), integer class labels 0/1/2.
        transform:  torchvision transform pipeline.
        n_channels: 1 = grayscale, 3 = replicate for pretrained RGB models.
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray,
                 transform=None, n_channels: int = 1):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.n_channels = n_channels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]  # (H, W, 1) float32 already in [0,1]

        # Remove trailing channel dim → (H, W)
        if img.ndim == 3:
            img = img[:, :, 0] if img.shape[-1] == 1 else img[0]

        # Add channel dim for PyTorch: (H, W) → (1, H, W)
        img = img[np.newaxis, :, :]    # shape (1, H, W)

        # Replicate to 3 channels if needed (for ImageNet-pretrained models)
        if self.n_channels == 3:
            img = np.repeat(img, 3, axis=0)   # (3, H, W)

        img_tensor = torch.from_numpy(img)     # float32 tensor

        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor, torch.tensor(self.labels[idx], dtype=torch.long)


# ─── Transforms ───────────────────────────────────────────────────────────────

def get_transforms(img_size: int, augment: bool = False, n_channels: int = 1):
    """
    Build a torchvision transform pipeline.

    Training (augment=True): random flips, rotations, brightness jitter.
      These create synthetic variations to prevent overfitting, especially
      beneficial for the minority classes (Class 1 and 2).

    Val/Test (augment=False): only resize (if needed) + normalize.
      Must be identical to training preprocessing minus augmentation.

    Args:
        img_size:   Target spatial size (H=W). 60 for custom CNNs, 224 for ResNet.
        augment:    Apply augmentation (training only).
        n_channels: 1 for grayscale, 3 for RGB pretrained models.
    """
    mean = [0.5] * n_channels   # images already [0,1]; center to [-1,1]
    std  = [0.5] * n_channels

    ops = []
    if img_size != 60:
        # Only resize if model requires a different size (e.g., ResNet → 224)
        ops.append(transforms.Resize((img_size, img_size), antialias=True))

    if augment:
        ops += [
            transforms.RandomHorizontalFlip(p=0.5),      # L-R symmetric X-rays
            transforms.RandomRotation(degrees=15),         # slight scanner tilt
            transforms.ColorJitter(brightness=0.2, contrast=0.2),  # scanner variability
        ]

    ops.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(ops)


# ─── Class Weights ────────────────────────────────────────────────────────────

def compute_class_weights(labels: np.ndarray, n_classes: int = 3) -> torch.Tensor:
    """
    Inverse-frequency class weights: weight_c = N / (K * N_c).

    Without weighting, CrossEntropyLoss treats all samples equally,
    so the model is overwhelmingly pushed to predict Class 0 (92% of data).
    Weighting makes errors on rare classes cost proportionally more.
    """
    n = len(labels)
    weights = []
    for c in range(n_classes):
        n_c = int((labels == c).sum())
        w = n / (n_classes * n_c)
        weights.append(w)
        print(f"  Class {c} ({n_c:6,} samples) → weight = {w:.4f}")
    return torch.tensor(weights, dtype=torch.float32)


# ─── Main DataLoader Builder ──────────────────────────────────────────────────

def get_data_loaders(images: np.ndarray, labels: np.ndarray,
                     config: dict = None, n_channels: int = 1):
    """
    Stratified 70/15/15 train/val/test split + DataLoaders.

    Returns: train_loader, val_loader, test_loader, class_weights, split_info
    """
    if config is None:
        config = CONFIG

    img_size   = config["img_size"]
    batch_size = config["batch_size"]
    seed       = config["random_seed"]

    idx = np.arange(len(labels))

    # ── Stratified split: preserves 92/6/2% in every fold ──
    idx_train, idx_valtest = train_test_split(
        idx, test_size=config["val_size"] + config["test_size"],
        stratify=labels, random_state=seed)

    test_frac = config["test_size"] / (config["val_size"] + config["test_size"])
    idx_val, idx_test = train_test_split(
        idx_valtest, test_size=test_frac,
        stratify=labels[idx_valtest], random_state=seed)

    print("\n--- Stratified Data Split ---")
    for name, idx_s in [("Train", idx_train), ("Val", idx_val), ("Test", idx_test)]:
        sl = labels[idx_s]
        dist = " | ".join(f"C{c}={int((sl==c).sum())}" for c in range(3))
        print(f"  {name:5s}: {len(idx_s):5,}  [{dist}]")

    # ── Datasets ──
    tr_ds = XRayDataset(images[idx_train], labels[idx_train],
                        get_transforms(img_size, augment=True,  n_channels=n_channels),
                        n_channels)
    va_ds = XRayDataset(images[idx_val],   labels[idx_val],
                        get_transforms(img_size, augment=False, n_channels=n_channels),
                        n_channels)
    te_ds = XRayDataset(images[idx_test],  labels[idx_test],
                        get_transforms(img_size, augment=False, n_channels=n_channels),
                        n_channels)

    # ── Class weights ──
    print("\n--- Class Weights (inverse-frequency) ---")
    class_weights = compute_class_weights(labels[idx_train])

    # ── WeightedRandomSampler: balanced mini-batches during training ──
    sampler = None
    if config.get("use_weighted_sampler", True):
        tl = labels[idx_train]
        counts = [(tl == c).sum() for c in range(3)]
        sw = torch.tensor([1.0 / counts[l] for l in tl], dtype=torch.float32)
        sampler = WeightedRandomSampler(sw, num_samples=len(sw), replacement=True)
        print("  WeightedRandomSampler: ON (balanced mini-batches)")

    # ── pin_memory=True: async CPU→GPU transfers on H200 ──
    kw = dict(num_workers=config["num_workers"], pin_memory=True)
    train_loader = DataLoader(tr_ds, batch_size=batch_size,
                              sampler=sampler, shuffle=(sampler is None), **kw)
    val_loader   = DataLoader(va_ds, batch_size=batch_size, shuffle=False, **kw)
    test_loader  = DataLoader(te_ds, batch_size=batch_size, shuffle=False, **kw)

    split_info = {"idx_train": idx_train, "idx_val": idx_val, "idx_test": idx_test,
                  "n_train": len(idx_train), "n_val": len(idx_val), "n_test": len(idx_test)}

    return train_loader, val_loader, test_loader, class_weights, split_info


def get_kaggle_loader(images: np.ndarray, config: dict = None, n_channels: int = 1):
    """DataLoader for Kaggle test set (dummy labels, no augmentation)."""
    if config is None:
        config = CONFIG
    dummy = np.zeros(len(images), dtype=np.int64)
    ds = XRayDataset(images, dummy,
                     get_transforms(config["img_size"], augment=False, n_channels=n_channels),
                     n_channels)
    return DataLoader(ds, batch_size=config["batch_size"], shuffle=False,
                      num_workers=config["num_workers"], pin_memory=True)


def get_balanced_test_loader(images, labels, idx_test, config=None, n_channels=1):
    """
    Balanced test DataLoader: undersample to equal count per class.
    Reports how performance changes when all 3 classes are equally represented —
    directly comparable to the Kaggle submission scoring.
    """
    if config is None:
        config = CONFIG
    tl = labels[idx_test]
    min_n = min((tl == c).sum() for c in range(3))
    print(f"\n  Balanced test: {min_n} per class = {min_n*3} total")

    np.random.seed(config["random_seed"])
    bal_idx = []
    for c in range(3):
        pool = idx_test[tl == c]
        bal_idx.extend(np.random.choice(pool, size=min_n, replace=False).tolist())

    bal_idx = np.array(bal_idx)
    ds = XRayDataset(images[bal_idx], labels[bal_idx],
                     get_transforms(config["img_size"], augment=False, n_channels=n_channels),
                     n_channels)
    return DataLoader(ds, batch_size=config["batch_size"], shuffle=False,
                      num_workers=config["num_workers"], pin_memory=True)


# ─── Sanity check ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  GATE G2: Data Pipeline Validation")
    print("=" * 60)

    images = np.load(CONFIG["trainvalid_images"])
    labels = pd.read_csv(CONFIG["trainvalid_labels"])["Label"].values
    print(f"Images: {images.shape} {images.dtype}  Labels: {labels.shape}")

    train_l, val_l, test_l, cw, si = get_data_loaders(images, labels)
    X, y = next(iter(train_l))
    print(f"\nBatch: X={X.shape} {X.dtype}  y={y.shape}")
    print(f"X range: [{X.min():.3f}, {X.max():.3f}]")
    print(f"Class weights: {cw.tolist()}")

    assert X.ndim == 4 and X.shape[1] == 1
    assert y.ndim == 1
    print("\n  GATE G2: PASSED ✓")
