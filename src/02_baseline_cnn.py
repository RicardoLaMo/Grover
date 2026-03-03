"""
02_baseline_cnn.py — Model A: Baseline CNN (H200 GPU optimized)

Environment: PyTorch 2.9.0+cu128 | NVIDIA H200 NVL 139 GB
Data:        (11414, 60, 60, 1) float32 — native size, no resize

Architecture: 3-block CNN → AdaptiveAvgPool → FC(256) → FC(3)
  - BatchNorm after every conv (training stability)
  - Dropout(0.5) before final FC (regularization)
  - Class-weighted CrossEntropyLoss (handles 92/6/2% imbalance)
  - WeightedRandomSampler (balanced mini-batches)

H200 settings: batch_size=512, num_workers=4, pin_memory=True

Gate G3: val macro-F1 > 0.40

Run: python src/02_baseline_cnn.py
"""

import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(__file__))
from data_pipeline import get_data_loaders, CONFIG as DATA_CONFIG
from utils import setup_logging, plot_training_curves, EarlyStopping

# ─── Config ───────────────────────────────────────────────────────────────────

CONFIG = {
    "img_size":      60,     # native image size — no resize
    "n_channels":    1,
    "dropout_rate":  0.5,

    "lr":            1e-3,
    "weight_decay":  0.0,    # no L2 for baseline
    "batch_size":    512,    # H200: huge VRAM, tiny images
    "n_epochs":      40,
    "patience":      8,

    "model_save_path":  "output/models/baseline.pth",
    "figure_save_path": "output/figures/baseline_training.png",
    "trainvalid_images": "ps4_trainvalid_images-2.npy",
    "trainvalid_labels": "ps4_trainvalid_labels.csv",
    "random_seed": 42,
}


# ─── Model ────────────────────────────────────────────────────────────────────

class BaselineCNN(nn.Module):
    """
    Baseline 3-block CNN for 60×60 grayscale chest X-ray classification.

    Block structure: Conv2d(k=3) → BatchNorm → ReLU → MaxPool(2×2)
      - 3 blocks double filters each time: 32 → 64 → 128
      - AdaptiveAvgPool(4,4) → size-agnostic, avoids Flatten dimension issues
      - FC(2048→256) + Dropout(0.5) → FC(256→3) raw logits

    Why this depth? 60×60 → after 3 MaxPools: 7×7 feature maps, 128 channels.
    AdaptiveAvgPool(4,4) then outputs 128×4×4=2048 features — sufficient for
    a simple 3-class problem.
    """

    def __init__(self, n_channels: int = 1, dropout_rate: float = 0.5, n_classes: int = 3):
        super().__init__()

        # Each block: Conv → BatchNorm → ReLU → MaxPool(2)
        # BatchNorm normalizes activations across the batch, accelerating
        # convergence and reducing sensitivity to weight initialization.
        self.block1 = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2))   # 60→30

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2))   # 30→15

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2))   # 15→7

        # AdaptiveAvgPool: maps any spatial size to a fixed (4,4) grid
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, n_classes))   # raw logits; CrossEntropyLoss adds softmax

    def forward(self, x):
        return self.classifier(self.pool(self.block3(self.block2(self.block1(x)))))


# ─── Train / Eval helpers ─────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, loss_fn, device):
    """One training epoch: forward → weighted loss → backward → optimizer step."""
    model.train()
    total = 0.0
    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)   # slightly faster than zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
    return total / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device):
    """Val/test pass: returns (avg_loss, macro_f1, y_true, y_pred)."""
    model.eval()
    total, preds, truths = 0.0, [], []
    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(X)
        total += loss_fn(logits, y).item() * len(y)
        preds.extend(logits.argmax(1).cpu().tolist())
        truths.extend(y.cpu().tolist())
    f1 = f1_score(truths, preds, average="macro", zero_division=0)
    return total / len(loader.dataset), f1, np.array(truths), np.array(preds)


# ─── Training pipeline ────────────────────────────────────────────────────────

def train_baseline(config=None):
    """Full training pipeline for Model A. Returns result dict."""
    if config is None:
        config = CONFIG

    logger = setup_logging()
    torch.manual_seed(config["random_seed"])
    np.random.seed(config["random_seed"])
    torch.backends.cudnn.benchmark = True   # auto-tune convolution algorithms on H200

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})")

    # ── Data ──
    images = np.load(config["trainvalid_images"])
    labels = pd.read_csv(config["trainvalid_labels"])["Label"].values

    dcfg = DATA_CONFIG.copy()
    dcfg.update({"batch_size": config["batch_size"], "img_size": config["img_size"]})
    train_l, val_l, test_l, cw, si = get_data_loaders(images, labels, dcfg,
                                                        n_channels=config["n_channels"])

    # ── Model, loss, optimizer ──
    model = BaselineCNN(n_channels=config["n_channels"],
                        dropout_rate=config["dropout_rate"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model A (Baseline): {n_params:,} parameters")

    # CrossEntropyLoss with class weights — rare classes penalized more heavily
    loss_fn   = nn.CrossEntropyLoss(weight=cw.to(device))
    optimizer = optim.Adam(model.parameters(), lr=config["lr"],
                           weight_decay=config["weight_decay"])
    early_stop = EarlyStopping(patience=config["patience"], mode="max",
                                save_path=config["model_save_path"])

    train_losses, val_losses, val_f1s = [], [], []
    logger.info("\n  === Training Baseline CNN (Model A) ===")
    logger.info(f"  {'Epoch':>6}  {'Train Loss':>11}  {'Val Loss':>9}  {'Val Macro-F1':>13}")

    for epoch in range(1, config["n_epochs"] + 1):
        tl = train_epoch(model, train_l, optimizer, loss_fn, device)
        vl, vf1, _, _ = eval_epoch(model, val_l, loss_fn, device)
        train_losses.append(tl); val_losses.append(vl); val_f1s.append(vf1)
        logger.info(f"  {epoch:6d}  {tl:11.4f}  {vl:9.4f}  {vf1:13.4f}")
        if early_stop(vf1, model):
            logger.info(f"  Early stopping at epoch {epoch}."); break

    os.makedirs("output/figures", exist_ok=True)
    plot_training_curves(train_losses, val_losses, val_f1s,
                         title="Model A — Baseline CNN Training History",
                         save_path=config["figure_save_path"])

    logger.info(f"\n  Best Val Macro-F1 (Baseline): {early_stop.best:.4f}")
    gate = "PASSED ✓" if early_stop.best > 0.40 else "NOT MET ✗"
    logger.info(f"  GATE G3 (val macro-F1 > 0.40): {gate}")

    # Reload best checkpoint
    model.load_state_dict(torch.load(config["model_save_path"], map_location=device))

    return {"best_val_f1": early_stop.best, "model": model, "device": device,
            "split_info": si, "train_losses": train_losses,
            "val_losses": val_losses, "val_f1s": val_f1s}


if __name__ == "__main__":
    r = train_baseline()
    print(f"\nBaseline done. Best Val Macro-F1: {r['best_val_f1']:.4f}")
