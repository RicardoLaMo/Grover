"""
03_improved_cnn.py — Models B & C: Improved CNNs (H200 GPU optimized)

Environment: PyTorch 2.9.0+cu128 | NVIDIA H200 NVL 139 GB
Data:        (11414, 60, 60, 1) float32 [0,1]

Model B — Deep Residual CNN (60×60, 1-channel input)
  4 residual blocks (64→128→256→512), GlobalAvgPool, 2xFC+Dropout, L2 reg
  Skip connections prevent vanishing gradients in deeper networks.

Model C — Transfer Learning: pretrained ResNet18 (224×224, 3-channel)
  Phase 1 (warm-up): freeze backbone, train head only (epochs 1-10)
  Phase 2 (fine-tune): unfreeze all, tiny LR for backbone

Gate G4: best val macro-F1 > 0.60

Run: python src/03_improved_cnn.py
"""

import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torchvision import models

sys.path.insert(0, os.path.dirname(__file__))
from data_pipeline import get_data_loaders, CONFIG as DATA_CONFIG
from utils import setup_logging, plot_training_curves, EarlyStopping

# ─── Configs ──────────────────────────────────────────────────────────────────

CONFIG_B = {
    "model_name":   "Model_B_ResidualCNN",
    "img_size":     60,     # native resolution
    "n_channels":   1,
    "dropout_rate": 0.5,
    "lr":           3e-4,
    "weight_decay": 1e-4,   # L2 regularization
    "batch_size":   512,    # H200 optimized
    "n_epochs":     60,
    "patience":     10,
    "sched_patience": 5,    # ReduceLROnPlateau patience
    "model_save_path":  "output/models/improved_B.pth",
    "figure_save_path": "output/figures/model_B_training.png",
    "trainvalid_images": "ps4_trainvalid_images-2.npy",
    "trainvalid_labels": "ps4_trainvalid_labels.csv",
    "random_seed": 42,
}

CONFIG_C = {
    "model_name":      "Model_C_ResNet18",
    "img_size":        224,   # ResNet18 pretrained at 224×224
    "n_channels":      3,     # ImageNet models expect RGB
    "lr_backbone":     1e-4,  # low LR for pretrained layers
    "lr_head":         1e-3,
    "weight_decay":    1e-4,
    "batch_size":      256,   # larger images, still fast on H200
    "n_epochs":        40,
    "unfreeze_epoch":  10,    # phase 2 starts at this epoch
    "patience":        8,
    "model_save_path":  "output/models/improved_C.pth",
    "figure_save_path": "output/figures/model_C_training.png",
    "trainvalid_images": "ps4_trainvalid_images-2.npy",
    "trainvalid_labels": "ps4_trainvalid_labels.csv",
    "random_seed": 42,
}


# ─── Model B: Residual Block + Deep CNN ──────────────────────────────────────

class ResidualBlock(nn.Module):
    """
    Residual block: output = ReLU(F(x) + skip(x))

    Instead of learning H(x) directly, we learn the residual F(x) = H(x) - x.
    The skip connection (identity or 1×1 proj) allows gradients to flow
    directly to earlier layers, solving the vanishing gradient problem in
    deep networks and enabling effective training of 4+ block networks.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch))
        # Project skip connection when channels change
        self.skip = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch)
        ) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.main(x) + self.skip(x))


class ImprovedCNN_B(nn.Module):
    """
    Model B: 4-block Residual CNN with strong regularization.

    60×60 → Block1(64)+Pool → 30×30 → Block2(128)+Pool → 15×15
         → Block3(256)+Pool → 7×7  → Block4(512)+Pool → 3×3
         → GlobalAvgPool → (512,) → FC(256)+Dropout(0.5) → FC(128)+Dropout(0.3) → FC(3)

    Regularization: BatchNorm (all blocks) + Dropout (FC) + L2 weight decay.
    """
    def __init__(self, n_channels: int = 1, dropout_rate: float = 0.5, n_classes: int = 3):
        super().__init__()
        self.layer1 = nn.Sequential(ResidualBlock(n_channels, 64),   nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(ResidualBlock(64, 128),          nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(ResidualBlock(128, 256),         nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(ResidualBlock(256, 512),         nn.MaxPool2d(2))
        self.gap    = nn.AdaptiveAvgPool2d(1)
        self.head   = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(dropout_rate),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(dropout_rate * 0.6),
            nn.Linear(128, n_classes))

    def forward(self, x):
        return self.head(self.gap(self.layer4(self.layer3(self.layer2(self.layer1(x))))))


# ─── Model C: Pretrained ResNet18 ─────────────────────────────────────────────

def build_resnet18(n_classes: int = 3, freeze: bool = True):
    """
    ResNet18 pretrained on ImageNet, adapted for chest X-ray classification.

    Transfer learning rationale: ImageNet pretraining gives the model
    low-level feature detectors (edges, gradients, textures) that transfer
    well to medical images, even though domains differ. This is especially
    valuable with our small minority-class sample count.

    Grayscale → RGB: XRayDataset replicates the 1-channel image to 3 channels,
    so ResNet18's first conv layer receives valid input.
    """
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    if freeze:
        for p in m.parameters():
            p.requires_grad = False
    # Replace final FC: 512 → 3 classes (this is always trainable)
    m.fc = nn.Sequential(
        nn.Linear(m.fc.in_features, 256), nn.ReLU(inplace=True),
        nn.Dropout(0.5), nn.Linear(256, n_classes))
    return m


# ─── Shared train/eval ────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total = 0.0
    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(X), y)
        loss.backward(); optimizer.step()
        total += loss.item() * len(y)
    return total / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total, preds, truths = 0.0, [], []
    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(X)
        total += loss_fn(logits, y).item() * len(y)
        preds.extend(logits.argmax(1).cpu().tolist())
        truths.extend(y.cpu().tolist())
    return total / len(loader.dataset), \
           f1_score(truths, preds, average="macro", zero_division=0), \
           np.array(truths), np.array(preds)


# ─── Train Model B ────────────────────────────────────────────────────────────

def train_model_B(config=None):
    if config is None: config = CONFIG_B
    logger = setup_logging()
    torch.manual_seed(config["random_seed"]); np.random.seed(config["random_seed"])
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"\n  === Training {config['model_name']} on {device} ===")

    images = np.load(config["trainvalid_images"])
    labels = pd.read_csv(config["trainvalid_labels"])["Label"].values

    dcfg = DATA_CONFIG.copy()
    dcfg.update({"batch_size": config["batch_size"], "img_size": config["img_size"]})
    train_l, val_l, _, cw, si = get_data_loaders(images, labels, dcfg,
                                                   n_channels=config["n_channels"])

    model     = ImprovedCNN_B(n_channels=config["n_channels"],
                              dropout_rate=config["dropout_rate"]).to(device)
    loss_fn   = nn.CrossEntropyLoss(weight=cw.to(device))
    optimizer = optim.Adam(model.parameters(), lr=config["lr"],
                           weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=config["sched_patience"])
    early_stop = EarlyStopping(patience=config["patience"], mode="max",
                                save_path=config["model_save_path"])

    logger.info(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"  {'Epoch':>6}  {'Train Loss':>11}  {'Val Loss':>9}  {'Val F1':>7}  {'LR':>8}")

    train_losses, val_losses, val_f1s = [], [], []
    for epoch in range(1, config["n_epochs"] + 1):
        tl = train_epoch(model, train_l, optimizer, loss_fn, device)
        vl, vf1, _, _ = eval_epoch(model, val_l, loss_fn, device)
        train_losses.append(tl); val_losses.append(vl); val_f1s.append(vf1)
        lr_now = optimizer.param_groups[0]["lr"]
        logger.info(f"  {epoch:6d}  {tl:11.4f}  {vl:9.4f}  {vf1:7.4f}  {lr_now:.2e}")
        scheduler.step(vf1)
        if early_stop(vf1, model): logger.info(f"  Early stop @ epoch {epoch}."); break

    plot_training_curves(train_losses, val_losses, val_f1s,
                         title=f"{config['model_name']} Training History",
                         save_path=config["figure_save_path"])
    logger.info(f"  Best Val Macro-F1 (B): {early_stop.best:.4f}")

    model.load_state_dict(torch.load(config["model_save_path"], map_location=device))
    return {"best_val_f1": early_stop.best, "model": model, "device": device,
            "split_info": si, "train_losses": train_losses,
            "val_losses": val_losses, "val_f1s": val_f1s}


# ─── Train Model C ────────────────────────────────────────────────────────────

def train_model_C(config=None):
    """
    Two-phase fine-tuning of ResNet18.
    Phase 1: backbone frozen, train head only → avoids corrupting ImageNet features.
    Phase 2: unfreeze all, backbone LR 10× smaller → careful fine-tuning.
    """
    if config is None: config = CONFIG_C
    logger = setup_logging()
    torch.manual_seed(config["random_seed"]); np.random.seed(config["random_seed"])
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"\n  === Training {config['model_name']} on {device} ===")

    images = np.load(config["trainvalid_images"])
    labels = pd.read_csv(config["trainvalid_labels"])["Label"].values

    dcfg = DATA_CONFIG.copy()
    dcfg.update({"batch_size": config["batch_size"], "img_size": config["img_size"]})
    train_l, val_l, _, cw, si = get_data_loaders(images, labels, dcfg, n_channels=3)

    # Phase 1: head only
    model   = build_resnet18(n_classes=3, freeze=True).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=cw.to(device))
    optimizer = optim.Adam(model.fc.parameters(), lr=config["lr_head"],
                           weight_decay=config["weight_decay"])
    early_stop = EarlyStopping(patience=config["patience"], mode="max",
                                save_path=config["model_save_path"])

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Phase 1: {trainable:,} trainable params (head only)")
    logger.info(f"  {'Epoch':>6}  {'Train Loss':>11}  {'Val Loss':>9}  {'Val F1':>7}")

    train_losses, val_losses, val_f1s = [], [], []
    unfrozen = False

    for epoch in range(1, config["n_epochs"] + 1):
        # Switch to full fine-tuning at unfreeze_epoch
        if epoch == config["unfreeze_epoch"] and not unfrozen:
            for p in model.parameters():
                p.requires_grad = True
            optimizer = optim.Adam([
                {"params": [p for n, p in model.named_parameters() if "fc" not in n],
                 "lr": config["lr_backbone"]},
                {"params": model.fc.parameters(), "lr": config["lr_head"]},
            ], weight_decay=config["weight_decay"])
            logger.info(f"\n  === Phase 2: full fine-tune from epoch {epoch} ===")
            logger.info(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")
            unfrozen = True

        tl = train_epoch(model, train_l, optimizer, loss_fn, device)
        vl, vf1, _, _ = eval_epoch(model, val_l, loss_fn, device)
        train_losses.append(tl); val_losses.append(vl); val_f1s.append(vf1)
        logger.info(f"  {epoch:6d}  {tl:11.4f}  {vl:9.4f}  {vf1:7.4f}")
        if early_stop(vf1, model): logger.info(f"  Early stop @ epoch {epoch}."); break

    plot_training_curves(train_losses, val_losses, val_f1s,
                         title=f"{config['model_name']} Training History",
                         save_path=config["figure_save_path"])
    logger.info(f"  Best Val Macro-F1 (C): {early_stop.best:.4f}")

    model.load_state_dict(torch.load(config["model_save_path"], map_location=device))
    return {"best_val_f1": early_stop.best, "model": model, "device": device,
            "split_info": si, "train_losses": train_losses,
            "val_losses": val_losses, "val_f1s": val_f1s}


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger = setup_logging()
    logger.info("\n" + "="*60)
    logger.info("  GATE G4: Improved Model Training")
    logger.info("="*60)

    res_B = train_model_B()
    res_C = train_model_C()

    best = max(res_B["best_val_f1"], res_C["best_val_f1"])
    logger.info(f"\n  Model B: {res_B['best_val_f1']:.4f}  |  Model C: {res_C['best_val_f1']:.4f}")
    gate = "PASSED ✓" if best > 0.60 else "NOT MET — best effort documented"
    logger.info(f"  GATE G4 (best macro-F1 > 0.60): {gate}")
