"""
run_champion_challenger.py — Champion vs Challenger: 10-Iteration Deep CNN Comparison
=======================================================================================
Platform : PyTorch 2.9.0+cu128 | NVIDIA H200 NVL 139 GB | Python 3.13
Data     : (11414, 60, 60, 1) float32 [0,1] | 3-class (92/6/2% imbalance)

Champion : ResNet18 (Model C) — Val Macro-F1=0.6976, Kaggle F1=0.723

10 Research-Grounded Challengers
----------------------------------
C01  EfficientNet-B0       Tan & Le, NeurIPS 2019 — compound scaling; 98% CXR accuracy
C02  DenseNet121           Rajpurkar et al., 2017 — CheXNet, chest-X-ray validated; 7.98M params
C03  EfficientNet-B4       Tan & Le, NeurIPS 2019 — 99.79% medical imaging accuracy; 19M params
C04  MobileNetV3-Large     Howard et al., ICCV 2019 — SE channel attention + Hard-Swish; 5.4M params
C05  ResNet50              He et al., CVPR 2016 — bottleneck blocks, 25.5M params (2× capacity)
C06  ConvNeXt-Tiny         Liu et al., CVPR 2022 — modernized CNN; 99.5% brain tumor 2024
C07  ResNet18 + FocalLoss  Lin et al., ICCV 2017 — γ=2 focuses on hard/rare examples
C08  EfficientNet-B0 + RandAugment  Cubuk et al., CVPR 2020 — best augmentation for chest X-ray
C09  ResNet18 + LabelSmooth + CosLR  LS+, MICCAI 2024 — calibration-aware training
C10  Ensemble(ResNet18 + EfficientNet-B0 + DenseNet121) — soft voting (+4-8% expected gain)

New Metric: PR-AUC (Precision-Recall AUC, macro-averaged)
------------------------------------------------------------
WHY: On a 92/6/2% imbalanced dataset, ROC-AUC can be inflated by the majority
class's high True Negative Rate. PR-AUC focuses on precision-recall trade-offs
for each class independently (one-vs-rest), measuring minority-class detection
quality directly. A random classifier has PR-AUC ≈ class_frequency (not 0.5),
making PR-AUC especially informative for Class 1 (5.7%) and Class 2 (2.3%).
Report: PR-AUC is most meaningful on the balanced test set; report both.

Research citations:
  - ROC vs PR-AUC: PMC11240176 (2024), MachineLearningMastery
  - Focal Loss: Lin et al. ICCV 2017, LMFLOSS arXiv 2212.12741
  - Ensemble gains: +4-8% (averaging), +10-13% (stacking) — PMC2201.11440
  - RandAugment for CXR: MDPI 2022/4/915 (best single augmentation strategy)
  - EfficientNet medical: PMC12453788 (99.79% EfficientNet-B4)
  - CheXNet/DenseNet121: Rajpurkar et al. arXiv 1711.05225

Run: python run_champion_challenger.py  (from hw4/ root)
"""

import os, json, logging, warnings, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import (f1_score, accuracy_score, roc_auc_score,
                              average_precision_score, classification_report,
                              confusion_matrix)
from sklearn.preprocessing import label_binarize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

# ─── Directories & Logging ────────────────────────────────────────────────────
for d in ["output/figures", "output/models", "output/predictions"]:
    os.makedirs(d, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("output/champion_challenger.log", "w"),
    ],
)
log = logging.getLogger("cc")

# ─── Global Config ────────────────────────────────────────────────────────────
SEED       = 42
VAL_FRAC   = 0.15
TEST_FRAC  = 0.15
NUM_WKRS   = 4
IMG_SIZE   = 224      # all pretrained models use 224×224
BATCH      = 256      # H200 optimised for 224×224 3-ch images
N_EPOCHS   = 40       # max epochs per challenger
PATIENCE   = 8        # early-stopping patience (val macro-F1)
UNFREEZE_EP = 10      # phase-2 starts here for all 2-phase models
N_CLS      = 3

CLASS_NAMES = ["Healthy(0)", "Pre-existing(1)", "Effusion/Mass(2)"]
COLORS      = ["#2196F3", "#FF9800", "#F44336"]

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
log.info(f"Device: {device} | "
         f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

ALL_RESULTS = {}  # name → metrics dict (written to JSON at end)

# ════════════════════════════════════════════════════════════════════════════
# DATA PIPELINE
# ════════════════════════════════════════════════════════════════════════════

class XRayDataset(Dataset):
    """Chest X-ray Dataset: numpy (N,H,W,1) → float32 tensor (n_ch,H,W)."""
    def __init__(self, imgs, lbls, transform=None, n_ch=3):
        self.imgs = imgs; self.lbls = lbls
        self.transform = transform; self.n_ch = n_ch

    def __len__(self): return len(self.lbls)

    def __getitem__(self, i):
        img = self.imgs[i]
        if img.ndim == 3:
            img = img[:, :, 0] if img.shape[-1] == 1 else img[0]
        img = img[np.newaxis, :, :]                     # (1, H, W)
        if self.n_ch == 3:
            img = np.repeat(img, 3, axis=0)             # (3, H, W)
        t = torch.from_numpy(img)
        if self.transform:
            t = self.transform(t)
        return t, torch.tensor(self.lbls[i], dtype=torch.long)


def make_transforms(img_size=224, augment=False, n_ch=3, randaug=False):
    """
    Build torchvision transform pipeline.
      augment  : training (flip + rotation + jitter)
      randaug  : additionally apply RandAugment (for C08)
                 Cubuk et al. CVPR 2020 — best single augmentation for chest X-ray.
                 num_ops=2, magnitude=9 (conservative for 60→224 upscale scenario).
    """
    mean, std = [0.5] * n_ch, [0.5] * n_ch
    ops = []
    if img_size != 60:
        ops.append(transforms.Resize((img_size, img_size), antialias=True))
    if augment:
        ops += [
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
        if randaug:
            # RandAugment: N=2 ops per image, magnitude M=9 (scale 0-30).
            # FIX: RandAugment's Solarize op expects uint8 input (0-255).
            # Our float tensor is in [0,1] → convert to uint8 → RandAugment → back to float.
            # Cubuk et al. CVPR 2020: best single augmentation for CXR classification.
            ops += [
                transforms.Lambda(lambda x: (x.clamp(0, 1) * 255).to(torch.uint8)),
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.Lambda(lambda x: x.to(torch.float32) / 255.0),
            ]
    ops.append(transforms.Normalize(mean, std))
    return transforms.Compose(ops)


def make_loaders(imgs, lbls, img_size=IMG_SIZE, batch=BATCH, n_ch=3,
                 randaug=False):
    """Stratified 70/15/15 split + WeightedRandomSampler for training."""
    idx = np.arange(len(lbls))
    i_tr, i_vt = train_test_split(idx, test_size=VAL_FRAC + TEST_FRAC,
                                   stratify=lbls, random_state=SEED)
    i_va, i_te = train_test_split(
        i_vt, test_size=TEST_FRAC / (VAL_FRAC + TEST_FRAC),
        stratify=lbls[i_vt], random_state=SEED)

    # Inverse-frequency class weights: w_c = N / (K * N_c)
    n_total = len(i_tr)
    cw = [n_total / (N_CLS * int((lbls[i_tr] == c).sum())) for c in range(N_CLS)]
    cw_t = torch.tensor(cw, dtype=torch.float32)

    # WeightedRandomSampler: balanced mini-batches expose model to rare classes
    sw = torch.tensor([1.0 / int((lbls[i_tr] == l).sum()) for l in lbls[i_tr]])
    sampler = WeightedRandomSampler(sw, len(sw), replacement=True)

    tr_ds = XRayDataset(imgs[i_tr], lbls[i_tr],
                        make_transforms(img_size, True,  n_ch, randaug), n_ch)
    va_ds = XRayDataset(imgs[i_va], lbls[i_va],
                        make_transforms(img_size, False, n_ch), n_ch)
    te_ds = XRayDataset(imgs[i_te], lbls[i_te],
                        make_transforms(img_size, False, n_ch), n_ch)

    kw = dict(num_workers=NUM_WKRS, pin_memory=True)
    tr_l = DataLoader(tr_ds, batch, sampler=sampler, **kw)
    va_l = DataLoader(va_ds, batch, shuffle=False, **kw)
    te_l = DataLoader(te_ds, batch, shuffle=False, **kw)
    return tr_l, va_l, te_l, cw_t, {"i_tr": i_tr, "i_va": i_va, "i_te": i_te}


def balanced_loader(imgs, lbls, i_te, img_size=IMG_SIZE, batch=BATCH, n_ch=3):
    """Subsample test indices to equal count per class (mirrors Kaggle balance)."""
    tl  = lbls[i_te]
    mn  = min((tl == c).sum() for c in range(N_CLS))
    np.random.seed(SEED)
    bi  = []
    for c in range(N_CLS):
        bi.extend(np.random.choice(i_te[tl == c], mn, replace=False).tolist())
    bi  = np.array(bi)
    ds  = XRayDataset(imgs[bi], lbls[bi],
                      make_transforms(img_size, False, n_ch), n_ch)
    return DataLoader(ds, batch, shuffle=False,
                      num_workers=NUM_WKRS, pin_memory=True)


# ─── Load Data ────────────────────────────────────────────────────────────────
log.info("Loading data...")
images     = np.load("ps4_trainvalid_images-2.npy")
labels     = pd.read_csv("ps4_trainvalid_labels.csv")["Label"].values
kag_images = np.load("ps4_kaggle_images-1.npy")
kag_labels = pd.read_csv("ps4_kaggle_labels.csv")["Predicted"].values
log.info(f"  TrainValid: {images.shape}  Kaggle: {kag_images.shape}")

# Standard loaders (used by most models)
tr_l, va_l, te_l, cw, splits = make_loaders(images, labels)
te_l_bal = balanced_loader(images, labels, splits["i_te"])
log.info(f"  Split: train={len(splits['i_tr'])} val={len(splits['i_va'])} "
         f"test={len(splits['i_te'])}  cw={[f'{w:.3f}' for w in cw.tolist()]}")

# RandAugment loaders (C08 only)
tr_l_ra, _, _, _, _ = make_loaders(images, labels, randaug=True)

# ════════════════════════════════════════════════════════════════════════════
# LOSS FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal Loss — Lin et al., "Focal Loss for Dense Object Detection," ICCV 2017.

    L_focal = -(1 - p_t)^γ * log(p_t)

    The (1-p_t)^γ modulation factor:
      • Easy examples (p_t → 1, majority class 0): factor → 0 → small gradient
      • Hard examples (p_t → 0, minority classes 1,2): factor → 1 → full gradient

    γ=2.0 (standard). Combined with inverse-frequency class weights for
    double-addressing of imbalance. 2024 research (LMFLOSS, arXiv 2212.12741)
    shows focal variants consistently yield +2-9% macro-F1 over weighted CE.
    """
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma  = gamma

    def forward(self, inputs, targets):
        ce  = F.cross_entropy(inputs, targets, weight=self.weight, reduction="none")
        pt  = torch.exp(-ce)                      # predicted probability of true class
        return ((1 - pt) ** self.gamma * ce).mean()


# ════════════════════════════════════════════════════════════════════════════
# HEAD REPLACEMENT UTILITIES
# ════════════════════════════════════════════════════════════════════════════

def replace_head(model, arch, n_cls=N_CLS):
    """
    Replace the final classification layer with a custom 2-layer head:
      backbone_features → FC(256) → ReLU → Dropout(0.5) → FC(n_cls)
    Handles the different head names across torchvision architectures.
    """
    if arch in ("resnet18", "resnet50"):
        in_f = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_f, 256), nn.ReLU(True),
            nn.Dropout(0.5), nn.Linear(256, n_cls))

    elif arch == "densenet121":
        in_f = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_f, 256), nn.ReLU(True),
            nn.Dropout(0.5), nn.Linear(256, n_cls))

    elif arch in ("efficientnet_b0", "efficientnet_b4"):
        in_f = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(in_f, 256),
            nn.ReLU(True), nn.Dropout(0.5), nn.Linear(256, n_cls))

    elif arch == "mobilenet_v3_large":
        # classifier: [Linear(960,1280), Hardswish, Dropout, Linear(1280,1000)]
        in_f = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_f, n_cls)

    elif arch == "convnext_tiny":
        # classifier: [LayerNorm2d(768), Flatten, Linear(768,1000)]
        in_f = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_f, n_cls)

    return model


def get_head(model, arch):
    """Return the head module for partial freezing and separate LR."""
    if arch in ("resnet18", "resnet50"):
        return model.fc
    return model.classifier


def freeze_except_head(model, arch):
    """Phase 1: freeze backbone, train head only."""
    head_name = "fc" if arch in ("resnet18", "resnet50") else "classifier"
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith(head_name)


# ════════════════════════════════════════════════════════════════════════════
# MODEL BUILDERS — Champion + 10 Challengers
# ════════════════════════════════════════════════════════════════════════════

def build_champion():
    """
    Champion: ResNet18 pretrained on ImageNet (He et al., CVPR 2016).
    Established baseline — val macro-F1=0.6976, Kaggle F1=0.723.
    2-phase training: freeze backbone 10 epochs → unfreeze all.
    """
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    return replace_head(m, "resnet18"), "resnet18"


def build_c01():
    """
    C01: EfficientNet-B0 (Tan & Le, NeurIPS 2019).

    RESEARCH BASIS: EfficientNet uses a compound coefficient φ to uniformly
    scale network width (d), depth (w), and resolution (r): d=α^φ, w=β^φ, r=γ^φ.
    Unlike ResNet's pure depth scaling, this achieves 8.4× fewer parameters
    than ResNet50 with higher accuracy. EfficientNet-B0 achieved 98.0% accuracy
    on multi-centric chest X-ray classification (PMC12453788), outperforming
    ResNet50 on the same task.

    EXPECTED GAIN: +2-4% macro-F1 over ResNet18 with fewer parameters (5.3M vs 11.3M).
    """
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    return replace_head(m, "efficientnet_b0"), "efficientnet_b0"


def build_c02():
    """
    C02: DenseNet121 (Huang et al., CVPR 2017; Rajpurkar et al. 2017 — CheXNet).

    RESEARCH BASIS: Dense connectivity: each layer receives feature maps from all
    preceding layers → maximum gradient flow, feature reuse, implicit deep supervision.
    CheXNet used DenseNet121 to achieve radiologist-level pneumonia detection on the
    ChestX-ray14 dataset (F1=0.435 vs radiologist 0.387), making it the canonical
    architecture for chest radiograph classification. 7.98M params.

    EXPECTED GAIN: Domain-specific inductive bias for chest X-rays. +1-3% macro-F1.
    """
    m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    return replace_head(m, "densenet121"), "densenet121"


def build_c03():
    """
    C03: EfficientNet-B4 (Tan & Le, NeurIPS 2019 — larger variant).

    RESEARCH BASIS: EfficientNet-B4 is the highest-performing EfficientNet variant
    for medical imaging: 99.79% accuracy on brain tumor classification (Nature 2025),
    99.54% on multi-class brain tumors, and 19M parameters vs B0's 5.3M.
    In comparative studies, EfficientNet-B4 outperforms ResNet50 on medical imaging
    by 3-7% accuracy across multiple modalities (PMC2024, dermatology+histopathology).

    EXPECTED GAIN: Best accuracy in the EfficientNet family, +4-6% over ResNet18.
    Note: Using 224×224 (vs native 380×380) — standard trade-off in transfer learning.
    """
    m = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
    return replace_head(m, "efficientnet_b4"), "efficientnet_b4"


def build_c04():
    """
    C04: MobileNetV3-Large (Howard et al., ICCV 2019).

    RESEARCH BASIS: MobileNetV3 introduces: (1) Squeeze-Excitation (SE) blocks
    for channel-wise attention — learns which feature channels are most informative
    for each input, (2) Hard-Swish activation (efficient approximation of Swish),
    (3) Neural architecture search for optimal block configuration. In medical imaging,
    MobileNetV3 achieved 96.94% accuracy on multi-class brain tumor classification
    (MedNetV3, 2025) and 99.38% on 7-class cancer with only 5.4M params.

    EXPECTED GAIN: SE attention may help with subtle pathological patterns.
    Efficient inference for deployment. +1-3% macro-F1 expected.
    """
    m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    return replace_head(m, "mobilenet_v3_large"), "mobilenet_v3_large"


def build_c05():
    """
    C05: ResNet50 (He et al., CVPR 2016 — direct scale-up of champion).

    RESEARCH BASIS: ResNet50 uses bottleneck blocks (1×1→3×3→1×1 conv) vs ResNet18's
    basic blocks (3×3→3×3), providing 25.5M parameters (vs 11.3M) and 50 layers (vs 18).
    On Alzheimer's MRI detection, ResNet50 slightly outperformed EfficientNet-B0
    (80.86% vs 79.33% — ScienceDirect 2024), showing domain-dependent advantages.
    EfficientNet-B0+ResNet50 ensemble achieved 99.45% accuracy (Nature 2025).

    EXPECTED GAIN: Higher capacity for complex feature learning. +1-3% macro-F1.
    Risk: potential overfitting on our 11K dataset (mitigated by early stopping).
    """
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    return replace_head(m, "resnet50"), "resnet50"


def build_c06():
    """
    C06: ConvNeXt-Tiny (Liu et al., CVPR 2022 — "A ConvNet for the 2020s").

    RESEARCH BASIS: ConvNeXt modernizes ResNet using Vision Transformer design choices:
    (1) 7×7 depthwise conv (vs 3×3 in ResNet) — larger receptive field per layer,
    (2) LayerNorm (vs BatchNorm) — better optimization dynamics,
    (3) GELU activation (vs ReLU) — smoother gradient flow,
    (4) Inverted bottleneck — 4× wider inner channels,
    (5) Fewer normalization layers — cleaner information flow.
    In 2024 medical imaging: 99.5% brain tumor classification (Journal of MMU Interfaces),
    MedNeXt (MICCAI 2023) extended ConvNeXt for medical segmentation with SOTA results.

    EXPECTED GAIN: Latest CNN design paradigm. +2-4% macro-F1. 28.6M params.
    """
    m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    return replace_head(m, "convnext_tiny"), "convnext_tiny"


# ════════════════════════════════════════════════════════════════════════════
# TRAINING & EVALUATION FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

class EarlyStopping:
    def __init__(self, patience=PATIENCE, save_path=None):
        self.patience  = patience
        self.save_path = save_path
        self.best      = -1e9
        self.counter   = 0
        self.stop      = False

    def __call__(self, val_metric, model):
        if val_metric > self.best + 1e-4:
            self.best    = val_metric
            self.counter = 0
            if self.save_path:
                os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
                torch.save(model.state_dict(), self.save_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


def train_epoch(model, loader, opt, loss_fn, mixup=False, mixup_alpha=0.4):
    """
    One training epoch. Optionally applies MixUp data augmentation.

    MixUp (Zhang et al., ICLR 2018): linear interpolation between training pairs:
      x_mix = λ·x_i + (1-λ)·x_j  where λ ~ Beta(α, α)
      loss  = λ·CE(x_mix, y_i) + (1-λ)·CE(x_mix, y_j)

    Creates synthetic training samples — especially valuable for minority classes,
    as it exposes the model to soft boundaries between Class 0 and Classes 1/2.
    +3% accuracy improvement documented on tuberculosis CXR detection (MDPI 2022).
    """
    model.train()
    total = 0.0
    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)

        if mixup:
            lam     = np.random.beta(mixup_alpha, mixup_alpha)
            idx     = torch.randperm(X.size(0), device=device)
            X_mix   = lam * X + (1 - lam) * X[idx]
            logits  = model(X_mix)
            loss    = lam * loss_fn(logits, y) + (1 - lam) * loss_fn(logits, y[idx])
        else:
            loss = loss_fn(model(X), y)

        loss.backward()
        opt.step()
        total += loss.item() * len(y)
    return total / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, loss_fn):
    """Evaluation pass — returns (avg_loss, macro_f1, y_true, y_pred, y_prob)."""
    model.eval()
    total, preds, truths, probas = 0.0, [], [], []
    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(X)
        total  += loss_fn(logits, y).item() * len(y)
        p = torch.softmax(logits, 1).cpu().numpy()
        preds.extend(np.argmax(p, 1).tolist())
        truths.extend(y.cpu().tolist())
        probas.extend(p.tolist())
    truths = np.array(truths)
    preds  = np.array(preds)
    probas = np.array(probas)
    mf1    = f1_score(truths, preds, average="macro", zero_division=0)
    return total / len(loader.dataset), mf1, truths, preds, probas


def compute_metrics(truths, preds, probas, tag=""):
    """
    Compute all metrics including PR-AUC (Precision-Recall AUC, macro).

    PR-AUC computation (macro):
      For each class c (one-vs-rest): compute AP = area under P-R curve for class c.
      Macro-average: PR-AUC = mean(AP_0, AP_1, AP_2).

    Interpretation on imbalanced data:
      • Random classifier PR-AUC ≈ class_frequency (not 0.5 like ROC-AUC).
      • Class 0 baseline: ~0.92,  Class 1: ~0.057,  Class 2: ~0.023
      • Macro random baseline: ~(0.92+0.057+0.023)/3 ≈ 0.33
      • PR-AUC > 0.33 indicates better-than-random across all classes.
      • More meaningful on balanced test set (equal class prior).
    """
    mf1 = f1_score(truths, preds, average="macro", zero_division=0)
    acc = accuracy_score(truths, preds)

    y_bin = label_binarize(truths, classes=[0, 1, 2])

    try:
        roc_auc = roc_auc_score(truths, probas,
                                 multi_class="ovr", average="macro")
    except Exception:
        roc_auc = float("nan")

    try:
        pr_auc = average_precision_score(y_bin, probas, average="macro")
    except Exception:
        pr_auc = float("nan")

    # Per-class PR-AUC for detailed analysis
    pr_auc_per = []
    for c in range(N_CLS):
        try:
            pr_auc_per.append(
                average_precision_score((truths == c).astype(int), probas[:, c]))
        except Exception:
            pr_auc_per.append(float("nan"))

    if tag:
        log.info(f"    {tag}: Macro-F1={mf1:.4f} | PR-AUC={pr_auc:.4f} | "
                 f"ROC-AUC={roc_auc:.4f} | Acc={acc:.4f}")
        log.info(f"      Per-class PR-AUC: "
                 f"C0={pr_auc_per[0]:.3f} C1={pr_auc_per[1]:.3f} C2={pr_auc_per[2]:.3f}")

    return {
        "macro_f1":    round(float(mf1),    4),
        "pr_auc":      round(float(pr_auc),  4),
        "roc_auc":     round(float(roc_auc), 4),
        "accuracy":    round(float(acc),     4),
        "pr_auc_c0":   round(float(pr_auc_per[0]), 4),
        "pr_auc_c1":   round(float(pr_auc_per[1]), 4),
        "pr_auc_c2":   round(float(pr_auc_per[2]), 4),
    }


def save_training_plot(tr_l, va_l, va_f1, title, path):
    ep = range(1, len(tr_l) + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(ep, tr_l, "b-o", ms=2.5, label="Train")
    a1.plot(ep, va_l, "r-o", ms=2.5, label="Val")
    a1.set(xlabel="Epoch", ylabel="Loss", title="Loss")
    a1.legend(); a1.grid(alpha=0.3)
    a2.plot(ep, va_f1, "g-o", ms=2.5, label="Val Macro-F1")
    a2.axhline(1/3, ls="--", color="gray", alpha=0.6, label="Random (0.33)")
    a2.set(xlabel="Epoch", ylabel="Macro-F1", title="Val Macro-F1")
    a2.legend(); a2.grid(alpha=0.3)
    fig.suptitle(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# GENERIC 2-PHASE TRAINING (Phase 1: head only → Phase 2: full fine-tune)
# ════════════════════════════════════════════════════════════════════════════

def train_2phase(name, build_fn,
                 tr_loader, va_loader, te_loader, te_bal_loader, cw_t,
                 loss_type="ce",       # "ce" | "focal"
                 label_smoothing=0.0,  # 0.0 | 0.1
                 mixup=False,          # True for C08
                 cosine_lr=False,      # True for C09
                 n_epochs=N_EPOCHS, patience=PATIENCE, unfreeze_ep=UNFREEZE_EP,
                 lr_head=1e-3, lr_backbone=1e-4, wd=1e-4):
    """
    Generic 2-phase transfer learning training.

    Phase 1 (epochs 1 → unfreeze_ep):
      Freeze backbone; train only the custom head.
      Rationale: prevents corrupting ImageNet features before the head converges.

    Phase 2 (epochs unfreeze_ep → n_epochs):
      Unfreeze all layers. Backbone gets lr_backbone (10× smaller than head).
      Rationale: fine-tunes pretrained features carefully.
    """
    log.info(f"\n{'='*60}\n  {'Champion' if name=='Champion_ResNet18' else 'Challenger'}: "
             f"{name}\n{'='*60}")
    t0 = time.time()
    ckpt_path = f"output/models/cc_{name}.pth"

    # ── Checkpoint resume: skip training if checkpoint already exists ─────────
    if os.path.exists(ckpt_path):
        log.info(f"  [RESUME] Checkpoint found: {ckpt_path}. Skipping training.")
        model, arch = build_fn()
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model = model.to(device)
        # Build loss fn for evaluation
        if loss_type == "focal":
            loss_fn = FocalLoss(weight=cw_t.to(device), gamma=2.0)
        else:
            loss_fn = nn.CrossEntropyLoss(weight=cw_t.to(device),
                                           label_smoothing=label_smoothing)
        _, _, yt_va, yp_va, ypr_va = eval_epoch(model, va_loader,     loss_fn)
        _, _, yt_ub, yp_ub, ypr_ub = eval_epoch(model, te_loader,     loss_fn)
        _, _, yt_b,  yp_b,  ypr_b  = eval_epoch(model, te_bal_loader, loss_fn)
        val_m = compute_metrics(yt_va, yp_va, ypr_va, "Val")
        ub_m  = compute_metrics(yt_ub, yp_ub, ypr_ub, "Test-Unbalanced")
        bal_m = compute_metrics(yt_b,  yp_b,  ypr_b,  "Test-Balanced")
        n_params = sum(p.numel() for p in model.parameters())
        return {"name": name, "val": val_m, "ub": ub_m, "bal": bal_m,
                "best_val_f1": val_m["macro_f1"], "n_params_M": round(n_params/1e6, 1),
                "arch": arch, "model": model}

    model, arch = build_fn()
    model = model.to(device)

    # Build loss function
    if loss_type == "focal":
        loss_fn = FocalLoss(weight=cw_t.to(device), gamma=2.0)
        log.info("  Loss: FocalLoss(γ=2.0) + class weights")
    else:
        loss_fn = nn.CrossEntropyLoss(weight=cw_t.to(device),
                                       label_smoothing=label_smoothing)
        if label_smoothing > 0:
            log.info(f"  Loss: CrossEntropyLoss(weight, label_smoothing={label_smoothing})")
        else:
            log.info("  Loss: CrossEntropyLoss(weight)")

    # Phase 1: head only
    freeze_except_head(model, arch)
    head = get_head(model, arch)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"  Phase 1 trainable params: {trainable:,}")

    opt = optim.Adam(head.parameters(), lr=lr_head, weight_decay=wd)
    es  = EarlyStopping(patience=patience, save_path=ckpt_path)

    tr_losses, va_losses, va_f1s = [], [], []
    unfrozen = False

    log.info(f"  {'Ep':>4}  {'TrainL':>8}  {'ValL':>7}  {'ValF1':>7}  {'LR':>9}")

    for ep in range(1, n_epochs + 1):

        # ── Phase 2 transition ──────────────────────────────────────────
        if ep == unfreeze_ep and not unfrozen:
            for p in model.parameters():
                p.requires_grad = True

            # Separate LR groups: backbone gets 10× smaller LR
            head_pname = "fc" if arch in ("resnet18", "resnet50") else "classifier"
            backbone_p = [p for n, p in model.named_parameters()
                          if not n.startswith(head_pname)]
            head_p     = list(head.parameters())

            opt = optim.Adam(
                [{"params": backbone_p, "lr": lr_backbone},
                 {"params": head_p,     "lr": lr_head}],
                weight_decay=wd)

            if cosine_lr:
                # CosineAnnealingLR: smooth decay from lr to 0 over remaining epochs
                sched = optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=n_epochs - unfreeze_ep, eta_min=1e-6)

            total_p = sum(p.numel() for p in model.parameters())
            log.info(f"\n  → Phase 2: unfreeze all ({total_p:,} params), "
                     f"lr_head={lr_head:.0e} | lr_backbone={lr_backbone:.0e}")
            unfrozen = True

        # ── Train + validate ────────────────────────────────────────────
        tl = train_epoch(model, tr_loader, opt, loss_fn, mixup=mixup)
        vl, vf1, _, _, _ = eval_epoch(model, va_loader, loss_fn)
        tr_losses.append(tl); va_losses.append(vl); va_f1s.append(vf1)
        lr_now = opt.param_groups[0]["lr"]
        log.info(f"  {ep:4d}  {tl:8.4f}  {vl:7.4f}  {vf1:7.4f}  {lr_now:9.2e}")

        if unfrozen and cosine_lr:
            sched.step()

        if es(vf1, model):
            log.info(f"  Early stop @ epoch {ep}  (best={es.best:.4f})")
            break

    # ── Load best checkpoint ─────────────────────────────────────────────────
    model.load_state_dict(
        torch.load(f"output/models/cc_{name}.pth", map_location=device))
    save_training_plot(tr_losses, va_losses, va_f1s,
                       f"{name} Training History",
                       f"output/figures/cc_{name}_training.png")

    # ── Evaluate ─────────────────────────────────────────────────────────────
    _, _, yt_va, yp_va, ypr_va = eval_epoch(model, va_loader,  loss_fn)
    _, _, yt_ub, yp_ub, ypr_ub = eval_epoch(model, te_loader,  loss_fn)
    _, _, yt_b,  yp_b,  ypr_b  = eval_epoch(model, te_bal_loader, loss_fn)

    val_m = compute_metrics(yt_va, yp_va, ypr_va, "Val")
    ub_m  = compute_metrics(yt_ub, yp_ub, ypr_ub, "Test-Unbalanced")
    bal_m = compute_metrics(yt_b,  yp_b,  ypr_b,  "Test-Balanced")

    n_params  = sum(p.numel() for p in model.parameters())
    elapsed   = time.time() - t0
    log.info(f"  Params: {n_params/1e6:.1f}M | Best Val Macro-F1: {es.best:.4f} | "
             f"Time: {elapsed/60:.1f} min")

    return {
        "name":          name,
        "val":           val_m,
        "ub":            ub_m,
        "bal":           bal_m,
        "best_val_f1":   es.best,
        "n_params_M":    round(n_params / 1e6, 1),
        "arch":          arch,
        "model":         model,
    }


# ════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING LOOP — Champion + 10 Challengers
# ════════════════════════════════════════════════════════════════════════════

log.info("\n" + "="*60)
log.info("  CHAMPION vs CHALLENGER — 10-Iteration Deep CNN Comparison")
log.info("="*60)

# ── Champion: ResNet18 (re-train here for fair side-by-side comparison) ──────
champ = train_2phase(
    "Champion_ResNet18", build_champion,
    tr_l, va_l, te_l, te_l_bal, cw,
    loss_type="ce", n_epochs=40, patience=8)
ALL_RESULTS["Champion_ResNet18"] = {
    k: v for k, v in champ.items() if k != "model"}

# ── C01: EfficientNet-B0 ──────────────────────────────────────────────────────
c01 = train_2phase(
    "C01_EfficientNet_B0", build_c01,
    tr_l, va_l, te_l, te_l_bal, cw,
    loss_type="ce")
ALL_RESULTS["C01_EfficientNet_B0"] = {k: v for k, v in c01.items() if k != "model"}

# ── C02: DenseNet121 (CheXNet) ────────────────────────────────────────────────
c02 = train_2phase(
    "C02_DenseNet121", build_c02,
    tr_l, va_l, te_l, te_l_bal, cw,
    loss_type="ce")
ALL_RESULTS["C02_DenseNet121"] = {k: v for k, v in c02.items() if k != "model"}

# ── C03: EfficientNet-B4 (highest medical imaging accuracy) ──────────────────
c03 = train_2phase(
    "C03_EfficientNet_B4", build_c03,
    tr_l, va_l, te_l, te_l_bal, cw,
    loss_type="ce")
ALL_RESULTS["C03_EfficientNet_B4"] = {k: v for k, v in c03.items() if k != "model"}

# ── C04: MobileNetV3-Large (SE attention) ────────────────────────────────────
c04 = train_2phase(
    "C04_MobileNetV3_Large", build_c04,
    tr_l, va_l, te_l, te_l_bal, cw,
    loss_type="ce")
ALL_RESULTS["C04_MobileNetV3_Large"] = {k: v for k, v in c04.items() if k != "model"}

# ── C05: ResNet50 (deeper backbone) ──────────────────────────────────────────
c05 = train_2phase(
    "C05_ResNet50", build_c05,
    tr_l, va_l, te_l, te_l_bal, cw,
    loss_type="ce")
ALL_RESULTS["C05_ResNet50"] = {k: v for k, v in c05.items() if k != "model"}

# ── C06: ConvNeXt-Tiny (modernized CNN) ──────────────────────────────────────
c06 = train_2phase(
    "C06_ConvNeXt_Tiny", build_c06,
    tr_l, va_l, te_l, te_l_bal, cw,
    loss_type="ce")
ALL_RESULTS["C06_ConvNeXt_Tiny"] = {k: v for k, v in c06.items() if k != "model"}

# ── C07: ResNet18 + Focal Loss (γ=2, same arch better loss for imbalance) ────
c07 = train_2phase(
    "C07_ResNet18_FocalLoss", build_champion,
    tr_l, va_l, te_l, te_l_bal, cw,
    loss_type="focal")
ALL_RESULTS["C07_ResNet18_FocalLoss"] = {k: v for k, v in c07.items() if k != "model"}

# ── C08: EfficientNet-B0 + RandAugment (best augmentation for chest X-ray) ───
# Uses RandAugment training loader (tr_l_ra) for stronger augmentation
c08 = train_2phase(
    "C08_EffB0_RandAugment", build_c01,
    tr_l_ra, va_l, te_l, te_l_bal, cw,   # RandAugment train loader
    loss_type="ce")
ALL_RESULTS["C08_EffB0_RandAugment"] = {k: v for k, v in c08.items() if k != "model"}

# ── C09: ResNet18 + Label Smoothing (ε=0.1) + Cosine Annealing LR ────────────
# Label smoothing: prevents overconfident predictions, improves calibration.
# CosineAnnealingLR: smoother LR decay vs ReduceLROnPlateau.
# LS+ (MICCAI 2024) showed notable calibration improvement on chest pathology.
c09 = train_2phase(
    "C09_ResNet18_LabelSmooth", build_champion,
    tr_l, va_l, te_l, te_l_bal, cw,
    loss_type="ce", label_smoothing=0.1, cosine_lr=True)
ALL_RESULTS["C09_ResNet18_LabelSmooth"] = {k: v for k, v in c09.items() if k != "model"}

# ── C10: Ensemble (soft voting of Champion + C01 + C02) ──────────────────────
log.info(f"\n{'='*60}\n  C10: Ensemble (Champion + C01_EffB0 + C02_DenseNet121)\n{'='*60}")
log.info("  Soft voting: average softmax probabilities, then argmax.")
log.info("  Research: +4-8% macro-F1 expected from 3-model soft ensemble "
         "(PMC2201.11440, Nature 2025 EfficientNet+ResNet50 ensemble=99.45%).")

@torch.no_grad()
def get_probas(model, loader):
    """Extract softmax probabilities from model on a loader."""
    model.eval(); truths, probas = [], []
    for X, y in loader:
        X = X.to(device, non_blocking=True)
        p = torch.softmax(model(X), 1).cpu().numpy()
        probas.extend(p.tolist())
        truths.extend(y.numpy().tolist())
    return np.array(truths), np.array(probas)

# Gather probabilities from 3 ensemble members
log.info("  Gathering probabilities from 3 ensemble members...")
yt_va_en,  pr_va_champ  = get_probas(champ["model"], va_l)
_,         pr_va_c01    = get_probas(c01["model"],   va_l)
_,         pr_va_c02    = get_probas(c02["model"],   va_l)

yt_ub_en,  pr_ub_champ  = get_probas(champ["model"], te_l)
_,         pr_ub_c01    = get_probas(c01["model"],   te_l)
_,         pr_ub_c02    = get_probas(c02["model"],   te_l)

yt_b_en,   pr_b_champ   = get_probas(champ["model"], te_l_bal)
_,         pr_b_c01     = get_probas(c01["model"],   te_l_bal)
_,         pr_b_c02     = get_probas(c02["model"],   te_l_bal)

# Soft voting: simple average (equal weights)
pr_va_ens  = (pr_va_champ + pr_va_c01 + pr_va_c02) / 3
pr_ub_ens  = (pr_ub_champ + pr_ub_c01 + pr_ub_c02) / 3
pr_b_ens   = (pr_b_champ  + pr_b_c01  + pr_b_c02)  / 3

yp_va_ens  = np.argmax(pr_va_ens, 1)
yp_ub_ens  = np.argmax(pr_ub_ens, 1)
yp_b_ens   = np.argmax(pr_b_ens,  1)

val_ens  = compute_metrics(yt_va_en, yp_va_ens, pr_va_ens, "Val")
ub_ens   = compute_metrics(yt_ub_en, yp_ub_ens, pr_ub_ens, "Test-Unbalanced")
bal_ens  = compute_metrics(yt_b_en,  yp_b_ens,  pr_b_ens,  "Test-Balanced")

ALL_RESULTS["C10_Ensemble"] = {
    "name":          "C10_Ensemble",
    "val":           val_ens,
    "ub":            ub_ens,
    "bal":           bal_ens,
    "best_val_f1":   val_ens["macro_f1"],
    "n_params_M":    round((champ["n_params_M"] + c01["n_params_M"] + c02["n_params_M"]), 1),
    "arch":          "ensemble(ResNet18+EffB0+DenseNet121)",
}
log.info(f"  C10 Ensemble → Val Macro-F1={val_ens['macro_f1']:.4f} | "
         f"Bal F1={bal_ens['macro_f1']:.4f} | PR-AUC(bal)={bal_ens['pr_auc']:.4f}")

# ════════════════════════════════════════════════════════════════════════════
# RESULTS AGGREGATION & VISUALISATION
# ════════════════════════════════════════════════════════════════════════════

log.info("\n" + "="*60)
log.info("  CHAMPION vs CHALLENGER — FINAL RESULTS")
log.info("="*60)

# Ordered list of model names for display
MODEL_ORDER = [
    "Champion_ResNet18", "C01_EfficientNet_B0", "C02_DenseNet121",
    "C03_EfficientNet_B4", "C04_MobileNetV3_Large", "C05_ResNet50",
    "C06_ConvNeXt_Tiny", "C07_ResNet18_FocalLoss",
    "C08_EffB0_RandAugment", "C09_ResNet18_LabelSmooth", "C10_Ensemble",
]

# ── Summary table ─────────────────────────────────────────────────────────────
header = (f"{'Model':<30}  {'Params':>7}  {'ValF1':>7}  "
          f"{'UbF1':>7}  {'BalF1':>7}  {'PR-AUC(b)':>10}  {'ROC-AUC(b)':>11}  "
          f"{'Outcome':<12}")
log.info("\n" + header)
log.info("-" * len(header))

champion_bal_f1 = ALL_RESULTS["Champion_ResNet18"]["bal"]["macro_f1"]
new_champion_name = "Champion_ResNet18"
new_champion_f1   = champion_bal_f1

csv_rows = []
for nm in MODEL_ORDER:
    r  = ALL_RESULTS[nm]
    is_champ = nm == "Champion_ResNet18"
    bal_f1   = r["bal"]["macro_f1"]
    outcome  = "CHAMPION" if is_champ else (
               "NEW CHAMP" if bal_f1 > new_champion_f1 + 1e-4 else
               "BEATS CHAMP" if bal_f1 > champion_bal_f1 + 1e-4 else "challenger")

    if bal_f1 > new_champion_f1 and not is_champ:
        new_champion_f1   = bal_f1
        new_champion_name = nm

    log.info(f"{nm:<30}  {r['n_params_M']:>6.1f}M  "
             f"{r['val']['macro_f1']:>7.4f}  "
             f"{r['ub']['macro_f1']:>7.4f}  "
             f"{bal_f1:>7.4f}  "
             f"{r['bal']['pr_auc']:>10.4f}  "
             f"{r['bal']['roc_auc']:>11.4f}  "
             f"{outcome:<12}")

    csv_rows.append({
        "Model":          nm,
        "Params_M":       r["n_params_M"],
        "Val_MacroF1":    r["val"]["macro_f1"],
        "Test_UB_MacroF1":r["ub"]["macro_f1"],
        "Test_Bal_MacroF1":bal_f1,
        "Test_Bal_PRAUC": r["bal"]["pr_auc"],
        "Test_Bal_ROCAUC":r["bal"]["roc_auc"],
        "Test_UB_PRAUC":  r["ub"]["pr_auc"],
        "Test_UB_ROCAUC": r["ub"]["roc_auc"],
        "Outcome":        outcome,
    })

log.info(f"\n  Original Champion (ResNet18)  Balanced Macro-F1 = {champion_bal_f1:.4f}")
log.info(f"  New Champion: {new_champion_name}  Balanced Macro-F1 = {new_champion_f1:.4f}")
log.info(f"  Improvement:  {new_champion_f1 - champion_bal_f1:+.4f}")

# ── Save CSV ──────────────────────────────────────────────────────────────────
df = pd.DataFrame(csv_rows)
df.to_csv("output/cc_results.csv", index=False)
log.info("\n  Saved: output/cc_results.csv")

# ── Save JSON ─────────────────────────────────────────────────────────────────
json_out = {k: {kk: vv for kk, vv in v.items() if kk != "model"}
            for k, v in ALL_RESULTS.items()}
json_out["_summary"] = {
    "champion": "Champion_ResNet18",
    "champion_bal_f1": champion_bal_f1,
    "new_champion": new_champion_name,
    "new_champion_bal_f1": new_champion_f1,
    "improvement": round(new_champion_f1 - champion_bal_f1, 4),
}
with open("output/cc_results.json", "w") as f:
    json.dump(json_out, f, indent=2)
log.info("  Saved: output/cc_results.json")

# ════════════════════════════════════════════════════════════════════════════
# COMPARISON FIGURES
# ════════════════════════════════════════════════════════════════════════════

short_labels = [
    "Champion\nResNet18", "C01\nEffB0", "C02\nDenseNet", "C03\nEffB4",
    "C04\nMobileV3", "C05\nResNet50", "C06\nConvNeXt", "C07\nFocal",
    "C08\nRandAug", "C09\nLabelSm", "C10\nEnsemble",
]

metrics_to_plot = [
    ("Test_Bal_MacroF1",  "Balanced Test Macro-F1 ↑", 0.33, "Primary metric"),
    ("Test_Bal_PRAUC",    "Balanced Test PR-AUC ↑",   0.33, "Imbalance-aware"),
    ("Test_Bal_ROCAUC",   "Balanced Test ROC-AUC ↑",  0.50, "Standard AUC"),
    ("Test_UB_MacroF1",   "Unbalanced Test Macro-F1 ↑", 0.33, "Real-world dist."),
]

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle("Champion vs Challenger — 10-Iteration CNN Comparison\n"
             "Dataset: Chest X-Ray 3-Class (92/6/2% imbalance) | H200 GPU",
             fontsize=13, fontweight="bold")

bar_colors = ["#D32F2F"] + ["#1976D2"] * 9 + ["#388E3C"]  # red=champ, blue=challengers, green=ensemble

for ax, (col, ylabel, baseline, subtitle) in zip(axes.flatten(), metrics_to_plot):
    vals = df[col].tolist()
    bars = ax.bar(range(len(vals)), vals, color=bar_colors, edgecolor="white",
                  linewidth=0.7, alpha=0.88)
    ax.axhline(baseline, ls="--", color="gray", lw=1.2, alpha=0.7,
               label=f"Baseline ({baseline:.2f})")
    ax.axhline(vals[0], ls=":", color="#D32F2F", lw=1.5, alpha=0.8,
               label=f"Champion ({vals[0]:.3f})")

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=7, fontweight="bold")

    ax.set_title(f"{ylabel}\n({subtitle})", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(short_labels)))
    ax.set_xticklabels(short_labels, fontsize=7.5, rotation=0)
    ax.set_ylim(0, min(1.0, max(vals) * 1.15))
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("output/figures/cc_comparison_4panel.png", dpi=150, bbox_inches="tight")
plt.close()
log.info("  Saved: output/figures/cc_comparison_4panel.png")

# ── PR-AUC per-class figure ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("PR-AUC Per Class — Champion vs Challenger (Balanced Test Set)\n"
             "PR-AUC random baseline ≈ class frequency: C0≈0.92, C1≈0.057, C2≈0.023",
             fontsize=11, fontweight="bold")

for ax, (ci, cname, rand_base) in zip(
        axes, [(0, "Healthy (C0)", 0.92), (1, "Pre-existing (C1)", 0.057),
               (2, "Effusion/Mass (C2)", 0.023)]):
    key = f"pr_auc_c{ci}"
    vals = [ALL_RESULTS[nm]["bal"].get(key, float("nan")) for nm in MODEL_ORDER]
    bars = ax.bar(range(len(vals)), vals, color=bar_colors, edgecolor="white",
                  linewidth=0.7, alpha=0.88)
    ax.axhline(rand_base, ls="--", color="gray", lw=1.2, alpha=0.7,
               label=f"Random ({rand_base:.3f})")
    for bar, val in zip(bars, vals):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals)*0.01,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=6.5, fontweight="bold")
    ax.set_title(cname, fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(short_labels)))
    ax.set_xticklabels(short_labels, fontsize=7, rotation=0)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    ax.set_ylabel("PR-AUC")

plt.tight_layout()
plt.savefig("output/figures/cc_prauc_perclass.png", dpi=150, bbox_inches="tight")
plt.close()
log.info("  Saved: output/figures/cc_prauc_perclass.png")

# ── Radar / Spider chart ─────────────────────────────────────────────────────
top3_names = sorted(MODEL_ORDER,
                    key=lambda n: ALL_RESULTS[n]["bal"]["macro_f1"],
                    reverse=True)[:3]
radar_metrics = ["Val\nMacro-F1", "Test-Bal\nMacro-F1", "Test-Bal\nPR-AUC",
                 "Test-Bal\nROC-AUC", "Test-UB\nMacro-F1"]
radar_keys    = [("val", "macro_f1"), ("bal", "macro_f1"), ("bal", "pr_auc"),
                 ("bal", "roc_auc"), ("ub",  "macro_f1")]

N_radar = len(radar_metrics)
angles  = [n / float(N_radar) * 2 * np.pi for n in range(N_radar)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 7), subplot_kw=dict(polar=True))
radar_colors = ["#D32F2F", "#1976D2", "#388E3C"]

for i, nm in enumerate(top3_names):
    vals_r = [ALL_RESULTS[nm][split][key] for split, key in radar_keys]
    vals_r += vals_r[:1]
    label  = nm.replace("Champion_", "").replace("_", " ")
    ax.plot(angles, vals_r, "o-", lw=2, color=radar_colors[i], label=label)
    ax.fill(angles, vals_r, alpha=0.12, color=radar_colors[i])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_metrics, fontsize=9)
ax.set_ylim(0, 1)
ax.set_yticks([0.3, 0.5, 0.7, 0.9])
ax.set_yticklabels(["0.3", "0.5", "0.7", "0.9"], fontsize=8)
ax.set_title("Top-3 Models — Multi-Metric Radar\n"
             "(all axes: higher is better)", fontsize=11, fontweight="bold", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
plt.tight_layout()
plt.savefig("output/figures/cc_radar_top3.png", dpi=150, bbox_inches="tight")
plt.close()
log.info("  Saved: output/figures/cc_radar_top3.png")

# ════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════════════════════

log.info("\n" + "="*60)
log.info("  CHAMPION vs CHALLENGER SUMMARY")
log.info("="*60)
log.info(f"  Champion (ResNet18):       Balanced Macro-F1 = {champion_bal_f1:.4f} | "
         f"PR-AUC = {ALL_RESULTS['Champion_ResNet18']['bal']['pr_auc']:.4f}")
log.info(f"  New Champion ({new_champion_name}): "
         f"Balanced Macro-F1 = {new_champion_f1:.4f} | "
         f"PR-AUC = {ALL_RESULTS[new_champion_name]['bal']['pr_auc']:.4f}")
log.info(f"  Improvement: {new_champion_f1 - champion_bal_f1:+.4f} macro-F1")
log.info(f"\n  Top-3 models by Balanced Macro-F1:")
for i, nm in enumerate(top3_names):
    r = ALL_RESULTS[nm]
    log.info(f"    {i+1}. {nm:<30} F1={r['bal']['macro_f1']:.4f} "
             f"PR-AUC={r['bal']['pr_auc']:.4f} "
             f"Params={r['n_params_M']:.1f}M")

log.info("\n  PR-AUC note: Balanced test set (33% per class) gives the most")
log.info("  reliable PR-AUC estimates since class priors are equal. On the")
log.info("  unbalanced set (92/6/2%), PR-AUC for Class 2 approaches the")
log.info("  random baseline (~0.023), making macro PR-AUC harder to interpret.")
log.info("\n  All results saved to output/cc_results.csv and output/cc_results.json")
log.info("  All figures saved to output/figures/cc_*.png")
log.info("\n  Run complete. Add the new champion to main.tex.")
