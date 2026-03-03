# Architecture Design Document
**PS4 — CNN Chest X-Ray Classifier**

---

## Problem Formulation

**Type**: Multi-class image classification (3 classes)
**Input**: Grayscale chest X-ray images (shape TBD from EDA)
**Output**: Class probabilities → argmax → {0: Healthy, 1: Pre-existing, 2: Effusion/Mass}

---

## Proposed CNN Architectures

### Model A: Baseline Custom CNN
**Purpose**: Establish a performance floor. Simple, interpretable, fast to train.

```
Input (C×H×W)
  → Conv2d(C→32, 3×3) + BN + ReLU + MaxPool(2×2)
  → Conv2d(32→64, 3×3) + BN + ReLU + MaxPool(2×2)
  → Conv2d(64→128, 3×3) + BN + ReLU + MaxPool(2×2)
  → AdaptiveAvgPool2d(4×4)   # makes it input-size-agnostic
  → Flatten → FC(128×16→256) + ReLU + Dropout(0.5)
  → FC(256→3) → LogSoftmax
```

**Hyperparameters**:
- Optimizer: Adam (lr=1e-3)
- Loss: CrossEntropyLoss with class weights
- Batch size: 64
- Epochs: 20
- No augmentation (baseline)

**Expected behavior**: Will likely be biased toward Class 0 even with weights; establishes reference

---

### Model B: Improved CNN with Residual Skip Connections
**Purpose**: Increase representational power, reduce vanishing gradients, add regularization.

```
Input (C×H×W)
  → ConvBlock1: [Conv(C→64, 3×3)+BN+ReLU] × 2 + MaxPool
  → ConvBlock2: [Conv(64→128, 3×3)+BN+ReLU] × 2 + MaxPool  [+ skip from block1 1×1 conv]
  → ConvBlock3: [Conv(128→256, 3×3)+BN+ReLU] × 2 + MaxPool [+ skip from block2]
  → ConvBlock4: [Conv(256→512, 3×3)+BN+ReLU] × 2 + MaxPool [+ skip from block3]
  → GlobalAvgPool2d()
  → FC(512→256) + ReLU + Dropout(0.5)
  → FC(256→128) + ReLU + Dropout(0.3)
  → FC(128→3)
```

**Regularization**:
- Dropout: 0.5 after first FC, 0.3 after second FC
- Weight decay (L2): 1e-4 in optimizer
- Batch Normalization in every conv block

**Hyperparameters**:
- Optimizer: Adam (lr=3e-4, weight_decay=1e-4)
- Loss: CrossEntropyLoss with class weights
- Scheduler: ReduceLROnPlateau (patience=5, factor=0.5)
- Batch size: 32
- Epochs: 50 with early stopping (patience=10)
- Augmentation: horizontal flip, rotation ±15°, brightness jitter

---

### Model C: Transfer Learning — ResNet18 / EfficientNet-B0
**Purpose**: Leverage ImageNet pretrained features for small dataset with large imbalance.

```
Base: torchvision.models.resnet18(pretrained=True)
  → Freeze all layers except layer3, layer4, fc
  → Replace fc: Linear(512→3)

OR

Base: torchvision.models.efficientnet_b0(pretrained=True)
  → Freeze features[0..5]
  → Fine-tune features[6:] + classifier
  → Replace classifier: Linear(1280→3)
```

**Preprocessing** (for pretrained models):
- Resize to 224×224
- Convert grayscale → RGB (repeat channel 3×)
- Normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

**Hyperparameters**:
- Optimizer: Adam with layer-wise LR (base=1e-4, new layers=1e-3)
- Loss: CrossEntropyLoss with class weights
- Epochs: 30 with early stopping
- Fine-tuning phase: unfreeze all layers at epoch 10 with lr=1e-5

---

## Handling Class Imbalance

### Strategy 1: Class-Weighted Loss (REQUIRED)
```python
# Compute inverse-frequency weights
N = 11414  # total samples
counts = [10506, 651, 257]  # per class
class_weights = torch.tensor([N/(3*c) for c in counts])
# → tensor([0.362, 5.843, 14.81])

loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
```

### Strategy 2: WeightedRandomSampler (for mini-batch balance)
```python
# Each sample gets weight = 1/count_of_its_class
sample_weights = [1/counts[label] for label in all_labels]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights))
train_loader = DataLoader(dataset, batch_size=32, sampler=sampler)
```

### Strategy 3: Data Augmentation for Minority Classes
Apply more aggressive augmentation to Class 1 and Class 2:
- Random horizontal flip (p=0.5)
- Random rotation ±15°
- Random brightness/contrast jitter (factor=0.2)
- Random crop + resize

---

## Evaluation Design

### Split Strategy
```
TrainValid (11,414 samples)
  → Stratified split (preserves class ratios)
  → Train: 70% (7,989 samples) [~9,762 class0, ~456 class1, ~180 class2]
  → Val:   15% (1,712 samples) [~1,576 class0, ~98 class1, ~39 class2]
  → Test:  15% (1,712 samples) [~1,576 class0, ~98 class1, ~39 class2]
```

### Two Test Conditions (MUST REPORT BOTH)

**Condition 1 — Unbalanced (mirrors real-world deployment)**
- Use the full stratified test split as-is (~92% class 0)
- Report: accuracy, macro-F1, micro-F1, macro-AUC
- Note: accuracy here is misleading (high due to class 0 dominance)

**Condition 2 — Balanced (mirrors Kaggle competition)**
- Subsample test set: take min(per-class test count) from each class
- Or: use the Kaggle test set directly (100 per class)
- Report same metrics
- Note: macro-F1 will be lower here since the model sees equal class representation

### Metrics to Report

| Metric | Formula | Why |
|--------|---------|-----|
| Macro-F1 | mean(F1_per_class) | Not dominated by majority class |
| Micro-F1 | Global TP/(TP+FP+FN) | Overall "accuracy" equivalent |
| Per-class F1 | 2×P×R/(P+R) | Diagnose per-class performance |
| Macro-AUC | mean(AUC_i) OvR | Probability-based, threshold-free |
| Confusion Matrix | N×N counts | Visualize error patterns |
| Accuracy | Correct/Total | Show WHY accuracy is misleading |

---

## Hyperparameter Search

### Grid/Random Search Space
```python
SEARCH_SPACE = {
    "lr": [1e-2, 1e-3, 3e-4, 1e-4],
    "weight_decay": [0, 1e-4, 1e-3],
    "dropout": [0.3, 0.5, 0.6],
    "batch_size": [32, 64],
    "augmentation": [True, False],
    "use_sampler": [True, False],  # WeightedRandomSampler
}
```

### Primary Metric for Selection: Val Macro-F1

---

## Training Protocol

1. Set `torch.manual_seed(42)` and `numpy.random.seed(42)` at start
2. Load data → compute class weights → create DataLoaders
3. Train baseline (Model A) → record val metrics per epoch → save best checkpoint
4. Train improved models (B, C) → compare with baseline
5. Select best model on val macro-F1
6. Evaluate selected model on test set (both conditions)
7. Run Kaggle prediction with best model

---

## Key Files & Their Roles

| File | Purpose |
|------|---------|
| `src/00_eda.py` | Load images, visualize, compute stats, class distribution |
| `src/01_data_pipeline.py` | XRayDataset class, stratified split, augmentation, DataLoaders |
| `src/02_baseline_cnn.py` | Model A definition, train loop, save checkpoint |
| `src/03_improved_cnn.py` | Models B & C, hyperparameter tuning, comparison |
| `src/04_evaluation.py` | Full evaluation: F1, AUC, confusion matrix, ROC plots |
| `src/05_kaggle_predict.py` | Load best model, predict on kaggle set, save CSV |
| `src/utils.py` | plot_confusion_matrix, plot_roc_curves, setup_logging, EarlyStopping |

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-02-27 | Use PyTorch over TensorFlow | More flexible for custom architectures; industry standard |
| 2026-02-27 | Use macro-F1 as primary metric | Imbalance makes accuracy uninformative |
| 2026-02-27 | Stratified 70/15/15 split | Preserves rare class representation in val/test |
| 2026-02-27 | Report BOTH balanced and unbalanced test results | Technical Manager explicitly required this |
| 2026-02-27 | Propose 3 architectures (A/B/C) | Assignment requires "a few CNN architectures" |
| 2026-02-27 | AdaptiveAvgPool instead of fixed Flatten | Makes models input-size-agnostic |
