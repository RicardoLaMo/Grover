# CLAUDE.md — PS4 Lung X-Ray Classifier Project

## Project Overview
**Course**: DTSC 8120: Fundamentals of Machine Learning
**Assignment**: Problem Set 4 — Chest X-Ray CNN Classifier
**Deadline**: Wednesday, March 4, 2026 at 5:30 PM ET
**Instructor**: Dr. Keith Burghardt

## Directory Structure
```
hw4/
├── CLAUDE.md                        # This file — agent rules & architecture
├── doc/
│   ├── PROJECT_BRIEF.md             # Business problem, stakeholders, criteria
│   ├── GRADING_RUBRIC.md            # Passing gates per role
│   ├── ARCHITECTURE.md              # CNN design decisions & rationale
│   └── report/
│       └── main.tex                 # LaTeX report (submit as PDF)
├── src/
│   ├── 00_eda.py                    # Exploratory Data Analysis
│   ├── 01_data_pipeline.py          # Data loading, augmentation, splits
│   ├── 02_baseline_cnn.py           # Baseline CNN model definition & train
│   ├── 03_improved_cnn.py           # Improved architectures (ResNet-style, etc.)
│   ├── 04_evaluation.py             # Metrics: macro-F1, AUC, confusion matrix
│   ├── 05_kaggle_predict.py         # Generate extra-credit submission CSV
│   └── utils.py                     # Shared utilities (logging, plotting)
├── output/
│   ├── figures/                     # EDA plots, confusion matrices, ROC curves
│   ├── models/                      # Saved model weights (.pth)
│   └── predictions/                 # kaggle_submission.csv
└── data/                            # Symlinks to raw data files
    ├── trainvalid_images.npy -> ../ps4_trainvalid_images-2.npy
    ├── trainvalid_labels.csv -> ../ps4_trainvalid_labels.csv
    ├── kaggle_images.npy    -> ../ps4_kaggle_images-1.npy
    └── kaggle_labels.csv    -> ../ps4_kaggle_labels.csv
```

## Agent Roles & Responsibilities

### Architect Agent (Claude — orchestrates)
- Reads problem spec, designs pipeline, sets passing gates
- Reviews outputs of other agents before proceeding to next stage
- Documents key decisions in `doc/ARCHITECTURE.md`

### Data Agent
- Runs `src/00_eda.py` → outputs figures to `output/figures/`
- Validates: image shape, dtype, range, class distribution, sample images
- GATE: must produce `output/figures/class_distribution.png` and `output/figures/sample_images.png`

### Pipeline Agent
- Runs `src/01_data_pipeline.py`
- Implements stratified 80/10/10 train/val/test split
- Implements class-weighted sampler and data augmentation
- GATE: DataLoader sanity check passes (shapes correct, labels match)

### Model Agent
- Runs `src/02_baseline_cnn.py` and `src/03_improved_cnn.py`
- Trains models, saves weights to `output/models/`
- GATE: val macro-F1 > 0.40 for baseline; > 0.60 for improved model

### Evaluation Agent
- Runs `src/04_evaluation.py`
- Produces confusion matrices, ROC curves, per-class F1 report
- GATE: test report must include balanced AND unbalanced split results
- GATE: must report both micro and macro averaged metrics

### Report Agent
- Compiles results into `doc/report/main.tex`
- CEO section: 1 figure + 2-3 sentences
- Manager section: full methodology + imbalance handling + metrics discussion
- Developer section: code architecture overview

## Passing Gates (Quality Control)

| Gate | Stage | Criterion | Who validates |
|------|-------|-----------|---------------|
| G1 | EDA | Images load, shapes confirmed, distribution plot saved | Architect |
| G2 | Pipeline | Stratified split verified, class weights computed | Architect |
| G3 | Baseline | Training converges, val macro-F1 > 0.40 | Architect |
| G4 | Improved | Val macro-F1 > 0.60 OR best effort with justification | Architect |
| G5 | Evaluation | Both balanced/unbalanced test results documented | Architect |
| G6 | Kaggle | submission.csv has Id 0-299, Predicted 0/1/2 | Architect |
| G7 | Report | PDF compiles, all sections present, GitHub link added | Architect |

## Critical Design Decisions

### Class Imbalance (MOST IMPORTANT)
- TrainValid: Class 0=92%, Class 1=5.7%, Class 2=2.3% (11,414 samples)
- Kaggle test: Perfectly balanced (100 each)
- Strategy: `class_weights = N_total / (N_classes * N_class_i)` → [0.36, 5.84, 14.8]
- Use `torch.nn.CrossEntropyLoss(weight=class_weights)`
- Use `WeightedRandomSampler` for balanced mini-batches

### Evaluation Strategy
1. **Stratified split** from trainvalid: 70% train / 15% val / 15% test
2. Report metrics on BOTH:
   - Unbalanced test set (mirrors real-world distribution)
   - Class-rebalanced test subset (mirrors Kaggle)
3. Primary metric: **Macro-averaged F1** (not accuracy — imbalance invalidates accuracy)
4. Secondary: Per-class precision/recall, ROC-AUC, confusion matrix

### CNN Architectures to Propose
- **Model A (Baseline)**: 3 conv-blocks [32→64→128 filters], GlobalAvgPool, FC→3
- **Model B (Improved)**: 5 conv-blocks with residual skip connections + Dropout(0.5) + L2
- **Model C (Transfer)**: Pretrained ResNet18/EfficientNet-B0, fine-tuned last 2 layers

## Code Standards (for Senior Developer)
- Every function has a docstring explaining purpose and args
- Non-trivial CNN layers explained with inline comments
- Training loop has progress logging (epoch, loss, val-F1)
- All hyperparameters defined in a CONFIG dict at top of each file
- Reproducibility: `torch.manual_seed(42)`, `numpy.random.seed(42)`
- Use `logging` module, not raw `print`

## Report Standards (for CEO/Manager)
- CEO section: abstract + best model metric + 1 figure (confusion matrix)
- Manager section: data description → imbalance analysis → methodology → results table → conclusion
- All figures must have captions and be referenced in text
- LaTeX `\label` and `\ref` for all figures and tables

## Environment
- **Python**: 3.13.9 (miniconda3)
- **PyTorch**: 2.9.0+cu128 (CUDA 12.8)
- **torchvision**: 0.24.0+cu128
- **scikit-learn**: 1.7.2 | **numpy**: 2.2.6 | **pandas**: 2.3.3 | **matplotlib**: 3.10.7

### GPU Hardware — NVIDIA H200 NVL × 8
| GPU | Memory | CUDA |
|-----|--------|------|
| 8× NVIDIA H200 NVL | 139 GB each (1.1 TB total) | 12.8 |

### Confirmed Data Specs (from EDA)
- Image shape: **(11414, 60, 60, 1)** — grayscale 60×60, float32, range [0.0, 1.0]
- Kaggle shape: **(300, 60, 60, 1)**
- `img_size = 60` (native, no resize needed for custom CNNs)
- `img_size = 224` for ResNet18/EfficientNet (upscale + 3-channel)

### H200-Optimized Batch Sizes
| Model | Batch Size | Rationale |
|-------|-----------|-----------|
| Baseline CNN (60×60) | 512 | Tiny images, huge VRAM |
| ResidualCNN-B (60×60) | 512 | Same |
| ResNet18-C (224×224, 3ch) | 256 | Larger images + model |

### Multi-GPU Strategy
- Primary: use `cuda:0` (single GPU sufficient for 11K × 60×60 dataset)
- `torch.nn.DataParallel` available if needed
- All scripts auto-detect `torch.cuda.is_available()`
