"""
05_kaggle_predict.py — Generate Kaggle Extra-Credit Submission

This script:
  1. Loads the best trained model (based on val macro-F1)
  2. Runs inference on the Kaggle test set (300 balanced images)
  3. Saves a submission CSV: columns [Id, Predicted]

IMPORTANT: Run this ONLY ONCE after the model is fully validated and optimized
using the trainvalid set. Running it multiple times while adjusting the model
to improve Kaggle performance would constitute data leakage (effectively using
the test set as a validation set, which is methodologically invalid).

Kaggle label encoding:
  0 → Healthy
  1 → Pre-existing Conditions
  2 → Effusion / Mass (Urgent)

Run: python src/05_kaggle_predict.py
Output: output/predictions/kaggle_submission.csv
Gate G6: passes when CSV has 300 rows and valid Id / Predicted columns
"""

import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, classification_report, accuracy_score

from data_pipeline import get_kaggle_loader, CONFIG as DATA_CONFIG
from baseline_cnn import BaselineCNN
from improved_cnn import ImprovedCNN_B, build_resnet18_classifier
from utils import setup_logging, plot_confusion_matrix

# ─── Configuration ────────────────────────────────────────────────────────────

CONFIG = {
    "kaggle_images":  "ps4_kaggle_images-1.npy",
    "kaggle_labels":  "ps4_kaggle_labels.csv",

    # Paths to saved model checkpoints — choose the best one
    "best_model_path": "output/models/improved_B.pth",   # update as needed
    "model_type":      "B",    # "A" for baseline, "B" for residual, "C" for ResNet18

    # Output
    "submission_path": "output/predictions/kaggle_submission.csv",
    "figures_dir":     "output/figures",

    # Must match the training configuration
    "img_size":    64,    # use 224 if model_type is "C"
    "n_channels":  1,     # use 3 if model_type is "C"
    "batch_size":  64,

    "random_seed": 42,
}

CLASS_NAMES = ["Healthy (0)", "Pre-existing (1)", "Effusion/Mass (2)"]


# ─── Load Model Helper ────────────────────────────────────────────────────────

def load_best_model(config: dict, device: torch.device):
    """
    Load the best model checkpoint based on model_type.

    Args:
        config: Configuration dict specifying model_type and best_model_path.
        device: Target device.

    Returns:
        Loaded model in eval mode.
    """
    model_type = config["model_type"]

    if model_type == "A":
        model = BaselineCNN(n_channels=config["n_channels"])
    elif model_type == "B":
        model = ImprovedCNN_B(n_channels=config["n_channels"])
    elif model_type == "C":
        # ResNet18: 3 channels, 224×224
        model = build_resnet18_classifier(n_classes=3, freeze_backbone=False)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose A, B, or C.")

    state_dict = torch.load(config["best_model_path"], map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()   # disable dropout, use BN running stats
    return model


# ─── Inference ────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_kaggle(model, loader, device) -> tuple:
    """
    Run inference on the Kaggle test set and collect predictions.

    Args:
        model:  Loaded trained model (eval mode).
        loader: Kaggle DataLoader (images only, dummy labels).
        device: Compute device.

    Returns:
        y_pred:  (300,) array of predicted class indices
        y_proba: (300, 3) array of softmax class probabilities
    """
    all_preds = []
    all_proba = []

    for X, _ in loader:   # _ = dummy labels (zeros), ignored here
        X = X.to(device)
        logits = model(X)

        # Softmax converts raw logits to class probabilities
        proba = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(proba, axis=1)

        all_preds.extend(preds.tolist())
        all_proba.extend(proba.tolist())

    return np.array(all_preds, dtype=int), np.array(all_proba)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    logger = setup_logging()
    torch.manual_seed(CONFIG["random_seed"])
    np.random.seed(CONFIG["random_seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    print("\n" + "="*60)
    print("  GATE G6: Kaggle Prediction — Extra Credit")
    print("="*60)

    # ── Load Kaggle Images ──
    logger.info(f"Loading Kaggle images from {CONFIG['kaggle_images']}")
    kag_images = np.load(CONFIG["kaggle_images"])
    kag_labels_df = pd.read_csv(CONFIG["kaggle_labels"])
    kag_labels_true = kag_labels_df["Predicted"].values   # ground truth for scoring
    logger.info(f"Kaggle images shape: {kag_images.shape}")
    logger.info(f"Kaggle ground-truth distribution: "
                + str({c: int((kag_labels_true == c).sum()) for c in range(3)}))

    # ── Build DataLoader ──
    data_cfg = DATA_CONFIG.copy()
    data_cfg["img_size"]   = CONFIG["img_size"]
    data_cfg["batch_size"] = CONFIG["batch_size"]
    kag_loader = get_kaggle_loader(kag_images, config=data_cfg,
                                   n_channels=CONFIG["n_channels"])

    # ── Load Model ──
    if not os.path.exists(CONFIG["best_model_path"]):
        logger.error(f"Model checkpoint not found: {CONFIG['best_model_path']}")
        logger.error("Run 02_baseline_cnn.py or 03_improved_cnn.py first.")
        return

    model = load_best_model(CONFIG, device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Loaded {CONFIG['model_type']} model ({total_params:,} params)"
                f" from {CONFIG['best_model_path']}")

    # ── Predict ──
    logger.info("Running inference on Kaggle test set...")
    y_pred, y_proba = predict_kaggle(model, kag_loader, device)

    # ── Score Against Ground Truth ──
    # (Only possible because GT labels are provided — in a real Kaggle, you'd submit blind)
    acc  = accuracy_score(kag_labels_true, y_pred)
    mf1  = f1_score(kag_labels_true, y_pred, average="macro", zero_division=0)
    print(f"\n  Kaggle Test Performance (n=300, balanced):")
    print(f"    Accuracy:  {acc:.4f}")
    print(f"    Macro-F1:  {mf1:.4f}")
    print(f"\n  Per-class report:")
    print(classification_report(kag_labels_true, y_pred, target_names=CLASS_NAMES,
                                zero_division=0))

    # ── Confusion Matrix on Kaggle Set ──
    plot_confusion_matrix(
        kag_labels_true, y_pred, normalize=True,
        title=f"Kaggle Test Set — Confusion Matrix ({CONFIG['model_type']} model)\n"
              f"Macro-F1 = {mf1:.4f}",
        save_path=os.path.join(CONFIG["figures_dir"],
                               f"kaggle_confusion_matrix_{CONFIG['model_type']}.png"),
    )

    # ── Save Submission CSV ──
    # Format: Id (0-299 integer), Predicted (0/1/2 integer)
    submission_df = pd.DataFrame({
        "Id":        np.arange(len(y_pred), dtype=int),   # 0, 1, 2, ..., 299
        "Predicted": y_pred.astype(int),
    })
    os.makedirs(os.path.dirname(CONFIG["submission_path"]), exist_ok=True)
    submission_df.to_csv(CONFIG["submission_path"], index=False)
    logger.info(f"\nSubmission saved → {CONFIG['submission_path']}")
    print(f"\nFirst 10 rows of submission:")
    print(submission_df.head(10).to_string(index=False))

    # ── Gate G6 Validation ──
    print("\n" + "="*60)
    print("  GATE G6 CHECKLIST:")
    assert len(submission_df) == 300, f"Expected 300 rows, got {len(submission_df)}"
    assert list(submission_df.columns) == ["Id", "Predicted"], \
        f"Wrong columns: {submission_df.columns.tolist()}"
    assert set(submission_df["Id"].tolist()) == set(range(300)), \
        "Id column must be 0-299"
    assert submission_df["Predicted"].isin([0, 1, 2]).all(), \
        "Predicted must be 0, 1, or 2"
    print(f"  [✓] 300 rows: OK")
    print(f"  [✓] Columns [Id, Predicted]: OK")
    print(f"  [✓] Id range 0-299: OK")
    print(f"  [✓] Predicted values ∈ {{0, 1, 2}}: OK")
    print(f"  [✓] File saved: {CONFIG['submission_path']}")
    print("  GATE G6: PASSED")
    print("="*60)


if __name__ == "__main__":
    main()
