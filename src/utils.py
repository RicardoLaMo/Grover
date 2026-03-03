"""
utils.py — Shared utility functions for PS4 Lung X-Ray Classifier

Provides: logging setup, plotting helpers (confusion matrix, ROC curves,
training curves), and early stopping callback.

Author: DTSC 8120 PS4
"""

import logging
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, ConfusionMatrixDisplay, confusion_matrix
)


# ─── Logging ───────────────────────────────────────────────────────────────

def setup_logging(log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    Configure a logger that writes to stdout and optionally to a file.

    Args:
        log_file: If given, also write logs to this file path.
        level:    Logging level (default: INFO).

    Returns:
        Configured Logger instance.
    """
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(level=level, format=fmt, handlers=handlers)
    return logging.getLogger("ps4")


# ─── Plotting: Confusion Matrix ─────────────────────────────────────────────

CLASS_NAMES = ["Healthy (0)", "Pre-existing (1)", "Effusion/Mass (2)"]


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalize: bool = True,
    title: str = "Confusion Matrix",
    save_path: str = None,
) -> None:
    """
    Plot a labeled confusion matrix with optional normalization.

    Args:
        y_true:    Ground-truth class labels.
        y_pred:    Predicted class labels.
        normalize: If True, show row-normalized values (recall per class).
        title:     Plot title.
        save_path: If given, save the figure to this path.
    """
    norm = "true" if normalize else None
    cm = confusion_matrix(y_true, y_pred, normalize=norm)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logging.getLogger("ps4").info(f"Confusion matrix saved → {save_path}")

    plt.show()


# ─── Plotting: ROC Curves (One-vs-Rest) ─────────────────────────────────────

def plot_roc_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str = "ROC Curves (One-vs-Rest)",
    save_path: str = None,
) -> dict:
    """
    Plot per-class ROC curves using the One-vs-Rest strategy.

    Because our problem is multi-class, we compute a separate ROC curve
    for each class by treating it as binary (class_i vs. all others).
    The Area Under the Curve (AUC) summarizes each curve in one number.
    Macro-AUC = mean of per-class AUCs, which is unaffected by class imbalance.

    Args:
        y_true:    1-D array of true class indices (0, 1, 2).
        y_proba:   2-D array of shape (N, 3) with softmax probabilities.
        title:     Plot title.
        save_path: If given, save figure to this path.

    Returns:
        Dict mapping class index to AUC score.
    """
    n_classes = y_proba.shape[1]
    colors = ["steelblue", "darkorange", "forestgreen"]
    auc_scores = {}

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random (AUC=0.50)")

    for i, (color, name) in enumerate(zip(colors, CLASS_NAMES)):
        # Binarize: class i is 1, all others are 0
        y_bin = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        auc_scores[i] = roc_auc
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={roc_auc:.3f})")

    macro_auc = np.mean(list(auc_scores.values()))
    ax.set_title(f"{title}\nMacro-AUC = {macro_auc:.3f}", fontsize=12, fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logging.getLogger("ps4").info(f"ROC curves saved → {save_path}")

    plt.show()
    return auc_scores


# ─── Plotting: Training Curves ───────────────────────────────────────────────

def plot_training_curves(
    train_losses: list,
    val_losses: list,
    val_f1s: list,
    title: str = "Training History",
    save_path: str = None,
) -> None:
    """
    Plot loss and validation macro-F1 over epochs side by side.

    Args:
        train_losses: List of training loss per epoch.
        val_losses:   List of validation loss per epoch.
        val_f1s:      List of validation macro-F1 per epoch.
        title:        Overall figure title.
        save_path:    If given, save figure to this path.
    """
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    # ─ Loss ─
    ax1.plot(epochs, train_losses, "b-o", ms=4, label="Train Loss")
    ax1.plot(epochs, val_losses, "r-o", ms=4, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("Loss over Epochs")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # ─ Macro-F1 ─
    ax2.plot(epochs, val_f1s, "g-o", ms=4, label="Val Macro-F1")
    ax2.axhline(1 / 3, ls="--", color="gray", alpha=0.7, label="Random (0.33)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Macro-F1")
    ax2.set_title("Validation Macro-F1 over Epochs")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logging.getLogger("ps4").info(f"Training curves saved → {save_path}")

    plt.show()


# ─── Early Stopping ──────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Stop training when a monitored metric stops improving.

    Tracks the best value seen so far. If no improvement is seen for
    `patience` consecutive epochs, sets the `stop` flag to True.
    Also saves the best model weights to disk.

    Args:
        patience:  Number of epochs to wait before stopping.
        min_delta: Minimum change to count as an improvement.
        mode:      'max' (higher is better, e.g. F1) or 'min' (lower is better, e.g. loss).
        save_path: Where to save the best model checkpoint.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = "max",
        save_path: str = None,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.save_path = save_path

        self.counter = 0
        self.best = float("-inf") if mode == "max" else float("inf")
        self.stop = False
        self._logger = logging.getLogger("ps4")

    def __call__(self, metric: float, model) -> bool:
        """
        Check if the model improved; update best checkpoint if so.

        Args:
            metric: Current epoch's metric value.
            model:  PyTorch model (saved if improved).

        Returns:
            True if training should stop, False otherwise.
        """
        import torch

        improved = (
            metric > self.best + self.min_delta
            if self.mode == "max"
            else metric < self.best - self.min_delta
        )

        if improved:
            self.best = metric
            self.counter = 0
            if self.save_path:
                os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
                torch.save(model.state_dict(), self.save_path)
                self._logger.info(
                    f"  ✓ New best {self.mode}={metric:.4f} — checkpoint saved"
                )
        else:
            self.counter += 1
            self._logger.info(
                f"  EarlyStopping: no improvement for {self.counter}/{self.patience} epochs"
            )
            if self.counter >= self.patience:
                self.stop = True
                self._logger.info("  ✗ Early stopping triggered.")

        return self.stop
