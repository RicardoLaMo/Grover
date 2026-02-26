"""
Replaced Token Detection (RTD) Pre-Training for TabTransformer.
===============================================================
Implements the semi-supervised pre-training procedure from Huang et al. (2020).

RTD randomly replaces k% of categorical features with random values from
the same column, then trains per-column binary classifiers on the contextual
embeddings to predict whether each feature was replaced.

Key design choices (from paper):
  - Dynamic replacement: different random replacements each epoch (Table 7)
  - Un-shared classifiers: one binary head per column (Table 7)
  - k=30% replacement rate (Table 6, not very sensitive)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional, Tuple

from src.models.tab_transformer import TabTransformer


class RTDPreTrainer(nn.Module):
    """Replaced Token Detection pre-training wrapper for TabTransformer.

    Wraps a TabTransformer and adds per-column binary classifiers
    for the RTD objective. After pre-training, discard the RTD heads
    and fine-tune the Transformer + MLP on labeled data.
    """

    def __init__(
        self,
        model: TabTransformer,
        replace_rate: float = 0.30,
    ):
        """
        Args:
            model: A TabTransformer instance (will be pre-trained in-place).
            replace_rate: fraction of features to replace per sample (default 30%).
        """
        super().__init__()
        self.model = model
        self.replace_rate = replace_rate
        self.num_cols = model.num_cat_cols

        # Per-column binary classifiers: contextual embedding → replaced?
        # Paper: "a different binary classifier is defined for each column"
        self.rtd_heads = nn.ModuleList([
            nn.Linear(model.d_model, 1)
            for _ in range(self.num_cols)
        ])

    def corrupt_batch(
        self,
        x_cat: torch.Tensor,
        num_categories_per_col: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dynamically replace features with random values from same column.

        Args:
            x_cat: (batch, num_cols) integer-encoded categoricals.
            num_categories_per_col: cardinality per column.

        Returns:
            x_corrupted: (batch, num_cols) with some features replaced.
            labels: (batch, num_cols) binary mask (1 = replaced, 0 = original).
        """
        batch_size, num_cols = x_cat.shape
        x_corrupted = x_cat.clone()
        labels = torch.zeros_like(x_cat, dtype=torch.float32)

        # Select which (sample, column) pairs to replace
        num_replace = max(1, int(num_cols * self.replace_rate))
        for i in range(batch_size):
            cols_to_replace = np.random.choice(num_cols, size=num_replace, replace=False)
            for col in cols_to_replace:
                # Replace with random class from [1, num_categories] (0 = missing)
                max_cat = num_categories_per_col[col]
                new_val = np.random.randint(1, max_cat + 1)
                if new_val != x_cat[i, col].item():
                    x_corrupted[i, col] = new_val
                    labels[i, col] = 1.0

        return x_corrupted, labels

    def forward(
        self,
        x_cat: torch.Tensor,
        x_cont: Optional[torch.Tensor] = None,
        num_categories_per_col: Optional[List[int]] = None,
    ) -> dict:
        """RTD forward pass: corrupt → encode → classify replacements.

        Args:
            x_cat: (batch, num_cols) original categoricals.
            x_cont: (batch, num_cont) continuous features (passed through but
                     not used in RTD loss — only categorical features are corrupted).
            num_categories_per_col: needed for generating random replacements.

        Returns:
            dict with:
                "loss": scalar RTD loss
                "rtd_logits": (batch, num_cols) per-column replacement predictions
                "labels": (batch, num_cols) ground truth replacement mask
                "accuracy": scalar accuracy of replacement detection
        """
        assert num_categories_per_col is not None, \
            "num_categories_per_col required for RTD corruption"

        # Dynamic corruption (different each call)
        x_corrupted, labels = self.corrupt_batch(x_cat, num_categories_per_col)
        x_corrupted = x_corrupted.to(x_cat.device)
        labels = labels.to(x_cat.device)

        # Get contextual embeddings from Transformer
        out = self.model(x_corrupted, x_cont, return_embeddings=True)
        embeddings = out["embeddings"]  # (batch, num_cols, d_model)

        # Per-column binary classification
        rtd_logits = []
        for col in range(self.num_cols):
            col_emb = embeddings[:, col, :]  # (batch, d_model)
            logit = self.rtd_heads[col](col_emb).squeeze(-1)  # (batch,)
            rtd_logits.append(logit)
        rtd_logits = torch.stack(rtd_logits, dim=1)  # (batch, num_cols)

        # Binary cross-entropy loss
        loss = nn.functional.binary_cross_entropy_with_logits(
            rtd_logits, labels, reduction="mean"
        )

        # Accuracy
        with torch.no_grad():
            preds = (rtd_logits > 0).float()
            accuracy = (preds == labels).float().mean()

        return {
            "loss": loss,
            "rtd_logits": rtd_logits,
            "labels": labels,
            "accuracy": accuracy,
        }

    def get_pretrained_model(self) -> TabTransformer:
        """Return the pre-trained TabTransformer (without RTD heads)."""
        return self.model


class MLMPreTrainer(nn.Module):
    """Masked Language Modeling pre-training wrapper for TabTransformer.

    Masks k% of categorical features and trains per-column multi-class
    classifiers to predict the original values.
    """

    def __init__(
        self,
        model: TabTransformer,
        num_categories_per_col: List[int],
        mask_rate: float = 0.30,
    ):
        super().__init__()
        self.model = model
        self.mask_rate = mask_rate
        self.num_cols = model.num_cat_cols

        # Per-column multi-class classifiers: embedding → original class
        self.mlm_heads = nn.ModuleList([
            nn.Linear(model.d_model, num_cat + 1)  # +1 for missing class
            for num_cat in num_categories_per_col
        ])

    def mask_batch(
        self,
        x_cat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mask random features (set to 0 = missing value embedding).

        Returns:
            x_masked: (batch, num_cols) with masked features → 0.
            mask: (batch, num_cols) boolean mask (True = masked).
            targets: (batch, num_cols) original values (only meaningful where masked).
        """
        batch_size, num_cols = x_cat.shape
        x_masked = x_cat.clone()
        mask = torch.zeros_like(x_cat, dtype=torch.bool)
        targets = x_cat.clone()

        num_mask = max(1, int(num_cols * self.mask_rate))
        for i in range(batch_size):
            cols_to_mask = np.random.choice(num_cols, size=num_mask, replace=False)
            for col in cols_to_mask:
                x_masked[i, col] = 0  # missing value embedding
                mask[i, col] = True

        return x_masked, mask, targets

    def forward(
        self,
        x_cat: torch.Tensor,
        x_cont: Optional[torch.Tensor] = None,
    ) -> dict:
        """MLM forward: mask → encode → predict originals.

        Returns:
            dict with "loss", "accuracy", "mask"
        """
        x_masked, mask, targets = self.mask_batch(x_cat)
        x_masked = x_masked.to(x_cat.device)
        mask = mask.to(x_cat.device)
        targets = targets.to(x_cat.device)

        out = self.model(x_masked, x_cont, return_embeddings=True)
        embeddings = out["embeddings"]  # (batch, num_cols, d_model)

        total_loss = 0.0
        correct = 0
        total = 0

        for col in range(self.num_cols):
            col_mask = mask[:, col]  # (batch,)
            if col_mask.sum() == 0:
                continue

            col_emb = embeddings[col_mask, col, :]  # (masked_count, d_model)
            col_targets = targets[col_mask, col]     # (masked_count,)
            logits = self.mlm_heads[col](col_emb)    # (masked_count, num_classes)

            total_loss += nn.functional.cross_entropy(logits, col_targets, reduction="sum")
            correct += (logits.argmax(dim=-1) == col_targets).sum().item()
            total += col_mask.sum().item()

        loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)

        return {
            "loss": loss,
            "accuracy": accuracy,
            "mask": mask,
        }

    def get_pretrained_model(self) -> TabTransformer:
        """Return the pre-trained TabTransformer (without MLM heads)."""
        return self.model
