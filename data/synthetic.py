"""Synthetic dataset generator for deterministic THORN experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from thorn.debug import assert_shape


@dataclass
class SyntheticBatch:
    """Synthetic graph batch.

    Shapes:
        x: [N, D]
        y: [N]
        t: [N]
    """

    x: torch.Tensor
    y: torch.Tensor
    t: torch.Tensor

    def validate(self) -> None:
        assert_shape(self.x, (-1, -1), "batch.x")
        assert_shape(self.y, (self.x.shape[0],), "batch.y")
        assert_shape(self.t, (self.x.shape[0],), "batch.t")


class SyntheticDatasetGenerator:
    """Generate synthetic node-classification data with temporal drift.

    Supports both binary and multi-class (C >= 2) label generation.
    """

    def generate(
        self,
        num_nodes: int,
        num_features: int,
        num_classes: int,
        seed: int,
        drift_strength: float = 0.75,
    ) -> SyntheticBatch:
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)

        t = torch.arange(num_nodes, dtype=torch.float32)
        tn = t / max(float(num_nodes - 1), 1.0)

        x = torch.randn(num_nodes, num_features, generator=g)
        x[:, 0] += 0.8 * torch.sin(6.0 * tn)
        x[:, 1] += 0.6 * torch.cos(4.0 * tn)
        if num_features > 2:
            x[:, 2] += drift_strength * (tn - 0.5)

        if num_classes == 2:
            y = self._binary_labels(x, tn, num_nodes, drift_strength, g)
        else:
            y = self._multiclass_labels(x, tn, num_nodes, num_classes, num_features, drift_strength, g)

        batch = SyntheticBatch(x=x, y=y, t=t)
        batch.validate()
        return batch

    def _binary_labels(
        self,
        x: torch.Tensor,
        tn: torch.Tensor,
        num_nodes: int,
        drift_strength: float,
        g: torch.Generator,
    ) -> torch.Tensor:
        """Original binary label generation."""
        raw = (
            1.2 * x[:, 0]
            - 0.8 * x[:, 1]
            + 0.7 * x[:, 2] * x[:, 3]
            + 1.5 * (tn > 0.75).float() * drift_strength
            + 0.1 * torch.randn(num_nodes, generator=g)
        )
        probs = torch.sigmoid(raw)
        threshold = torch.quantile(probs, 0.68)
        return (probs > threshold).long()

    def _multiclass_labels(
        self,
        x: torch.Tensor,
        tn: torch.Tensor,
        num_nodes: int,
        num_classes: int,
        num_features: int,
        drift_strength: float,
        g: torch.Generator,
    ) -> torch.Tensor:
        """Multi-class labels via C cluster centers with temporal drift.

        Each class has a center that drifts over time. Soft assignment based
        on distance + noise creates overlapping decision boundaries.
        """
        # Generate class centers with temporal drift
        feat_dim = min(num_features, x.shape[1])
        centers = torch.randn(num_classes, feat_dim, generator=g) * 1.5

        # Temporal drift on centers: each class drifts in different directions
        drift_dirs = torch.randn(num_classes, feat_dim, generator=g)
        drift_dirs = drift_dirs / drift_dirs.norm(dim=1, keepdim=True).clamp_min(1e-6)

        # Compute per-node distance to each drifted center
        x_feat = x[:, :feat_dim]  # [N, feat_dim]
        logits = torch.zeros(num_nodes, num_classes)

        for c in range(num_classes):
            # Center drifts over time
            center_t = centers[c].unsqueeze(0) + drift_strength * tn.unsqueeze(1) * drift_dirs[c].unsqueeze(0)  # [N, feat_dim]
            dist = torch.norm(x_feat - center_t, dim=1)  # [N]
            logits[:, c] = -dist

        # Late-time decision boundary shift
        late_mask = (tn > 0.75).float()
        shift = torch.randn(num_classes, generator=g) * drift_strength
        logits += late_mask.unsqueeze(1) * shift.unsqueeze(0)

        # Add noise
        logits += 0.3 * torch.randn(num_nodes, num_classes, generator=g)

        return logits.argmax(dim=1).long()
