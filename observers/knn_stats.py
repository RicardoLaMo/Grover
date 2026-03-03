"""kNN distance summary features."""

from __future__ import annotations

import torch

from thorn.debug import assert_shape


def compute_knn_distance_stats(knn_distances: torch.Tensor) -> torch.Tensor:
    """Compute per-node kNN distance stats: mean/var/min/max.

    Args:
        knn_distances: [N, k]

    Returns:
        stats: [N, 4]
    """
    assert_shape(knn_distances, (-1, -1), "knn_distances")
    mean = knn_distances.mean(dim=1)
    var = knn_distances.var(dim=1, unbiased=False)
    min_v = knn_distances.min(dim=1).values
    max_v = knn_distances.max(dim=1).values
    return torch.stack([mean, var, min_v, max_v], dim=1)
