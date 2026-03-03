"""LOF-like density ratio baseline."""

from __future__ import annotations

import torch

from thorn.debug import assert_shape


def compute_lof_ratio(knn_distances: torch.Tensor) -> torch.Tensor:
    """Compute a LOF-like local density ratio baseline.

    Args:
        knn_distances: [N, k]

    Returns:
        lof_ratio: [N]
    """
    assert_shape(knn_distances, (-1, -1), "knn_distances")
    local_density = 1.0 / (knn_distances.mean(dim=1) + 1e-6)
    neighbor_density = local_density.mean().expand_as(local_density)
    return neighbor_density / (local_density + 1e-6)
