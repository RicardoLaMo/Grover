r"""Local Intrinsic Dimensionality (LID) estimator.

Levina-Bickel MLE baseline:
    LID(x) = -\left( \frac{1}{k-1} \sum_{j=1}^{k-1} \log \frac{r_j(x)}{r_k(x)} \right)^{-1}
"""

from __future__ import annotations

import torch

from thorn.debug import assert_shape


def estimate_lid_levina_bickel(knn_distances: torch.Tensor, k: int) -> torch.Tensor:
    """Estimate per-node LID from sorted kNN distances.

    Args:
        knn_distances: [N, k] sorted ascending by neighbor rank.
        k: Number of neighbors.

    Returns:
        lid: [N]
    """
    assert_shape(knn_distances, (-1, k), "knn_distances")
    eps = 1e-8
    r_k = knn_distances[:, -1].clamp_min(eps)
    ratios = (knn_distances[:, :-1].clamp_min(eps) / r_k.unsqueeze(1)).clamp_min(eps)
    mean_log = torch.log(ratios).mean(dim=1)
    lid = -1.0 / mean_log.clamp_max(-eps)
    return lid


def multi_scale_lid(knn_distances: torch.Tensor, scales: list[int] | None = None) -> torch.Tensor:
    """Compute LID at multiple scales by slicing knn_distances.

    Args:
        knn_distances: [N, K] sorted ascending (K >= max(scales))
        scales: list of k values to compute LID at. Default: [5, 10, 20]

    Returns:
        lid_multi: [N, len(scales)]
    """
    if scales is None:
        scales = [5, 10, 20]

    k_max = knn_distances.shape[1]
    results = []
    for k in scales:
        k_use = min(k, k_max)
        lid = estimate_lid_levina_bickel(knn_distances[:, :k_use], k=k_use)
        results.append(lid)

    return torch.stack(results, dim=1)  # [N, len(scales)]
