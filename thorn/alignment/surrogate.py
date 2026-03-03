"""Verified surrogate alignment baselines."""

from __future__ import annotations

import torch


def neighborhood_overlap_score(src_neighbors: torch.Tensor, dst_neighbors: torch.Tensor) -> torch.Tensor:
    """Compute per-node neighborhood Jaccard overlap (vectorized).

    Args:
        src_neighbors: [N, K]
        dst_neighbors: [N, K]

    Returns:
        overlap: [N]
    """
    if src_neighbors.shape != dst_neighbors.shape:
        raise ValueError("Neighbor tensors must have equal shape")

    n, k = src_neighbors.shape

    # Expand for pairwise comparison: [N, K, 1] vs [N, 1, K]
    src_exp = src_neighbors.unsqueeze(2)  # [N, K, 1]
    dst_exp = dst_neighbors.unsqueeze(1)  # [N, 1, K]

    # Valid (non-padding) masks
    src_valid = src_neighbors != -1  # [N, K]
    dst_valid = dst_neighbors != -1  # [N, K]

    # Match matrix: [N, K, K] — True where src[i] == dst[j] and both valid
    match = (src_exp == dst_exp) & src_valid.unsqueeze(2) & dst_valid.unsqueeze(1)

    # Intersection: number of src elements that match any dst element
    inter = match.any(dim=2).sum(dim=1).float()  # [N]

    # Union: |valid_src| + |valid_dst| - |intersection|
    n_src_valid = src_valid.sum(dim=1).float()
    n_dst_valid = dst_valid.sum(dim=1).float()
    union = n_src_valid + n_dst_valid - inter

    overlap = torch.where(union > 0, inter / union, torch.zeros_like(inter))
    return overlap
