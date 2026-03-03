"""Union-view utilities."""

from __future__ import annotations

from typing import Dict, Literal

import torch

from thorn.contracts import UnifiedNeighborhood, ViewEdges


def merge_views(
    views: Dict[str, ViewEdges],
    degree_norm: Literal["none", "symmetric", "row"] = "none",
) -> UnifiedNeighborhood:
    """Merge view edges into a deduplicated union representation.

    Vectorized: encodes edges as src * N_max + dst, uses torch.unique for dedup.

    Args:
        views: Named view edges to merge.
        degree_norm: Normalization mode for view weights.
            - "none": raw weights (default, best empirical performance)
            - "symmetric": Â = A / sqrt(deg_i * deg_j)  (THORN++ spec eq. routedA)
            - "row": Â = A / deg_i  (row-stochastic normalization)

    Returns:
        UnifiedNeighborhood with per-view masks and per-view weights over union edges.
    """
    if not views:
        raise ValueError("No views provided")

    for view in views.values():
        view.validate()

    # Collect all edges and compute N_max for encoding
    all_src = []
    all_dst = []
    for view in views.values():
        all_src.append(view.edge_index[0])
        all_dst.append(view.edge_index[1])

    all_src_cat = torch.cat(all_src)
    all_dst_cat = torch.cat(all_dst)
    n_max = max(int(all_src_cat.max().item()), int(all_dst_cat.max().item())) + 1

    # Encode as unique edge IDs
    all_edge_ids = all_src_cat * n_max + all_dst_cat
    union_edge_ids, inverse = torch.unique(all_edge_ids, return_inverse=True)

    # Decode back to src, dst
    union_src = union_edge_ids // n_max
    union_dst = union_edge_ids % n_max
    edge_index_union = torch.stack([union_src, union_dst], dim=0).long()
    num_union = edge_index_union.shape[1]

    # Build per-view masks and weights using the same encoding
    view_masks: Dict[str, torch.Tensor] = {}
    view_weights: Dict[str, torch.Tensor] = {}

    eps_deg = 1e-12  # degree clamp for numerical safety

    for name, view in views.items():
        view_edge_ids = view.edge_index[0] * n_max + view.edge_index[1]
        # Find positions in union via searchsorted (union_edge_ids is sorted from torch.unique)
        positions = torch.searchsorted(union_edge_ids, view_edge_ids)

        mask = torch.zeros(num_union, dtype=torch.bool)
        raw_w = torch.zeros(num_union, dtype=torch.float32)

        mask[positions] = True
        if view.edge_weight is not None:
            raw_w[positions] = view.edge_weight.float()
        else:
            raw_w[positions] = 1.0

        if degree_norm == "symmetric":
            # THORN++ spec §1: Â_ij = A_ij / sqrt(deg_i * deg_j)
            deg = torch.zeros(n_max, dtype=torch.float32)
            deg.scatter_add_(0, union_dst.long(), raw_w)
            deg = deg.clamp_min(eps_deg)
            inv_sqrt_deg = deg.rsqrt()
            weights = raw_w * inv_sqrt_deg[union_dst.long()] * inv_sqrt_deg[union_src.long()]
        elif degree_norm == "row":
            # Row normalization: Â_ij = A_ij / deg_i (target-normalized)
            deg = torch.zeros(n_max, dtype=torch.float32)
            deg.scatter_add_(0, union_dst.long(), raw_w)
            deg = deg.clamp_min(eps_deg)
            weights = raw_w / deg[union_dst.long()]
        else:
            weights = raw_w

        view_masks[name] = mask
        view_weights[name] = weights

    out = UnifiedNeighborhood(
        edge_index_union=edge_index_union,
        view_masks=view_masks,
        view_weights=view_weights,
    )
    out.validate()
    return out
