"""Ollivier-Ricci curvature proxy from local neighborhood structure."""

from __future__ import annotations

import torch

from thorn.debug import assert_shape


def compute_curvature_proxy(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Approximate Ollivier-Ricci curvature per node from neighbors-of-neighbors ratio.

    Positive curvature: neighbors are well-connected to each other (clustered).
    Negative curvature: neighbors are not connected (tree-like).

    The proxy is: curvature_i = (triangles at i) / (possible triangles at i)

    Args:
        edge_index: [2, E] edge indices
        num_nodes: number of nodes

    Returns:
        curvature: [N, 1]
    """
    assert_shape(edge_index, (2, -1), "edge_index")

    # Build adjacency as sparse set membership
    src, dst = edge_index[0], edge_index[1]

    # Build dense adjacency for triangle counting
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    adj[src, dst] = 1.0
    adj = torch.maximum(adj, adj.T)
    adj.fill_diagonal_(0.0)

    # Triangle count per node: diag(A^3) / 2
    A2 = adj @ adj
    triangles = (A2 * adj).sum(dim=1) / 2.0  # [N]

    # Degree
    deg = adj.sum(dim=1)  # [N]

    # Possible triangles: deg * (deg - 1) / 2
    possible = deg * (deg - 1.0) / 2.0

    # Clustering coefficient as curvature proxy
    curvature = torch.where(
        possible > 0,
        triangles / possible,
        torch.zeros_like(triangles),
    )

    return curvature.unsqueeze(1)  # [N, 1]


def compute_forman_ricci_curvature(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Forman-Ricci curvature per node (scalable alternative to ORC, critique 2.1).

    Forman-Ricci curvature for an edge (i,j) in an unweighted graph:
        F(i,j) = 4 - deg(i) - deg(j) + 3 * |triangles containing (i,j)|

    We aggregate per-node as the mean Forman curvature of incident edges.

    Complexity: O(E * k_max) vs O(E * N) for exact ORC.

    Args:
        edge_index: [2, E] edge indices
        num_nodes: number of nodes

    Returns:
        forman_curv: [N, 1]
    """
    assert_shape(edge_index, (2, -1), "edge_index")

    src, dst = edge_index[0], edge_index[1]

    # Degree
    deg = torch.bincount(dst, minlength=num_nodes).float()
    deg = deg + torch.bincount(src, minlength=num_nodes).float()
    # Avoid double counting for undirected
    deg = deg / 2.0
    deg = deg.clamp_min(1.0)

    # Build adjacency for triangle counting
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    adj[src, dst] = 1.0
    adj = torch.maximum(adj, adj.T)
    adj.fill_diagonal_(0.0)

    # Per-edge triangle count: number of common neighbors of (i,j)
    # common_neighbors(i,j) = (A[i] * A[j]).sum()
    # For edges in edge_index:
    common = (adj[src] * adj[dst]).sum(dim=1)  # [E]

    # Forman curvature per edge
    forman_edge = 4.0 - deg[src] - deg[dst] + 3.0 * common  # [E]

    # Aggregate per node: mean of incident edge curvatures
    node_sum = torch.zeros(num_nodes, dtype=torch.float32)
    node_count = torch.zeros(num_nodes, dtype=torch.float32)
    node_sum.scatter_add_(0, dst, forman_edge)
    node_count.scatter_add_(0, dst, torch.ones_like(forman_edge))

    forman_curv = node_sum / node_count.clamp_min(1.0)

    return forman_curv.unsqueeze(1)  # [N, 1]
