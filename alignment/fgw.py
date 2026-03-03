"""Fused Gromov-Wasserstein (FGW) alignment for cross-view consistency."""

from __future__ import annotations

import torch

from thorn.alignment.sinkhorn import sinkhorn


def _structure_cost(adj: torch.Tensor) -> torch.Tensor:
    """Compute structure cost matrix from adjacency via shortest-path approximation.

    Uses A + A^2 / 2 as a proxy for shortest-path distances (captures 1-hop and 2-hop).
    """
    # Normalize adjacency
    deg = adj.sum(dim=1, keepdim=True).clamp_min(1.0)
    A_norm = adj / deg
    A2 = A_norm @ A_norm
    # Structure distance: nodes that share neighbors are close
    S = A_norm + 0.5 * A2
    # Convert similarity to distance
    C = 1.0 - S / S.max().clamp_min(1e-8)
    return C


def _gw_cost_matrix(
    C_a: torch.Tensor,
    C_b: torch.Tensor,
    T: torch.Tensor,
) -> torch.Tensor:
    """Compute the Gromov-Wasserstein cost tensor contracted with T.

    GW cost: sum_{i',j'} |C_a[i,i'] - C_b[j,j']|^2 * T[i',j']

    Efficient computation:
        L = C_a^2 @ mu_a @ 1^T + 1 @ nu_b^T @ C_b^2^T - 2 * C_a @ T @ C_b^T
    where mu_a = T @ 1, nu_b = T^T @ 1
    """
    n = C_a.shape[0]
    device = C_a.device

    mu = T.sum(dim=1)  # [N]
    nu = T.sum(dim=0)  # [N]

    Ca2 = C_a ** 2
    Cb2 = C_b ** 2

    # term1: Ca2 @ mu broadcast to [N, N]
    term1 = (Ca2 @ mu).unsqueeze(1).expand(n, n)
    # term2: Cb2^T @ nu broadcast to [N, N]
    term2 = (Cb2 @ nu).unsqueeze(0).expand(n, n)
    # term3: cross term
    term3 = 2.0 * C_a @ T @ C_b.T

    return term1 + term2 - term3


def fused_gromov_wasserstein(
    node_features_a: torch.Tensor,
    node_features_b: torch.Tensor,
    adj_a: torch.Tensor,
    adj_b: torch.Tensor,
    alpha: float = 0.5,
    sinkhorn_reg: float = 0.05,
    sinkhorn_iters: int = 30,
    fgw_iters: int = 5,
) -> torch.Tensor:
    """Compute Fused Gromov-Wasserstein transport plan.

    FGW = alpha * GW(C_A, C_B, T) + (1 - alpha) * <M_feat, T>

    Uses Frank-Wolfe outer loop with Sinkhorn inner solver.

    Args:
        node_features_a: [N, D] node embeddings from view A
        node_features_b: [N, D] node embeddings from view B
        adj_a: [N, N] adjacency of view A
        adj_b: [N, N] adjacency of view B
        alpha: trade-off between structure (GW) and feature (W) cost
        sinkhorn_reg: entropic regularization for Sinkhorn
        sinkhorn_iters: inner Sinkhorn iterations
        fgw_iters: outer Frank-Wolfe iterations

    Returns:
        T: [N, N] transport plan
    """
    n = node_features_a.shape[0]
    device = node_features_a.device

    # Structure cost matrices
    C_a = _structure_cost(adj_a)
    C_b = _structure_cost(adj_b)

    # Feature cost matrix: Euclidean distance
    M_feat = torch.cdist(node_features_a, node_features_b, p=2.0)

    # Uniform marginals
    mu = torch.ones(n, device=device) / n
    nu = torch.ones(n, device=device) / n

    # Initialize T with pure feature OT
    T = sinkhorn(M_feat, mu, nu, reg=sinkhorn_reg, num_iters=sinkhorn_iters)

    # Frank-Wolfe iterations
    for iteration in range(fgw_iters):
        # Compute GW gradient w.r.t. T
        gw_cost = _gw_cost_matrix(C_a, C_b, T)

        # Combined cost for FGW linearization
        M_combined = alpha * gw_cost + (1.0 - alpha) * M_feat

        # Solve linearized OT subproblem
        T_new = sinkhorn(M_combined, mu, nu, reg=sinkhorn_reg, num_iters=sinkhorn_iters)

        # Frank-Wolfe step size (diminishing)
        gamma = 2.0 / (iteration + 2.0)
        T = (1.0 - gamma) * T + gamma * T_new

    return T


def fgw_alignment_signals(
    node_features: torch.Tensor,
    adj_a: torch.Tensor,
    adj_b: torch.Tensor,
    alpha: float = 0.5,
    sinkhorn_reg: float = 0.05,
    sinkhorn_iters: int = 30,
    fgw_iters: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute alignment signals from FGW transport plan.

    Args:
        node_features: [N, D] shared node features
        adj_a, adj_b: [N, N] adjacency matrices for two views

    Returns:
        pairwise_agreement: [N] per-node transport self-mass (diagonal of T)
        per_view_confidence: [N, 2] marginal transport mass per view
    """
    T = fused_gromov_wasserstein(
        node_features, node_features,
        adj_a, adj_b,
        alpha=alpha,
        sinkhorn_reg=sinkhorn_reg,
        sinkhorn_iters=sinkhorn_iters,
        fgw_iters=fgw_iters,
    )

    # Diagonal: how much node i maps to itself across views
    overlap = T.diag() * T.shape[0]  # scale to [0, ~1] range

    # Marginal confidence: row and column sums indicate transportability
    conf_a = T.sum(dim=1) * T.shape[0]  # [N]
    conf_b = T.sum(dim=0) * T.shape[0]  # [N]
    per_view_conf = torch.stack([conf_a, conf_b], dim=1)  # [N, 2]

    return overlap, per_view_conf
