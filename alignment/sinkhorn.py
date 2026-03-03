"""Entropy-regularized optimal transport via Sinkhorn-Knopp algorithm."""

from __future__ import annotations

import torch


def sinkhorn(
    cost: torch.Tensor,
    mu: torch.Tensor,
    nu: torch.Tensor,
    reg: float = 0.05,
    num_iters: int = 30,
) -> torch.Tensor:
    """Compute entropy-regularized OT plan via Sinkhorn-Knopp in log-domain.

    Args:
        cost: [N, N] cost matrix
        mu: [N] source marginal (sums to 1)
        nu: [N] target marginal (sums to 1)
        reg: entropic regularization strength
        num_iters: number of Sinkhorn iterations

    Returns:
        T: [N, N] transport plan
    """
    n = cost.shape[0]
    device = cost.device

    # Log-domain Sinkhorn for numerical stability
    log_K = -cost / reg  # [N, N]
    log_mu = torch.log(mu.clamp_min(1e-10))
    log_nu = torch.log(nu.clamp_min(1e-10))

    # Initialize dual variables
    f = torch.zeros(n, device=device)  # [N]
    g = torch.zeros(n, device=device)  # [N]

    for _ in range(num_iters):
        # Update f: f = log_mu - logsumexp(log_K + g, dim=1)
        f = log_mu - torch.logsumexp(log_K + g.unsqueeze(0), dim=1)
        # Update g: g = log_nu - logsumexp(log_K + f, dim=0)
        g = log_nu - torch.logsumexp(log_K + f.unsqueeze(1), dim=0)

    # Compute transport plan
    log_T = f.unsqueeze(1) + log_K + g.unsqueeze(0)
    T = torch.exp(log_T)

    return T
