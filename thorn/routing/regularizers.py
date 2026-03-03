"""Router regularizers including load balancing loss."""

from __future__ import annotations

import torch

from thorn.debug import assert_shape


def entropy_regularizer(pi: torch.Tensor) -> torch.Tensor:
    """Entropy control regularizer.

    Args:
        pi: [N, H, M]

    Returns:
        scalar tensor (-entropy so minimizing encourages peaky routing).
    """
    assert_shape(pi, (-1, -1, -1), "pi")
    ent = -(pi.clamp_min(1e-8) * torch.log(pi.clamp_min(1e-8))).sum(dim=2).mean()
    return -ent


def temporal_smoothness_regularizer(pi_t: torch.Tensor, pi_t1: torch.Tensor) -> torch.Tensor:
    """Temporal smoothness regularizer on routing probabilities.

    Args:
        pi_t: [N, H, M]
        pi_t1: [N, H, M]
    """
    assert_shape(pi_t, (-1, -1, -1), "pi_t")
    assert_shape(pi_t1, tuple(pi_t.shape), "pi_t1")
    return (pi_t - pi_t1).pow(2).mean()


def head_view_orthogonality_loss(pi: torch.Tensor) -> torch.Tensor:
    """Head-view orthogonality regularizer (THORN++ spec §5.2).

    Penalizes redundant routing across heads by encouraging orthogonal
    routing patterns per node:

        L_orth = Σ_i ||P_i P_i^T / ||P_i||_F^2 - I_H / H||_F^2

    where P_i ∈ R^{H×M} is the routing matrix at node i.

    Args:
        pi: [N, H, M] routing probabilities

    Returns:
        scalar loss (mean over nodes)
    """
    assert_shape(pi, (-1, -1, -1), "pi")
    n, h, m = pi.shape
    eps = 1e-8

    # P: [N, H, M]
    gram = torch.bmm(pi, pi.transpose(1, 2))  # [N, H, H] = P @ P^T
    norm_sq = (pi * pi).sum(dim=(1, 2)).clamp_min(eps)  # [N] = ||P_i||_F^2
    normalized = gram / norm_sq.unsqueeze(1).unsqueeze(2)  # [N, H, H]

    target = torch.eye(h, device=pi.device, dtype=pi.dtype).unsqueeze(0) / h  # [1, H, H]
    diff = normalized - target  # [N, H, H]
    loss = (diff * diff).sum(dim=(1, 2)).mean()  # mean over nodes of ||·||_F^2

    return loss


def load_balancing_loss(pi: torch.Tensor, gate_logits: torch.Tensor) -> torch.Tensor:
    """Switch Transformer-style load balancing loss.

    Encourages uniform load across views to prevent routing collapse.

    L_balance = M * sum_m(f_m * P_m)

    where:
        f_m = fraction of tokens routed to view m (from pi)
        P_m = mean gate probability for view m (from softmax of gate_logits)

    Args:
        pi: [N, H, M] routing probabilities (after top-k)
        gate_logits: [N, H, M] raw logits before top-k selection

    Returns:
        scalar loss
    """
    assert_shape(pi, (-1, -1, -1), "pi")
    assert_shape(gate_logits, tuple(pi.shape), "gate_logits")

    m = pi.shape[2]

    # f_m: fraction of tokens where view m gets highest routing weight
    # Average over heads
    f = pi.mean(dim=(0, 1))  # [M]

    # P_m: mean softmax probability from full gate logits (before top-k)
    p = torch.softmax(gate_logits, dim=2).mean(dim=(0, 1))  # [M]

    return float(m) * (f * p).sum()
