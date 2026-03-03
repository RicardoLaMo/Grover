"""Debug utilities for tensor-shape and invariant checks."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DebugSettings:
    """Runtime debug settings.

    Attributes:
        enabled: If true, run shape/invariant assertions.
    """

    enabled: bool = True


def assert_rank(tensor: torch.Tensor, expected_rank: int, name: str) -> None:
    """Assert tensor rank.

    Args:
        tensor: Tensor to validate.
        expected_rank: Expected number of dimensions.
        name: Name used in error messages.
    """
    if tensor.ndim != expected_rank:
        raise ValueError(f"{name} rank mismatch: got {tensor.ndim}, expected {expected_rank}")


def assert_shape(tensor: torch.Tensor, expected: tuple[int, ...], name: str) -> None:
    """Assert tensor shape with optional wildcards.

    Use `-1` in `expected` as a wildcard for any positive dimension.
    """
    assert_rank(tensor, len(expected), name)
    for idx, (got_dim, exp_dim) in enumerate(zip(tensor.shape, expected)):
        if exp_dim != -1 and got_dim != exp_dim:
            raise ValueError(
                f"{name} shape mismatch at dim {idx}: got {got_dim}, expected {exp_dim}"
            )


def assert_probability_simplex(
    probs: torch.Tensor,
    *,
    dim: int,
    atol: float = 1e-5,
    name: str = "probs",
) -> None:
    """Assert a tensor lies on a probability simplex along `dim`.

    Checks:
    1. non-negativity
    2. sum to one along specified axis
    """
    if torch.any(probs < -atol):
        raise ValueError(f"{name} has negative entries below tolerance {atol}")
    row_sums = probs.sum(dim=dim)
    ones = torch.ones_like(row_sums)
    if not torch.allclose(row_sums, ones, atol=atol, rtol=0.0):
        raise ValueError(f"{name} does not sum to 1 along dim={dim}")
