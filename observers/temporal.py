"""Temporal observer features."""

from __future__ import annotations

import torch

from thorn.debug import assert_shape


def compute_temporal_features(timestamps: torch.Tensor) -> torch.Tensor:
    """Compute temporal node features: delta, recency, burstiness proxy.

    Args:
        timestamps: [N]

    Returns:
        features: [N, 3]
    """
    assert_shape(timestamps, (-1,), "timestamps")
    n = timestamps.shape[0]
    sorted_t, idx = torch.sort(timestamps)

    deltas = torch.zeros_like(sorted_t)
    deltas[1:] = sorted_t[1:] - sorted_t[:-1]
    recency = (sorted_t - sorted_t.min()) / (sorted_t.max() - sorted_t.min() + 1e-6)

    window = 5
    burst = torch.zeros_like(sorted_t)
    for i in range(n):
        start = max(0, i - window + 1)
        local = deltas[start : i + 1]
        burst[i] = local.std(unbiased=False) / (local.mean().abs() + 1e-6)

    out_sorted = torch.stack([deltas, recency, burst], dim=1)
    out = torch.zeros_like(out_sorted)
    out[idx] = out_sorted
    return out
