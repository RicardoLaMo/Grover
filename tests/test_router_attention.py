"""Tests for router simplex and attention normalization invariants."""

from __future__ import annotations

import torch

from thorn.config import AttentionConfig, RouterConfig
from thorn.contracts import RouterOutput, UnifiedNeighborhood
from thorn.layers.routed_attention import RoutedNeighborhoodAttention
from thorn.routing.router import ObserverRouter


def test_router_simplex() -> None:
    cfg = RouterConfig(num_heads=3, num_views=3, hidden_dim=16)
    router = ObserverRouter(cfg)
    obs = torch.randn(12, 10)
    out = router(obs)
    assert out.pi.shape == (12, 3, 3)
    sums = out.pi.sum(dim=2)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)


def test_attention_normalizes_over_incoming_neighbors() -> None:
    # Union graph with three views and 6 edges.
    edge_index = torch.tensor(
        [[0, 2, 3, 1, 0, 2], [1, 1, 1, 3, 3, 3]], dtype=torch.long
    )
    union = UnifiedNeighborhood(
        edge_index_union=edge_index,
        view_masks={
            "knn": torch.tensor([True, True, True, False, False, False]),
            "time": torch.tensor([False, False, False, True, True, False]),
            "diffusion": torch.tensor([True, False, True, False, True, True]),
        },
        view_weights={
            "knn": torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
            "time": torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 0.0]),
            "diffusion": torch.tensor([0.5, 0.0, 0.4, 0.0, 0.2, 0.9]),
        },
    )

    x = torch.randn(4, 7)
    logits = torch.randn(4, 2, 3)
    pi = torch.softmax(logits, dim=2)
    router_out = RouterOutput(logits=logits, pi=pi)

    layer = RoutedNeighborhoodAttention(AttentionConfig(model_dim=16, head_dim=8, adjacency_mode="bias"))
    out = layer(x=x, union=union, router_output=router_out, edge_features=torch.randn(6, 3))

    attn = out.attn_weights
    dst = edge_index[1]
    for h in range(attn.shape[1]):
        for node in [1, 3]:
            mask = dst == node
            s = float(attn[mask, h].sum().item())
            assert abs(s - 1.0) < 1e-5
