"""Stage 1 tests for shape contracts and invariants."""

from __future__ import annotations

import pytest
import torch

from thorn.contracts import RouterOutput, UnifiedNeighborhood, ViewEdges
from thorn.debug import assert_probability_simplex


def test_view_edges_validate_passes() -> None:
    edges = ViewEdges(
        name="knn",
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
        edge_weight=torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
    )
    edges.validate()


def test_unified_neighborhood_validate_passes() -> None:
    edge_index = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 2]], dtype=torch.long)
    union = UnifiedNeighborhood(
        edge_index_union=edge_index,
        view_masks={
            "knn": torch.tensor([True, True, False, True]),
            "time": torch.tensor([False, True, True, False]),
        },
        view_weights={
            "knn": torch.tensor([1.0, 1.0, 0.0, 1.0]),
            "time": torch.tensor([0.0, 1.0, 1.0, 0.0]),
        },
    )
    union.validate()


def test_router_output_simplex_passes() -> None:
    logits = torch.randn(5, 2, 3)
    pi = torch.softmax(logits, dim=2)
    out = RouterOutput(logits=logits, pi=pi)
    out.validate()


def test_simplex_assert_fails_for_invalid_probs() -> None:
    bad = torch.tensor([[[0.2, 0.2, 0.2]]], dtype=torch.float32)
    with pytest.raises(ValueError):
        assert_probability_simplex(bad, dim=2)
