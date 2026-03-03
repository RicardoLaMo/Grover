"""Tests for neighborhood builders and deterministic behavior."""

from __future__ import annotations

import torch

from thorn.data.synthetic import SyntheticDatasetGenerator
from thorn.views.knn import KNNBuilder
from thorn.views.time import TimeBuilder
from thorn.views.diffusion import DiffusionBuilder


def test_knn_builder_shapes() -> None:
    batch = SyntheticDatasetGenerator().generate(64, 8, 2, seed=42)
    out = KNNBuilder().build(batch.x, k=5, scalable_mode=False)
    assert out.edges.edge_index.shape == (2, 64 * 5)
    assert out.distances.shape == (64 * 5,)


def test_time_builder_deterministic() -> None:
    t = torch.arange(32, dtype=torch.float32)
    b = TimeBuilder()
    out1 = b.build(t, window=4)
    out2 = b.build(t, window=4)
    assert torch.equal(out1.edges.edge_index, out2.edges.edge_index)
    assert torch.allclose(out1.time_deltas, out2.time_deltas)


def test_diffusion_builder_shapes() -> None:
    # Ring graph
    n = 20
    src = torch.arange(n)
    dst = (src + 1) % n
    edge_index = torch.stack([src, dst], dim=0)
    out = DiffusionBuilder().build(edge_index, num_nodes=n, output_dim=6)
    assert out.coords.shape == (n, 6)
    assert out.pairwise_kernel is not None
    assert out.pairwise_kernel.shape == (n, n)
