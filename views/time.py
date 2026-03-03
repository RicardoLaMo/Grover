"""Temporal view builder implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from thorn.contracts import ViewEdges
from thorn.debug import assert_shape


@dataclass
class TimeOutput:
    """Temporal edges and deltas.

    Shapes:
        edges.edge_index: [2, E_time]
        time_deltas: [E_time]
        burstiness: [N] or None
    """

    edges: ViewEdges
    time_deltas: torch.Tensor
    burstiness: Optional[torch.Tensor]

    def validate(self, num_nodes: int) -> None:
        self.edges.validate()
        assert_shape(self.time_deltas, (self.edges.edge_index.shape[1],), "time.time_deltas")
        if self.burstiness is not None:
            assert_shape(self.burstiness, (num_nodes,), "time.burstiness")


class TimeBuilder:
    """Builder for deterministic temporal edges."""

    def build(self, timestamps: torch.Tensor, window: int, lag_k: int | None = None) -> TimeOutput:
        """Build temporal edges from sorted timestamps.

        Args:
            timestamps: [N] timestamps.
            window: Sliding window size for predecessor links.
            lag_k: Optional fixed-lag edge instead of full window.
        """
        assert_shape(timestamps, (-1,), "timestamps")
        n_nodes = timestamps.shape[0]
        order = torch.argsort(timestamps)

        src_list: list[int] = []
        dst_list: list[int] = []
        delta_list: list[float] = []

        for pos in range(n_nodes):
            dst_idx = int(order[pos].item())
            if lag_k is not None:
                begin = max(0, pos - lag_k)
            else:
                begin = max(0, pos - window)
            for prev in range(begin, pos):
                src_idx = int(order[prev].item())
                delta = float((timestamps[dst_idx] - timestamps[src_idx]).item())
                src_list.append(src_idx)
                dst_list.append(dst_idx)
                delta_list.append(delta)

        if not src_list:
            raise ValueError("Temporal builder produced no edges; increase window or number of nodes")

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        time_deltas = torch.tensor(delta_list, dtype=torch.float32)

        incoming_counts = torch.bincount(edge_index[1], minlength=n_nodes).float()
        burstiness = incoming_counts / max(incoming_counts.mean().item(), 1e-6)

        out = TimeOutput(
            edges=ViewEdges(name="time", edge_index=edge_index, edge_weight=torch.exp(-time_deltas / (time_deltas.std() + 1e-6))),
            time_deltas=time_deltas,
            burstiness=burstiness,
        )
        out.validate(n_nodes)
        return out
