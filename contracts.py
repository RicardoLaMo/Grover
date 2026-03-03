"""Module contracts and tensor-shape specifications for THORN.

Notation:
- N: number of nodes
- E: number of edges in a specific view
- E_u: number of edges in union graph
- H: number of attention heads
- M: number of views
- D: feature dimension
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from thorn.debug import assert_probability_simplex, assert_shape


@dataclass
class ViewEdges:
    """Single-view edges.

    Shapes:
        edge_index: [2, E]
        edge_weight: [E] or None
    """

    name: str
    edge_index: torch.Tensor
    edge_weight: Optional[torch.Tensor] = None

    def validate(self) -> None:
        assert_shape(self.edge_index, (2, -1), f"{self.name}.edge_index")
        if self.edge_weight is not None:
            assert_shape(self.edge_weight, (-1,), f"{self.name}.edge_weight")
            if self.edge_weight.shape[0] != self.edge_index.shape[1]:
                raise ValueError(
                    f"{self.name}: edge_weight length {self.edge_weight.shape[0]} "
                    f"!= num_edges {self.edge_index.shape[1]}"
                )


@dataclass
class UnifiedNeighborhood:
    """Union-graph representation with per-view masks and weights.

    Shapes:
        edge_index_union: [2, E_u]
        view_masks[m]: [E_u] (bool)
        view_weights[m]: [E_u] (float)
    """

    edge_index_union: torch.Tensor
    view_masks: Dict[str, torch.Tensor]
    view_weights: Dict[str, torch.Tensor]

    def validate(self) -> None:
        assert_shape(self.edge_index_union, (2, -1), "edge_index_union")
        num_union_edges = self.edge_index_union.shape[1]
        for name, mask in self.view_masks.items():
            assert_shape(mask, (num_union_edges,), f"view_masks[{name}]")
            if mask.dtype != torch.bool:
                raise ValueError(f"view_masks[{name}] must be bool")
        for name, weights in self.view_weights.items():
            assert_shape(weights, (num_union_edges,), f"view_weights[{name}]")


@dataclass
class AlignmentSignals:
    """Alignment signals used to condition router and attention.

    Shapes:
        pairwise_agreement: [N, M, M]
        per_view_confidence: [N, M]
    """

    pairwise_agreement: torch.Tensor
    per_view_confidence: torch.Tensor

    def validate(self, num_nodes: int, num_views: int) -> None:
        assert_shape(self.pairwise_agreement, (num_nodes, num_views, num_views), "pairwise_agreement")
        assert_shape(self.per_view_confidence, (num_nodes, num_views), "per_view_confidence")


@dataclass
class RouterOutput:
    """Router output with logits and probabilities.

    Equation:
        pi_{i,h,m} = softmax_m(logits_{i,h,m})

    Shapes:
        logits: [N, H, M]
        pi: [N, H, M]
        gate_logits: [N, H, M] (raw logits before top-k, for load balancing loss)
    """

    logits: torch.Tensor
    pi: torch.Tensor
    gate_logits: Optional[torch.Tensor] = None

    def validate(self) -> None:
        assert_shape(self.logits, (-1, -1, -1), "router.logits")
        assert_shape(self.pi, tuple(self.logits.shape), "router.pi")
        assert_probability_simplex(self.pi, dim=2, name="router.pi")
        if self.gate_logits is not None:
            assert_shape(self.gate_logits, tuple(self.logits.shape), "router.gate_logits")


@dataclass
class RoutedAttentionOutput:
    """Routed attention output.

    Shapes:
        node_states: [N, H, D_h]
        attn_weights: [E_u, H]
    """

    node_states: torch.Tensor
    attn_weights: torch.Tensor

    def validate(self, num_nodes: int, num_edges: int, num_heads: int) -> None:
        assert_shape(self.node_states, (num_nodes, num_heads, -1), "attn.node_states")
        assert_shape(self.attn_weights, (num_edges, num_heads), "attn.attn_weights")
