"""Observer-routed MoE mechanism with top-k gating and load balancing."""

from __future__ import annotations

import torch
from torch import nn

from thorn.config import RouterConfig
from thorn.contracts import RouterOutput
from thorn.debug import assert_shape


class ObserverRouter(nn.Module):
    """Route node-wise attention over view mixtures with MoE top-k gating.

    Architecture:
        observer_features -> LayerNorm -> Linear -> GELU -> Linear -> GELU -> Linear -> logits
        logits -> (optional noise) -> top-k selection -> renormalize -> pi

    Input shape:
        observer_features: [N, F_obs]
    Output shape:
        logits, pi: [N, H, M]
    """

    def __init__(self, config: RouterConfig) -> None:
        super().__init__()
        self.config = config
        out_dim = config.num_heads * config.num_views

        # 3-layer gating network with LayerNorm + GELU
        self.norm = nn.LazyLinear(config.hidden_dim)  # acts as input proj with LazyLinear
        self.net = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, out_dim),
        )

    def forward(self, observer_features: torch.Tensor) -> RouterOutput:
        assert_shape(observer_features, (-1, -1), "observer_features")
        n_nodes = observer_features.shape[0]
        h = self.config.num_heads
        m = self.config.num_views
        k = min(self.config.top_k, m)

        # Compute gate logits
        projected = self.norm(observer_features)
        raw_logits = self.net(projected).view(n_nodes, h, m)

        # Save raw logits for load balancing loss
        gate_logits = raw_logits.clone()

        # Add training noise for exploration
        if self.training and self.config.noise_std > 0:
            noise = torch.randn_like(raw_logits) * self.config.noise_std
            raw_logits = raw_logits + noise

        if k < m:
            # Top-k gating: select top-k views per head, zero others, renormalize
            topk_vals, topk_idx = torch.topk(raw_logits, k, dim=2)
            # Create mask
            mask = torch.zeros_like(raw_logits, dtype=torch.bool)
            mask.scatter_(2, topk_idx, True)
            # Zero out non-top-k
            masked_logits = raw_logits.masked_fill(~mask, float("-inf"))
            pi = torch.softmax(masked_logits, dim=2)
            # Ensure numerical stability — replace NaN from all-inf rows
            pi = torch.nan_to_num(pi, nan=0.0)
        else:
            # Dense softmax (no top-k)
            pi = torch.softmax(raw_logits, dim=2)

        out = RouterOutput(logits=raw_logits, pi=pi, gate_logits=gate_logits)
        out.validate()
        return out
