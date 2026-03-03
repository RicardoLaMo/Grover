"""Top-level THORN model with stacked transformer-style blocks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from thorn.config import AttentionConfig, ExperimentConfig
from thorn.contracts import RouterOutput, UnifiedNeighborhood
from thorn.layers.routed_attention import RoutedNeighborhoodAttention
from thorn.routing.router import ObserverRouter
from thorn.debug import assert_shape


@dataclass
class GraphContext:
    """Graph context used by THORN forward pass."""

    union: UnifiedNeighborhood
    edge_features: torch.Tensor
    observer_features: torch.Tensor
    gw_consensus: torch.Tensor | None = None  # [N, N] GW consensus for edge gating


class THORNBlock(nn.Module):
    """Pre-norm transformer block: LayerNorm -> RoutedAttention -> Residual -> LayerNorm -> FFN -> Residual."""

    def __init__(self, config: AttentionConfig, num_views: int = 3) -> None:
        super().__init__()
        self.config = config
        self.attn = RoutedNeighborhoodAttention(config, num_views=num_views)

        if config.use_layer_norm:
            self.norm1 = nn.LayerNorm(config.model_dim)
            self.norm2 = nn.LayerNorm(config.model_dim)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        # FFN: Linear -> GELU -> Dropout -> Linear
        ffn_dim = config.model_dim * config.ffn_multiplier
        self.ffn = nn.Sequential(
            nn.Linear(config.model_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(ffn_dim, config.model_dim),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        union: UnifiedNeighborhood,
        router_output: RouterOutput,
        edge_features: torch.Tensor | None = None,
        gw_gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass for a single THORN block.

        Args:
            x: [N, model_dim]
            union: unified neighborhood
            router_output: routing probabilities
            edge_features: optional [E_u, F_edge]
            gw_gate: optional [N, N] GW consensus for edge gating

        Returns:
            x: [N, model_dim]
        """
        # Pre-norm attention with residual
        residual = x
        x_normed = self.norm1(x)
        attn_out = self.attn(
            x=x_normed,
            union=union,
            router_output=router_output,
            edge_features=edge_features,
            gw_gate=gw_gate,
        )
        # Project from [N, H, D_h] -> [N, model_dim]
        attn_flat = attn_out.node_states.reshape(x.shape[0], -1)
        attn_proj = self.attn.out_proj(attn_flat)
        x = residual + self.dropout(attn_proj) if self.config.use_residual else self.dropout(attn_proj)

        # Pre-norm FFN with residual
        residual = x
        x_normed = self.norm2(x)
        ffn_out = self.ffn(x_normed)
        x = residual + self.dropout(ffn_out) if self.config.use_residual else self.dropout(ffn_out)

        return x


class THORNModel(nn.Module):
    """THORN model for node classification with stacked transformer blocks."""

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        self.config = config

        self.router = ObserverRouter(config.router)

        # Input projection to model_dim
        self.input_proj = nn.LazyLinear(config.attention.model_dim)

        # Stacked THORN blocks
        self.blocks = nn.ModuleList([
            THORNBlock(config.attention, num_views=config.router.num_views)
            for _ in range(config.attention.num_layers)
        ])

        # Final layer norm before classification
        if config.attention.use_layer_norm:
            self.final_norm = nn.LayerNorm(config.attention.model_dim)
        else:
            self.final_norm = nn.Identity()

        self.classifier = nn.Linear(config.attention.model_dim, config.data.num_classes)

    def forward(
        self,
        x: torch.Tensor,
        context: GraphContext,
        routing_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, RouterOutput]:
        """Forward pass.

        Args:
            x: [N, D]
            context: graph context with union edges and observer features.
            routing_override: Optional [N, H, M] routing probabilities.

        Returns:
            logits: [N, C]
            router_out: RouterOutput
        """
        assert_shape(x, (-1, -1), "x")
        assert_shape(context.observer_features, (x.shape[0], -1), "observer_features")

        # Route once; share pi across all layers
        router_out = self.router(context.observer_features)
        if routing_override is not None:
            assert_shape(routing_override, tuple(router_out.pi.shape), "routing_override")
            router_out = RouterOutput(logits=router_out.logits, pi=routing_override)
            router_out.validate()

        # Project input to model_dim
        h = self.input_proj(x)

        # Stack of THORN blocks
        for block in self.blocks:
            h = block(
                x=h,
                union=context.union,
                router_output=router_out,
                edge_features=context.edge_features,
                gw_gate=context.gw_consensus,
            )

        # Final norm + classify
        h = self.final_norm(h)
        logits = self.classifier(h)
        return logits, router_out
