"""Routed neighborhood attention layer.

Mixture adjacency per node/head:
    A_tilde_{ij}^{(i,h)} = sum_m pi_{i,h,m} * A_{ij}^{(m)}
"""

from __future__ import annotations

import math

import torch
from torch import nn

from thorn.config import AttentionConfig
from thorn.contracts import RoutedAttentionOutput, RouterOutput, UnifiedNeighborhood
from thorn.debug import assert_shape


class RoutedNeighborhoodAttention(nn.Module):
    """Multi-head neighborhood attention over routed view mixtures.

    Input shapes:
        x: [N, D]
        router.pi: [N, H, M]
        union.edge_index_union: [2, E_u]

    Output shapes:
        node_states: [N, H, D_h]
        attn_weights: [E_u, H]
    """

    def __init__(self, config: AttentionConfig, num_views: int = 3) -> None:
        super().__init__()
        self.config = config
        self.num_views = num_views
        self.q_proj = nn.LazyLinear(config.model_dim)
        self.k_proj = nn.LazyLinear(config.model_dim)
        self.v_proj = nn.LazyLinear(config.model_dim)

        # View-specific Q/K/V adapters (THORN++ critique 4.2):
        # LoRA-style low-rank per-view delta: W_Q^(m) = W_Q + ΔW_Q^(m)
        if config.view_specific_projections:
            rank = config.lora_rank if config.lora_rank > 0 else max(config.head_dim // 2, 8)
            self.view_q_down = nn.ModuleList([nn.Linear(config.model_dim, rank) for _ in range(num_views)])
            self.view_q_up = nn.ModuleList([nn.Linear(rank, config.model_dim) for _ in range(num_views)])
            self.view_k_down = nn.ModuleList([nn.Linear(config.model_dim, rank) for _ in range(num_views)])
            self.view_k_up = nn.ModuleList([nn.Linear(rank, config.model_dim) for _ in range(num_views)])
            self.view_v_down = nn.ModuleList([nn.Linear(config.model_dim, rank) for _ in range(num_views)])
            self.view_v_up = nn.ModuleList([nn.Linear(rank, config.model_dim) for _ in range(num_views)])
            # Initialize up projections near zero for stable start
            for ups in [self.view_q_up, self.view_k_up, self.view_v_up]:
                for up in ups:
                    nn.init.zeros_(up.weight)
                    nn.init.zeros_(up.bias)

        # Adaptive temperature for log-barrier (critique 3.3/8.1):
        # τ_h = softplus(s_h), initialized so τ ≈ 1.0
        if config.adaptive_tau:
            # softplus(0.5413) ≈ 1.0; n_heads determined at first forward
            self._tau_logit = None  # lazy init

        # Per-head edge bias MLP: φ_h(e_ij), maps edge features to per-head scalars
        # Hidden dim = head_dim, output = n_heads (determined by first forward via lazy)
        self.edge_bias_hidden = nn.LazyLinear(config.head_dim)
        self._edge_bias_out = None  # init on first forward when n_heads is known
        # Output projection: [N, H, D_h] -> [N, model_dim]
        self.out_proj = nn.LazyLinear(config.model_dim)

    def _edge_softmax(
        self,
        scores: torch.Tensor,
        dst: torch.Tensor,
        valid: torch.Tensor,
        n_nodes: int,
        n_heads: int,
    ) -> torch.Tensor:
        """Vectorized edge softmax: normalize scores over incoming neighbors per node/head.

        Uses scatter-based max subtraction and sum for numerical stability.
        All ops are [E, H]-shaped tensor operations with zero Python loops.
        """
        # Mask invalid edges with large negative (non-inplace for autograd)
        masked_scores = torch.where(valid, scores, torch.tensor(-1e9, dtype=scores.dtype, device=scores.device))

        # Per-destination max for numerical stability: [N, H]
        neg_inf = torch.full((n_nodes, n_heads), -1e9, dtype=scores.dtype, device=scores.device)
        dst_max = neg_inf.scatter_reduce(0, dst.unsqueeze(1).expand(-1, n_heads), masked_scores, reduce="amax", include_self=True)

        # Subtract max and exponentiate
        shifted = masked_scores - dst_max[dst]
        exp_scores = torch.where(valid, torch.exp(shifted), torch.tensor(0.0, dtype=scores.dtype, device=scores.device))

        # Per-destination sum: [N, H]
        dst_sum = torch.zeros((n_nodes, n_heads), dtype=scores.dtype, device=scores.device)
        dst_sum.scatter_add_(0, dst.unsqueeze(1).expand(-1, n_heads), exp_scores)

        # Normalize
        attn = exp_scores / dst_sum[dst].clamp_min(1e-8)
        return attn

    def forward(
        self,
        x: torch.Tensor,
        union: UnifiedNeighborhood,
        router_output: RouterOutput,
        edge_features: torch.Tensor | None = None,
        gw_gate: torch.Tensor | None = None,
    ) -> RoutedAttentionOutput:
        assert_shape(x, (-1, -1), "x")
        union.validate()
        router_output.validate()

        n_nodes = x.shape[0]
        edge_index = union.edge_index_union
        src = edge_index[0]
        dst = edge_index[1]
        n_edges = edge_index.shape[1]

        view_names = list(union.view_weights.keys())
        m_views = len(view_names)
        n_heads = router_output.pi.shape[1]

        if router_output.pi.shape[2] != m_views:
            raise ValueError(
                f"Router num_views ({router_output.pi.shape[2]}) != union views ({m_views})"
            )

        q = self.q_proj(x).view(n_nodes, n_heads, -1)
        k = self.k_proj(x).view(n_nodes, n_heads, -1)
        v = self.v_proj(x).view(n_nodes, n_heads, -1)
        d_h = q.shape[2]

        view_w = torch.stack([union.view_weights[name] for name in view_names], dim=1)
        pi_dst = router_output.pi[dst]  # [E, H, M]
        mix = (pi_dst * view_w.unsqueeze(1)).sum(dim=2)  # [E, H]

        # GW gating: A_tilde *= g_ij — trust edges with multi-geometry consistency
        if gw_gate is not None:
            g_ij = gw_gate[src, dst].unsqueeze(1)  # [E, 1]
            mix = mix * g_ij  # broadcast over heads

        # Adaptive temperature τ_h for log-barrier stability (critique 3.3/8.1)
        if self.config.adaptive_tau:
            if self._tau_logit is None:
                self._tau_logit = nn.Parameter(
                    torch.full((n_heads,), 0.5413, device=x.device)
                )
            tau = torch.nn.functional.softplus(self._tau_logit)  # [H], ≈1.0 at init

        raw_scores = (q[dst] * k[src]).sum(dim=2) / math.sqrt(float(d_h))

        if edge_features is not None:
            assert_shape(edge_features, (n_edges, -1), "edge_features")
            # Per-head edge bias: φ_h(e_ij) — each head specializes on geometry
            bias_h = torch.nn.functional.gelu(self.edge_bias_hidden(edge_features))  # [E, head_dim]
            # Lazy-init output projection to match actual n_heads
            if self._edge_bias_out is None:
                self._edge_bias_out = nn.Linear(self.config.head_dim, n_heads, device=bias_h.device)
            raw_scores = raw_scores + self._edge_bias_out(bias_h)  # [E, H]

        if self.config.adjacency_mode == "post_softmax":
            # Post-softmax view mixing (THORN++ spec §5.1):
            #   α_ij^(h,m) = softmax_{j∈N^(m)}(s_ij + b_ij^(m))
            #   y_i^(h) = Σ_m π_{i,h,m} * Σ_j α_ij^(h,m) * v_j^(m)
            # Each view gets its own attention normalization, then outputs are mixed.
            pi_dst_node = router_output.pi  # [N, H, M]
            node_states = torch.zeros((n_nodes, n_heads, d_h), dtype=v.dtype, device=v.device)
            # Combined attn for diagnostics: weighted sum of per-view attentions
            attn = torch.zeros((n_edges, n_heads), dtype=v.dtype, device=v.device)

            for m_idx, name in enumerate(view_names):
                view_mask = union.view_masks[name]  # [E_u] bool
                valid_m = view_mask.unsqueeze(1).expand(-1, n_heads)  # [E, H]

                # View-specific Q/K/V via LoRA adapters (critique 4.2)
                if self.config.view_specific_projections:
                    dq = self.view_q_up[m_idx](self.view_q_down[m_idx](x))
                    dk = self.view_k_up[m_idx](self.view_k_down[m_idx](x))
                    dv = self.view_v_up[m_idx](self.view_v_down[m_idx](x))
                    q_m = (self.q_proj(x) + dq).view(n_nodes, n_heads, -1)
                    k_m = (self.k_proj(x) + dk).view(n_nodes, n_heads, -1)
                    v_m = (self.v_proj(x) + dv).view(n_nodes, n_heads, -1)
                    scores_m = (q_m[dst] * k_m[src]).sum(dim=2) / math.sqrt(float(d_h))
                    if edge_features is not None:
                        scores_m = scores_m + self._edge_bias_out(
                            torch.nn.functional.gelu(self.edge_bias_hidden(edge_features))
                        )
                    attn_m = self._edge_softmax(scores_m, dst, valid_m, n_nodes, n_heads)
                    v_src = v_m[src]
                else:
                    attn_m = self._edge_softmax(raw_scores, dst, valid_m, n_nodes, n_heads)
                    v_src = v[src]

                # Route-weighted value aggregation for this view
                pi_m = pi_dst_node[:, :, m_idx]  # [N, H]
                weighted_v_m = v_src * attn_m.unsqueeze(2) * pi_m[dst].unsqueeze(2)  # [E, H, D_h]
                wv_flat = weighted_v_m.view(n_edges, n_heads * d_h)
                node_states_flat_m = torch.zeros((n_nodes, n_heads * d_h), dtype=v.dtype, device=v.device)
                node_states_flat_m.scatter_add_(0, dst.unsqueeze(1).expand(-1, n_heads * d_h), wv_flat)
                node_states = node_states + node_states_flat_m.view(n_nodes, n_heads, d_h)

                attn = attn + attn_m * pi_dst_node[dst, :, m_idx]  # weighted for diagnostics
        else:
            if self.config.adjacency_mode == "bias":
                if self.config.adaptive_tau:
                    # τ_h scales the mask before log: log(τ_h * Ã_ij + ε)
                    scaled_mix = tau.unsqueeze(0) * mix  # [E, H]
                    scores = raw_scores + torch.log(scaled_mix.clamp_min(1e-8))
                else:
                    scores = raw_scores + torch.log(mix.clamp_min(1e-8))
                valid = mix > 0
            else:
                scores = raw_scores.clone()
                valid = mix > 0
                scores[~valid] = -1e9

            attn = self._edge_softmax(scores, dst, valid, n_nodes, n_heads)

            # Vectorized value aggregation: single scatter_add_ over [E, H*D_h]
            weighted_v = v[src] * attn.unsqueeze(2)  # [E, H, D_h]
            weighted_v_flat = weighted_v.view(n_edges, n_heads * d_h)  # [E, H*D_h]
            node_states_flat = torch.zeros((n_nodes, n_heads * d_h), dtype=v.dtype, device=v.device)
            node_states_flat.scatter_add_(0, dst.unsqueeze(1).expand(-1, n_heads * d_h), weighted_v_flat)
            node_states = node_states_flat.view(n_nodes, n_heads, d_h)

        out = RoutedAttentionOutput(node_states=node_states, attn_weights=attn)
        out.validate(num_nodes=n_nodes, num_edges=n_edges, num_heads=n_heads)
        return out
