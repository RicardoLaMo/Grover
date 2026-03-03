"""Diffusion view builder with Laplacian eigendecomposition and power iteration fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from thorn.debug import assert_shape


@dataclass
class DiffusionOutput:
    """Diffusion coordinates and optional kernels.

    Shapes:
        coords: [N, D_diff]
        pairwise_kernel: [N, N] or None
    """

    coords: torch.Tensor
    pairwise_kernel: torch.Tensor | None

    def validate(self, num_nodes: int) -> None:
        assert_shape(self.coords, (num_nodes, -1), "diffusion.coords")
        if self.pairwise_kernel is not None:
            assert_shape(self.pairwise_kernel, (num_nodes, num_nodes), "diffusion.pairwise_kernel")


class DiffusionBuilder:
    """Builder for diffusion features via Laplacian eigenvectors or power iteration."""

    def build(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        output_dim: int,
        method: Literal["power_iteration", "laplacian_eigenvectors"] = "laplacian_eigenvectors",
        scales: list[float] | None = None,
        device: torch.device | None = None,
    ) -> DiffusionOutput:
        """Compute diffusion coordinates from graph adjacency.

        Args:
            edge_index: [2, E] edge indices
            num_nodes: number of nodes
            output_dim: base dimension for eigenvectors
            method: "laplacian_eigenvectors" or "power_iteration"
            scales: diffusion time scales for multi-scale features
            device: target device for computation
        """
        assert_shape(edge_index, (2, -1), "edge_index")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")

        if device is None:
            device = edge_index.device

        if method == "laplacian_eigenvectors":
            return self._laplacian_eigenvectors(edge_index, num_nodes, output_dim, scales, device)
        else:
            return self._power_iteration(edge_index, num_nodes, output_dim, device)

    def _laplacian_eigenvectors(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        output_dim: int,
        scales: list[float] | None,
        device: torch.device,
    ) -> DiffusionOutput:
        """Multi-scale Laplacian eigenvector diffusion coordinates."""
        if scales is None:
            scales = [1.0]

        src, dst = edge_index[0].to(device), edge_index[1].to(device)

        # Build symmetric adjacency
        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)
        adj[src, dst] = 1.0
        adj = torch.maximum(adj, adj.T)
        adj.fill_diagonal_(0.0)  # no self-loops for Laplacian

        # Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
        deg = adj.sum(dim=1)
        deg_inv_sqrt = torch.zeros_like(deg)
        nonzero = deg > 0
        deg_inv_sqrt[nonzero] = 1.0 / torch.sqrt(deg[nonzero])
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        L = torch.eye(num_nodes, device=device) - D_inv_sqrt @ adj @ D_inv_sqrt

        # Eigendecomposition — bottom-k non-trivial eigenvectors
        # eigh returns eigenvalues in ascending order
        num_eigvecs = min(output_dim + 1, num_nodes)
        eigenvalues, eigenvectors = torch.linalg.eigh(L)

        # Skip the trivial zero eigenvalue (index 0), take next output_dim
        start = 1
        end = min(start + output_dim, num_nodes)
        lambdas = eigenvalues[start:end]  # [K]
        phi = eigenvectors[:, start:end]  # [N, K]

        # Multi-scale: for each scale t, compute exp(-lambda_i * t) * phi_i
        scale_coords = []
        for t in scales:
            heat_kernel = torch.exp(-lambdas * t)  # [K]
            scaled = phi * heat_kernel.unsqueeze(0)  # [N, K]
            scale_coords.append(scaled)

        coords = torch.cat(scale_coords, dim=1)  # [N, K * num_scales]

        pairwise_kernel = coords @ coords.T
        out = DiffusionOutput(coords=coords, pairwise_kernel=pairwise_kernel)
        out.validate(num_nodes)
        return out

    def _power_iteration(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        output_dim: int,
        device: torch.device,
    ) -> DiffusionOutput:
        """Original random-walk power iteration from anchor basis vectors."""
        src, dst = edge_index[0].to(device), edge_index[1].to(device)

        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)
        adj[src, dst] = 1.0
        adj = torch.maximum(adj, adj.T)
        adj.fill_diagonal_(1.0)

        deg = adj.sum(dim=1, keepdim=True)
        transition = adj / deg.clamp_min(1e-6)

        anchors = torch.linspace(0, num_nodes - 1, steps=output_dim).round().long().to(device)
        basis = torch.zeros((num_nodes, output_dim), dtype=torch.float32, device=device)
        basis[anchors, torch.arange(output_dim, device=device)] = 1.0

        coords = basis.clone()
        for _ in range(3):
            coords = transition @ coords

        pairwise_kernel = coords @ coords.T
        out = DiffusionOutput(coords=coords, pairwise_kernel=pairwise_kernel)
        out.validate(num_nodes)
        return out
