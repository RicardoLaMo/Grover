"""kNN view builder implementation.

Exact mode uses torch.cdist.
Scalable mode tries FAISS and falls back to exact mode when unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from thorn.contracts import ViewEdges
from thorn.debug import assert_shape


@dataclass
class KNNOutput:
    """kNN graph outputs.

    Shapes:
        edges.edge_index: [2, E_knn]
        distances: [E_knn]
    """

    edges: ViewEdges
    distances: torch.Tensor
    stats: dict[str, float | str]

    def validate(self) -> None:
        self.edges.validate()
        assert_shape(self.distances, (self.edges.edge_index.shape[1],), "knn.distances")


class KNNBuilder:
    """Builder for kNN neighborhoods."""

    def build(self, node_features: torch.Tensor, k: int, scalable_mode: bool = False) -> KNNOutput:
        """Build directed kNN edges from node features.

        Edges use `src -> dst` convention where each destination node attends to its neighbors.
        """
        assert_shape(node_features, (-1, -1), "node_features")
        n_nodes = node_features.shape[0]
        if k <= 0 or k >= n_nodes:
            raise ValueError(f"k must satisfy 1 <= k < N; got k={k}, N={n_nodes}")

        backend = "torch_cdist"
        used_faiss = False

        if scalable_mode:
            try:
                import faiss  # type: ignore

                x_np = node_features.detach().cpu().numpy().astype("float32")
                index = faiss.IndexFlatL2(x_np.shape[1])
                index.add(x_np)
                # k+1 because nearest neighbor is self.
                d2, nn_idx = index.search(x_np, k + 1)
                nn_idx = torch.from_numpy(nn_idx[:, 1:]).long()
                dists = torch.from_numpy(d2[:, 1:]).sqrt().float()
                used_faiss = True
                backend = "faiss"
            except Exception:
                # Verified fallback to exact mode.
                used_faiss = False

        if not used_faiss:
            dist_matrix = torch.cdist(node_features, node_features, p=2)
            dist_matrix.fill_diagonal_(float("inf"))
            dists, nn_idx = torch.topk(dist_matrix, k=k, dim=1, largest=False)

        dst = torch.arange(n_nodes, dtype=torch.long).repeat_interleave(k)
        src = nn_idx.reshape(-1)
        edge_index = torch.stack([src, dst], dim=0)
        distances = dists.reshape(-1).float()

        out = KNNOutput(
            edges=ViewEdges(name="knn", edge_index=edge_index, edge_weight=torch.exp(-distances)),
            distances=distances,
            stats={
                "n_nodes": float(n_nodes),
                "k": float(k),
                "mean_distance": float(distances.mean().item()),
                "max_distance": float(distances.max().item()),
                "backend": backend,
                "scalable_mode": str(bool(scalable_mode)),
            },
        )
        out.validate()
        return out
