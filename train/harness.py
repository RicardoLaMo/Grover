"""Unified THORN experiment harness."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, UTC
from pathlib import Path
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from thorn.alignment.interface import align_views, compute_gw_consensus_matrix
from thorn.config import ExperimentConfig
from thorn.data.synthetic import SyntheticBatch, SyntheticDatasetGenerator
from thorn.models.thorn import GraphContext, THORNModel
from thorn.observers.knn_stats import compute_knn_distance_stats
from thorn.observers.curvature import compute_curvature_proxy
from thorn.observers.lid import estimate_lid_levina_bickel, multi_scale_lid
from thorn.observers.lof_ratio import compute_lof_ratio
from thorn.observers.temporal import compute_temporal_features
from thorn.routing.regularizers import entropy_regularizer, head_view_orthogonality_loss, load_balancing_loss
from thorn.train.eval import ClassificationMetrics, DriftMetrics, classification_metrics
from thorn.utils.io import save_json
from thorn.utils.seed import set_global_seed
from thorn.views.diffusion import DiffusionBuilder
from thorn.views.knn import KNNBuilder
from thorn.views.time import TimeBuilder
from thorn.views.union import merge_views
from thorn.contracts import RouterOutput, ViewEdges


@dataclass
class RunArtifacts:
    """Paths emitted by a run."""

    run_dir: Path
    config_snapshot: Path
    metrics_json: Path
    checkpoint: Path
    profile_json: Path
    routing_json: Path


def _time_masks(timestamps: torch.Tensor, train_frac: float, val_frac: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = timestamps.shape[0]
    order = torch.argsort(timestamps)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)

    train_idx = order[:n_train]
    val_idx = order[n_train : n_train + n_val]
    test_idx = order[n_train + n_val :]

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask


def _reshape_knn_distances(edge_index: torch.Tensor, distances: torch.Tensor, num_nodes: int, k: int) -> torch.Tensor:
    """Convert edge list distances into [N, k] by destination node (vectorized)."""
    dst = edge_index[1]

    # Sort by destination to group neighbors
    sort_idx = torch.argsort(dst)
    sorted_dst = dst[sort_idx]
    sorted_dist = distances[sort_idx]

    # Count per-node edges and cap at k
    counts = torch.zeros(num_nodes, dtype=torch.long)
    counts.scatter_add_(0, sorted_dst, torch.ones_like(sorted_dst))

    out = torch.zeros((num_nodes, k), dtype=torch.float32)
    # Place sorted distances into the output matrix
    cum = torch.zeros(num_nodes, dtype=torch.long)
    for e in range(sorted_dst.shape[0]):
        d = int(sorted_dst[e].item())
        c = int(cum[d].item())
        if c < k:
            out[d, c] = sorted_dist[e]
            cum[d] += 1
    return out


def _build_diffusion_view(knn_edges: ViewEdges, diff_coords: torch.Tensor) -> ViewEdges:
    src = knn_edges.edge_index[0]
    dst = knn_edges.edge_index[1]
    d = torch.norm(diff_coords[src] - diff_coords[dst], dim=1)
    w = torch.exp(-d)
    return ViewEdges(name="diffusion", edge_index=knn_edges.edge_index.clone(), edge_weight=w)


def _alignment_features(views: dict[str, ViewEdges], node_features: torch.Tensor, config: ExperimentConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Build global alignment tensors across all views via pairwise surrogate alignment."""
    names = list(views.keys())
    m = len(names)
    n = node_features.shape[0]

    pairwise = torch.zeros((n, m, m), dtype=torch.float32)
    per_view = torch.zeros((n, m), dtype=torch.float32)

    for i in range(m):
        pairwise[:, i, i] = 1.0

    for i in range(m):
        for j in range(i + 1, m):
            sig = align_views(views[names[i]], views[names[j]], node_features, {}, config.alignment)
            overlap = sig.pairwise_agreement[:, 0, 1]
            pairwise[:, i, j] = overlap
            pairwise[:, j, i] = overlap
            per_view[:, i] += sig.per_view_confidence[:, 0]
            per_view[:, j] += sig.per_view_confidence[:, 1]

    # Normalize confidence to simplex over views.
    per_view = per_view / per_view.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return pairwise, per_view


def _edge_features(
    union_edge_index: torch.Tensor,
    views: dict[str, ViewEdges],
    time_out_deltas: torch.Tensor,
    lid: torch.Tensor | None = None,
    gw_consensus: torch.Tensor | None = None,
) -> torch.Tensor:
    """Construct enriched edge features.

    Output columns:
        [0-2]: knn_weight, time_weight, diffusion_weight  (per-view distances)
        [3]:   GW consensus g_ij  (cross-view structural agreement)
        [4-5]: LID_i, LID_j      (local intrinsic dimensionality at endpoints)
    """
    e = union_edge_index.shape[1]
    src_idx = union_edge_index[0]
    dst_idx = union_edge_index[1]
    feat = torch.zeros((e, 6), dtype=torch.float32)

    # Compute N_max for edge-id encoding
    n_max = max(int(union_edge_index.max().item()), 0) + 1
    union_ids = union_edge_index[0] * n_max + union_edge_index[1]

    names = ["knn", "time", "diffusion"]
    for col, name in enumerate(names):
        if name in views and views[name].edge_weight is not None:
            view = views[name]
            view_ids = view.edge_index[0] * n_max + view.edge_index[1]
            sort_idx = torch.argsort(union_ids)
            sorted_union = union_ids[sort_idx]
            positions = torch.searchsorted(sorted_union, view_ids)
            positions = positions.clamp(max=e - 1)
            matches = sorted_union[positions] == view_ids
            original_positions = sort_idx[positions[matches]]
            feat[original_positions, col] = view.edge_weight[matches].float()

    if "time" in views:
        time_col = feat[:, 1]
        feat[:, 1] = (time_col + (time_col > 0).float() * 0.5) / 1.5

    # Column 3: GW consensus g_ij — per-edge cross-view agreement
    if gw_consensus is not None:
        feat[:, 3] = gw_consensus[src_idx, dst_idx]

    # Columns 4-5: LID at source and destination endpoints
    if lid is not None:
        lid_flat = lid.squeeze(-1) if lid.dim() > 1 else lid
        feat[:, 4] = lid_flat[src_idx]
        feat[:, 5] = lid_flat[dst_idx]

    return feat


def _profile_peak_memory_mb() -> float:
    if torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
    # CPU proxy: not a true process-wide peak, but consistent for gate reporting.
    return 0.0


def _to_device(context: GraphContext, device: torch.device) -> GraphContext:
    """Move all graph context tensors to the specified device."""
    union = context.union
    new_masks = {k: v.to(device) for k, v in union.view_masks.items()}
    new_weights = {k: v.to(device) for k, v in union.view_weights.items()}
    from thorn.contracts import UnifiedNeighborhood
    new_union = UnifiedNeighborhood(
        edge_index_union=union.edge_index_union.to(device),
        view_masks=new_masks,
        view_weights=new_weights,
    )
    return GraphContext(
        union=new_union,
        edge_features=context.edge_features.to(device),
        observer_features=context.observer_features.to(device),
        gw_consensus=context.gw_consensus.to(device) if context.gw_consensus is not None else None,
    )


class ExperimentHarness:
    """Unified harness for THORN and all baselines."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _prepare_batch(self) -> SyntheticBatch:
        n_nodes = self.config.train.tiny_nodes if self.config.train.overfit_tiny else self.config.data.num_nodes
        gen = SyntheticDatasetGenerator()
        batch = gen.generate(
            num_nodes=n_nodes,
            num_features=self.config.data.num_features,
            num_classes=self.config.data.num_classes,
            seed=self.config.train.seed,
            drift_strength=self.config.data.drift_strength,
        )
        if self.config.train.overfit_tiny:
            # Make tiny-mode labels explicitly learnable for sanity overfit gate.
            easy_logit = 2.5 * batch.x[:, 0] - 1.5 * batch.x[:, 1] + 0.25 * batch.x[:, 2]
            batch.y = (easy_logit > 0.0).long()
        return batch

    def _prepare_graph(self, batch: SyntheticBatch) -> tuple[GraphContext, dict[str, float | str], torch.Tensor, torch.Tensor]:
        build_stats: dict[str, float | str] = {}

        t0 = time.perf_counter()
        knn = KNNBuilder().build(batch.x, k=self.config.views.knn_k, scalable_mode=self.config.views.scalable_mode)
        build_stats["knn_sec"] = time.perf_counter() - t0
        build_stats["knn_backend"] = str(knn.stats.get("backend", "unknown"))
        build_stats["knn_scalable_mode"] = str(knn.stats.get("scalable_mode", "False"))

        t1 = time.perf_counter()
        time_view = TimeBuilder().build(batch.t, window=self.config.views.time_window)
        build_stats["time_sec"] = time.perf_counter() - t1

        t2 = time.perf_counter()
        diff_out = DiffusionBuilder().build(
            knn.edges.edge_index,
            num_nodes=batch.x.shape[0],
            output_dim=self.config.views.diffusion_dim,
            method=self.config.views.diffusion_method,
            scales=self.config.views.diffusion_scales,
        )
        diff_view = _build_diffusion_view(knn.edges, diff_out.coords)
        build_stats["diffusion_sec"] = time.perf_counter() - t2

        views: dict[str, ViewEdges] = {}
        if self.config.views.enable_knn:
            views["knn"] = knn.edges
        if self.config.views.enable_time:
            views["time"] = time_view.edges
        if self.config.views.enable_diffusion:
            views["diffusion"] = diff_view
        if not views:
            raise ValueError("At least one view must be enabled")

        # THORN++ spec §1: "Self-loops should be enforced BEFORE constructing E_union"
        # Configurable: self-loops can change attention dynamics significantly
        if self.config.views.add_self_loops:
            n_nodes_sl = batch.x.shape[0]
            self_idx = torch.arange(n_nodes_sl, dtype=torch.long)
            self_edge = torch.stack([self_idx, self_idx], dim=0)
            for name in views:
                v = views[name]
                ei = torch.cat([v.edge_index, self_edge], dim=1)
                if v.edge_weight is not None:
                    ew = torch.cat([v.edge_weight, torch.ones(n_nodes_sl, dtype=v.edge_weight.dtype)])
                else:
                    ew = None
                views[name] = ViewEdges(name=name, edge_index=ei, edge_weight=ew)

        union = merge_views(views, degree_norm=self.config.views.degree_norm)

        knn_dist_2d = _reshape_knn_distances(
            edge_index=knn.edges.edge_index,
            distances=knn.distances,
            num_nodes=batch.x.shape[0],
            k=self.config.views.knn_k,
        )

        lid = estimate_lid_levina_bickel(knn_dist_2d, k=self.config.views.knn_k).unsqueeze(1)
        lid_multi = multi_scale_lid(knn_dist_2d)  # [N, 3]
        knn_stats = compute_knn_distance_stats(knn_dist_2d)
        temporal = compute_temporal_features(batch.t)
        lof = compute_lof_ratio(knn_dist_2d).unsqueeze(1)

        # Curvature proxy from kNN graph
        curvature = compute_curvature_proxy(knn.edges.edge_index, num_nodes=batch.x.shape[0])  # [N, 1]

        # Per-view degree features
        view_degrees = []
        for name in views:
            deg = torch.bincount(views[name].edge_index[1], minlength=batch.x.shape[0]).float().unsqueeze(1)
            view_degrees.append(deg)
        view_deg_feat = torch.cat(view_degrees, dim=1)  # [N, M]

        # Non-alignment observer features
        non_align_parts = [batch.x, diff_out.coords, lid, lid_multi, knn_stats, temporal, lof, curvature, view_deg_feat]
        non_align_observer = torch.cat(non_align_parts, dim=1)
        non_align_dim = non_align_observer.shape[1]

        # Compute GW consensus matrix for edge features (before alignment features)
        gw_consensus = compute_gw_consensus_matrix(views, batch.x, self.config.alignment)

        align_pair, align_conf = _alignment_features(views, batch.x, self.config)
        align_flat = align_pair.view(batch.x.shape[0], -1)

        observer = torch.cat([non_align_observer, align_conf, align_flat], dim=1)
        edge_features = _edge_features(
            union.edge_index_union, views, time_view.time_deltas,
            lid=lid,
            gw_consensus=gw_consensus,
        )

        context = GraphContext(
            union=union,
            edge_features=edge_features,
            observer_features=observer,
            gw_consensus=gw_consensus,
        )
        # Store non-alignment dimension for router_no_alignment mode
        build_stats["non_align_dim"] = non_align_dim
        return context, build_stats, diff_out.coords, align_conf

    def _routing_override(self, mode: str, n_nodes: int, n_heads: int, m_views: int) -> torch.Tensor | None:
        if mode == "uniform_multi":
            return torch.full((n_nodes, n_heads, m_views), fill_value=1.0 / m_views)
        if mode == "single_view_knn":
            pi = torch.zeros((n_nodes, n_heads, m_views), dtype=torch.float32)
            pi[:, :, 0] = 1.0
            return pi
        if mode == "single_view_time":
            pi = torch.zeros((n_nodes, n_heads, m_views), dtype=torch.float32)
            pi[:, :, 1] = 1.0
            return pi
        if mode == "single_view_diffusion":
            pi = torch.zeros((n_nodes, n_heads, m_views), dtype=torch.float32)
            pi[:, :, 2] = 1.0
            return pi
        if mode == "gat":
            # approximate GAT baseline with uniform mix over all available views on union graph.
            return torch.full((n_nodes, n_heads, m_views), fill_value=1.0 / m_views)
        return None

    def _compute_loss(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
        train_mask: torch.Tensor,
        router_out: "RouterOutput",
        per_view_conf: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ce = F.cross_entropy(logits[train_mask], y[train_mask])
        if self.config.mode in {"thorn", "router_no_alignment"} or self.config.mode.startswith("thorn_"):
            ce = ce + self.config.router.entropy_reg * entropy_regularizer(router_out.pi)
            if router_out.gate_logits is not None:
                ce = ce + self.config.router.load_balance_reg * load_balancing_loss(router_out.pi, router_out.gate_logits)
            # Head-view orthogonality (THORN++ §5.2): prevent head collapse
            if self.config.router.orth_reg > 0:
                ce = ce + self.config.router.orth_reg * head_view_orthogonality_loss(router_out.pi)
            # Alignment regularization (THORN++ §Pillar III): L_align = Σ π * D
            if self.config.router.align_reg > 0 and per_view_conf is not None:
                # D_{i,m} = 1 - confidence_{i,m} (discordance = inverse of transportability)
                d_im = 1.0 - per_view_conf  # [N, M]
                pi_mean = router_out.pi.mean(dim=1)  # [N, M] average over heads
                align_loss = (pi_mean * d_im).sum(dim=1).mean()
                ce = ce + self.config.router.align_reg * align_loss
        return ce

    def _compute_classification_metrics(self, logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> ClassificationMetrics:
        prob = torch.softmax(logits[mask], dim=1).detach().cpu().numpy()
        yt = y[mask].detach().cpu().numpy().astype(np.int64)
        num_classes = self.config.data.num_classes
        if num_classes == 2:
            # Binary: pass positive class probabilities
            return classification_metrics(yt, prob[:, 1], frac_k=self.config.eval.precision_at_k)
        else:
            return classification_metrics(yt, prob, frac_k=self.config.eval.precision_at_k, num_classes=num_classes)

    def _drift_metrics(self, router_pi: torch.Tensor, timestamps: torch.Tensor, test_mask: torch.Tensor) -> DriftMetrics:
        test_t = timestamps[test_mask]
        pi_test = router_pi[test_mask]
        median_t = torch.median(test_t)
        early_mask = test_t <= median_t
        late_mask = test_t > median_t
        if not torch.any(early_mask) or not torch.any(late_mask):
            early_mask = torch.arange(test_t.shape[0]) % 2 == 0
            late_mask = ~early_mask

        p_early = pi_test[early_mask].mean(dim=(0, 1))
        p_late = pi_test[late_mask].mean(dim=(0, 1))
        shift = torch.abs(p_early - p_late).sum().item()

        ent_early = float((-(pi_test[early_mask].clamp_min(1e-8) * torch.log(pi_test[early_mask].clamp_min(1e-8))).sum(dim=2).mean()).item())
        ent_late = float((-(pi_test[late_mask].clamp_min(1e-8) * torch.log(pi_test[late_mask].clamp_min(1e-8))).sum(dim=2).mean()).item())
        collapse = float(max(p_early.max().item(), p_late.max().item()))

        return DriftMetrics(
            routing_shift_l1=float(shift),
            routing_entropy_early=ent_early,
            routing_entropy_late=ent_late,
            collapse_score=collapse,
        )

    def _apply_ablation_overrides(self, mode: str) -> None:
        """Apply config overrides for ablation modes. Modifies self.config in place."""
        if mode == "thorn_no_fgw":
            self.config.alignment.method = "surrogate_overlap"
        elif mode == "thorn_no_depth":
            self.config.attention.num_layers = 1
            self.config.attention.use_residual = False
            self.config.attention.use_layer_norm = False
        elif mode == "thorn_no_moe":
            self.config.router.top_k = self.config.router.num_views  # dense softmax, no top-k
            self.config.router.noise_std = 0.0
            self.config.router.load_balance_reg = 0.0
            self.config.router.orth_reg = 0.0  # L_orth conflicts with dense softmax
            self.config.router.align_reg = 0.0
        elif mode == "thorn_single_scale":
            self.config.views.diffusion_scales = [1.0]

    def run(self, mode: str, output_dir: Path | None = None) -> RunArtifacts:
        """Run a configured experiment."""
        if output_dir is None:
            ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
            run_id = f"{ts}_{self.config.name}_{mode}_{self.config.sha256()[:8]}"
            output_dir = Path(self.config.output_root) / run_id

        output_dir.mkdir(parents=True, exist_ok=True)
        set_global_seed(self.config.train.seed, deterministic=self.config.train.deterministic)

        # Apply ablation overrides before building anything
        if mode.startswith("thorn_"):
            self._apply_ablation_overrides(mode)

        batch = self._prepare_batch()
        train_mask, val_mask, test_mask = _time_masks(
            batch.t,
            train_frac=self.config.eval.train_frac,
            val_frac=self.config.eval.val_frac,
        )
        if self.config.train.overfit_tiny:
            train_mask = torch.ones_like(train_mask)
            val_mask = torch.ones_like(val_mask)
            test_mask = torch.ones_like(test_mask)

        t_graph = time.perf_counter()
        context, build_stats, diff_coords, align_conf = self._prepare_graph(batch)
        graph_sec = time.perf_counter() - t_graph

        # Move data to device
        device = self.device
        batch.x = batch.x.to(device)
        batch.y = batch.y.to(device)
        batch.t = batch.t.to(device)
        train_mask = train_mask.to(device)
        val_mask = val_mask.to(device)
        test_mask = test_mask.to(device)
        context = _to_device(context, device)
        align_conf = align_conf.to(device)

        model = THORNModel(self.config).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=self.config.train.lr, weight_decay=self.config.train.weight_decay)

        best_val = -1.0
        best_state: dict[str, torch.Tensor] | None = None
        best_router_pi: torch.Tensor | None = None

        train_curve: list[dict[str, float]] = []

        t_train = time.perf_counter()
        for epoch in range(self.config.train.epochs):
            model.train()
            opt.zero_grad()

            if mode == "router_no_alignment":
                # remove alignment contributions from observer features (last block).
                nad = int(build_stats["non_align_dim"])
                context_local = GraphContext(
                    union=context.union,
                    edge_features=context.edge_features,
                    observer_features=context.observer_features[:, :nad],
                    gw_consensus=None,  # no GW in no-alignment mode
                )
            else:
                context_local = context

            override = self._routing_override(mode, batch.x.shape[0], self.config.router.num_heads, self.config.router.num_views)
            if override is not None:
                override = override.to(device)
            logits, router_out = model(batch.x, context_local, routing_override=override)
            loss = self._compute_loss(logits, batch.y, train_mask, router_out, per_view_conf=align_conf)
            loss.backward()
            opt.step()

            model.eval()
            with torch.no_grad():
                logits_eval, router_eval = model(batch.x, context_local, routing_override=override)
                val_metrics = self._compute_classification_metrics(logits_eval, batch.y, val_mask)

            train_curve.append(
                {
                    "epoch": float(epoch),
                    "loss": float(loss.item()),
                    "val_pr_auc": val_metrics.pr_auc,
                    "val_roc_auc": val_metrics.roc_auc,
                }
            )

            if val_metrics.pr_auc > best_val:
                best_val = val_metrics.pr_auc
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                best_router_pi = router_eval.pi.detach().clone()

        train_sec = time.perf_counter() - t_train

        if best_state is None or best_router_pi is None:
            raise RuntimeError("Training did not produce a best checkpoint")

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            if mode == "router_no_alignment":
                nad = int(build_stats["non_align_dim"])
                context_local = GraphContext(
                    union=context.union,
                    edge_features=context.edge_features,
                    observer_features=context.observer_features[:, :nad],
                )
            else:
                context_local = context

            override = self._routing_override(mode, batch.x.shape[0], self.config.router.num_heads, self.config.router.num_views)
            if override is not None:
                override = override.to(device)
            logits, router_out = model(batch.x, context_local, routing_override=override)

        train_metrics = self._compute_classification_metrics(logits, batch.y, train_mask)
        val_metrics = self._compute_classification_metrics(logits, batch.y, val_mask)
        test_metrics = self._compute_classification_metrics(logits, batch.y, test_mask)
        drift = self._drift_metrics(router_out.pi.detach(), batch.t, test_mask)

        metrics_payload = {
            "mode": mode,
            "seed": self.config.train.seed,
            "config_hash": self.config.sha256(),
            "splits": {
                "train_nodes": int(train_mask.sum().item()),
                "val_nodes": int(val_mask.sum().item()),
                "test_nodes": int(test_mask.sum().item()),
            },
            "train": asdict(train_metrics),
            "val": asdict(val_metrics),
            "test": asdict(test_metrics),
            "drift": asdict(drift),
            "training_curve": train_curve,
            "overfit_success": bool(train_metrics.accuracy >= 0.98) if self.config.train.overfit_tiny else None,
        }

        profile_payload = {
            "graph_build": {
                **build_stats,
                "total_sec": graph_sec,
            },
            "train_sec": train_sec,
            "router_sec_approx": train_sec / max(self.config.train.epochs, 1),
            "attention_sec_approx": train_sec / max(self.config.train.epochs, 1),
            "peak_memory_mb": _profile_peak_memory_mb(),
            "small_mode": bool(self.config.train.overfit_tiny),
            "scalable_mode": bool(self.config.views.scalable_mode),
        }

        routing_stats = {
            "mean_pi": router_out.pi.mean(dim=(0, 1)).tolist(),
            "entropy": float((-(router_out.pi.clamp_min(1e-8) * torch.log(router_out.pi.clamp_min(1e-8))).sum(dim=2).mean()).item()),
            "collapse_score": float(router_out.pi.mean(dim=(0, 1)).max().item()),
            "drift": asdict(drift),
        }

        config_path = output_dir / "config_snapshot.json"
        metrics_path = output_dir / "metrics.json"
        ckpt_path = output_dir / "checkpoint.pt"
        profile_path = output_dir / "profile.json"
        routing_path = output_dir / "routing_stats.json"

        self.config.save_json(config_path)
        save_json(metrics_payload, metrics_path)
        torch.save({"state_dict": best_state, "mode": mode}, ckpt_path)
        save_json(profile_payload, profile_path)
        save_json(routing_stats, routing_path)

        manifest_path = Path(self.config.output_root) / "index.jsonl"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"run_dir": str(output_dir), "mode": mode, "config_hash": self.config.sha256()}) + "\n")

        return RunArtifacts(
            run_dir=output_dir,
            config_snapshot=config_path,
            metrics_json=metrics_path,
            checkpoint=ckpt_path,
            profile_json=profile_path,
            routing_json=routing_path,
        )
