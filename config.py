"""Dataclass configs for THORN experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal
import hashlib
import json


NodeMode = Literal["transactions", "entity_time"]
ModelMode = Literal[
    "thorn",
    "gat",
    "single_view_knn",
    "single_view_time",
    "single_view_diffusion",
    "uniform_multi",
    "router_no_alignment",
    "thorn_no_fgw",
    "thorn_no_depth",
    "thorn_no_moe",
    "thorn_single_scale",
]


@dataclass
class DataConfig:
    node_mode: NodeMode = "transactions"
    num_nodes: int = 512
    num_features: int = 16
    num_classes: int = 2
    use_time: bool = True
    drift_strength: float = 0.75


@dataclass
class ViewConfig:
    enable_knn: bool = True
    enable_time: bool = True
    enable_diffusion: bool = True
    knn_k: int = 10
    time_window: int = 6
    diffusion_dim: int = 8
    scalable_mode: bool = False
    degree_norm: Literal["none", "symmetric", "row"] = "none"
    add_self_loops: bool = False
    diffusion_method: Literal["power_iteration", "laplacian_eigenvectors"] = "laplacian_eigenvectors"
    diffusion_scales: list[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 5.0])


@dataclass
class AlignmentConfig:
    enabled: bool = True
    method: Literal["surrogate_overlap", "fgw", "online_proxy"] = "fgw"
    temperature: float = 0.1
    fgw_alpha: float = 0.5
    sinkhorn_reg: float = 0.05
    sinkhorn_iters: int = 30
    fgw_iters: int = 5


@dataclass
class RouterConfig:
    num_heads: int = 4
    num_views: int = 3
    hidden_dim: int = 64
    per_head: bool = True
    entropy_reg: float = 1e-3
    temporal_smoothness_reg: float = 1e-3
    top_k: int = 2
    noise_std: float = 0.1
    load_balance_reg: float = 1e-2
    orth_reg: float = 5e-3
    align_reg: float = 1e-3


@dataclass
class AttentionConfig:
    model_dim: int = 128
    head_dim: int = 24
    use_edge_bias: bool = True
    adjacency_mode: Literal["mask", "bias", "post_softmax"] = "bias"
    adaptive_tau: bool = False  # learned per-head temperature for log-barrier
    view_specific_projections: bool = False  # per-view Q/K/V (THORN++ critique 4.2)
    lora_rank: int = 0  # LoRA rank for view-specific; 0 = auto (head_dim//2, min 8)
    debug_assertions: bool = True
    num_layers: int = 3
    dropout: float = 0.1
    use_residual: bool = True
    use_layer_norm: bool = True
    ffn_multiplier: int = 4


@dataclass
class TrainConfig:
    seed: int = 42
    deterministic: bool = True
    epochs: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    overfit_tiny: bool = False
    tiny_nodes: int = 200
    log_every: int = 10


@dataclass
class EvalConfig:
    precision_at_k: float = 0.1
    train_frac: float = 0.6
    val_frac: float = 0.2


@dataclass
class ExperimentConfig:
    name: str = "thorn"
    mode: ModelMode = "thorn"
    output_root: str = "artifacts/runs"
    data: DataConfig = field(default_factory=DataConfig)
    views: ViewConfig = field(default_factory=ViewConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def save_json(self, path: Path) -> None:
        payload = self.to_dict()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def sha256(self) -> str:
        data = json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(data).hexdigest()
