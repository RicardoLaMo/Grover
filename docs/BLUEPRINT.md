# THORN Blueprint

This file is the implementation contract for THORN and must remain aligned with code.

## A) Data model
- Node definition supports:
  - transactions as nodes OR
  - entity-time states as nodes.
- Edge sets (views): V = {kNN, time, diffusion} at minimum.
- Unified representation: union edge_index plus per-view masks/weights.
- Config-driven toggles.

## B) Neighborhood builders (view generators)
### B1) kNN builder
- Exact mode: `torch.cdist` for small/synthetic.
- Scalable mode: ANN optional (FAISS if installed); fallback must run.
- Outputs: `edge_index_knn`, `knn_distances`, stats.

### B2) Time builder
- Sliding window edges OR lag-k edges; deterministic.
- Outputs: `edge_index_time`, `time_deltas`, optional burstiness proxy features.

### B3) Diffusion builder
- Build diffusion operator from a graph (kNN or union).
- Compute diffusion embeddings (approx OK): truncated eigens, random-walk features, or power iteration.
- Outputs: diffusion coordinates + optional diffusion distances or kernels.

## C) Observer signals (node/edge features)
- LID estimator per node (Levina-Bickel MLE baseline).
- Diffusion coords/dist proxy.
- Temporal features: delta, recency, burstiness proxy.
- kNN distance stats: mean/var/min/max of neighbor distances per node.
- Optional baseline: LOF-like density ratio.

## D) Transport alignment module
- Stable interface:
  `align_views(view_a, view_b, node_features, edge_features, config) -> alignment_signals`
- Provide at least ONE implemented option:
  - FGW approximate (entropic / sampled / low-rank) if feasible and verifiable
  - Otherwise verified surrogate alignment baseline (e.g., neighborhood overlap, spectral alignment, diffusion-kernel agreement)
- Alignment signals must condition router and/or attention.
- Must be ablatable via config flag (`alignment off`).

## E) Router (Observer-Routed mechanism)
- Small GNN/MPNN consumes observer signals and outputs routing weights:
  `pi_{i,h,m} = softmax_m(logits(i,h,m))`
- Prefer per-head routing; allow per-layer fallback.
- Regularizers:
  - entropy/sparsity control
  - temporal smoothness (if time present)
- Logging: routing distribution stats; detect collapse.

## F) Routed Neighborhood Attention layer
- Multi-head attention restricted to mixture neighborhood:
  `A_tilde_{ij}^{(i,h)} = sum_m pi_{i,h,m} A_{ij}^{(m)}`
- Support adjacency-as-mask (hard) and adjacency-as-bias (soft log-bias).
- Support edge bias from observer features: `b_{ij}^{(h)} = phi_h(e_{ij})`.
- Provide clear tensor shape contracts and assertions.

## G) Training & evaluation harness
- Config-driven experiments with Hydra or argparse (reproducible).
- Baselines in same harness:
  - GAT baseline
  - single-view attention (each view alone)
  - multi-view without router (uniform pi)
  - router without transport alignment
  - router + alignment (THORN)
- Metrics:
  - PR-AUC / ROC-AUC / precision@K (if labels)
  - drift stability across time splits
- Logging:
  - always save metrics JSON + config snapshot + checkpoint.

## H) Documentation artifacts
- `docs/trace.md` for design decisions + references.
- `docs/math.md` for equations and code mapping.
- `docs/positioning.md` for related-work positioning (no fabricated citations).
- `docs/gates.md` for gate definitions and evidence format.
- README with one-command baseline/THORN/ablation runs.
- Scripts: `run_baselines.sh`, `run_thorn.sh`.
- Results: auto-generated table comparing baselines vs THORN.
