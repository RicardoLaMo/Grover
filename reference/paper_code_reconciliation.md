# THORN Paper-to-Code Reconciliation

## Scope
- Paper reference: `thorn/paper/thorn_paper.pdf` (checked against `thorn/paper/thorn_paper.tex`)
- Code scope: `thorn/` package and saved run artifacts under `artifacts/runs/` and `artifacts/reports/`
- Goal: determine whether implementation flow reconciles with paper design and reported behavior

## Executive Assessment
- The implementation strongly matches the paper's core architecture and flow.
- The largest gaps are not architectural breaks; they are implementation approximations and a few spec/code mismatches in details.
- Reported headline drift results are consistent with saved artifacts.

## End-to-End Flow (Paper vs Code)
1. Synthetic temporal data generation
   - Paper: 512 nodes, 16-D, binary, drifted temporal signal and regime shift at `t_n=0.75`.
   - Code: `thorn/data/synthetic.py` implements this directly (`x[:,0:3]` signal, `+1.5*drift_strength` late logit shift).
2. Multi-view graph construction (`kNN`, `time`, `diffusion`)
   - Code: `thorn/train/harness.py` builds all three views, merges union via `thorn/views/union.py`.
3. Observer feature construction
   - Code includes node observers (LID, multi-scale LID, kNN stats, temporal features, LOF-like proxy, curvature proxy, per-view degree, alignment features).
4. Observer-routed MoE gating
   - Code router in `thorn/routing/router.py`: LN/GELU MLP + Gaussian noise + top-k softmax.
5. Routed neighborhood attention
   - Pre-softmax (`bias`) and post-softmax (`post_softmax`) both implemented in `thorn/layers/routed_attention.py`.
6. Transformer-style stacked THORN blocks
   - Pre-norm + residual + FFN implemented in `thorn/models/thorn.py`.
7. Auxiliary regularizers
   - Orthogonality, load-balancing, and alignment regularization in `thorn/routing/regularizers.py` and `thorn/train/harness.py`.
8. Evaluation and artifact emission
   - Metrics and reproducibility artifacts emitted by `thorn/train/harness.py` and scripts in `scripts/`.

## Reconciliation Matrix

### Fully Reconciled
- Observer-routed multi-view attention architecture is present end-to-end.
- Log-barrier pre-softmax masking is implemented (`scores + log(mix)`).
- Post-softmax view-isolated attention is implemented and selectable.
- Top-k per-head routing with exploration noise is implemented.
- Orthogonality and alignment losses are implemented and wired into training.
- Drift-based synthetic benchmark protocol and temporal split logic exist.

### Reconciled with Approximation
- Curvature:
  - Paper describes Ollivier-Ricci.
  - Code uses clustering-coefficient proxy (`compute_curvature_proxy`).
- LOF:
  - Paper describes LOF.
  - Code uses LOF-like density ratio baseline (`compute_lof_ratio`).
- FGW:
  - Paper describes ego-neighborhood/offline cached FGW alignment.
  - Code computes dense/global approximations during graph prep each run, then uses consensus features.

### Material Detail Mismatches
- Per-layer vs shared routing
  - Paper algorithm recomputes router each layer.
  - Code routes once and shares `pi` across all layers (`THORNModel.forward` comment: "Route once; share pi across all layers").
- Post-softmax bias granularity
  - Paper equation allows `b_{ij}^{(h,m)}` (view-specific).
  - Code computes one edge-bias tensor and reuses it across views in post-softmax.
- Edge feature ordering
  - Paper edge vector order: `[kNN, time, diff, LID_i, LID_j, g_ij]`.
  - Code order: `[kNN, time, diff, g_ij, LID_i, LID_j]`.
- Load-balancing formulation
  - Paper equation states L2-to-uniform usage.
  - Code uses Switch-style `M * sum(f_m * p_m)`.
- Training hyperparameters
  - Paper setup section states `AdamW`, `lr=3e-3`, `wd=1e-2`, `d=64`.
  - Current configs/runs use `Adam`, `lr=1e-3`, `wd=1e-4`, and `model_dim=128`.

## Reported Results vs Saved Artifacts
- Saved `final_*` runs match paper-level drift claims closely.
- Extracted test PR-AUC by drift:
  - `0.00`: bias `0.885`, post-softmax `0.892`, time `0.856`, gat `0.873`
  - `0.25`: bias `0.925`, post-softmax `0.887`, time `0.936`, gat `0.909`
  - `0.50`: bias `0.920`, post-softmax `0.984`, time `0.948`, gat `0.886`
  - `0.75`: bias `0.939`, post-softmax `0.932`, time `0.887`, gat `0.919`
  - `1.00`: bias `0.924`, post-softmax `0.920`, time `0.896`, gat `0.922`
  - `1.50`: bias `0.837`, post-softmax `0.855`, time `0.921`, gat `0.798`
- These values are consistent with the table and narrative in `thorn_paper.tex`.

## Practical Interpretation
- If the question is "is this the THORN architecture described in the paper?":
  - Yes, for core design and experimental flow.
- If the question is "is every equation implemented literally?":
  - Not fully; several parts are engineered approximations or deliberate deviations.
- If the question is "do paper behavior claims reconcile with produced runs?":
  - Yes, especially the isolation-vs-interaction drift pattern and headline `0.984` (post-softmax, drift `0.5`) and `0.939` (bias, drift `0.75`).

## Key File Anchors
- Paper design/equations: `thorn/paper/thorn_paper.tex`
- Model and shared-routing behavior: `thorn/models/thorn.py`
- Attention modes/log-barrier: `thorn/layers/routed_attention.py`
- Pipeline/observer assembly/loss wiring: `thorn/train/harness.py`
- Router/top-k logic: `thorn/routing/router.py`
- Regularizers: `thorn/routing/regularizers.py`
- FGW/surrogate alignment: `thorn/alignment/interface.py`, `thorn/alignment/fgw.py`
- Drift result artifacts: `artifacts/runs/final_*/*` and `artifacts/reports/results_table.*`
