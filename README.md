# THORN

Transport-aligned Heterogeneous Observer-Routed Neighborhood Attention (`THORN`) is a graph-attention framework that routes heads using geometric observers (e.g., curvature, intrinsic dimension, density) instead of content-only gating.

This repository contains the PyTorch implementation, experiment scripts, and reproducibility checks used in the THORN project.

## Introduction
- Problem: content-only graph attention underuses graph geometry.
- Core idea: observer-routed, multi-view attention with differentiable structural masking.
- Goal: improve robustness and performance under view drift and class imbalance.

## Related Work
- Graph neural networks and graph transformers.
- Mixture-of-experts style routing.
- Optimal transport and fused Gromov-Wasserstein alignment.
- Geometric descriptors (LID, curvature, LOF).
- Low-rank adaptation for parameter-efficient view specialization.

## THORN Architecture
Main components (from `thorn/paper/thorn_paper.tex`):
- Preliminaries and notation.
- Observer features.
- Multi-view graph construction.
- MoE-style observer routing.
- Log-barrier masking.
- Pre-softmax vs. post-softmax view mixing.
- Fused Gromov-Wasserstein cross-view alignment.
- THORNBlock architecture.
- Regularizers.
- End-to-end algorithm.

## Theoretical Analysis
- Formal properties of routing/masking behavior.
- Subsumption perspective (connections to GAT/Graphormer/MoE settings).
- Isolation-versus-interaction tradeoff framing for view mixing.

## Experiments
Evaluation sections include:
- Experimental setup.
- Drift-strength experiment.
- Ablation study.
- Routing analysis.
- Theory versus practice.
- Baseline comparison.
- Log-barrier stability study.
- THORN++ evaluation.
- Observer orthogonality analysis.

## Discussion
- Practical lessons from observer routing and multi-view mixing.
- Failure modes, stability constraints, and deployment tradeoffs.

## Conclusion and Future Work
- THORN benefits from geometry-aware routing and differentiable structural control.
- Future directions include stronger scalable alignment, online proxies, and broader benchmarks.

## Quickstart

Run environment checks:
```bash
bash scripts/stage1_verify.sh
```

Run THORN smoke:
```bash
bash scripts/run_thorn.sh --smoke
```

Run THORN full:
```bash
bash scripts/run_thorn.sh --full
```

Run baselines and comparison:
```bash
bash scripts/run_baselines.sh
```
