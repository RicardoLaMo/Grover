# GROVER

Geometric Observer-Routed View-aware Edge Reasoning (`GROVER`) is the project framing used in this repository for geometric observer routing in multi-view graph attention.

## Why the Abbreviation GROVER
`GROVER` emphasizes what the method does:
- `G`: Geometric signals drive routing.
- `RO`: Observer-routed decision layer selects useful views.
- `VER`: View-aware edge reasoning integrates multi-view neighborhood evidence.

The paper source currently uses `GoRA` as a method macro (`Geometric Observer-Routed Attention`). In this repository, `GROVER` is used as the broader project name for the same core idea plus its practical multi-view edge reasoning pipeline.

## Introduction
- Multi-view graphs create a routing challenge: when to isolate views and when to let them interact.
- The method routes attention using geometry (curvature, intrinsic dimension, density), not only content features.

## The Isolation-Interaction Tradeoff
- Formalizes pre-softmax (interaction) versus post-softmax (isolation) view mixing.
- Shows each mode wins under different signal and overlap regimes.

### Setup
- Defines view neighborhoods, overlap ratio, and per-view signal quality.

### Two Mixing Modes
- Interaction mode: merge views before normalization for cross-view reinforcement.
- Isolation mode: normalize per view, then combine for robustness to noisy views.

### When Each Mode Dominates
- Mode preference depends on view SNR and neighborhood overlap.

## Observer-Routed Multi-View Attention
- Geometry observers produce routing features.
- Router assigns view weights per node/head.
- Differentiable masking relaxes hard graph constraints.

### Notation
### The Observer Layer: Geometric Perception
### Multi-View Graph Construction
### MoE-Style Observer Routing
### Log-Barrier Differentiable Masking
### Instantiating the Tradeoff
### Cross-View Alignment
### Block Architecture and Regularization

## Related Work
- Graph attention and graph transformers.
- Mixture-of-experts routing.
- Geometric descriptors and alignment methods.

## Experiments
- Setup.
- Drift-strength tradeoff study.
- Observer necessity and component ablations.
- Routing analysis and stability checks.
- Negative results and bottleneck analysis.

### Setup
### The Tradeoff in Action: Drift-Strength Experiment
### Geometric Observers are Essential
### Component Ablation
### Routing Analysis
### Log-Barrier Stability
### Negative Results: Capacity vs. Bottleneck

## Practitioner's Guide: Choosing the Attention Mode
- Use interaction mode when complementary views reinforce one another.
- Use isolation mode when one view is much cleaner than the others.

## Discussion
- Practical deployment tradeoffs and failure modes.

## Conclusion
- Geometry-aware observer routing gives a principled control knob for multi-view attention behavior.
