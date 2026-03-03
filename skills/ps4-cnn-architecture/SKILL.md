---
name: ps4-cnn-architecture
description: Apply the PS4 architecture design guidance for CNN modeling, imbalance handling, split strategy, and evaluation criteria. Use when implementing or revising baseline/improved/transfer models, tuning hyperparameters, or justifying model choices in code and report text.
---

# PS4 CNN Architecture

## Overview
Use this skill to keep implementation and model decisions aligned with the architecture document and assignment constraints.

## Workflow
1. Open `doc/ARCHITECTURE.md` and `CLAUDE.md`.
2. Select a model tier (baseline, improved residual CNN, or transfer learning).
3. Apply imbalance controls in training (`CrossEntropyLoss` class weights and/or `WeightedRandomSampler`).
4. Preserve stratified split logic and keep evaluation on both unbalanced and balanced test conditions.
5. Compare models using macro-F1 as the primary selection criterion.
6. Log final hyperparameters, rationale, and expected tradeoffs.

## Implementation Guardrails
- Keep reproducibility controls active (fixed seeds, explicit config, deterministic split routine).
- Prefer architecture changes that are easy to explain in report sections.
- Do not present accuracy as the only success metric on this dataset.
- Include per-class performance interpretation, especially for urgent Class 2 behavior.

## Source and Reference
- Primary source: `doc/ARCHITECTURE.md`
- Supporting source: `CLAUDE.md`
- Quick reference: `references/architecture-keypoints.md`
