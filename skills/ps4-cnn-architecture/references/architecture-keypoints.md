# Architecture Keypoints

Source document: `doc/ARCHITECTURE.md`

## Model Options
- Model A: baseline custom CNN for performance floor.
- Model B: deeper residual CNN with regularization.
- Model C: transfer learning (ResNet18/EfficientNet-B0).

## Training and Evaluation Requirements
- Preserve stratified split strategy.
- Handle imbalance with weighted loss and/or weighted sampling.
- Track macro-F1 as primary metric for model selection.
- Report both unbalanced and balanced test-condition performance.

## Reporting Requirements
- Include confusion matrix and per-class interpretation.
- Include macro and micro views of model quality.

## Prior Progress Memory
- Baseline pipeline (`output/pipeline_results.json`):
  - Model A val macro-F1: `0.1749`
  - Model B val macro-F1: `0.6397`
  - Model C (ResNet18) val macro-F1: `0.6976`
  - Model C internal test macro-F1: `0.709` (unbalanced), `0.704` (balanced)
  - Model C Kaggle macro-F1: `0.723`
- Challenger sweep (`output/cc_results.json`):
  - Original champion: `Champion_ResNet18` (balanced macro-F1 `0.6903`)
  - Final champion: `C07_ResNet18_FocalLoss` (balanced macro-F1 `0.7251`)
  - Best pure-architecture challenger by balanced macro-F1: `C06_ConvNeXt_Tiny` (`0.7169`)
- Practical implication:
  - Loss-function choice (Focal Loss) currently gives larger gain than architecture swap alone.
