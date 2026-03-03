# AGENTS.md instructions for /home/wliu23/google_drive/machine_learning/hw4_thorn

## Workspace Lock
- Workspace root: `/home/wliu23/google_drive/machine_learning/hw4_thorn`
- Default working directory for commands: `/home/wliu23/google_drive/machine_learning/hw4_thorn`
- Treat this directory as the only active workspace unless the user explicitly asks to include additional paths.
- Resolve project files relative to this root first.

## Canonical Source Docs
- `CLAUDE.md`
- `doc/PROJECT_BRIEF.md`
- `doc/ARCHITECTURE.md`
- `doc/GRADING_RUBRIC.md`

If guidance conflicts, prioritize:
1. `doc/GRADING_RUBRIC.md` (grading gates and submission criteria)
2. `doc/PROJECT_BRIEF.md` (stakeholder and business requirements)
3. `doc/ARCHITECTURE.md` (model design and evaluation plan)
4. `CLAUDE.md` (orchestration and environment notes)

## Local Skills
- `ps4-project-brief`: Use for stakeholder requirements, class labels/priorities, deliverables, and imbalance framing.  
  File: `skills/ps4-project-brief/SKILL.md`
- `ps4-cnn-architecture`: Use for model selection, training protocol, split strategy, and imbalance-aware metrics.  
  File: `skills/ps4-cnn-architecture/SKILL.md`
- `ps4-grading-gates`: Use for rubric-driven reviews, pass/fail checks, and release-readiness before submission.  
  File: `skills/ps4-grading-gates/SKILL.md`

## Project Guardrails
- Treat class imbalance as a first-class design constraint (Class 0 dominates train/val).
- Prefer macro-F1 and macro-AUC as primary metrics; do not rely on raw accuracy alone.
- Report both unbalanced and balanced test evaluations whenever producing final model comparisons.
- Keep outputs organized under `output/figures`, `output/models`, and `output/predictions`.
- Keep model/report artifacts reproducible (fixed seeds, explicit config/hyperparameters, traceable commands).

## Prior Progress Memory (Synced 2026-02-27)
- Pipeline artifact: `output/pipeline_results.json`
  - Best validation model in baseline pipeline: `ResNet18_C`, val macro-F1 `0.6976`.
  - Internal test (ResNet18_C): unbalanced macro-F1 `0.709`, balanced macro-F1 `0.704`.
  - Kaggle (ResNet18_C): accuracy `0.747`, macro-F1 `0.723`, macro-AUC `0.918`.
- Champion/challenger artifact: `output/cc_results.json` and `output/cc_results.csv`
  - Original champion: `Champion_ResNet18` (balanced macro-F1 `0.6903`).
  - New champion: `C07_ResNet18_FocalLoss` (balanced macro-F1 `0.7251`, +`0.0348` absolute).
  - Strong architecture challenger: `C06_ConvNeXt_Tiny` (balanced macro-F1 `0.7169`, best PR-AUC `0.8157`).
- Report artifact: `doc/report/main.tex` compiled to `doc/report/main.pdf`.

## Current Gate Snapshot
- `G1` PASS: `output/figures/class_distribution.png`, `output/figures/sample_images.png` exist.
- `G2` PASS: split and class weights recorded in `output/pipeline_results.json`.
- `G3` FAIL (strict rubric threshold): baseline val macro-F1 is `0.1749` (< `0.40` target).
- `G4` PASS: improved models exceed `0.60` val macro-F1.
- `G5` PASS: both balanced and unbalanced test conditions reported.
- `G6` PASS: `output/predictions/kaggle_submission.csv` format validated (`300` rows, `Id` 0-299, classes 0/1/2).
- `G7` PARTIAL: PDF compiles, but title-page and repository placeholders remain in `doc/report/main.tex`.
