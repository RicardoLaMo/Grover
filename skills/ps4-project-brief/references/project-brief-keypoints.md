# Project Brief Keypoints

Source document: `doc/PROJECT_BRIEF.md`

## Critical Facts
- Business objective: triage chest X-rays into classes 0/1/2 for hospital prioritization.
- Class priority: Class 2 is urgent and clinically highest priority.
- Data imbalance: train/validation set is heavily skewed toward Class 0; Kaggle set is balanced.
- Metric implication: macro-level metrics are required because raw accuracy can be misleading.
- Deliverables: report PDF, code repo, and optional Kaggle submission CSV (`Id,Predicted`).

## Stakeholder Focus
- CEO: plain-language impact and one clear figure.
- Manager: explicit test methodology and imbalance handling.
- Senior Developer: code clarity, comments, docstrings, and reproducibility.

## Prior Progress Memory
- Current report file: `doc/report/main.tex` with compiled `doc/report/main.pdf`.
- Current KPI narrative in report:
  - ResNet18 baseline pipeline result: internal macro-F1 `0.709` (unbalanced), Kaggle macro-F1 `0.723`.
  - Balanced Kaggle-style reporting is already included in the report.
- Open deliverable cleanup:
  - Replace placeholders `[Your Name]`, `[your-username]`, and `[repo-name]` before final submission.
