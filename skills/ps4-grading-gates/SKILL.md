---
name: ps4-grading-gates
description: Enforce PS4 rubric and quality gates for report/code deliverables. Use when reviewing readiness before submission, auditing model outputs against required metrics/artifacts, or producing pass/fail findings tied to CEO/Manager/Senior Developer expectations.
---

# PS4 Grading Gates

## Overview
Use this skill to run rubric-driven checks before final submission and to identify missing artifacts early.

## Workflow
1. Open `doc/GRADING_RUBRIC.md`.
2. Evaluate deliverables in reviewer order: CEO, Manager, Senior Developer.
3. Run gate checks G1 through G7 sequentially.
4. Mark each gate as pass/fail with concrete missing evidence when failing.
5. Produce an action list that prioritizes blocking failures first.

## Required Review Output
- CEO report readiness (clear non-technical summary + key figure).
- Manager methodology readiness (test-set construction, imbalance strategy, metric rationale).
- Developer code-quality readiness (comments/docstrings/structure/reproducibility).
- Gate status matrix for G1-G7 with artifact paths.

## Source and Reference
- Primary source: `doc/GRADING_RUBRIC.md`
- Quick reference: `references/grading-gates-keypoints.md`
