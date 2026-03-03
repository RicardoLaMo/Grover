---
name: ps4-project-brief
description: Ground DTSC 8120 PS4 work in the official project brief and stakeholder requirements. Use when planning scope, writing CEO/Manager report sections, defining label semantics and clinical priority, justifying imbalance-aware metrics, or validating required deliverables for the chest X-ray classifier.
---

# PS4 Project Brief

## Overview
Use the project brief as the requirements baseline for the assignment. Extract constraints that affect modeling, evaluation, reporting, and delivery.

## Workflow
1. Open `doc/PROJECT_BRIEF.md` first.
2. Extract and restate the business problem in one sentence.
3. Map class IDs to clinical meaning and urgency.
4. Capture stakeholder-specific expectations (Executive Director, CEO, Manager, Senior Developer).
5. Quantify class imbalance and state why accuracy alone is insufficient.
6. Translate requirements into concrete deliverables and acceptance checks.

## Required Output Checklist
- State dataset split context: imbalanced train/validation vs balanced Kaggle test.
- State required metrics: macro and micro perspectives, plus per-class behavior.
- Include explicit mention of imbalance handling (class weights and/or sampler).
- Keep executive summary non-technical.
- Keep manager/developer sections explicit about methodology and reproducibility.

## Source and Reference
- Primary source: `doc/PROJECT_BRIEF.md`
- Quick reference: `references/project-brief-keypoints.md`
