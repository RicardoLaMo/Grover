# Project Brief: Chest X-Ray Lung Classifier
**PS4 — DTSC 8120 | Deadline: March 4, 2026 5:30 PM ET**

---

## Business Problem

A city hospital is overwhelmed by the surge in lung complication cases post-pandemic. Radiologists and specialists cannot process the volume of X-rays manually. The hospital needs an **automated triage classifier** to sort chest X-rays into three priority categories, enabling specialists to focus on the most urgent cases first.

## Classification Target

| Class | Label | Description | Clinical Priority |
|-------|-------|-------------|-------------------|
| Healthy | 0 | No visible lung abnormality | Low |
| Pre-existing Conditions | 1 | Aortic enlargement, cardiomegaly, pulmonary fibrosis | Medium |
| Effusion/Mass (Urgent) | 2 | Pleural effusion, mass lesions | **HIGH — immediate attention** |

## Stakeholders & Their Requirements

### Executive Director (Hospital)
- **Needs**: A working system by March 4, 2026
- **Cares about**: Does it work? What conditions does it detect?
- **Report section**: CEO (1 figure + 2-3 sentences about accuracy/impact)

### CEO (Your Company)
- **Reads**: High-level summary only — a few sentences + 1 key figure
- **Grading**: 4 pts for report
- **Deliverable**: Clear, non-technical summary of what was built and how well it works

### Technical Manager
- **Reads**: Full methodology section
- **Specifically asked**: Explain imbalance handling, test set construction, choice of metrics
- **Grading**: 6 pts report + 2 pts code (can look at parts of code)
- **Key concern**: "Depending on how you construct your test set, your measured accuracy might be very different"

### Senior Developer
- **Reads**: Code primarily, may skip report
- **Grading**: 8 pts for code
- **Needs**: Comments on all non-trivial code, logical structure, markdown cells in notebooks
- **Key concern**: "I'm not a specialist in neural networks — explain any nontrivial part"

---

## Data Summary

### Training/Validation Data (`ps4_trainvalid_images-2.npy` + `ps4_trainvalid_labels.csv`)
- **Total samples**: 11,414 chest X-ray images
- **Class distribution**:
  - Class 0 (Healthy): 10,506 samples (92.0%) ← **severe imbalance**
  - Class 1 (Pre-existing): 651 samples (5.7%)
  - Class 2 (Effusion): 257 samples (2.3%)
- **File size**: 164 MB (numpy array of image data)
- **Label column**: `Label` (values: 0, 1, 2)
- **Use for**: Training, validation, and internal test evaluation

### Kaggle Test Data (`ps4_kaggle_images-1.npy` + `ps4_kaggle_labels.csv`)
- **Total samples**: 300 chest X-ray images
- **Class distribution**: Perfectly balanced (100 per class = 33.3% each)
- **File size**: 4.3 MB
- **Label column**: `Predicted` (values: 0, 1, 2)
- **Use for**: Final evaluation ONLY after model is validated and optimized
- **Extra credit**: Submit predictions as `Id,Predicted` CSV

---

## Critical Data Insight: The Imbalance Problem

The training data is **92% healthy patients** (Class 0). This creates a critical modeling trap:

> A model that predicts **"healthy" for every scan** would achieve **92% accuracy** — but would completely miss all urgent cases (Class 2), which is clinically catastrophic.

**Implications**:
1. **Accuracy is a misleading metric** — must use macro-F1, macro-AUC
2. **Naive models will be biased** toward Class 0
3. **Test set choice matters**: Unbalanced internal test set (~92% class 0) vs. balanced Kaggle set (33% each) will give dramatically different numbers — both must be reported
4. **Must implement** class-weighted loss or balanced sampling

### Class Weights (for weighted loss):
```
weight_0 = 11414 / (3 × 10506) = 0.362
weight_1 = 11414 / (3 × 651)   = 5.843
weight_2 = 11414 / (3 × 257)   = 14.81
```

---

## Technical Requirements

### Must Do
- [ ] Propose **at least 2 CNN architectures** and compare them
- [ ] Train models with proper train/val split
- [ ] Tune hyperparameters (LR, dropout, weight decay, architecture depth)
- [ ] Evaluate with appropriate metrics for imbalanced data
- [ ] Describe testing methodology in detail (this is "one of the central parts" — grader priority)
- [ ] Handle class imbalance explicitly

### Must Report
- [ ] How the test set was constructed and why
- [ ] Both micro and macro averaged metrics (or F1 and AUC)
- [ ] Per-class breakdown (precision, recall, F1 for each of 3 classes)
- [ ] Confusion matrix
- [ ] Comparison: balanced test vs. unbalanced test metrics
- [ ] Comparison: baseline model vs. improved model

### Extra Credit
- [ ] Run optimized model on `ps4_kaggle_images-1.npy`
- [ ] Submit `Id,Predicted` CSV (300 rows, classes 0/1/2)
- [ ] Report Kaggle results in Manager section
- [ ] Use techniques: dropout, L1/L2 regularization, architecture modifications

---

## Deliverables

| Item | Format | Where |
|------|--------|-------|
| Report | PDF (from LaTeX) | Canvas submission |
| Code | Python / Jupyter | GitHub (link in PDF) |
| Kaggle predictions | CSV (`Id,Predicted`) | Extra credit |

---

## Hints Alignment

| Hint Source | Hint | Implementation |
|-------------|------|----------------|
| PDF Hint (general) | Propose few CNN architectures, tune hyperparameters | Models A, B, C in `src/02_baseline_cnn.py` and `src/03_improved_cnn.py` |
| PDF Hint (general) | Testing methodology section is central — grader looks here first | Full section in Manager report |
| Technical Manager | Labels unbalanced — be careful with training AND measurement | Weighted loss + macro metrics |
| Technical Manager | Describe how you measured performance (test set construction) | Explicit subsection on eval methodology |
| Senior Developer | Write comments, explain non-trivial code | Docstrings + inline comments |
| Senior Developer | Use Jupyter markdown cells to split logically | Notebook structure |
