# Grading Rubric & Passing Gates
**PS4 — DTSC 8120 | Total: 20 points**

---

## Score Breakdown

| Reviewer | Report | Code | Total |
|----------|--------|------|-------|
| CEO | 4 pts | — | 4 pts |
| Manager | 6 pts | 2 pts | 8 pts |
| Senior Developer | — | 8 pts | 8 pts |
| **Total** | **10 pts** | **10 pts** | **20 pts** |

---

## CEO Evaluation (4 pts for report)

**What the CEO reads**: A few sentences and one figure.

### Checklist
- [ ] **Executive Summary** (1-2 sentences): What was built? What problem does it solve?
- [ ] **Key Result** (1 sentence): Best model performance in plain language (e.g., "The model correctly identifies urgent lung conditions X% of the time")
- [ ] **Key Figure**: Confusion matrix or bar chart of per-class performance — visually clear, labeled
- [ ] **No jargon**: No mention of "conv layers", "gradient descent", etc. in this section
- [ ] **Impact statement**: Why does this matter for the hospital?

### CEO Section Template
```
Problem: [1 sentence — what are we classifying and why?]
Solution: [1 sentence — what approach did we take?]
Result: [1 sentence — how well does it work, in plain terms?]
[1 Figure: confusion matrix or class-performance bar chart]
```

---

## Manager Evaluation (6 pts report + 2 pts code)

**What the Manager reads**: Full detailed methodology. May look at code structure.

### Report Checklist (6 pts)
- [ ] **Data Description**: N samples, image shapes, class distribution table/plot
- [ ] **Imbalance Analysis**: Quantify imbalance, explain why it matters for accuracy
- [ ] **Test Set Construction** ← CRITICAL (grader explicitly looks here):
  - [ ] Explain HOW the split was done (stratified? random? ratio?)
  - [ ] Explain WHY (stratified preserves distribution)
  - [ ] Report results on BOTH balanced and unbalanced test sets
  - [ ] Note that Kaggle set is balanced vs. internal test which is unbalanced
- [ ] **Metric Choice Justification**: Why macro-F1 (not accuracy)? Why AUC?
- [ ] **Model Comparison Table**: Baseline vs. improved — val metrics side by side
- [ ] **Hyperparameter Tuning**: What was tuned, range explored, best values
- [ ] **Imbalance Handling Strategy**: class weights, sampler, augmentation
- [ ] **Final Model Results**: Full classification report (per-class P/R/F1)
- [ ] **Confusion Matrix**: With interpretation
- [ ] **Extra Credit (if done)**: Kaggle results vs. baseline

### Code Checklist for Manager (2 pts)
- [ ] Code is logically organized (clear file/notebook structure)
- [ ] Training/evaluation pipeline is understandable at a high level
- [ ] Results (metrics, figures) are reproducible from the code

---

## Senior Developer Evaluation (8 pts for code)

**What the Developer reads**: Code quality, comments, structure.

### Code Checklist (8 pts)
- [ ] **Comments on all non-trivial code**:
  - CNN architecture layers: why this number of filters? what does this do?
  - Loss function choice: explain CrossEntropyLoss with class weights
  - Training loop: explain what optimizer.zero_grad(), loss.backward(), optimizer.step() do
  - Data augmentation: explain each transform
- [ ] **Function docstrings**: Every function/class has a docstring with args/returns
- [ ] **Logical structure**: Notebook/script split into clear sections with headers
- [ ] **Jupyter markdown cells**: If notebook, use markdown to separate: Data → EDA → Model → Evaluation
- [ ] **Reproducibility**: Random seeds set (`torch.manual_seed(42)`)
- [ ] **Config dict**: Hyperparameters in a CONFIG dict at top, not scattered as magic numbers
- [ ] **No dead code**: Remove unused imports, commented-out experiments
- [ ] **Readable variable names**: `conv_block1`, `train_loader`, `class_weights` (not `x1`, `dl`, `w`)

---

## Quality Gates (Sequential — must pass each before proceeding)

### Gate 1: EDA Complete
**Criterion**: Can answer these questions from output:
- What is the image shape (H × W × C)?
- What is the pixel value range (0-1 float? 0-255 uint8?)
- Are there any NaN/corrupt images?
- What is the exact class distribution?

**Output**: `output/figures/class_distribution.png`, `output/figures/sample_images.png`

---

### Gate 2: Data Pipeline Validated
**Criterion**: DataLoader returns correct shapes and balanced batches

**Check**:
```python
for X, y in train_loader:
    assert X.shape == (BATCH_SIZE, C, H, W)
    assert y.shape == (BATCH_SIZE,)
    assert set(y.numpy()).issubset({0, 1, 2})
    break  # passes
```

**Output**: Class weights computed and logged

---

### Gate 3: Baseline Model Trains
**Criterion**:
- Training loss decreases over epochs (not stuck/diverging)
- Val macro-F1 > 0.40 (better than random for 3-class = 0.33)
- Per-class recall for Class 2 (urgent) > 0.20 (not completely missed)

**Output**: `output/models/baseline.pth`, training curve saved

---

### Gate 4: Improved Model Outperforms Baseline
**Criterion**:
- Val macro-F1 > 0.60 (or best effort with documented hyperparameter search)
- Improvement over baseline is demonstrated and quantified
- At least one regularization technique used (dropout, L2, augmentation)

**Output**: `output/models/improved.pth`, comparison table

---

### Gate 5: Evaluation Report Complete
**Criterion**: Must include ALL of:
- [ ] Classification report (P/R/F1 per class + macro avg)
- [ ] Confusion matrix (normalized)
- [ ] ROC curves (one vs. rest, per class)
- [ ] Results on UNBALANCED internal test set
- [ ] Results on BALANCED test subset (sampled equally per class)
- [ ] Written interpretation of each metric

**Output**: `output/figures/confusion_matrix.png`, `output/figures/roc_curves.png`

---

### Gate 6: Kaggle Prediction (Extra Credit)
**Criterion**:
- CSV has exactly 300 rows + header
- Columns: `Id` (0-299 int), `Predicted` (0/1/2 int)
- No NaN values

**Output**: `output/predictions/kaggle_submission.csv`

---

### Gate 7: LaTeX Report Compiles
**Criterion**:
- PDF compiles without errors
- All three sections present (CEO, Manager, Developer/Methods)
- All figures included with captions
- GitHub link in document
- < 10 pages (or reasonable length)

**Output**: `doc/report/main.pdf`
