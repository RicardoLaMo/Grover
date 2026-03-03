# Grading Gates Keypoints

Source document: `doc/GRADING_RUBRIC.md`

## Reviewer Breakdown
- CEO: 4 points (report summary + figure clarity).
- Manager: 8 points (6 report + 2 code; methodology is central).
- Senior Developer: 8 points (code quality and explanation quality).

## Sequential Gates
- G1: EDA artifacts and data sanity.
- G2: data pipeline and class-weight setup.
- G3: baseline training convergence and minimum macro-F1.
- G4: improved model gains or justified best effort.
- G5: complete evaluation package including balanced/unbalanced views.
- G6: Kaggle CSV format validity (extra credit).
- G7: final report compile and completeness.

## Review Priority
Failing G2-G5 should block release; resolve before polishing narrative sections.

## Prior Progress Memory (Current Status)
- `G1` PASS: EDA figures exist (`class_distribution.png`, `sample_images.png`).
- `G2` PASS: split and class weights are logged in `output/pipeline_results.json`.
- `G3` FAIL under strict threshold: baseline val macro-F1 `0.1749` is below rubric target `0.40`.
- `G4` PASS: improved models surpass `0.60` validation macro-F1.
- `G5` PASS: both balanced and unbalanced test metrics are present.
- `G6` PASS: Kaggle CSV validated (`300` rows, `Id` 0-299, `Predicted` in {0,1,2}, no NaNs).
- `G7` PARTIAL: `doc/report/main.pdf` exists, but report still includes name/repo placeholders.
