# THORN Goals

## North Star
Build a research-grade, end-to-end runnable PyTorch system for THORN (Transport-aligned Heterogeneous Observer-Routed Neighborhood attention) with baselines, strict traceability, and reproducibility.

## Non-Negotiables
- No hallucinated claims, APIs, or results.
- Every major design decision must be logged in `docs/trace.md`.
- Every experiment must emit config snapshot + metrics JSON.
- Reproducibility from one command per workflow.
- All quality gates G0-G7 must pass before completion.

## Milestones
1. Stage 1 Planner artifacts and module contracts.
2. Stage 2 Minimal end-to-end synthetic training (G0-G2).
3. Stage 3 Geometry enrichment (diffusion + LID + observers).
4. Stage 4 Transport alignment + ablations.
5. Stage 5 Baselines + drift evaluation.
6. Stage 6 Gate completion and reproducibility validation.

## Definition of Done
- `bash scripts/run_baselines.sh` and `bash scripts/run_thorn.sh` reproduce stored metrics.
- Gates G0-G7 all PASS with evidence paths in `docs/gates.md` and `BUILD_LOG.md`.
- `docs/STATUS.md` final state is complete with zero blocking risks.
