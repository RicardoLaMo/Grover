# THORN Status

## Current Stage
- Stage 6: Gate completion and reproducibility validation

## Iteration
- Iteration: 2
- Timestamp: 2026-02-27

## Gate Status
| Gate | Status | Evidence |
|---|---|---|
| G0 | PASS | `bash scripts/stage1_verify.sh`, `artifacts/env/pip_freeze.txt` |
| G1 | PASS | `tests/test_views_builders.py`, `tests/test_router_attention.py`, `thorn/debug.py` |
| G2 | PASS | `bash scripts/run_thorn.sh --smoke` + run artifacts |
| G3 | PASS | `bash scripts/run_thorn.sh --overfit_tiny` (`train accuracy=0.995`) |
| G4 | PASS | `bash scripts/run_baselines.sh`, `artifacts/reports/results_table.*`, `alignment_effect.json` |
| G5 | PASS | `profile.json` with component runtime/memory and small/scalable flags |
| G6 | PASS | time-split drift metrics in run `metrics.json` + `routing_stats.json` |
| G7 | PASS | `bash scripts/run_thorn.sh --repro`, `artifacts/reports/reproducibility.json` |

## Open Risks
- Current transport alignment implementation uses verified surrogate overlap baseline; FGW approximation remains optional future work.
- CPU-oriented implementation is deterministic and reproducible; GPU throughput optimizations are not yet prioritized.

## Next 3 Actions
1. Optional: implement FGW-approx alignment backend under existing alignment interface.
2. Optional: extend synthetic benchmarks to larger N and multi-class labels.
3. Optional: add richer efficiency profiling (per-op CUDA memory timeline).
