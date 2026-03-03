# Quality Gates (G0-G7)

## Evidence Format
For each gate, record:
- command(s) executed
- PASS/FAIL
- artifact paths
- notes on residual risk

## G0 Environment gate
- Capture `pip freeze` or conda env in artifacts.
- Document deterministic seed policy.
- Run `pytest`.

## G1 Shapes & invariants gate
- Tests validate:
  - view graph construction outputs
  - router output pi shape and simplex along view dimension
  - attention normalization over neighbors
- Shape assertions present in debug mode.

## G2 End-to-end smoke gate
- Single command trains on synthetic data for N epochs.
- Must save: metrics JSON + config snapshot + checkpoint.

## G3 Overfit gate
- Tiny dataset overfit sanity (or justified objective decrease for unsupervised).

## G4 Ablation gate
- Baselines run via same harness.
- Results table auto-generated.
- Alignment on/off changes behavior and is documented.

## G5 Efficiency gate
- Report runtime + peak memory for neighborhood construction, router, and attention.
- Provide `small mode` and `scalable mode` flags.

## G6 Drift gate
- Time-split evaluation implemented.
- Routing distributions vary over time and do not trivially collapse unless justified.

## G7 Reproducibility gate
- Two reruns with same seed match within tolerance.
- Config snapshot saved with hash.

## Gate Evidence (Current)

### G0 PASS
- Commands:
  - `bash scripts/stage1_verify.sh`
- Evidence:
  - `artifacts/env/pip_freeze.txt`
  - pytest summary: `10 passed`

### G1 PASS
- Commands:
  - `pytest -q`
- Evidence:
  - `tests/test_views_builders.py` (view construction invariants)
  - `tests/test_router_attention.py::test_router_simplex`
  - `tests/test_router_attention.py::test_attention_normalizes_over_incoming_neighbors`
  - shape asserts in `thorn/debug.py`

### G2 PASS
- Commands:
  - `bash scripts/run_thorn.sh --smoke`
- Evidence:
  - `artifacts/runs/20260227-214233_thorn_smoke_thorn_e984123c/config_snapshot.json`
  - `artifacts/runs/20260227-214233_thorn_smoke_thorn_e984123c/metrics.json`
  - `artifacts/runs/20260227-214233_thorn_smoke_thorn_e984123c/checkpoint.pt`

### G3 PASS
- Commands:
  - `bash scripts/run_thorn.sh --overfit_tiny`
- Evidence:
  - `artifacts/runs/20260227-214517_thorn_overfit_thorn_93cb373f/metrics.json`
  - train accuracy `0.995`, overfit flag `true`

### G4 PASS
- Commands:
  - `bash scripts/run_baselines.sh`
- Evidence:
  - `artifacts/reports/results_table.csv`
  - `artifacts/reports/results_table.md`
  - `artifacts/reports/alignment_effect.json`
  - measurable alignment effect (`delta_routing_shift_l1` non-zero)

### G5 PASS
- Commands:
  - `PYTHONPATH="$(pwd):${PYTHONPATH:-}" python scripts/run_experiment.py --mode thorn --name thorn_scalable --epochs 5 --small-mode --scalable-mode`
- Evidence:
  - `artifacts/runs/20260227-214630_thorn_scalable_thorn_dd66befa/profile.json`
  - includes runtime for graph construction/router/attention and `peak_memory_mb`
  - includes `small_mode=true` and `scalable_mode=true`

### G6 PASS
- Commands:
  - `bash scripts/run_baselines.sh`
- Evidence:
  - `artifacts/runs/20260227-214358_baseline_suite_thorn_c1b959a7/metrics.json` (drift section)
  - `artifacts/runs/20260227-214358_baseline_suite_thorn_c1b959a7/routing_stats.json`
  - non-collapse (`collapse_score` < 0.95) and non-zero routing shift

### G7 PASS
- Commands:
  - `bash scripts/run_thorn.sh --repro`
- Evidence:
  - `artifacts/reports/reproducibility.json` with `passed=true`
  - run metrics include `config_hash` and saved `config_snapshot.json`
