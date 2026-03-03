# BUILD_LOG

## Iteration 1 - 2026-02-27 - Stage 1 bootstrap
- Scope: create persistent planner artifacts and THORN interfaces without model implementation.
- Added: `docs/GOALS.md`, `docs/BLUEPRINT.md`, `docs/gates.md`, `docs/STATUS.md`, `docs/CHECKPOINT.md`, `docs/trace.md`, `docs/math.md`, `docs/positioning.md`.
- Added: `thorn/` package scaffolding with explicit contracts and stubs.
- Added: `pyproject.toml` and gate-mapping placeholders.
- GOALS sha256: `5b3926b53979745bcd5daf98a72b8271d6a7d040865a6e9a263b762b632d8dc4`
- Runtime dependency verification:
  - `python=3.13.9`
  - `torch=2.9.0+cu128`
  - `numpy=2.2.6`
  - `pytest=9.0.2`
- Commands executed:
  - `bash scripts/stage1_verify.sh` (first run): FAIL (`ModuleNotFoundError: thorn`)
  - Added `tests/conftest.py` path bootstrap
  - `bash scripts/stage1_verify.sh` (second run): PASS (`7 passed`)
  - Added `.gitignore` for Python cache files
  - `bash scripts/stage1_verify.sh` (final run): PASS (`7 passed`)
- Artifacts produced:
  - `artifacts/env/pip_freeze.txt`
  - `artifacts/env/runtime_versions.txt`

## Iteration 2 - 2026-02-27 - Stage 2 through Stage 6 completion
- Scope: implement runnable THORN + baselines, execute gate checks, and finalize reproducibility artifacts.
- GOALS sha256: `5b3926b53979745bcd5daf98a72b8271d6a7d040865a6e9a263b762b632d8dc4`
- Implemented modules:
  - concrete builders: `thorn/views/{knn.py,time.py,diffusion.py,union.py}`
  - observer features: `thorn/observers/*.py`
  - alignment surrogate: `thorn/alignment/{interface.py,surrogate.py}`
  - router + attention: `thorn/routing/router.py`, `thorn/layers/routed_attention.py`
  - harness + eval: `thorn/train/{harness.py,eval.py}`
  - CLIs: `scripts/run_experiment.py`, `scripts/run_baseline_suite.py`, `scripts/check_reproducibility.py`, `scripts/check_gates.py`
- Commands executed:
  - `pytest -q` -> PASS (`10 passed`)
  - `bash scripts/run_thorn.sh --smoke` -> PASS (G2 artifacts written)
  - `bash scripts/run_baselines.sh` -> PASS (`artifacts/reports/results_table.*`)
  - `bash scripts/run_thorn.sh --overfit_tiny` -> PASS (`train accuracy=0.995`)
  - `PYTHONPATH="$(pwd):${PYTHONPATH:-}" python scripts/run_experiment.py --mode thorn --name thorn_scalable --epochs 5 --small-mode --scalable-mode`
  - `bash scripts/run_thorn.sh --repro` -> PASS (`artifacts/reports/reproducibility.json`, `passed=true`)
  - `PYTHONPATH="$(pwd):${PYTHONPATH:-}" python scripts/check_gates.py` -> `overall_pass=true`
- Key evidence:
  - `artifacts/reports/gate_check.json`
  - `artifacts/reports/results_table.csv`
  - `artifacts/reports/reproducibility.json`
  - `artifacts/runs/20260227-214517_thorn_overfit_thorn_93cb373f/metrics.json`
