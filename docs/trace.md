# Design Trace

## Decision 0001: Introduce isolated `thorn/` package within existing repo
- Date: 2026-02-27
- Decision: Add THORN implementation in a new package path (`thorn/`) instead of modifying existing chest X-ray pipeline modules.
- Rationale: Existing repo already contains a separate project; isolation reduces regression risk and keeps traceability clear.
- Alternatives considered:
  - Reuse existing `src/` tree directly: rejected due to coupling and higher migration risk.
  - New repository: rejected for now because user requested updates in current folder/workspace.
- References:
  - Local repository structure inspection (`find . -maxdepth 3 -type d`).

## Decision 0002: Use dataclass-based configuration with deterministic config hash
- Date: 2026-02-27
- Decision: Define experiment config via Python dataclasses and expose `sha256()` for config snapshots.
- Rationale: Required for reproducibility and artifact traceability in G7.
- Alternatives considered:
  - Raw dict configs only: rejected due to weak typing and weaker contracts.
  - External config dependency before Stage 2: deferred to keep Stage 1 minimal.
- References:
  - Python dataclasses documentation: https://docs.python.org/3/library/dataclasses.html
  - hashlib documentation: https://docs.python.org/3/library/hashlib.html

## Decision 0003: Enforce explicit tensor-shape contracts in central module
- Date: 2026-02-27
- Decision: Centralize shape and probability-simplex checks in `thorn/debug.py` and `thorn/contracts.py`.
- Rationale: Directly satisfies shape invariant requirements (G1) and prevents silent interface drift.
- Alternatives considered:
  - Ad-hoc asserts per module only: rejected due to inconsistency risk.
- References:
  - PyTorch tensor shape semantics: https://pytorch.org/docs/stable/tensors.html
  - `torch.nn.functional.softmax`: https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html

## Decision 0004: Verify runtime dependency versions before relying on interfaces
- Date: 2026-02-27
- Decision: Capture runtime versions for Python, torch, numpy, and pytest before Stage 1 verification.
- Rationale: Satisfies anti-hallucination rule (no assumptions about local runtime behavior).
- Alternatives considered:
  - Infer versions from prior project docs: rejected because current environment must be observed directly.
- References:
  - Runtime probe command output persisted to `artifacts/env/runtime_versions.txt`.

## Decision 0005: Add pytest import bootstrap via `tests/conftest.py`
- Date: 2026-02-27
- Decision: Insert repository root into `sys.path` from `tests/conftest.py`.
- Rationale: `pytest` collection initially failed with `ModuleNotFoundError: thorn`; this fix allows local package imports without editable install in Stage 1.
- Alternatives considered:
  - `pip install -e .` during every verification: rejected for unnecessary setup overhead in early iterations.
  - Modify environment `PYTHONPATH` manually per command: rejected due to reproducibility risk.
- References:
  - Pytest collection failure from `bash scripts/stage1_verify.sh` (first run).

## Decision 0006: Implement surrogate transport alignment baseline first
- Date: 2026-02-27
- Decision: Implement `align_views(...)` using neighborhood-overlap agreement instead of FGW in the initial runnable system.
- Rationale: Required to deliver a verifiable alignment module with stable interface and ablation support without introducing unverified heavy OT dependencies.
- Alternatives considered:
  - FGW approximation immediately: deferred due verification and dependency risk.
  - No alignment module: rejected because blueprint requires alignment-conditioned routing/attention path.
- References:
  - THORN blueprint requirement in `docs/BLUEPRINT.md` section D.

## Decision 0007: Use deterministic synthetic time-split protocol for drift gate
- Date: 2026-02-27
- Decision: Train on early timestamps, evaluate on later timestamps, and log routing shift/collapse metrics.
- Rationale: Satisfies drift gate with controlled and reproducible temporal distribution shift.
- Alternatives considered:
  - Random split only: rejected because it does not test temporal drift.
  - External dataset dependency: rejected to keep one-command reproducibility local.
- References:
  - Gate requirement G6 in `docs/gates.md`.

## Decision 0008: Full-batch deterministic training for reproducibility gate
- Date: 2026-02-27
- Decision: Use fixed seeds + deterministic torch algorithms and a reproducibility checker that reruns with same seed and compares metrics.
- Rationale: Produces stable G7 evidence with explicit tolerance report.
- Alternatives considered:
  - GPU-first stochastic dataloaders: deferred to avoid nondeterministic variance.
- References:
  - PyTorch deterministic API: https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
