# THORN Checkpoint

## Iteration Window
- Generated at iteration 2.

## Architecture Summary
- Data model supports transaction/entity-time style node semantics via config.
- Views implemented: kNN, time, diffusion.
- Unified neighborhood merge implemented with per-view masks/weights.
- Observer features implemented: LID, temporal features, kNN stats, LOF-like ratio, diffusion coords.
- Alignment implemented via verified surrogate neighborhood-overlap baseline (`align_views` interface stable).
- Router implemented (`ObserverRouter`) with per-head softmax routing.
- Routed attention implemented with adjacency bias/mask behavior and neighbor-wise normalization.
- Unified harness implemented for THORN + baselines with shared artifact schema.

## Module Contracts
- `thorn/contracts.py`: canonical shape contracts for views/union/router/alignment/attention.
- `thorn/train/harness.py`: end-to-end run orchestration and artifact persistence.
- `scripts/run_thorn.sh` and `scripts/run_baselines.sh`: one-command entrypoints.

## Gate Snapshot
- G0: PASS
- G1: PASS
- G2: PASS
- G3: PASS
- G4: PASS
- G5: PASS
- G6: PASS
- G7: PASS

## Notes
- Surrogate alignment is the verified baseline for this build; FGW remains an extensible backend.
