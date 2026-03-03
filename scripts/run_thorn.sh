#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

MODE="smoke"
if [[ $# -ge 1 ]]; then
  MODE="$1"
fi

case "$MODE" in
  --smoke|smoke)
    python scripts/run_experiment.py \
      --mode thorn \
      --name thorn_smoke \
      --epochs 12 \
      --small-mode
    ;;
  --full|full)
    python scripts/run_experiment.py \
      --mode thorn \
      --name thorn_full \
      --epochs 40
    ;;
  --overfit_tiny|overfit_tiny)
    python scripts/run_experiment.py \
      --mode thorn \
      --name thorn_overfit \
      --epochs 220 \
      --overfit-tiny
    ;;
  --repro|repro)
    python scripts/check_reproducibility.py --epochs 25
    ;;
  *)
    echo "Unknown mode: $MODE"
    echo "Usage: bash scripts/run_thorn.sh [--smoke|--full|--overfit_tiny|--repro]"
    exit 2
    ;;
esac
