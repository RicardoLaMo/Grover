#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

python scripts/run_baseline_suite.py --epochs 25
