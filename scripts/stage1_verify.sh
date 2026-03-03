#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

python -m pip freeze > artifacts/env/pip_freeze.txt
pytest -q
