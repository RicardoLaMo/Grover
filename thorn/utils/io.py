"""IO utilities."""

from __future__ import annotations

from pathlib import Path
import json


def save_json(payload: dict, path: Path) -> None:
    """Save JSON payload with stable key ordering."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
