"""Shared helper for reading metrics from run artifacts."""

from __future__ import annotations
import json
from pathlib import Path


def read_run_metrics(metrics_json: Path) -> dict:
    """Read a metrics.json file and return flattened metric dict."""
    data = json.loads(metrics_json.read_text())
    return {
        "test_pr_auc": data.get("test", {}).get("pr_auc", 0),
        "test_roc_auc": data.get("test", {}).get("roc_auc", 0),
        "val_pr_auc": data.get("val", {}).get("pr_auc", 0),
        "collapse_score": data.get("drift", {}).get("collapse_score", 0),
        "routing_shift_l1": data.get("drift", {}).get("routing_shift_l1", 0),
    }
