"""Smoke test for end-to-end harness artifact generation."""

from __future__ import annotations

import json
from pathlib import Path

from thorn.config import ExperimentConfig
from thorn.train.harness import ExperimentHarness


def test_harness_writes_artifacts(tmp_path: Path) -> None:
    cfg = ExperimentConfig()
    cfg.name = "pytest_smoke"
    cfg.output_root = str(tmp_path)
    cfg.train.epochs = 2
    cfg.train.overfit_tiny = True
    cfg.train.tiny_nodes = 80

    harness = ExperimentHarness(cfg)
    art = harness.run(mode="thorn")

    assert art.config_snapshot.exists()
    assert art.metrics_json.exists()
    assert art.checkpoint.exists()
    assert art.profile_json.exists()
    assert art.routing_json.exists()

    metrics = json.loads(art.metrics_json.read_text(encoding="utf-8"))
    assert "test" in metrics
    assert "pr_auc" in metrics["test"]
