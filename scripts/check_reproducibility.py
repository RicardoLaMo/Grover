#!/usr/bin/env python
"""Run two same-seed THORN runs and compare metrics within tolerance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from thorn.config import ExperimentConfig
from thorn.train.harness import ExperimentHarness


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check THORN reproducibility")
    p.add_argument("--output-root", type=str, default="artifacts/runs")
    p.add_argument("--report-path", type=str, default="artifacts/reports/reproducibility.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--tolerance", type=float, default=1e-9)
    return p.parse_args()


def _run_once(cfg: ExperimentConfig) -> dict:
    harness = ExperimentHarness(cfg)
    art = harness.run(mode="thorn")
    return json.loads(Path(art.metrics_json).read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig()
    cfg.name = "repro"
    cfg.output_root = args.output_root
    cfg.train.seed = args.seed
    cfg.train.epochs = args.epochs

    m1 = _run_once(cfg)
    m2 = _run_once(cfg)

    keys = ["pr_auc", "roc_auc", "precision_at_k", "accuracy"]
    diffs = {}
    for split in ["train", "val", "test"]:
        for k in keys:
            diffs[f"{split}.{k}"] = abs(float(m1[split][k]) - float(m2[split][k]))

    passed = all(v <= args.tolerance for v in diffs.values())
    payload = {
        "seed": args.seed,
        "tolerance": args.tolerance,
        "passed": passed,
        "diffs": diffs,
    }

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"report={report_path}")
    print(f"passed={passed}")


if __name__ == "__main__":
    main()
