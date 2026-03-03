#!/usr/bin/env python
"""Run a single THORN or baseline experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from thorn.config import ExperimentConfig
from thorn.train.harness import ExperimentHarness


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run THORN experiment")
    p.add_argument("--mode", type=str, default="thorn")
    p.add_argument("--name", type=str, default="thorn")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--num-nodes", type=int, default=512)
    p.add_argument("--num-features", type=int, default=16)
    p.add_argument("--knn-k", type=int, default=10)
    p.add_argument("--time-window", type=int, default=6)
    p.add_argument("--diffusion-dim", type=int, default=8)
    p.add_argument("--output-root", type=str, default="artifacts/runs")
    p.add_argument("--run-dir", type=str, default="")
    p.add_argument("--small-mode", action="store_true")
    p.add_argument("--scalable-mode", action="store_true")
    p.add_argument("--overfit-tiny", action="store_true")
    p.add_argument("--alignment-off", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = ExperimentConfig()
    cfg.name = args.name
    cfg.mode = args.mode  # type: ignore[assignment]
    cfg.output_root = args.output_root
    cfg.data.num_nodes = args.num_nodes
    cfg.data.num_features = args.num_features
    cfg.views.knn_k = args.knn_k
    cfg.views.time_window = args.time_window
    cfg.views.diffusion_dim = args.diffusion_dim
    cfg.views.scalable_mode = args.scalable_mode
    cfg.train.seed = args.seed
    cfg.train.epochs = args.epochs
    cfg.train.overfit_tiny = bool(args.overfit_tiny or args.small_mode)
    cfg.alignment.enabled = not args.alignment_off

    harness = ExperimentHarness(cfg)
    run_dir = Path(args.run_dir) if args.run_dir else None
    artifacts = harness.run(mode=args.mode, output_dir=run_dir)

    print(f"run_dir={artifacts.run_dir}")
    print(f"metrics={artifacts.metrics_json}")
    print(f"config={artifacts.config_snapshot}")
    print(f"checkpoint={artifacts.checkpoint}")
    print(f"profile={artifacts.profile_json}")
    print(f"routing={artifacts.routing_json}")


if __name__ == "__main__":
    main()
