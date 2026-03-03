"""Sweep LoRA rank for view-specific Q/K/V to find if smaller rank reduces overfitting.

Tests rank=2, 4, 8 (auto=12 already tested) in post_softmax mode at drift=0.50 and 0.75.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from thorn.config import ExperimentConfig
from thorn.train.harness import ExperimentHarness
from thorn.utils.seed import set_global_seed as set_seed


def make_config(name, adj_mode, drift, lora_rank, epochs=80):
    config = ExperimentConfig()
    config.name = name
    config.mode = "thorn"
    config.data.drift_strength = drift
    config.attention.adjacency_mode = adj_mode
    config.attention.view_specific_projections = True
    config.attention.lora_rank = lora_rank
    config.train.epochs = epochs
    config.train.seed = 42
    config.train.deterministic = True
    return config


VARIANTS = []
for rank in [2, 4, 8]:
    VARIANTS.append((f"viewqkv_r{rank}_post_050", dict(adj_mode="post_softmax", drift=0.50, lora_rank=rank)))
    VARIANTS.append((f"viewqkv_r{rank}_post_075", dict(adj_mode="post_softmax", drift=0.75, lora_rank=rank)))


def main():
    output_root = Path("artifacts/runs_lora_sweep")
    report_dir = Path("artifacts/reports_lora_sweep")
    report_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for name, kwargs in VARIANTS:
        print(f"\n{'='*60}")
        print(f"Running variant: {name}")
        print(f"{'='*60}")

        set_seed(42)
        config = make_config(name, **kwargs)
        harness = ExperimentHarness(config)

        run_dir = output_root / name
        artifacts = harness.run(mode=config.mode, output_dir=run_dir)

        from _metrics_helper import read_run_metrics
        m = read_run_metrics(artifacts.metrics_json)

        results[name] = {
            "test_pr_auc": m["test_pr_auc"],
            "test_roc_auc": m["test_roc_auc"],
            "lora_rank": kwargs["lora_rank"],
            "drift": kwargs["drift"],
            "adj_mode": kwargs["adj_mode"],
        }
        print(f"  PR-AUC: {results[name]['test_pr_auc']:.4f}")

    report_path = report_dir / "lora_rank_sweep.json"
    report_path.write_text(json.dumps(results, indent=2))

    # Baselines for comparison
    baselines = {"post_050": 0.9839, "post_075": 0.9322}

    print(f"\n{'='*70}")
    print("LORA RANK SWEEP RESULTS")
    print(f"{'='*70}")
    print(f"{'Variant':<30} {'Rank':>5} {'PR-AUC':>8} {'Δ vs base':>10}")
    print("-" * 55)
    for name, r in results.items():
        drift_key = f"post_{int(r['drift']*100):03d}"
        base = baselines.get(drift_key, 0)
        delta = r["test_pr_auc"] - base
        print(f"{name:<30} {r['lora_rank']:>5} {r['test_pr_auc']:>8.4f} {delta:>+10.4f}")

    print(f"\nResults saved to {report_path}")


if __name__ == "__main__":
    main()
