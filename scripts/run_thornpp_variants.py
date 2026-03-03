"""Run THORN++ variant experiments for revision checklist.

Tests: adaptive tau, view-specific Q/K/V, online proxy, and combinations.
Compares against baseline THORN at drift=0.75 (bias mode) and drift=0.50 (post_softmax).
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from thorn.config import ExperimentConfig
from thorn.train.harness import ExperimentHarness
from thorn.utils.seed import set_global_seed as set_seed


def make_config(
    name: str,
    mode: str = "thorn",
    adj_mode: str = "bias",
    drift: float = 0.75,
    adaptive_tau: bool = False,
    view_specific: bool = False,
    alignment_method: str = "fgw",
    epochs: int = 80,
) -> ExperimentConfig:
    config = ExperimentConfig()
    config.name = name
    config.mode = mode
    config.data.drift_strength = drift
    config.attention.adjacency_mode = adj_mode
    config.attention.adaptive_tau = adaptive_tau
    config.attention.view_specific_projections = view_specific
    config.alignment.method = alignment_method
    config.train.epochs = epochs
    config.train.seed = 42
    config.train.deterministic = True
    return config


VARIANTS = [
    # Baseline references (should match prior runs)
    ("baseline_bias_075", dict(adj_mode="bias", drift=0.75)),
    ("baseline_post_050", dict(adj_mode="post_softmax", drift=0.50)),

    # Adaptive tau only
    ("tau_bias_075", dict(adj_mode="bias", drift=0.75, adaptive_tau=True)),
    ("tau_post_050", dict(adj_mode="post_softmax", drift=0.50, adaptive_tau=True)),

    # View-specific Q/K/V only (post-softmax, where it matters)
    ("viewqkv_post_050", dict(adj_mode="post_softmax", drift=0.50, view_specific=True)),
    ("viewqkv_post_075", dict(adj_mode="post_softmax", drift=0.75, view_specific=True)),

    # Online proxy only
    ("proxy_bias_075", dict(adj_mode="bias", drift=0.75, alignment_method="online_proxy")),
    ("proxy_post_050", dict(adj_mode="post_softmax", drift=0.50, alignment_method="online_proxy")),

    # Full THORN++ (all three)
    ("thornpp_bias_075", dict(
        adj_mode="bias", drift=0.75, adaptive_tau=True,
        view_specific=True, alignment_method="online_proxy")),
    ("thornpp_post_050", dict(
        adj_mode="post_softmax", drift=0.50, adaptive_tau=True,
        view_specific=True, alignment_method="online_proxy")),

    # Adaptive tau + view-specific (no proxy change)
    ("tau_viewqkv_post_050", dict(
        adj_mode="post_softmax", drift=0.50, adaptive_tau=True, view_specific=True)),
]


def main():
    output_root = Path("artifacts/runs_thornpp")
    report_dir = Path("artifacts/reports_thornpp")
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

        # Read metrics from saved JSON
        from _metrics_helper import read_run_metrics
        m = read_run_metrics(artifacts.metrics_json)

        results[name] = {
            "test_pr_auc": m["test_pr_auc"],
            "test_roc_auc": m["test_roc_auc"],
            "val_pr_auc": m["val_pr_auc"],
            "collapse_score": m["collapse_score"],
            "routing_shift_l1": m["routing_shift_l1"],
            "config": {
                "adj_mode": kwargs.get("adj_mode", "bias"),
                "drift": kwargs.get("drift", 0.75),
                "adaptive_tau": kwargs.get("adaptive_tau", False),
                "view_specific": kwargs.get("view_specific", False),
                "alignment_method": kwargs.get("alignment_method", "fgw"),
            }
        }

        print(f"  PR-AUC: {results[name]['test_pr_auc']:.4f}")
        print(f"  ROC-AUC: {results[name]['test_roc_auc']:.4f}")

    # Save results
    report_path = report_dir / "thornpp_variants.json"
    report_path.write_text(json.dumps(results, indent=2))

    # Print summary table
    print(f"\n{'='*80}")
    print("THORN++ VARIANT RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Variant':<30} {'PR-AUC':>8} {'ROC-AUC':>9} {'Collapse':>9} {'Shift':>8}")
    print("-" * 70)
    for name, r in results.items():
        print(f"{name:<30} {r['test_pr_auc']:>8.4f} {r['test_roc_auc']:>9.4f} "
              f"{r['collapse_score']:>9.3f} {r['routing_shift_l1']:>8.4f}")

    print(f"\nResults saved to {report_path}")


if __name__ == "__main__":
    main()
