"""Log-barrier stability study (critique 3.5 / checklist item 4).

Tests adaptive tau under normalization variants:
  (a) none, (b) row norm, (c) sym norm, (d) self-loops
With and without adaptive tau.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from thorn.config import ExperimentConfig
from thorn.train.harness import ExperimentHarness
from thorn.utils.seed import set_global_seed as set_seed


CONFIGS = [
    # (name, degree_norm, self_loops, adaptive_tau)
    ("none_baseline", "none", False, False),
    ("none_tau", "none", False, True),
    ("sym_baseline", "symmetric", False, False),
    ("sym_tau", "symmetric", False, True),
    ("row_baseline", "row", False, False),
    ("row_tau", "row", False, True),
    ("selfloops_baseline", "none", True, False),
    ("selfloops_tau", "none", True, True),
]


def main():
    output_root = Path("artifacts/runs_stability")
    report_dir = Path("artifacts/reports_stability")
    report_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for name, deg_norm, self_loops, adaptive_tau in CONFIGS:
        print(f"\n{'='*60}")
        print(f"Stability study: {name}")
        print(f"  degree_norm={deg_norm}, self_loops={self_loops}, adaptive_tau={adaptive_tau}")
        print(f"{'='*60}")

        set_seed(42)
        config = ExperimentConfig()
        config.name = name
        config.mode = "thorn"
        config.data.drift_strength = 0.75
        config.attention.adjacency_mode = "bias"
        config.attention.adaptive_tau = adaptive_tau
        config.views.degree_norm = deg_norm
        config.views.add_self_loops = self_loops
        config.train.epochs = 80
        config.train.seed = 42
        config.train.deterministic = True

        harness = ExperimentHarness(config)
        run_dir = output_root / name
        artifacts = harness.run(mode="thorn", output_dir=run_dir)

        # Read metrics from saved JSON
        from _metrics_helper import read_run_metrics
        m = read_run_metrics(artifacts.metrics_json)

        results[name] = {
            "test_pr_auc": m["test_pr_auc"],
            "test_roc_auc": m["test_roc_auc"],
            "degree_norm": deg_norm,
            "self_loops": self_loops,
            "adaptive_tau": adaptive_tau,
        }
        print(f"  PR-AUC: {results[name]['test_pr_auc']:.4f}")

    # Save
    report_path = report_dir / "stability_study.json"
    report_path.write_text(json.dumps(results, indent=2))

    # Summary
    print(f"\n{'='*70}")
    print("LOG-BARRIER STABILITY STUDY")
    print(f"{'='*70}")
    print(f"{'Config':<25} {'PR-AUC':>8} {'ROC-AUC':>9} {'tau':>5} {'norm':>10} {'loops':>6}")
    print("-" * 65)
    for name, r in results.items():
        print(f"{name:<25} {r['test_pr_auc']:>8.4f} {r['test_roc_auc']:>9.4f} "
              f"{'Y' if r['adaptive_tau'] else 'N':>5} {r['degree_norm']:>10} "
              f"{'Y' if r['self_loops'] else 'N':>6}")

    # Compute recovery ratios
    print(f"\nRecovery Analysis:")
    baseline_none = results.get("none_baseline", {}).get("test_pr_auc", 0)
    for norm_type in ["sym", "row", "selfloops"]:
        base_key = f"{norm_type}_baseline"
        tau_key = f"{norm_type}_tau"
        if base_key in results and tau_key in results:
            degradation = baseline_none - results[base_key]["test_pr_auc"]
            recovery = results[tau_key]["test_pr_auc"] - results[base_key]["test_pr_auc"]
            pct = (recovery / degradation * 100) if degradation > 0.001 else 0
            print(f"  {norm_type}: degradation={degradation:.4f}, "
                  f"recovery={recovery:.4f} ({pct:.1f}%)")

    print(f"\nResults saved to {report_path}")


if __name__ == "__main__":
    main()
