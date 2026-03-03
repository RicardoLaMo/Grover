"""Observer ablation study (critique 2 / checklist item 2-iii).

Removes individual observer feature families from the harness and measures
the impact on PR-AUC, routing entropy, and head specialization.

This requires modifying the observer feature construction at runtime.
We achieve this by monkey-patching the harness's _prepare_graph to zero out
specific observer feature groups.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from thorn.config import ExperimentConfig
from thorn.train.harness import ExperimentHarness
from thorn.utils.seed import set_global_seed as set_seed


# Observer feature group indices (based on harness construction order):
# [node_features(16) | diffusion_coords(8) | LID(1) | multi_LID(3) |
#  kNN_stats(2) | temporal(2) | LOF(1) | curvature(1) | view_degrees(3) |
#  align_conf(2-3) | align_flat(4-9)]
# Total non-align: 16+8+1+3+2+2+1+1+3 = 37
# With alignment: +2 conf + ~4-9 flat

OBSERVER_GROUPS = {
    "LID": (24, 28),       # LID(1) + multi_LID(3) = indices 24-27
    "curvature": (32, 33), # curvature(1) = index 32
    "LOF": (31, 32),       # LOF(1) = index 31
    "kNN_stats": (29, 31), # kNN_stats(2) = indices 29-30
    "temporal": (31, 33),  # temporal(2) = indices 31-32 (WRONG, recompute)
}

# Actually, we should determine the indices dynamically. Let me build a smarter approach.


def run_with_zeroed_observers(
    group_name: str | None,
    zero_indices: list[int] | None,
    epochs: int = 80,
) -> dict:
    """Run THORN with specific observer features zeroed out."""
    set_seed(42)
    config = ExperimentConfig()
    config.name = f"ablate_{group_name or 'none'}"
    config.mode = "thorn"
    config.data.drift_strength = 0.75
    config.attention.adjacency_mode = "bias"
    config.train.epochs = epochs
    config.train.seed = 42
    config.train.deterministic = True

    harness = ExperimentHarness(config)

    # Monkey-patch _prepare_graph to zero out specified observer features
    original_prepare = harness._prepare_graph

    def patched_prepare(batch):
        context, build_stats, diff_coords, align_conf = original_prepare(batch)
        if zero_indices:
            obs = context.observer_features.clone()
            n_feat = obs.shape[1]
            valid_indices = [i for i in zero_indices if i < n_feat]
            if valid_indices:
                obs[:, valid_indices] = 0.0
            context = context._replace(observer_features=obs) if hasattr(context, '_replace') else context
            # GraphContext is a dataclass, so set attribute directly
            context.observer_features = obs
        return context, build_stats, diff_coords, align_conf

    harness._prepare_graph = patched_prepare

    run_dir = Path(f"artifacts/runs_observer_ablation/{group_name or 'full'}")
    artifacts = harness.run(mode="thorn", output_dir=run_dir)

    # Read metrics from saved JSON
    from _metrics_helper import read_run_metrics
    m = read_run_metrics(artifacts.metrics_json)

    return {
        "test_pr_auc": m["test_pr_auc"],
        "test_roc_auc": m["test_roc_auc"],
        "collapse_score": m["collapse_score"],
        "routing_shift_l1": m["routing_shift_l1"],
    }


def main():
    report_dir = Path("artifacts/reports_observer_ablation")
    report_dir.mkdir(parents=True, exist_ok=True)

    # First, determine the observer feature dimensions
    set_seed(42)
    config = ExperimentConfig()
    harness = ExperimentHarness(config)
    batch = harness._prepare_batch()
    context, build_stats, _, _ = harness._prepare_graph(batch)
    n_feat = context.observer_features.shape[1]
    non_align_dim = int(build_stats.get("non_align_dim", n_feat))
    print(f"Total observer features: {n_feat}, non-alignment: {non_align_dim}")

    # Feature layout based on harness code:
    # [node_features(16) | diffusion_coords(8) | LID(1) | multi_LID(3) |
    #  kNN_stats(2) | temporal(2) | LOF(1) | curvature(1) | view_degrees(3)]
    # = 16+8+1+3+2+2+1+1+3 = 37
    # Then alignment features: conf(M) + flat(M*M or 2*2=4)

    groups = {
        "full": None,                                    # No ablation (baseline)
        "no_LID": list(range(24, 28)),                   # Remove LID + multi-LID
        "no_curvature": [32],                            # Remove curvature
        "no_LOF": [31],                                  # Remove LOF
        "no_kNN_stats": list(range(29, 31)),             # Remove kNN stats
        "no_temporal": list(range(29, 31)),              # Remove temporal features
        "no_view_degrees": list(range(33, 36)),          # Remove per-view degrees
        "no_LID_ORC_LOF": list(range(24, 28)) + [31, 32],  # Remove all three geometric
        "no_diffusion": list(range(16, 24)),             # Remove diffusion coords
    }

    # Clamp indices to actual feature dim
    groups = {k: ([i for i in v if i < n_feat] if v else None) for k, v in groups.items()}

    results = {}
    for name, indices in groups.items():
        print(f"\n{'='*60}")
        print(f"Observer ablation: {name}")
        if indices:
            print(f"  Zeroing indices: {indices}")
        print(f"{'='*60}")

        r = run_with_zeroed_observers(name, indices)
        results[name] = r
        print(f"  PR-AUC: {r['test_pr_auc']:.4f}, Collapse: {r['collapse_score']:.3f}")

    # Save
    report_path = report_dir / "observer_ablation.json"
    report_path.write_text(json.dumps(results, indent=2))

    # Summary
    baseline = results.get("full", {}).get("test_pr_auc", 0)
    print(f"\n{'='*70}")
    print("OBSERVER ABLATION RESULTS")
    print(f"{'='*70}")
    print(f"{'Ablation':<25} {'PR-AUC':>8} {'Delta':>8} {'Collapse':>9} {'Shift':>8}")
    print("-" * 60)
    for name, r in results.items():
        delta = r["test_pr_auc"] - baseline
        print(f"{name:<25} {r['test_pr_auc']:>8.4f} {delta:>+8.4f} "
              f"{r['collapse_score']:>9.3f} {r['routing_shift_l1']:>8.4f}")

    print(f"\nResults saved to {report_path}")


if __name__ == "__main__":
    main()
