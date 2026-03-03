"""Observer orthogonality analysis (critique 2.4).

Analyzes the redundancy/orthogonality of geometric observer features
(LID, ORC proxy, LOF, kNN stats, view degrees) used by the MoE router.

Outputs:
  - Pairwise Pearson/Spearman correlation matrix
  - PCA effective rank and variance explained
  - Per-observer permutation importance for routing decisions
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from thorn.config import ExperimentConfig
from thorn.train.harness import ExperimentHarness
from thorn.utils.seed import set_global_seed as set_seed


def compute_observer_correlation(observer_features: np.ndarray, names: list[str]) -> dict:
    """Compute pairwise Pearson and Spearman correlations."""
    from scipy.stats import spearmanr

    n_feat = observer_features.shape[1]
    pearson = np.corrcoef(observer_features.T)

    spearman_corr, _ = spearmanr(observer_features)
    if spearman_corr.ndim == 0:
        spearman_corr = np.array([[1.0, spearman_corr], [spearman_corr, 1.0]])

    return {
        "pearson": pearson.tolist(),
        "spearman": spearman_corr.tolist(),
        "feature_names": names,
    }


def compute_pca_analysis(observer_features: np.ndarray) -> dict:
    """PCA analysis: variance explained, effective rank."""
    centered = observer_features - observer_features.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues, 0)

    total_var = eigenvalues.sum()
    if total_var < 1e-10:
        return {"effective_rank": 1, "variance_explained": [1.0]}

    normalized = eigenvalues / total_var
    cumulative = np.cumsum(normalized)

    # Effective rank (exponential of entropy of normalized eigenvalues)
    nonzero = normalized[normalized > 1e-10]
    entropy = -np.sum(nonzero * np.log(nonzero))
    effective_rank = np.exp(entropy)

    # How many PCs for 95% variance
    pcs_95 = int(np.searchsorted(cumulative, 0.95)) + 1

    return {
        "effective_rank": float(effective_rank),
        "pcs_for_95pct": pcs_95,
        "total_features": observer_features.shape[1],
        "variance_explained_top5": cumulative[:5].tolist(),
        "eigenvalues_top10": eigenvalues[:10].tolist(),
    }


def compute_router_attribution(
    model: torch.nn.Module,
    observer_features: torch.Tensor,
    n_permutations: int = 10,
) -> dict:
    """Permutation importance for each observer feature group on routing decisions."""
    model.eval()
    with torch.no_grad():
        base_out = model.router(observer_features)
        base_pi = base_out.pi.clone()

    n_nodes, n_feat = observer_features.shape

    # Define feature groups (approximate indices based on harness construction)
    # The exact mapping depends on the harness, but typical order:
    # [node_features(16) | diffusion_coords(8) | LID(3) | kNN_stats(2) |
    #  temporal(2) | LOF(1) | curvature(1) | view_degrees(3) | align_conf(3) | align_flat(9)]
    groups = {
        "node_features": list(range(0, 16)),
        "diffusion_coords": list(range(16, 24)),
        "LID": list(range(24, 27)),
        "kNN_stats": list(range(27, 29)),
        "temporal": list(range(29, 31)),
        "LOF": [31],
        "curvature": [32],
        "view_degrees": list(range(33, 36)),
    }

    # Clamp to actual feature dim
    groups = {k: [i for i in v if i < n_feat] for k, v in groups.items()}
    groups = {k: v for k, v in groups.items() if len(v) > 0}

    importance = {}
    for group_name, indices in groups.items():
        diffs = []
        for _ in range(n_permutations):
            perm_features = observer_features.clone()
            perm_idx = torch.randperm(n_nodes)
            perm_features[:, indices] = perm_features[perm_idx][:, indices]

            with torch.no_grad():
                perm_out = model.router(perm_features)
                perm_pi = perm_out.pi

            # L1 routing shift
            diff = (base_pi - perm_pi).abs().mean().item()
            diffs.append(diff)

        importance[group_name] = {
            "mean_routing_shift": float(np.mean(diffs)),
            "std_routing_shift": float(np.std(diffs)),
            "feature_indices": indices,
        }

    return importance


def main():
    set_seed(42)
    config = ExperimentConfig()

    print("=== Observer Orthogonality Analysis (Critique 2.4) ===\n")

    harness = ExperimentHarness(config)
    batch = harness._prepare_batch()
    context, build_stats, diff_coords, align_conf = harness._prepare_graph(batch)

    obs_features = context.observer_features.numpy()
    n_nodes, n_feat = obs_features.shape
    non_align_dim = int(build_stats.get("non_align_dim", n_feat))
    print(f"Observer features shape: {n_nodes} nodes x {n_feat} features")
    print(f"Non-alignment features: {non_align_dim}, Alignment features: {n_feat - non_align_dim}\n")

    # Analyze ONLY the geometric observer features (non-alignment part)
    # This is what the router actually differentiates on
    geo_features = obs_features[:, :non_align_dim]
    geo_n_feat = geo_features.shape[1]

    # Normalize features to zero-mean unit-variance for fair PCA
    from sklearn.preprocessing import StandardScaler
    geo_normalized = StandardScaler().fit_transform(geo_features)

    # 1. Correlation analysis (on geometric features only)
    print("--- Correlation Analysis (geometric observers only) ---")
    names = [f"f{i}" for i in range(geo_n_feat)]
    corr_results = compute_observer_correlation(geo_normalized, names)

    pearson = np.array(corr_results["pearson"])
    # Find highly correlated pairs (|r| > 0.8, excluding diagonal)
    high_corr = []
    for i in range(geo_n_feat):
        for j in range(i + 1, geo_n_feat):
            if not np.isnan(pearson[i, j]) and abs(pearson[i, j]) > 0.8:
                high_corr.append((i, j, pearson[i, j]))

    print(f"Feature pairs with |Pearson r| > 0.8: {len(high_corr)} / {geo_n_feat*(geo_n_feat-1)//2}")
    for i, j, r in high_corr[:10]:
        print(f"  f{i} <-> f{j}: r = {r:.3f}")

    # 2. PCA analysis (on normalized geometric features)
    print("\n--- PCA Analysis (normalized geometric observers) ---")
    pca_results = compute_pca_analysis(geo_normalized)
    print(f"Effective rank: {pca_results['effective_rank']:.2f} / {geo_n_feat}")
    print(f"PCs for 95% variance: {pca_results['pcs_for_95pct']}")
    print(f"Cumulative variance (top 5 PCs): {[f'{v:.3f}' for v in pca_results['variance_explained_top5']]}")

    # Also analyze full features for comparison
    print("\n--- PCA Analysis (ALL features, normalized) ---")
    all_normalized = StandardScaler().fit_transform(obs_features)
    pca_all = compute_pca_analysis(all_normalized)
    print(f"Effective rank: {pca_all['effective_rank']:.2f} / {n_feat}")
    print(f"PCs for 95% variance: {pca_all['pcs_for_95pct']}")

    # 3. Router attribution (if model available)
    print("\n--- Router Attribution ---")
    from thorn.models.thorn import THORNModel
    model = THORNModel(config)
    obs_tensor = torch.tensor(obs_features, dtype=torch.float32)
    # Warm up lazy modules
    x_dummy = torch.randn(n_nodes, config.data.num_features)
    model.input_proj(x_dummy)
    model.router(obs_tensor)

    attrib = compute_router_attribution(model, obs_tensor)
    print(f"{'Group':<20} {'Routing Shift':>15} {'Std':>10}")
    print("-" * 50)
    sorted_groups = sorted(attrib.items(), key=lambda x: x[1]["mean_routing_shift"], reverse=True)
    for name, vals in sorted_groups:
        print(f"{name:<20} {vals['mean_routing_shift']:>15.4f} {vals['std_routing_shift']:>10.4f}")

    # Save results
    output_dir = Path("artifacts/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "correlation": {"high_corr_pairs": len(high_corr), "total_pairs": n_feat * (n_feat - 1) // 2},
        "pca": pca_results,
        "router_attribution": {k: {"mean": v["mean_routing_shift"], "std": v["std_routing_shift"]} for k, v in attrib.items()},
    }
    (output_dir / "observer_orthogonality.json").write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {output_dir / 'observer_orthogonality.json'}")


if __name__ == "__main__":
    main()
