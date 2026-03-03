"""Generate diagnostic plots for THORN paper.

1. Isolation vs Interaction: Δ(post-softmax − bias) PR-AUC vs drift
2. Routing entropy and collapse vs drift
3. Observer ablation bar chart
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARTIFACTS = Path("/home/wliu23/google_drive/machine_learning/hw4_thorn/artifacts")
FIGURES = ARTIFACTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

DRIFTS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]


def load_drift_data():
    """Load PR-AUC and routing stats for both modes across drift levels."""
    bias_pr, post_pr = [], []
    bias_entropy, bias_collapse = [], []
    post_entropy, post_collapse = [], []

    for d in DRIFTS:
        bias_dir = ARTIFACTS / "runs" / f"adjmode_bias_drift_{d}"
        post_dir = ARTIFACTS / "runs" / f"adjmode_post_softmax_drift_{d}"

        bm = json.loads((bias_dir / "metrics.json").read_text())
        pm = json.loads((post_dir / "metrics.json").read_text())
        bias_pr.append(bm["test"]["pr_auc"])
        post_pr.append(pm["test"]["pr_auc"])

        bs = json.loads((bias_dir / "routing_stats.json").read_text())
        ps = json.loads((post_dir / "routing_stats.json").read_text())
        bias_entropy.append(bs["entropy"])
        bias_collapse.append(bs["collapse_score"])
        post_entropy.append(ps["entropy"])
        post_collapse.append(ps["collapse_score"])

    return {
        "bias_pr": bias_pr, "post_pr": post_pr,
        "bias_entropy": bias_entropy, "bias_collapse": bias_collapse,
        "post_entropy": post_entropy, "post_collapse": post_collapse,
    }


def plot_isolation_vs_interaction(data):
    """Fig 1: PR-AUC for both modes + delta, across drift."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

    ax1.plot(DRIFTS, data["bias_pr"], "o-", color="#2196F3", linewidth=2, markersize=6, label="Pre-softmax (bias)")
    ax1.plot(DRIFTS, data["post_pr"], "s-", color="#FF5722", linewidth=2, markersize=6, label="Post-softmax (isolated)")
    ax1.set_ylabel("PR-AUC", fontsize=12)
    ax1.legend(fontsize=10, loc="lower left")
    ax1.set_ylim(0.82, 1.0)
    ax1.grid(alpha=0.3)
    ax1.set_title("Isolation vs Interaction Tradeoff", fontsize=13, fontweight="bold")

    # Shade regions
    delta = np.array(data["post_pr"]) - np.array(data["bias_pr"])
    ax2.bar(DRIFTS, delta, width=0.12, color=["#FF5722" if d > 0 else "#2196F3" for d in delta], alpha=0.8)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_xlabel("Drift Strength", fontsize=12)
    ax2.set_ylabel("Δ (post − bias)", fontsize=11)
    ax2.grid(alpha=0.3)

    # Annotate winner regions
    ax2.text(0.25, 0.03, "Post-softmax\nwins", fontsize=8, ha="center", color="#FF5722")
    ax2.text(0.75, -0.01, "Bias\nwins", fontsize=8, ha="center", color="#2196F3", va="top")

    plt.tight_layout()
    fig.savefig(FIGURES / "isolation_vs_interaction.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(FIGURES / "isolation_vs_interaction.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved: {FIGURES / 'isolation_vs_interaction.pdf'}")


def plot_routing_diagnostics(data):
    """Fig 2: Routing entropy and collapse vs drift for both modes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(DRIFTS, data["bias_entropy"], "o-", color="#2196F3", linewidth=2, label="Bias mode")
    ax1.plot(DRIFTS, data["post_entropy"], "s-", color="#FF5722", linewidth=2, label="Post-softmax")
    ax1.set_xlabel("Drift Strength", fontsize=11)
    ax1.set_ylabel("Routing Entropy", fontsize=11)
    ax1.set_title("Head Routing Entropy", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    ax2.plot(DRIFTS, data["bias_collapse"], "o-", color="#2196F3", linewidth=2, label="Bias mode")
    ax2.plot(DRIFTS, data["post_collapse"], "s-", color="#FF5722", linewidth=2, label="Post-softmax")
    ax2.axhline(1/3, color="gray", linestyle="--", alpha=0.5, label="Uniform (1/3)")
    ax2.set_xlabel("Drift Strength", fontsize=11)
    ax2.set_ylabel("Max π (Collapse Score)", fontsize=11)
    ax2.set_title("Routing Collapse", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURES / "routing_diagnostics.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(FIGURES / "routing_diagnostics.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved: {FIGURES / 'routing_diagnostics.pdf'}")


def plot_observer_ablation():
    """Fig 3: Observer ablation horizontal bar chart."""
    ablation_path = ARTIFACTS / "reports_observer_ablation" / "observer_ablation.json"
    data = json.loads(ablation_path.read_text())

    # Sort by delta (most impactful first), exclude "full"
    items = [(k, v) for k, v in data.items() if k != "full"]
    items.sort(key=lambda x: x[1]["test_pr_auc"] - data["full"]["test_pr_auc"])

    names = [k.replace("no_", "−").replace("_", " ") for k, _ in items]
    deltas = [v["test_pr_auc"] - data["full"]["test_pr_auc"] for _, v in items]

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#d32f2f" if d < -0.05 else "#ff9800" if d < -0.01 else "#4caf50" for d in deltas]
    bars = ax.barh(range(len(names)), deltas, color=colors, height=0.6)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Δ PR-AUC (vs full observers)", fontsize=11)
    ax.set_title("Observer Feature Ablation", fontsize=12, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.grid(alpha=0.3, axis="x")

    # Add value labels
    for bar, delta in zip(bars, deltas):
        ax.text(bar.get_width() - 0.002, bar.get_y() + bar.get_height()/2,
                f"{delta:.3f}", ha="right", va="center", fontsize=8, fontweight="bold", color="white")

    plt.tight_layout()
    fig.savefig(FIGURES / "observer_ablation.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(FIGURES / "observer_ablation.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved: {FIGURES / 'observer_ablation.pdf'}")


def plot_stability():
    """Fig 4: Stability study — tau recovery under normalization."""
    stability_path = ARTIFACTS / "reports_stability" / "stability_study.json"
    data = json.loads(stability_path.read_text())

    configs = ["none", "sym", "row", "selfloops"]
    labels = ["None\n(default)", "Symmetric\nnorm", "Row\nnorm", "Self-\nloops"]

    baseline_vals = [data[f"{c}_baseline"]["test_pr_auc"] for c in configs]
    tau_vals = [data[f"{c}_tau"]["test_pr_auc"] for c in configs]

    x = np.arange(len(configs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    bars1 = ax.bar(x - width/2, baseline_vals, width, label="Without τ_h", color="#90CAF9")
    bars2 = ax.bar(x + width/2, tau_vals, width, label="With τ_h", color="#2196F3")

    ax.set_ylabel("PR-AUC", fontsize=11)
    ax.set_title("Adaptive Temperature Recovery", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(fontsize=10)
    ax.set_ylim(0.82, 0.96)
    ax.grid(alpha=0.3, axis="y")

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig.savefig(FIGURES / "stability_tau.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(FIGURES / "stability_tau.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved: {FIGURES / 'stability_tau.pdf'}")


if __name__ == "__main__":
    data = load_drift_data()
    plot_isolation_vs_interaction(data)
    plot_routing_diagnostics(data)
    plot_observer_ablation()
    plot_stability()
    print("\nAll diagnostic plots generated.")
