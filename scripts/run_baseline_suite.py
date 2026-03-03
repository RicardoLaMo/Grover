#!/usr/bin/env python
"""Run baseline suite + THORN and write comparison table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from thorn.config import ExperimentConfig
from thorn.train.harness import ExperimentHarness


MODES = [
    "gat",
    "single_view_knn",
    "single_view_time",
    "single_view_diffusion",
    "uniform_multi",
    "router_no_alignment",
    "thorn",
    "thorn_no_fgw",
    "thorn_no_depth",
    "thorn_no_moe",
    "thorn_single_scale",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run THORN baseline suite")
    p.add_argument("--output-root", type=str, default="artifacts/runs")
    p.add_argument("--report-dir", type=str, default="artifacts/reports")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--small-mode", action="store_true")
    p.add_argument("--scalable-mode", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | str]] = []

    for mode in MODES:
        cfg = ExperimentConfig()
        cfg.name = "baseline_suite"
        cfg.mode = mode  # type: ignore[assignment]
        cfg.output_root = args.output_root
        cfg.train.seed = args.seed
        cfg.train.epochs = args.epochs
        cfg.train.overfit_tiny = args.small_mode
        cfg.views.scalable_mode = args.scalable_mode
        if mode == "router_no_alignment":
            cfg.alignment.enabled = False

        harness = ExperimentHarness(cfg)
        artifacts = harness.run(mode=mode)
        metrics = json.loads(Path(artifacts.metrics_json).read_text(encoding="utf-8"))
        routing = json.loads(Path(artifacts.routing_json).read_text(encoding="utf-8"))

        rows.append(
            {
                "mode": mode,
                "run_dir": str(artifacts.run_dir),
                "val_pr_auc": metrics["val"]["pr_auc"],
                "test_pr_auc": metrics["test"]["pr_auc"],
                "test_roc_auc": metrics["test"]["roc_auc"],
                "test_p_at_k": metrics["test"]["precision_at_k"],
                "routing_shift_l1": metrics["drift"]["routing_shift_l1"],
                "collapse_score": routing["collapse_score"],
            }
        )

    csv_path = report_dir / "results_table.csv"
    md_path = report_dir / "results_table.md"
    json_path = report_dir / "results_table.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with md_path.open("w", encoding="utf-8") as f:
        headers = list(rows[0].keys())
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(str(r[h]) for h in headers) + " |\n")

    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    # Alignment effect summary required by G4.
    thorn_row = next(r for r in rows if r["mode"] == "thorn")
    no_align_row = next(r for r in rows if r["mode"] == "router_no_alignment")
    effect = {
        "delta_test_pr_auc": float(thorn_row["test_pr_auc"]) - float(no_align_row["test_pr_auc"]),
        "delta_routing_shift_l1": float(thorn_row["routing_shift_l1"]) - float(no_align_row["routing_shift_l1"]),
    }
    (report_dir / "alignment_effect.json").write_text(json.dumps(effect, indent=2), encoding="utf-8")

    print(f"results_csv={csv_path}")
    print(f"results_md={md_path}")
    print(f"results_json={json_path}")
    print(f"alignment_effect={report_dir / 'alignment_effect.json'}")


if __name__ == "__main__":
    main()
