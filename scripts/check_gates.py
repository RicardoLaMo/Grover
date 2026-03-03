#!/usr/bin/env python
"""Evaluate G0-G7 status from produced artifacts."""

from __future__ import annotations

import glob
import json
from pathlib import Path


def latest(pattern: str) -> str:
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No matches for pattern: {pattern}")
    return matches[-1]


def main() -> None:
    report = {"gates": {}, "overall_pass": False}

    # G0
    g0 = Path("artifacts/env/pip_freeze.txt").exists()
    report["gates"]["G0"] = {"pass": g0, "evidence": ["artifacts/env/pip_freeze.txt"]}

    # G1 (tests exist + debug asserts)
    g1 = (
        Path("tests/test_views_builders.py").exists()
        and Path("tests/test_router_attention.py").exists()
        and Path("thorn/debug.py").exists()
    )
    report["gates"]["G1"] = {
        "pass": g1,
        "evidence": [
            "tests/test_views_builders.py",
            "tests/test_router_attention.py",
            "thorn/debug.py",
        ],
    }

    # G2
    smoke_metrics = latest("artifacts/runs/*thorn_smoke_thorn*/metrics.json")
    smoke_cfg = Path(smoke_metrics).with_name("config_snapshot.json")
    smoke_ckpt = Path(smoke_metrics).with_name("checkpoint.pt")
    g2 = Path(smoke_metrics).exists() and smoke_cfg.exists() and smoke_ckpt.exists()
    report["gates"]["G2"] = {
        "pass": g2,
        "evidence": [smoke_metrics, str(smoke_cfg), str(smoke_ckpt)],
    }

    # G3
    overfit_metrics = latest("artifacts/runs/*thorn_overfit_thorn*/metrics.json")
    om = json.loads(Path(overfit_metrics).read_text(encoding="utf-8"))
    g3 = bool(om.get("overfit_success", False))
    report["gates"]["G3"] = {
        "pass": g3,
        "train_accuracy": om["train"]["accuracy"],
        "evidence": [overfit_metrics],
    }

    # G4
    result_table = Path("artifacts/reports/results_table.csv")
    align_effect = Path("artifacts/reports/alignment_effect.json")
    g4 = result_table.exists() and align_effect.exists()
    if g4:
        ae = json.loads(align_effect.read_text(encoding="utf-8"))
        g4 = g4 and abs(float(ae["delta_routing_shift_l1"])) > 0.0
    report["gates"]["G4"] = {
        "pass": g4,
        "evidence": [str(result_table), str(align_effect)],
    }

    # G5
    profile = latest("artifacts/runs/*thorn_scalable_thorn*/profile.json")
    p = json.loads(Path(profile).read_text(encoding="utf-8"))
    g5 = (
        "graph_build" in p
        and "router_sec_approx" in p
        and "attention_sec_approx" in p
        and "peak_memory_mb" in p
        and p.get("small_mode", False)
        and p.get("scalable_mode", False)
    )
    report["gates"]["G5"] = {"pass": g5, "evidence": [profile]}

    # G6
    thorn_metrics = latest("artifacts/runs/*baseline_suite_thorn*/metrics.json")
    tm = json.loads(Path(thorn_metrics).read_text(encoding="utf-8"))
    shift = float(tm["drift"]["routing_shift_l1"])
    collapse = float(tm["drift"]["collapse_score"])
    g6 = shift > 0.0 and collapse < 0.95
    report["gates"]["G6"] = {
        "pass": g6,
        "routing_shift_l1": shift,
        "collapse_score": collapse,
        "evidence": [thorn_metrics],
    }

    # G7
    repro = Path("artifacts/reports/reproducibility.json")
    if repro.exists():
        rep = json.loads(repro.read_text(encoding="utf-8"))
        g7 = bool(rep.get("passed", False))
    else:
        g7 = False
    report["gates"]["G7"] = {"pass": g7, "evidence": [str(repro)]}

    report["overall_pass"] = all(v["pass"] for v in report["gates"].values())

    out = Path("artifacts/reports/gate_check.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"report={out}")
    print(f"overall_pass={report['overall_pass']}")


if __name__ == "__main__":
    main()
