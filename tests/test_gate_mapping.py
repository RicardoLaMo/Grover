"""Stage 1 tests for planner documentation and gate mapping presence."""

from __future__ import annotations

from pathlib import Path


REQUIRED_DOCS = [
    "docs/GOALS.md",
    "docs/BLUEPRINT.md",
    "docs/gates.md",
    "docs/STATUS.md",
    "docs/CHECKPOINT.md",
    "docs/trace.md",
    "docs/math.md",
]


def test_required_planner_docs_exist() -> None:
    for rel_path in REQUIRED_DOCS:
        assert Path(rel_path).exists(), f"Missing required doc: {rel_path}"


def test_all_gates_are_listed() -> None:
    text = Path("docs/gates.md").read_text(encoding="utf-8")
    for gate in ["G0", "G1", "G2", "G3", "G4", "G5", "G6", "G7"]:
        assert gate in text, f"Gate {gate} missing from docs/gates.md"


def test_stage1_verify_script_exists() -> None:
    assert Path("scripts/stage1_verify.sh").exists()
