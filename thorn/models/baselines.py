"""Baseline model registry definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict


@dataclass
class BaselineRegistry:
    """Registry for baseline constructors."""

    constructors: Dict[str, Callable]

    def build(self, name: str):
        if name not in self.constructors:
            raise KeyError(f"Unknown baseline: {name}")
        return self.constructors[name]()
