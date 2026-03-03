"""Runtime/memory profiling interfaces."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProfileStats:
    wall_seconds: float
    peak_memory_mb: float
