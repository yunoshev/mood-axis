"""Volatility metric for mood axis analysis.

Volatility measures how much a model's mood values fluctuate
within a single scenario or conversation.
Formula: volatility[axis] = std(values_per_turn)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np


@dataclass
class VolatilityResult:
    """Result of volatility computation."""
    per_axis: Dict[str, float]
    overall: float
    num_turns: int
    raw_values: Optional[Dict[str, List[float]]] = None

    def to_dict(self) -> dict:
        return {
            "per_axis": self.per_axis,
            "overall": self.overall,
            "num_turns": self.num_turns,
        }


def compute_volatility(
    turn_values: List[Dict[str, float]],
    include_raw: bool = False,
) -> VolatilityResult:
    """Compute volatility from a sequence of mood readings."""
    if not turn_values:
        return VolatilityResult(per_axis={}, overall=0.0, num_turns=0)

    axes = list(turn_values[0].keys())
    values_per_axis: Dict[str, List[float]] = {axis: [] for axis in axes}

    for turn in turn_values:
        for axis in axes:
            if axis in turn:
                values_per_axis[axis].append(turn[axis])

    per_axis_volatility: Dict[str, float] = {}
    for axis, values in values_per_axis.items():
        if len(values) >= 2:
            per_axis_volatility[axis] = float(np.std(values))
        else:
            per_axis_volatility[axis] = 0.0

    overall = float(np.mean(list(per_axis_volatility.values()))) if per_axis_volatility else 0.0

    return VolatilityResult(
        per_axis=per_axis_volatility,
        overall=overall,
        num_turns=len(turn_values),
        raw_values=values_per_axis if include_raw else None,
    )
