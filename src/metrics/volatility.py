"""Volatility metric for mood axis analysis.

Volatility measures how much a model's mood values fluctuate
within a single scenario or conversation.

Formula: volatility[axis] = std(values_per_turn)

High volatility = "nervous" model that changes mood frequently
Low volatility = "stable" model with consistent mood
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np


@dataclass
class VolatilityResult:
    """Result of volatility computation."""

    # Per-axis volatility (std of values across turns)
    per_axis: Dict[str, float]

    # Overall volatility (mean of per-axis volatilities)
    overall: float

    # Number of turns analyzed
    num_turns: int

    # Raw values used for computation
    raw_values: Optional[Dict[str, List[float]]] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "per_axis": self.per_axis,
            "overall": self.overall,
            "num_turns": self.num_turns,
        }


def compute_volatility(
    turn_values: List[Dict[str, float]],
    include_raw: bool = False,
) -> VolatilityResult:
    """Compute volatility from a sequence of mood readings.

    Args:
        turn_values: List of mood value dictionaries, one per turn.
                    Each dict maps axis name to value (e.g., {"warm_cold": 0.3, ...})
        include_raw: Whether to include raw values in result

    Returns:
        VolatilityResult with per-axis and overall volatility

    Example:
        >>> turns = [
        ...     {"warm_cold": 0.1, "patient_irritated": 0.2},
        ...     {"warm_cold": 0.3, "patient_irritated": 0.1},
        ...     {"warm_cold": -0.1, "patient_irritated": 0.15},
        ... ]
        >>> result = compute_volatility(turns)
        >>> print(result.per_axis["warm_cold"])  # ~0.16
    """
    if not turn_values:
        return VolatilityResult(
            per_axis={},
            overall=0.0,
            num_turns=0,
        )

    # Collect values per axis
    axes = list(turn_values[0].keys())
    values_per_axis: Dict[str, List[float]] = {axis: [] for axis in axes}

    for turn in turn_values:
        for axis in axes:
            if axis in turn:
                values_per_axis[axis].append(turn[axis])

    # Compute std for each axis
    per_axis_volatility: Dict[str, float] = {}
    for axis, values in values_per_axis.items():
        if len(values) >= 2:
            per_axis_volatility[axis] = float(np.std(values))
        else:
            per_axis_volatility[axis] = 0.0

    # Overall volatility is mean of per-axis volatilities
    overall = float(np.mean(list(per_axis_volatility.values()))) if per_axis_volatility else 0.0

    return VolatilityResult(
        per_axis=per_axis_volatility,
        overall=overall,
        num_turns=len(turn_values),
        raw_values=values_per_axis if include_raw else None,
    )


def compute_volatility_comparison(
    model_results: Dict[str, List[Dict[str, float]]],
) -> Dict[str, VolatilityResult]:
    """Compute volatility for multiple models on the same scenario.

    Args:
        model_results: Dict mapping model name to list of turn values

    Returns:
        Dict mapping model name to VolatilityResult
    """
    return {
        model: compute_volatility(turns)
        for model, turns in model_results.items()
    }
