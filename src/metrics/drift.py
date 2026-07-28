"""Drift metric for mood axis analysis.

Drift measures how a model's mood values change over a long conversation.
Formula: drift[axis] = slope(values vs turn_number)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import stats


@dataclass
class AxisDrift:
    """Drift analysis for a single axis."""
    slope: float
    r_squared: float
    p_value: float
    intercept: float
    is_significant: bool

    def to_dict(self) -> dict:
        return {
            "slope": self.slope,
            "r_squared": self.r_squared,
            "p_value": self.p_value,
            "intercept": self.intercept,
            "is_significant": self.is_significant,
        }


@dataclass
class DriftResult:
    """Result of drift computation for all axes."""
    per_axis: Dict[str, AxisDrift]
    num_turns: int
    raw_values: Optional[Dict[str, List[float]]] = None

    def to_dict(self) -> dict:
        return {
            "per_axis": {axis: drift.to_dict() for axis, drift in self.per_axis.items()},
            "num_turns": self.num_turns,
        }

    def get_significant_drifts(self) -> Dict[str, AxisDrift]:
        return {axis: drift for axis, drift in self.per_axis.items() if drift.is_significant}

    def get_strongest_drift(self) -> Optional[Tuple[str, AxisDrift]]:
        significant = self.get_significant_drifts()
        if not significant:
            return None
        return max(significant.items(), key=lambda x: abs(x[1].slope))


def compute_drift(
    turn_values: List[Dict[str, float]],
    significance_level: float = 0.05,
    include_raw: bool = False,
) -> DriftResult:
    """Compute drift from a sequence of mood readings."""
    if len(turn_values) < 3:
        return DriftResult(per_axis={}, num_turns=len(turn_values))

    axes = list(turn_values[0].keys())
    values_per_axis: Dict[str, List[float]] = {axis: [] for axis in axes}

    for turn in turn_values:
        for axis in axes:
            if axis in turn:
                values_per_axis[axis].append(turn[axis])

    per_axis_drift: Dict[str, AxisDrift] = {}
    turn_numbers = np.arange(len(turn_values))

    for axis, values in values_per_axis.items():
        if len(values) < 3:
            continue

        values_array = np.array(values)
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            turn_numbers[:len(values)], values_array
        )

        per_axis_drift[axis] = AxisDrift(
            slope=float(slope),
            r_squared=float(r_value ** 2),
            p_value=float(p_value),
            intercept=float(intercept),
            is_significant=p_value < significance_level,
        )

    return DriftResult(
        per_axis=per_axis_drift,
        num_turns=len(turn_values),
        raw_values=values_per_axis if include_raw else None,
    )


def compute_drift_comparison(
    model_results: Dict[str, List[Dict[str, float]]],
) -> Dict[str, DriftResult]:
    """Compute drift for multiple models on the same scenario."""
    return {model: compute_drift(turns) for model, turns in model_results.items()}
