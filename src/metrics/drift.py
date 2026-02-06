"""Drift metric for mood axis analysis.

Drift measures how a model's mood values change over a long conversation.
It captures the tendency of models to "drift" in a certain direction.

Formula: drift[axis] = slope(values vs turn_number)

Positive drift = mood increases over time
Negative drift = mood decreases over time
Near-zero drift = stable mood over time
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import stats


@dataclass
class AxisDrift:
    """Drift analysis for a single axis."""

    # Slope of linear regression (drift rate per turn)
    slope: float

    # R-squared (how well drift explains variance)
    r_squared: float

    # P-value (statistical significance)
    p_value: float

    # Intercept (starting value)
    intercept: float

    # Is drift statistically significant? (p < 0.05)
    is_significant: bool

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
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

    # Per-axis drift analysis
    per_axis: Dict[str, AxisDrift]

    # Number of turns analyzed
    num_turns: int

    # Raw values used for computation (optional)
    raw_values: Optional[Dict[str, List[float]]] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "per_axis": {
                axis: drift.to_dict()
                for axis, drift in self.per_axis.items()
            },
            "num_turns": self.num_turns,
        }

    def get_significant_drifts(self) -> Dict[str, AxisDrift]:
        """Get only axes with statistically significant drift."""
        return {
            axis: drift
            for axis, drift in self.per_axis.items()
            if drift.is_significant
        }

    def get_strongest_drift(self) -> Optional[Tuple[str, AxisDrift]]:
        """Get the axis with the strongest (most significant) drift."""
        if not self.per_axis:
            return None

        # Sort by absolute slope, considering only significant drifts
        significant = self.get_significant_drifts()
        if not significant:
            return None

        strongest = max(
            significant.items(),
            key=lambda x: abs(x[1].slope)
        )
        return strongest


def compute_drift(
    turn_values: List[Dict[str, float]],
    significance_level: float = 0.05,
    include_raw: bool = False,
) -> DriftResult:
    """Compute drift from a sequence of mood readings.

    Uses linear regression to measure how each axis changes over turns.

    Args:
        turn_values: List of mood value dictionaries, one per turn.
                    Each dict maps axis name to value
        significance_level: P-value threshold for significance
        include_raw: Whether to include raw values in result

    Returns:
        DriftResult with per-axis drift analysis

    Example:
        >>> # Simulate warming trend over 10 turns
        >>> turns = [{"warm_cold": 0.1 + i * 0.02} for i in range(10)]
        >>> result = compute_drift(turns)
        >>> print(result.per_axis["warm_cold"].slope)  # ~0.02
    """
    if len(turn_values) < 3:
        # Need at least 3 points for meaningful regression
        return DriftResult(
            per_axis={},
            num_turns=len(turn_values),
        )

    # Collect values per axis
    axes = list(turn_values[0].keys())
    values_per_axis: Dict[str, List[float]] = {axis: [] for axis in axes}

    for turn in turn_values:
        for axis in axes:
            if axis in turn:
                values_per_axis[axis].append(turn[axis])

    # Compute linear regression for each axis
    per_axis_drift: Dict[str, AxisDrift] = {}
    turn_numbers = np.arange(len(turn_values))

    for axis, values in values_per_axis.items():
        if len(values) < 3:
            continue

        values_array = np.array(values)

        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            turn_numbers[:len(values)],
            values_array
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
    """Compute drift for multiple models on the same scenario.

    Args:
        model_results: Dict mapping model name to list of turn values

    Returns:
        Dict mapping model name to DriftResult
    """
    return {
        model: compute_drift(turns)
        for model, turns in model_results.items()
    }


def describe_drift(drift: AxisDrift, axis_name: str) -> str:
    """Generate human-readable description of drift.

    Args:
        drift: AxisDrift to describe
        axis_name: Name of the axis (e.g., "warm_cold")

    Returns:
        Human-readable description
    """
    if not drift.is_significant:
        return f"{axis_name}: No significant drift (stable)"

    direction = "increasing" if drift.slope > 0 else "decreasing"
    strength = abs(drift.slope)

    if strength < 0.005:
        intensity = "very slowly"
    elif strength < 0.01:
        intensity = "slowly"
    elif strength < 0.02:
        intensity = "moderately"
    else:
        intensity = "rapidly"

    # Parse axis name for readable description
    parts = axis_name.split("_")
    if len(parts) == 2:
        positive, negative = parts
        target = positive if drift.slope > 0 else negative
    else:
        target = direction

    return (
        f"{axis_name}: {intensity} {direction} "
        f"(slope={drift.slope:.4f}, R²={drift.r_squared:.2f})"
    )
