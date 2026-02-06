"""Mood projection module for Mood Axis."""

import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(__file__).rsplit("/src/", 1)[0])
from config.settings import MOOD_AXES, AXIS_LABELS, AXES_FILE
from src.calibration.axis_computer import load_axis_vectors


@dataclass
class MoodReading:
    """A single mood reading across all axes."""
    values: Dict[str, float]  # axis_name -> value in [-1, 1]
    descriptions: Dict[str, str]  # axis_name -> text description

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "values": self.values,
            "descriptions": self.descriptions,
        }


class MoodProjector:
    """Projects hidden states onto mood axes."""

    def __init__(
        self,
        axis_vectors: Optional[Dict[str, np.ndarray]] = None,
        axes_file: Optional[Path] = None,
        normalization_scales: Optional[Dict[str, float]] = None,
    ):
        """Initialize the projector.

        Args:
            axis_vectors: Pre-loaded axis vectors
            axes_file: Path to load axis vectors from
            normalization_scales: Per-axis scale factors for normalization
        """
        axes_path = axes_file or AXES_FILE
        if axis_vectors is not None:
            self.axis_vectors = axis_vectors
        else:
            self.axis_vectors = load_axis_vectors(axes_path)

        # Load or compute normalization scales
        if normalization_scales is not None:
            self.normalization_scales = normalization_scales
        else:
            self.normalization_scales = self._load_or_compute_scales(axes_path)

        self._calibration_stats: Dict[str, Dict] = {}

    def _load_or_compute_scales(self, axes_path: Path) -> Dict[str, float]:
        """Load normalization scales from calibration data or use defaults."""
        scales = {}
        try:
            data = np.load(axes_path)
            for axis in MOOD_AXES:
                scale_key = f"{axis}_scale"
                if scale_key in data:
                    scales[axis] = float(data[scale_key])
                else:
                    # Default: estimate from axis vector norm and typical hidden state norm
                    # Typical hidden state norm ~20, axis is normalized, so projection ~20
                    scales[axis] = 2.0  # Map [-2, +2] to [-1, +1]
        except Exception:
            for axis in MOOD_AXES:
                scales[axis] = 2.0
        return scales

    def project(
        self,
        hidden_state: np.ndarray,
        normalize: bool = True,
        clip: bool = True,
    ) -> MoodReading:
        """Project a hidden state onto all mood axes.

        Args:
            hidden_state: The hidden state vector to project
            normalize: Whether to normalize to [-1, 1] range
            clip: Whether to clip values to [-1, 1]

        Returns:
            MoodReading with values and descriptions
        """
        values = {}
        descriptions = {}

        for axis in MOOD_AXES:
            if axis not in self.axis_vectors:
                continue

            axis_vector = self.axis_vectors[axis]

            # Dot product gives projection
            raw_value = np.dot(hidden_state, axis_vector)

            # Apply per-axis normalization
            if normalize:
                scale = self.normalization_scales.get(axis, 2.0)
                raw_value = raw_value / scale

            # Clip to valid range
            if clip:
                raw_value = np.clip(raw_value, -1.0, 1.0)

            values[axis] = float(raw_value)
            descriptions[axis] = self._value_to_description(axis, raw_value)

        return MoodReading(values=values, descriptions=descriptions)

    def _value_to_description(self, axis: str, value: float) -> str:
        """Convert a numeric value to text description.

        Args:
            axis: The axis name
            value: Value in [-1, 1]

        Returns:
            Text description
        """
        positive_label, negative_label = AXIS_LABELS.get(axis, ("Positive", "Negative"))

        abs_value = abs(value)

        if abs_value < 0.1:
            intensity = "neutral"
        elif abs_value < 0.3:
            intensity = "slightly"
        elif abs_value < 0.5:
            intensity = "somewhat"
        elif abs_value < 0.7:
            intensity = "moderately"
        elif abs_value < 0.9:
            intensity = "quite"
        else:
            intensity = "very"

        if abs_value < 0.1:
            return f"Neutral on {positive_label.lower()}/{negative_label.lower()}"
        elif value > 0:
            return f"{intensity.capitalize()} {positive_label.lower()}"
        else:
            return f"{intensity.capitalize()} {negative_label.lower()}"

    def get_summary(self, reading: MoodReading) -> str:
        """Get a summary text of a mood reading.

        Args:
            reading: The mood reading

        Returns:
            Summary text
        """
        parts = []
        for axis in MOOD_AXES:
            if axis in reading.descriptions:
                parts.append(reading.descriptions[axis])
        return ", ".join(parts)

    def calibrate_normalization(
        self,
        sample_states: List[np.ndarray],
        target_range: float = 0.8,
    ) -> None:
        """Calibrate normalization scale from sample data.

        Sets the normalization scale so that typical values fall
        within the target range.

        Args:
            sample_states: List of hidden states to calibrate from
            target_range: Target max absolute value for typical samples
        """
        all_projections = []

        for state in sample_states:
            for axis, vector in self.axis_vectors.items():
                projection = np.dot(state, vector)
                all_projections.append(abs(projection))

        if all_projections:
            # Use 90th percentile as the normalization scale
            percentile_90 = np.percentile(all_projections, 90)
            self.normalization_scale = percentile_90 / target_range
            print(f"Calibrated normalization scale: {self.normalization_scale:.4f}")


def get_mood_for_response(
    hidden_state: np.ndarray,
    projector: Optional[MoodProjector] = None,
) -> Tuple[MoodReading, MoodProjector]:
    """Convenience function to get mood reading for a response.

    Args:
        hidden_state: The hidden state from generation
        projector: Optional pre-initialized projector

    Returns:
        Tuple of (MoodReading, MoodProjector used)
    """
    if projector is None:
        projector = MoodProjector()

    reading = projector.project(hidden_state)
    return reading, projector
