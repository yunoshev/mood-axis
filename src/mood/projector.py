"""Mood projection module for Mood Axis."""

import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from pathlib import Path

from config.settings import MOOD_AXES, AXIS_LABELS, AXES_FILE
from src.calibration.axis_computer import load_axis_vectors


@dataclass
class MoodReading:
    """A single mood reading across all axes."""
    values: Dict[str, float]  # axis_name -> value in [-1, 1]
    descriptions: Dict[str, str]  # axis_name -> text description

    def to_dict(self) -> Dict:
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
        normalization_centers: Optional[Dict[str, float]] = None,
    ):
        axes_path = axes_file or AXES_FILE
        if axis_vectors is not None:
            self.axis_vectors = axis_vectors
        else:
            self.axis_vectors = load_axis_vectors(axes_path)

        if normalization_scales is not None:
            self.normalization_scales = normalization_scales
        else:
            self.normalization_scales = self._load_params(axes_path, "scale")

        if normalization_centers is not None:
            self.normalization_centers = normalization_centers
        else:
            self.normalization_centers = self._load_params(axes_path, "center")

    def _load_params(self, axes_path: Path, suffix: str) -> Dict[str, float]:
        """Load per-axis parameters (scale or center) from NPZ.

        Raises if a key is missing — silently defaulting (scale=2.0/center=0.0)
        produces wrong, hard-to-diagnose projections.
        """
        params = {}
        data = np.load(axes_path)
        for axis in MOOD_AXES:
            if axis not in self.axis_vectors:
                continue  # axis not calibrated in this NPZ
            key = f"{axis}_{suffix}"
            if key not in data:
                raise KeyError(
                    f"Axes NPZ {axes_path} is missing '{key}'. "
                    f"Recalibrate with scripts/01_calibrate.py "
                    f"(it saves scale+center); refusing to default."
                )
            params[axis] = float(data[key])
        return params

    def project(
        self,
        hidden_state: np.ndarray,
        normalize: bool = True,
        clip: bool = True,
    ) -> MoodReading:
        """Project a hidden state onto all mood axes."""
        values = {}
        descriptions = {}

        for axis in MOOD_AXES:
            if axis not in self.axis_vectors:
                continue

            axis_vector = self.axis_vectors[axis]
            raw_value = np.dot(hidden_state, axis_vector)

            if normalize:
                center = self.normalization_centers.get(axis, 0.0)
                scale = self.normalization_scales.get(axis, 2.0)
                raw_value = (raw_value - center) / scale

            if clip:
                raw_value = np.clip(raw_value, -1.0, 1.0)

            values[axis] = float(raw_value)
            descriptions[axis] = self._value_to_description(axis, raw_value)

        return MoodReading(values=values, descriptions=descriptions)

    def _value_to_description(self, axis: str, value: float) -> str:
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
        parts = []
        for axis in MOOD_AXES:
            if axis in reading.descriptions:
                parts.append(reading.descriptions[axis])
        return ", ".join(parts)


def get_mood_for_response(
    hidden_state: np.ndarray,
    projector: Optional[MoodProjector] = None,
) -> Tuple[MoodReading, MoodProjector]:
    """Convenience function to get mood reading for a response."""
    if projector is None:
        projector = MoodProjector()
    reading = projector.project(hidden_state)
    return reading, projector
