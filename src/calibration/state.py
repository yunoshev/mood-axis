"""Calibration state management.

Handles loading/saving calibration state, scanning existing calibrations,
and determining what work needs to be done.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class AxisCalibration:
    """Information about a calibrated axis."""
    model: str
    axis: str
    has_file: bool = False
    val_accuracy: Optional[float] = None
    val_separation: Optional[float] = None
    needs_recalibration: bool = False

    @property
    def accuracy_percent(self) -> float:
        """Validation accuracy as percentage."""
        return (self.val_accuracy or 0) * 100

    @property
    def is_problem(self) -> bool:
        """Check if this axis has low accuracy (< 80%)."""
        return self.val_accuracy is not None and self.val_accuracy < 0.80


@dataclass
class ModelCalibrationStatus:
    """Calibration status for a model."""
    model: str
    axes: dict[str, AxisCalibration] = field(default_factory=dict)

    @property
    def calibrated_count(self) -> int:
        return sum(1 for a in self.axes.values() if a.has_file)

    @property
    def total_count(self) -> int:
        return len(self.axes)

    @property
    def avg_accuracy(self) -> float:
        accuracies = [a.val_accuracy for a in self.axes.values() if a.val_accuracy is not None]
        return sum(accuracies) / len(accuracies) if accuracies else 0

    @property
    def problem_axes(self) -> list[AxisCalibration]:
        return [a for a in self.axes.values() if a.is_problem]


class CalibrationStateManager:
    """Manages calibration state and determines work items."""

    def __init__(self, axes_dir: Path):
        self.axes_dir = axes_dir
        self.state_file = axes_dir / "calibration_state.json"
        self.results_file = axes_dir / "calibration_results.json"
        self._state: Optional[dict] = None
        self._results: Optional[dict] = None

    def load_state(self) -> dict:
        """Load calibration state from file."""
        if self._state is not None:
            return self._state

        if self.state_file.exists():
            with open(self.state_file) as f:
                self._state = json.load(f)
        else:
            self._state = {"version": "v3.0", "calibrated": {}, "needs_recalibration": {}}

        return self._state

    def save_state(self, state: dict):
        """Save calibration state to file."""
        self._state = state
        self.axes_dir.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)

    def load_results(self) -> dict:
        """Load calibration results (accuracy metrics)."""
        if self._results is not None:
            return self._results

        if self.results_file.exists():
            with open(self.results_file) as f:
                self._results = json.load(f)
        else:
            self._results = {}

        return self._results

    def get_axis_file(self, model: str, axis: str) -> Path:
        """Get path to individual axis file."""
        return self.axes_dir / f"{model}_{axis}.npz"

    def get_combined_file(self, model: str) -> Path:
        """Get path to combined axes file for a model."""
        return self.axes_dir / f"{model}_axes.npz"

    def scan_calibrations(self, models: list[str], axes: list[str]) -> dict[str, ModelCalibrationStatus]:
        """Scan existing calibrations and build status."""
        state = self.load_state()
        results = self.load_results()

        statuses = {}

        for model in models:
            status = ModelCalibrationStatus(model=model)
            model_results = results.get(model, {})

            for axis in axes:
                axis_file = self.get_axis_file(model, axis)
                axis_results = model_results.get(axis, {})

                calib = AxisCalibration(
                    model=model,
                    axis=axis,
                    has_file=axis_file.exists(),
                    val_accuracy=axis_results.get("val_accuracy"),
                    val_separation=axis_results.get("val_separation"),
                    needs_recalibration=axis in state.get("needs_recalibration", {}).get(model, []),
                )

                # If we have the file but no results, try to extract from npz
                if calib.has_file and calib.val_accuracy is None:
                    try:
                        data = np.load(axis_file, allow_pickle=True)
                        if "val_accuracy" in data:
                            calib.val_accuracy = float(data["val_accuracy"])
                        if "val_separation" in data:
                            calib.val_separation = float(data["val_separation"])
                    except Exception:
                        pass

                status.axes[axis] = calib

            statuses[model] = status

        return statuses

    def get_work_items(
        self,
        models: list[str],
        axes: list[str],
        problem_only: bool = False,
        force: bool = False,
    ) -> list[tuple[str, str]]:
        """Determine what needs to be calibrated.

        Args:
            models: Models to consider
            axes: Axes to consider
            problem_only: Only include axes with accuracy < 80%
            force: Force recalibration even if already done

        Returns:
            List of (model, axis) tuples needing calibration
        """
        if force:
            # Force recalibration of everything requested
            return [(m, a) for m in models for a in axes]

        statuses = self.scan_calibrations(models, axes)
        work_items = []

        for model in models:
            status = statuses.get(model)
            if not status:
                continue

            for axis in axes:
                calib = status.axes.get(axis)
                if not calib:
                    continue

                if problem_only:
                    # Only recalibrate if accuracy < 80%
                    if calib.is_problem:
                        work_items.append((model, axis))
                else:
                    # Recalibrate if missing file or marked for recalibration
                    if not calib.has_file or calib.needs_recalibration:
                        work_items.append((model, axis))

        return work_items

    def mark_calibrated(self, model: str, axis: str):
        """Mark an axis as successfully calibrated."""
        state = self.load_state()

        # Add to calibrated
        if model not in state["calibrated"]:
            state["calibrated"][model] = []
        if axis not in state["calibrated"][model]:
            state["calibrated"][model].append(axis)

        # Remove from needs_recalibration
        if model in state.get("needs_recalibration", {}):
            if axis in state["needs_recalibration"][model]:
                state["needs_recalibration"][model].remove(axis)

        self.save_state(state)

    def mark_for_recalibration(self, model: str, axis: str):
        """Mark an axis as needing recalibration."""
        state = self.load_state()

        if model not in state["needs_recalibration"]:
            state["needs_recalibration"][model] = []
        if axis not in state["needs_recalibration"][model]:
            state["needs_recalibration"][model].append(axis)

        self.save_state(state)

    def update_results(self, model: str, axis: str, accuracy: float, separation: float):
        """Update results for an axis."""
        results = self.load_results()

        if model not in results:
            results[model] = {}

        results[model][axis] = {
            "status": "success",
            "val_accuracy": accuracy,
            "val_separation": separation,
        }

        self._results = results
        with open(self.results_file, "w") as f:
            json.dump(results, f, indent=2)

    def build_combined_axes_file(self, model: str, axes: list[str]) -> bool:
        """Build or update combined axes file from individual files."""
        full_path = self.get_combined_file(model)
        axis_vectors = {}
        scales = {}

        # Load from existing combined file
        if full_path.exists():
            try:
                data = np.load(full_path, allow_pickle=True)
                for axis in axes:
                    if axis in data:
                        axis_vectors[axis] = data[axis]
                    scale_key = f"{axis}_scale"
                    if scale_key in data:
                        scales[axis] = float(data[scale_key])
            except Exception:
                pass

        # Load from individual files
        for axis in axes:
            axis_file = self.get_axis_file(model, axis)
            if not axis_file.exists():
                continue

            try:
                d = np.load(axis_file, allow_pickle=True)
                if "axis_vector" in d:
                    axis_vectors[axis] = d["axis_vector"]
                elif axis in d:
                    axis_vectors[axis] = d[axis]

                if "normalization_scale" in d:
                    scales[axis] = float(d["normalization_scale"])
                else:
                    scale_key = f"{axis}_scale"
                    if scale_key in d:
                        scales[axis] = float(d[scale_key])
            except Exception:
                pass

        # Check we have all axes
        missing_vectors = [a for a in axes if a not in axis_vectors]
        if missing_vectors:
            return False

        # Default scale for missing
        for axis in axes:
            if axis not in scales:
                scales[axis] = 2.0

        # Save combined file
        save_dict = {"_axes": np.array(axes)}
        save_dict.update(axis_vectors)
        for axis in axes:
            save_dict[f"{axis}_scale"] = np.array(scales[axis])

        try:
            np.savez(full_path, **save_dict)
            return True
        except Exception:
            return False
