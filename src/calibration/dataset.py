"""Calibration dataset generation for Mood Axis.

Reads questions and style instructions from data/calibration/*.json files.
"""

import json
from typing import List, Dict, Iterator, Tuple
from dataclasses import dataclass
from pathlib import Path

from config.settings import (
    MOOD_AXES,
    CALIBRATION_DIR,
    CALIBRATION_SAMPLES_PER_STYLE,
    CALIBRATION_EVAL_SAMPLES,
)


@dataclass
class CalibrationSample:
    """A single calibration sample."""
    axis: str
    pole: str  # 'positive' or 'negative'
    question: str
    system_prompt: str
    user_prompt: str


def _load_axis_data(axis: str) -> dict:
    """Load questions and style instructions for an axis from JSON."""
    path = CALIBRATION_DIR / f"{axis}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Calibration data not found: {path}. "
            f"Expected JSON file for axis '{axis}'."
        )
    with open(path) as f:
        return json.load(f)


def _load_baseline_data() -> dict:
    """Load baseline questions from JSON."""
    path = CALIBRATION_DIR / "baseline.json"
    if not path.exists():
        raise FileNotFoundError(f"Baseline data not found: {path}")
    with open(path) as f:
        return json.load(f)


def get_style_instruction(axis: str, pole: str) -> str:
    """Get style instruction for an axis and pole."""
    data = _load_axis_data(axis)
    return data["style_instructions"][pole]


def get_calibration_questions(axis: str, num_questions: int = 50) -> list:
    """Get calibration questions for an axis."""
    data = _load_axis_data(axis)
    return data["calibration_questions"][:num_questions]


def get_eval_questions(axis: str, num_questions: int = 15) -> list:
    """Get eval questions for an axis."""
    data = _load_axis_data(axis)
    return data["eval_questions"][:num_questions]


def get_baseline_questions(num_questions: int = 50) -> list:
    """Get neutral baseline questions."""
    data = _load_baseline_data()
    return data["questions"][:num_questions]


def get_calibration_prompt(axis: str, pole: str, question: str) -> dict:
    """Generate a calibration prompt for a given axis, pole, and question."""
    style_instruction = get_style_instruction(axis, pole)
    return {
        "system": style_instruction,
        "user": question,
    }


def generate_calibration_dataset(
    num_samples_per_style: int = CALIBRATION_SAMPLES_PER_STYLE,
    axes: List[str] = None,
) -> List[CalibrationSample]:
    """Generate the full calibration dataset from JSON files."""
    if axes is None:
        axes = MOOD_AXES
    samples = []

    for axis in axes:
        questions = get_calibration_questions(axis, num_samples_per_style)

        for pole in ["positive", "negative"]:
            for question in questions:
                prompt = get_calibration_prompt(axis, pole, question)
                samples.append(
                    CalibrationSample(
                        axis=axis,
                        pole=pole,
                        question=question,
                        system_prompt=prompt["system"],
                        user_prompt=prompt["user"],
                    )
                )

    return samples


def iterate_by_axis(
    samples: List[CalibrationSample],
) -> Iterator[Tuple[str, List[CalibrationSample], List[CalibrationSample]]]:
    """Iterate through samples grouped by axis."""
    axes_in_samples = []
    seen = set()
    for s in samples:
        if s.axis not in seen:
            axes_in_samples.append(s.axis)
            seen.add(s.axis)

    for axis in axes_in_samples:
        positive = [s for s in samples if s.axis == axis and s.pole == "positive"]
        negative = [s for s in samples if s.axis == axis and s.pole == "negative"]
        yield axis, positive, negative


def get_messages_for_sample(sample: CalibrationSample) -> List[Dict[str, str]]:
    """Convert a calibration sample to chat messages format."""
    return [
        {"role": "system", "content": sample.system_prompt},
        {"role": "user", "content": sample.user_prompt},
    ]


def get_dataset_stats(samples: List[CalibrationSample]) -> Dict:
    """Get statistics about the calibration dataset."""
    stats = {
        "total_samples": len(samples),
        "axes": {},
    }

    for axis in MOOD_AXES:
        axis_samples = [s for s in samples if s.axis == axis]
        stats["axes"][axis] = {
            "total": len(axis_samples),
            "positive": len([s for s in axis_samples if s.pole == "positive"]),
            "negative": len([s for s in axis_samples if s.pole == "negative"]),
        }

    return stats


def split_calibration_dataset(
    samples: List[CalibrationSample],
    train_size: int = CALIBRATION_SAMPLES_PER_STYLE,
) -> Tuple[List[CalibrationSample], List[CalibrationSample]]:
    """Split samples into train and validation sets per axis/pole.

    NOTE: Prefer using all cal questions for train and eval questions for val
    (see 01_calibrate.py). This function is kept for backwards compatibility.
    """
    train_samples = []
    val_samples = []

    for axis in MOOD_AXES:
        for pole in ["positive", "negative"]:
            axis_pole_samples = [
                s for s in samples if s.axis == axis and s.pole == pole
            ]
            train_samples.extend(axis_pole_samples[:train_size])
            val_samples.extend(axis_pole_samples[train_size:])

    return train_samples, val_samples
