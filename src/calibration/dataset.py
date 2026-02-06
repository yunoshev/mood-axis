"""Calibration dataset generation for Mood Axis."""

from typing import List, Dict, Iterator, Tuple
from dataclasses import dataclass
import sys

sys.path.insert(0, str(__file__).rsplit("/src/", 1)[0])
from config.prompts import (
    NEUTRAL_QUESTIONS,
    STYLE_INSTRUCTIONS,
    get_calibration_prompt,
    get_calibration_questions,  # NEW: Use calibration-specific questions
)
from config.settings import MOOD_AXES, CALIBRATION_SAMPLES_PER_STYLE, CALIBRATION_TRAIN_SAMPLES


@dataclass
class CalibrationSample:
    """A single calibration sample."""
    axis: str
    pole: str  # 'positive' or 'negative'
    question: str
    system_prompt: str
    user_prompt: str


def generate_calibration_dataset(
    num_samples_per_style: int = CALIBRATION_SAMPLES_PER_STYLE,
    use_axis_specific_questions: bool = True,
) -> List[CalibrationSample]:
    """Generate the full calibration dataset.

    Creates samples for each axis, each pole, using calibration-specific
    questions that are SEPARATE from evaluation questions (to prevent leakage).

    Args:
        num_samples_per_style: Number of samples per (axis, pole) combination
                               Max 20 for calibration set
        use_axis_specific_questions: Whether to use axis-specific questions

    Returns:
        List of CalibrationSample objects
    """
    samples = []

    for axis in MOOD_AXES:
        # Get calibration-specific questions (max 20 per axis)
        # These are SEPARATE from eval questions to prevent data leakage
        if use_axis_specific_questions:
            questions = get_calibration_questions(axis, num_samples_per_style)
        else:
            questions = NEUTRAL_QUESTIONS[:num_samples_per_style]

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
    """Iterate through samples grouped by axis.

    Yields:
        Tuple of (axis_name, positive_samples, negative_samples)
    """
    for axis in MOOD_AXES:
        positive = [s for s in samples if s.axis == axis and s.pole == "positive"]
        negative = [s for s in samples if s.axis == axis and s.pole == "negative"]
        yield axis, positive, negative


def get_messages_for_sample(sample: CalibrationSample) -> List[Dict[str, str]]:
    """Convert a calibration sample to chat messages format.

    Args:
        sample: The calibration sample

    Returns:
        List of message dicts suitable for chat template
    """
    return [
        {"role": "system", "content": sample.system_prompt},
        {"role": "user", "content": sample.user_prompt},
    ]


def get_dataset_stats(samples: List[CalibrationSample]) -> Dict:
    """Get statistics about the calibration dataset.

    Args:
        samples: List of calibration samples

    Returns:
        Dict with statistics
    """
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
    train_size: int = CALIBRATION_TRAIN_SAMPLES,
) -> Tuple[List[CalibrationSample], List[CalibrationSample]]:
    """Split samples into train and validation sets per axis/pole.

    Args:
        samples: Full list of calibration samples
        train_size: Number of samples for training per (axis, pole) combination

    Returns:
        Tuple of (train_samples, val_samples)
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
