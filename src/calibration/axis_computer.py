"""Axis vector computation for Mood Axis calibration."""

import numpy as np
from scipy.stats import trim_mean
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from config.settings import (
    MOOD_AXES,
    AXES_DIR,
    AXES_FILE,
    CALIBRATION_MAX_NEW_TOKENS,
    CALIBRATION_SAMPLES_PER_STYLE,
)
from src.calibration.dataset import (
    generate_calibration_dataset,
    iterate_by_axis,
    get_messages_for_sample,
    CalibrationSample,
)
from src.model.inference import get_hidden_state_for_prompt


TRIM_PROPORTION = 0.1  # Remove top/bottom 10% of samples for robust estimation


def compute_normalization_scale(projections: List[float]) -> float:
    """Compute normalization scale using IQR for robustness."""
    projections_arr = np.array(projections)
    q75, q25 = np.percentile(projections_arr, [75, 25])
    iqr = q75 - q25
    scale = iqr / 1.0
    return max(scale, 0.1)


def compute_normalization_center(projections: List[float]) -> float:
    """Compute normalization center (median of training projections).

    Centering ensures that the normalized projection is zero-mean,
    preventing saturation when the axis vector has a large dot product
    with the global mean hidden state direction.
    """
    return float(np.median(projections))


def compute_axis_vector(
    positive_states: List[np.ndarray],
    negative_states: List[np.ndarray],
) -> np.ndarray:
    """Compute the axis direction vector using contrastive method.

    axis_direction = trimmed_mean(positive_states) - trimmed_mean(negative_states)
    """
    positive_mean = trim_mean(positive_states, proportiontocut=TRIM_PROPORTION, axis=0)
    negative_mean = trim_mean(negative_states, proportiontocut=TRIM_PROPORTION, axis=0)

    direction = positive_mean - negative_mean

    norm = np.linalg.norm(direction)
    if norm > 0:
        direction = direction / norm

    return direction


def compute_dprime(
    positive_values: List[float],
    negative_values: List[float],
) -> float:
    """Compute d-prime (Cohen's d) between two distributions."""
    pos = np.array(positive_values)
    neg = np.array(negative_values)

    mean_diff = np.mean(pos) - np.mean(neg)

    n_pos, n_neg = len(pos), len(neg)
    var_pos = np.var(pos, ddof=1) if n_pos > 1 else 0
    var_neg = np.var(neg, ddof=1) if n_neg > 1 else 0

    if n_pos + n_neg > 2:
        pooled_var = ((n_pos - 1) * var_pos + (n_neg - 1) * var_neg) / (n_pos + n_neg - 2)
        pooled_std = np.sqrt(pooled_var)
    else:
        pooled_std = 1.0

    if pooled_std < 1e-10:
        return 0.0

    return float(mean_diff / pooled_std)


def compute_validation_metrics(
    axis_vector: np.ndarray,
    val_positive_states: List[np.ndarray],
    val_negative_states: List[np.ndarray],
) -> Dict[str, float]:
    """Compute separation and consistency metrics on validation set."""
    pos_projections = [np.dot(s, axis_vector) for s in val_positive_states]
    neg_projections = [np.dot(s, axis_vector) for s in val_negative_states]

    separation = np.mean(pos_projections) - np.mean(neg_projections)

    correct = sum(1 for p in pos_projections for n in neg_projections if p > n)
    total = len(pos_projections) * len(neg_projections)
    accuracy = correct / total if total > 0 else 0

    dprime = compute_dprime(pos_projections, neg_projections)

    return {
        "separation": float(separation),
        "accuracy": float(accuracy),
        "correct_pairs": correct,
        "total_pairs": total,
        "pos_mean": float(np.mean(pos_projections)),
        "neg_mean": float(np.mean(neg_projections)),
        "pos_std": float(np.std(pos_projections)),
        "neg_std": float(np.std(neg_projections)),
        "dprime": dprime,
    }


def collect_hidden_states_for_samples(
    model,
    tokenizer,
    samples: List[CalibrationSample],
    max_new_tokens: int = CALIBRATION_MAX_NEW_TOKENS,
    show_progress: bool = True,
) -> List[np.ndarray]:
    """Collect hidden states for a list of samples."""
    hidden_states = []
    iterator = tqdm(samples, desc="Collecting hidden states") if show_progress else samples

    for sample in iterator:
        messages = get_messages_for_sample(sample)
        _, hidden_state = get_hidden_state_for_prompt(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            max_new_tokens=max_new_tokens,
        )
        hidden_states.append(hidden_state)

    return hidden_states


def calibrate_all_axes(
    model,
    tokenizer,
    num_samples_per_style: int = 30,
    max_new_tokens: int = CALIBRATION_MAX_NEW_TOKENS,
    show_progress: bool = True,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, Dict[str, float]]]:
    """Run full calibration for all mood axes with train/validation split.

    NOTE: This is a legacy function. Prefer scripts/01_calibrate.py which uses
    all cal questions for train and separate eval questions for validation.
    """
    train_size = CALIBRATION_SAMPLES_PER_STYLE
    val_size = num_samples_per_style - train_size

    print(f"Generating calibration dataset with {num_samples_per_style} samples per style...", flush=True)
    print(f"  Train: {train_size}, Validation: {val_size}", flush=True)
    samples = generate_calibration_dataset(num_samples_per_style)

    axis_vectors = {}
    normalization_scales = {}
    validation_metrics = {}

    for axis, positive_samples, negative_samples in iterate_by_axis(samples):
        print(f"\nCalibrating axis: {axis}", flush=True)

        train_pos = positive_samples[:train_size]
        val_pos = positive_samples[train_size:]
        train_neg = negative_samples[:train_size]
        val_neg = negative_samples[train_size:]

        print(f"  Collecting positive train samples ({len(train_pos)} samples)...", flush=True)
        train_pos_states = collect_hidden_states_for_samples(
            model, tokenizer, train_pos, max_new_tokens=max_new_tokens, show_progress=show_progress,
        )

        print(f"  Collecting negative train samples ({len(train_neg)} samples)...", flush=True)
        train_neg_states = collect_hidden_states_for_samples(
            model, tokenizer, train_neg, max_new_tokens=max_new_tokens, show_progress=show_progress,
        )

        print(f"  Collecting positive val samples ({len(val_pos)} samples)...", flush=True)
        val_pos_states = collect_hidden_states_for_samples(
            model, tokenizer, val_pos, max_new_tokens=max_new_tokens, show_progress=show_progress,
        )

        print(f"  Collecting negative val samples ({len(val_neg)} samples)...", flush=True)
        val_neg_states = collect_hidden_states_for_samples(
            model, tokenizer, val_neg, max_new_tokens=max_new_tokens, show_progress=show_progress,
        )

        print("  Computing axis vector from training data...", flush=True)
        axis_vector = compute_axis_vector(train_pos_states, train_neg_states)
        axis_vectors[axis] = axis_vector

        train_states = train_pos_states + train_neg_states
        train_projections = [np.dot(s, axis_vector) for s in train_states]
        scale = compute_normalization_scale(train_projections)
        normalization_scales[axis] = float(scale)

        train_pos_proj = [np.dot(s, axis_vector) for s in train_pos_states]
        train_neg_proj = [np.dot(s, axis_vector) for s in train_neg_states]
        train_separation = np.mean(train_pos_proj) - np.mean(train_neg_proj)

        val_metrics = compute_validation_metrics(axis_vector, val_pos_states, val_neg_states)
        validation_metrics[axis] = val_metrics

        print(f"  Train separation: {train_separation:.4f}", flush=True)
        print(f"  Val separation: {val_metrics['separation']:.4f}", flush=True)
        print(f"  Val accuracy: {val_metrics['accuracy']*100:.1f}%", flush=True)
        print(f"  Normalization scale: {normalization_scales[axis]:.4f}", flush=True)

    return axis_vectors, normalization_scales, validation_metrics


def save_axis_vectors(
    axis_vectors: Dict[str, np.ndarray],
    normalization_scales: Optional[Dict[str, float]] = None,
    normalization_centers: Optional[Dict[str, float]] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """Save computed axis vectors, normalization scales, and centers to file."""
    if output_path is None:
        output_path = AXES_FILE

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {**axis_vectors, "_axes": np.array(list(axis_vectors.keys()))}

    if normalization_scales:
        for axis, scale in normalization_scales.items():
            save_dict[f"{axis}_scale"] = np.array(scale)

    if normalization_centers:
        for axis, center in normalization_centers.items():
            save_dict[f"{axis}_center"] = np.array(center)

    np.savez(output_path, **save_dict)

    print(f"Axis vectors saved to {output_path}", flush=True)
    return output_path


def load_axis_vectors(input_path: Optional[Path] = None) -> Dict[str, np.ndarray]:
    """Load axis vectors from file."""
    if input_path is None:
        input_path = AXES_FILE

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Axis vectors file not found at {input_path}. "
            "Please run calibration first with 'python scripts/01_calibrate.py'"
        )

    data = np.load(input_path)
    axes = data["_axes"].tolist()

    return {axis: data[axis] for axis in axes}
