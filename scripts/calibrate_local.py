#!/usr/bin/env python3
"""Local calibration script for Mood Axis.

Calibrates axis vectors for a single model locally.
Requires GPU with ~24GB VRAM for 7-9B models.

Usage:
    python scripts/calibrate_local.py --model qwen_7b
    python scripts/calibrate_local.py --model llama_8b --axes warm_cold,patient_irritated
"""

import argparse
import sys
import gc
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import MOOD_AXES, AXES_DIR
from config.models import MODELS
from config.prompts import STYLE_INSTRUCTIONS
from src.model.loader import load_model
from src.model.inference import get_hidden_state_for_prompt
from src.calibration.dataset import generate_calibration_dataset, iterate_by_axis
from src.calibration.axis_computer import (
    compute_axis_vector,
    compute_normalization_scale,
    save_axis_vectors,
)


def calibrate_axis(model, tokenizer, axis: str, samples_per_pole: int = 20) -> dict:
    """Calibrate a single axis.

    Returns dict with axis_vector, scale, and validation metrics.
    """
    logger.info(f"Calibrating axis: {axis}")

    # Generate calibration samples for this axis
    dataset = generate_calibration_dataset(num_samples_per_style=samples_per_pole)
    axis_samples = [s for s in dataset if s.axis == axis]

    positive_states = []
    negative_states = []

    for sample in axis_samples:
        messages = [
            {"role": "system", "content": sample.system_prompt},
            {"role": "user", "content": sample.user_prompt},
        ]

        text, hidden_state = get_hidden_state_for_prompt(model, tokenizer, messages)

        if sample.pole == "positive":
            positive_states.append(hidden_state)
        else:
            negative_states.append(hidden_state)

    # Compute axis vector
    axis_vector = compute_axis_vector(positive_states, negative_states)

    # Compute normalization scale
    all_projections = []
    for state in positive_states + negative_states:
        proj = float(np.dot(state, axis_vector))
        all_projections.append(proj)

    scale = compute_normalization_scale(all_projections)

    # Compute validation accuracy (simple split)
    n_val = len(positive_states) // 5  # 20% validation
    if n_val > 0:
        val_pos = positive_states[-n_val:]
        val_neg = negative_states[-n_val:]

        correct = 0
        total = 0
        for state in val_pos:
            if np.dot(state, axis_vector) > 0:
                correct += 1
            total += 1
        for state in val_neg:
            if np.dot(state, axis_vector) < 0:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0
    else:
        accuracy = None

    return {
        "axis_vector": axis_vector,
        "scale": scale,
        "accuracy": accuracy,
        "n_samples": len(axis_samples),
    }


def calibrate_model(model_key: str, axes: list = None):
    """Calibrate all axes for a model."""
    if model_key not in MODELS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(MODELS.keys())}")

    model_config = MODELS[model_key]
    model_id = model_config.model_id
    axes = axes or MOOD_AXES

    logger.info(f"Calibrating model: {model_id}")
    logger.info(f"Axes: {axes}")

    # Load model
    logger.info("Loading model...")
    model, tokenizer = load_model(model_id)

    # Calibrate each axis
    results = {}
    axis_vectors = {}
    scales = {}

    for axis in axes:
        result = calibrate_axis(model, tokenizer, axis)
        results[axis] = result
        axis_vectors[axis] = result["axis_vector"]
        scales[axis] = result["scale"]

        if result["accuracy"]:
            logger.info(f"  {axis}: accuracy={result['accuracy']*100:.1f}%")

        # Clear cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save results
    AXES_DIR.mkdir(parents=True, exist_ok=True)
    output_file = AXES_DIR / f"{model_key}_axes.npz"
    save_axis_vectors(axis_vectors, scales, output_file)
    logger.info(f"Saved to {output_file}")

    # Cleanup
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="Calibrate axis vectors locally")
    parser.add_argument("--model", required=True, help="Model key (e.g., qwen_7b)")
    parser.add_argument("--axes", help="Comma-separated list of axes (default: all)")
    args = parser.parse_args()

    axes = args.axes.split(",") if args.axes else None

    results = calibrate_model(args.model, axes)

    print("\n=== Calibration Results ===")
    for axis, result in results.items():
        acc = result["accuracy"]
        acc_str = f"{acc*100:.1f}%" if acc else "N/A"
        print(f"{axis}: accuracy={acc_str}, samples={result['n_samples']}")


if __name__ == "__main__":
    main()
