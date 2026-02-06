#!/usr/bin/env python3
"""Collect baseline temperament measurements.

Measures model temperament on neutral questions without any style prompting.
This establishes the model's "default personality".

Usage:
    python scripts/collect_baseline.py --model qwen_7b
    python scripts/collect_baseline.py --model all
"""

import argparse
import json
import sys
import gc
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import MOOD_AXES, AXES_DIR
from config.models import MODELS
from config.prompts import BASELINE_QUESTIONS
from src.model.loader import load_model
from src.model.inference import get_hidden_state_for_prompt
from src.mood.projector import MoodProjector


OUTPUT_DIR = PROJECT_ROOT / "data" / "article" / "baselines"


def collect_baseline(model_key: str) -> dict:
    """Collect baseline measurements for a model."""
    if model_key not in MODELS:
        raise ValueError(f"Unknown model: {model_key}")

    model_config = MODELS[model_key]
    model_id = model_config.model_id

    print(f"Collecting baseline for {model_key} ({model_id})")

    # Load model
    print("Loading model...")
    model, tokenizer = load_model(model_id)

    # Load calibrated axes
    axes_file = AXES_DIR / f"{model_key}_axes.npz"
    if not axes_file.exists():
        raise FileNotFoundError(f"Calibration not found: {axes_file}")

    projector = MoodProjector(axes_file=axes_file)

    # Collect measurements
    measurements = {axis: [] for axis in MOOD_AXES}

    for question in tqdm(BASELINE_QUESTIONS, desc="Measuring"):
        messages = [{"role": "user", "content": question}]

        text, hidden_state = get_hidden_state_for_prompt(model, tokenizer, messages)
        reading = projector.project(hidden_state)

        for axis in MOOD_AXES:
            measurements[axis].append(reading.values.get(axis, 0))

    # Compute statistics
    results = {
        "model": model_key,
        "model_id": model_id,
        "timestamp": datetime.now().isoformat(),
        "n_questions": len(BASELINE_QUESTIONS),
        "axes": {},
    }

    for axis in MOOD_AXES:
        values = measurements[axis]
        results["axes"][axis] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "values": values,
        }

    # Cleanup
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="Collect baseline temperament")
    parser.add_argument("--model", required=True, help="Model key or 'all'")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.model == "all":
        models = list(MODELS.keys())
    else:
        models = [args.model]

    for model_key in models:
        try:
            results = collect_baseline(model_key)

            output_file = OUTPUT_DIR / f"{model_key}_baseline.json"
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)

            print(f"Saved to {output_file}")

            # Print summary
            print(f"\n=== {model_key} Baseline ===")
            for axis, stats in results["axes"].items():
                print(f"  {axis}: {stats['mean']:.3f} (±{stats['std']:.3f})")

        except Exception as e:
            print(f"Error with {model_key}: {e}")


if __name__ == "__main__":
    main()
