#!/usr/bin/env python3
"""Local calibration script for Mood Axis.

Calibrates axis vectors for a single model locally.
Requires GPU with ~24GB VRAM for 7-9B models.

Usage:
    python scripts/calibrate_local.py --model qwen_7b
    python scripts/calibrate_local.py --model llama_8b --axes warm_cold,patient_irritated
"""

import argparse
import json
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
from src.model.inference import get_full_result_for_prompt
from src.calibration.dataset import generate_calibration_dataset, iterate_by_axis
from src.calibration.axis_computer import (
    compute_axis_vector,
    compute_normalization_scale,
    save_axis_vectors,
)


def calibrate_axis(model, tokenizer, axis: str, samples_per_pole: int = 30) -> dict:
    """Calibrate a single axis.

    Returns dict with axis_vector, scale, and validation metrics.
    """
    logger.info(f"Calibrating axis: {axis}")

    # Generate calibration samples for this axis
    dataset = generate_calibration_dataset(num_samples_per_style=samples_per_pole, axes=[axis])
    axis_samples = [s for s in dataset if s.axis == axis]

    positive_states = []
    negative_states = []
    positive_last_token_states = []
    negative_last_token_states = []
    per_layer_states = []
    token_states = []
    top_k_ids = []
    top_k_logprobs = []
    gen_times = []
    responses = []

    for sample in axis_samples:
        messages = [
            {"role": "system", "content": sample.system_prompt},
            {"role": "user", "content": sample.user_prompt},
        ]

        result = get_full_result_for_prompt(model, tokenizer, messages)

        if sample.pole == "positive":
            positive_states.append(result.hidden_state)
            positive_last_token_states.append(result.hidden_state_last_token)
        else:
            negative_states.append(result.hidden_state)
            negative_last_token_states.append(result.hidden_state_last_token)

        if result.per_layer_states is not None:
            per_layer_states.append(result.per_layer_states)
        if result.token_states is not None:
            token_states.append(result.token_states)
        if result.top_k_ids is not None:
            top_k_ids.append(result.top_k_ids)
        if result.top_k_logprobs is not None:
            top_k_logprobs.append(result.top_k_logprobs)
        gen_times.append(result.generation_time_s)

        responses.append({
            "axis": axis,
            "pole": sample.pole,
            "question": sample.user_prompt,
            "system_prompt": sample.system_prompt,
            "response": result.text,
            "n_tokens": result.n_generated_tokens,
            "n_words": result.n_words,
            "generation_time_s": result.generation_time_s,
        })

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
        "responses": responses,
        "decay_states": np.array(positive_states + negative_states),
        "last_token_states": np.array(positive_last_token_states + negative_last_token_states),
        "per_layer_states": per_layer_states,
        "token_states": token_states,
        "top_k_ids": top_k_ids,
        "top_k_logprobs": top_k_logprobs,
        "gen_times": gen_times,
    }


def calibrate_model(model_key: str, axes: list = None):
    """Calibrate axes for a model, merging with existing calibration data."""
    if model_key not in MODELS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(MODELS.keys())}")

    model_config = MODELS[model_key]
    model_id = model_config.model_id
    axes = axes or MOOD_AXES

    # Load existing calibration if present — merge new axes with old
    output_file = AXES_DIR / f"{model_key}_axes.npz"
    existing_vectors = {}
    existing_scales = {}
    if output_file.exists():
        data = np.load(output_file)
        existing_axes = data["_axes"].tolist()
        for a in existing_axes:
            existing_vectors[a] = data[a]
            scale_key = f"{a}_scale"
            if scale_key in data:
                existing_scales[a] = float(data[scale_key])
        logger.info(f"Loaded existing axes: {existing_axes}")

        # Only calibrate axes not already present
        missing = [a for a in axes if a not in existing_axes]
        if not missing:
            logger.info("All requested axes already calibrated, nothing to do")
            return {}
        logger.info(f"Will calibrate missing axes: {missing}")
        axes = missing

    logger.info(f"Calibrating model: {model_id}")
    logger.info(f"Axes: {axes}")

    # Load model
    logger.info("Loading model...")
    model, tokenizer = load_model(model_id)

    # Calibrate each axis
    results = {}
    axis_vectors = dict(existing_vectors)
    scales = dict(existing_scales)

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

    # Save merged results
    AXES_DIR.mkdir(parents=True, exist_ok=True)
    save_axis_vectors(axis_vectors, scales, output_file)
    logger.info(f"Saved to {output_file} ({len(axis_vectors)} axes)")

    # Save text responses as JSONL
    all_responses = []
    for axis in axes:
        all_responses.extend(results[axis]["responses"])
    if all_responses:
        responses_file = AXES_DIR / f"{model_key}_calibration_responses.jsonl"
        with open(responses_file, "a") as f:
            for entry in all_responses:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(all_responses)} responses to {responses_file}")

    # Save hidden states NPZ (decay + last_token + per_layer + token_level + top_k + timing)
    all_decay = []
    all_last_token = []
    all_per_layer = []
    all_token_states = []
    all_top_k_ids = []
    all_top_k_logprobs = []
    all_gen_times = []
    for axis in axes:
        all_decay.append(results[axis]["decay_states"])
        all_last_token.append(results[axis]["last_token_states"])
        all_per_layer.extend(results[axis]["per_layer_states"])
        all_token_states.extend(results[axis]["token_states"])
        all_top_k_ids.extend(results[axis]["top_k_ids"])
        all_top_k_logprobs.extend(results[axis]["top_k_logprobs"])
        all_gen_times.extend(results[axis]["gen_times"])
    if all_decay:
        hs_file = AXES_DIR / f"{model_key}_calibration_hidden_states.npz"
        hs_save = {
            "decay_states": np.concatenate(all_decay, axis=0),
            "last_token_states": np.concatenate(all_last_token, axis=0),
            "generation_times": np.array(all_gen_times),
        }
        if all_per_layer:
            hs_save["per_layer_states"] = np.stack(all_per_layer)
        if all_token_states:
            token_offsets = np.cumsum([0] + [t.shape[0] for t in all_token_states])
            hs_save["token_states"] = np.concatenate(all_token_states)
            hs_save["token_offsets"] = token_offsets
        if all_top_k_ids:
            hs_save["top_k_ids"] = np.concatenate(all_top_k_ids)
            hs_save["top_k_logprobs"] = np.concatenate(all_top_k_logprobs)
        np.savez(hs_file, **hs_save)
        logger.info(f"Saved hidden states to {hs_file}")

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
