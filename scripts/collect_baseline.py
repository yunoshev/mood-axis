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

from config.settings import MOOD_AXES, AXES_DIR, MAX_NEW_TOKENS
from config.models import MODELS
from config.prompts import BASELINE_QUESTIONS
from src.model.loader import load_model
from src.model.inference import get_full_result_for_prompt
from src.calibration.axis_computer import load_axis_vectors


OUTPUT_DIR = PROJECT_ROOT / "data" / "article" / "baselines"



def collect_baseline(model_key: str, save_extra: bool = True,
                     axes_dir: Path = None, output_dir: Path = None) -> dict:
    """Collect baseline measurements for a model."""
    if model_key not in MODELS:
        raise ValueError(f"Unknown model: {model_key}")

    model_config = MODELS[model_key]
    model_id = model_config.model_id
    _axes_dir = axes_dir or AXES_DIR

    print(f"Collecting baseline for {model_key} ({model_id})")

    # Load calibrated axes — use MOOD_AXES as single source of truth
    axes_file = _axes_dir / f"{model_key}_axes.npz"
    if not axes_file.exists():
        raise FileNotFoundError(f"Calibration not found: {axes_file}")

    all_axes = MOOD_AXES
    axis_vectors = load_axis_vectors(axes_file)

    # Load normalization scales
    npz_data = np.load(axes_file)
    scales = {}
    for axis in all_axes:
        scale_key = f"{axis}_scale"
        if scale_key in npz_data:
            scales[axis] = float(npz_data[scale_key])
        else:
            scales[axis] = 2.0

    print(f"Using {len(all_axes)} axes: {all_axes}")

    # Load model
    print("Loading model...")
    model, tokenizer = load_model(model_id)

    # Collect measurements
    measurements = {axis: [] for axis in all_axes}
    responses = []
    decay_states = []
    last_token_states = []
    per_layer_states = []
    token_states_list = []
    top_k_ids_list = []
    top_k_logprobs_list = []
    gen_times = []

    for question in tqdm(BASELINE_QUESTIONS, desc="Measuring"):
        messages = [{"role": "user", "content": question}]

        result = get_full_result_for_prompt(
            model, tokenizer, messages, max_new_tokens=MAX_NEW_TOKENS,
            chat_template_kwargs=model_config.chat_template_kwargs,
            save_extra=save_extra,
        )

        responses.append({
            "question": question,
            "response": result.text,
            "n_tokens": result.n_generated_tokens,
            "n_words": result.n_words,
            "generation_time_s": result.generation_time_s,
        })
        decay_states.append(result.hidden_state)
        if result.hidden_state_last_token is not None:
            last_token_states.append(result.hidden_state_last_token)
        if result.per_layer_states is not None:
            per_layer_states.append(result.per_layer_states)
        if result.token_states is not None:
            token_states_list.append(result.token_states)
        if result.top_k_ids is not None:
            top_k_ids_list.append(result.top_k_ids)
        if result.top_k_logprobs is not None:
            top_k_logprobs_list.append(result.top_k_logprobs)
        gen_times.append(result.generation_time_s)

        # Project onto each axis manually
        for axis in all_axes:
            if axis not in axis_vectors:
                continue
            raw_value = float(np.dot(result.hidden_state, axis_vectors[axis]))
            raw_value = raw_value / scales.get(axis, 2.0)
            raw_value = np.clip(raw_value, -1.0, 1.0)
            measurements[axis].append(float(raw_value))

    # Compute statistics
    results = {
        "model": model_key,
        "model_id": model_id,
        "timestamp": datetime.now().isoformat(),
        "n_questions": len(BASELINE_QUESTIONS),
        "axes": {},
        "responses": responses,
    }

    for axis in all_axes:
        values = measurements[axis]
        results["axes"][axis] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "values": values,
        }

    # Save hidden states NPZ
    _output_dir = output_dir or OUTPUT_DIR
    _output_dir.mkdir(parents=True, exist_ok=True)
    hs_save = {
        "decay_states": np.array(decay_states),
        "generation_times": np.array(gen_times),
    }
    if last_token_states:
        hs_save["last_token_states"] = np.array(last_token_states)
    if per_layer_states:
        hs_save["per_layer_states"] = np.stack(per_layer_states)
    if token_states_list:
        token_offsets = np.cumsum([0] + [t.shape[0] for t in token_states_list])
        hs_save["token_states"] = np.concatenate(token_states_list)
        hs_save["token_offsets"] = token_offsets
    if top_k_ids_list:
        hs_save["top_k_ids"] = np.concatenate(top_k_ids_list)
        hs_save["top_k_logprobs"] = np.concatenate(top_k_logprobs_list)
    hs_file = _output_dir / f"{model_key}_baseline_hidden_states.npz"
    np.savez(hs_file, **hs_save)
    print(f"Saved hidden states to {hs_file}")

    # Cleanup
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="Collect baseline temperament")
    parser.add_argument("--model", required=True, help="Model key or 'all'")
    parser.add_argument("--no-extra", action="store_true",
                        help="Skip per_layer_states, token_states, top_k (saves memory)")
    parser.add_argument("--axes-dir", type=str, default=None,
                        help="Directory with calibrated axes (default: data/axes/)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for baselines (default: data/article/baselines/)")
    args = parser.parse_args()

    _axes_dir = Path(args.axes_dir) if args.axes_dir else None
    _output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    _output_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "all":
        models = list(MODELS.keys())
    else:
        models = [args.model]

    for model_key in models:
        try:
            results = collect_baseline(model_key, save_extra=not args.no_extra,
                                       axes_dir=_axes_dir, output_dir=_output_dir)

            output_file = _output_dir / f"{model_key}_baseline.json"
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
