#!/usr/bin/env python3
"""Step 07: Base vs Instruct comparison experiments.

Produces: data/experiments/<model>_exp_base_instruct.json
Depends on: 01_calibrate (needs axes for both base and instruct)

Usage:
    python scripts/07_experiments.py --model llama_8b
    python scripts/07_experiments.py --model all-base
    python scripts/07_experiments.py --check
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

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import MOOD_AXES, AXES_DIR, MAX_NEW_TOKENS
from config.models import MODELS, MODEL_SETS
from config.prompts import BASELINE_QUESTIONS
from src.model.loader import load_model
from src.model.inference import get_full_result_for_prompt
from src.calibration.axis_computer import load_axis_vectors

OUTPUT_DIR = PROJECT_ROOT / "data" / "experiments"

# Base models with their instruct counterparts
BASE_PAIRS = {
    "llama_8b_base": "llama_8b",
    "yi_9b_base": "yi_9b",
    "qwen_7b_base": "qwen_7b",
    "mistral_7b_base": "mistral_7b",
    "deepseek_7b_base": "deepseek_7b",
    "gemma_9b_base": "gemma_9b",
}


def preflight_check():
    print(f"Base model pairs: {len(BASE_PAIRS)}", flush=True)
    for base, instruct in BASE_PAIRS.items():
        base_axes = AXES_DIR / f"{base}_axes.npz"
        inst_axes = AXES_DIR / f"{instruct}_axes.npz"
        print(f"  {base} -> {instruct}", flush=True)
        print(f"    base axes: {'OK' if base_axes.exists() else 'MISSING'}", flush=True)
        print(f"    instruct axes: {'OK' if inst_axes.exists() else 'MISSING'}", flush=True)
    print("Preflight OK", flush=True)


def run_base_instruct_comparison(base_key: str, force: bool = False) -> dict:
    """Compare base vs instruct model on same baseline questions."""
    instruct_key = BASE_PAIRS.get(base_key)
    if not instruct_key:
        raise ValueError(f"{base_key} has no instruct counterpart")

    output_file = OUTPUT_DIR / f"{base_key}_exp_base_instruct.json"
    if output_file.exists() and not force:
        print(f"Output exists: {output_file}. Use --force to recompute.", flush=True)
        return json.load(open(output_file))

    # Load instruct model's axes (use as reference frame)
    inst_axes_file = AXES_DIR / f"{instruct_key}_axes.npz"
    if not inst_axes_file.exists():
        raise FileNotFoundError(f"Missing instruct axes: {inst_axes_file}")

    axis_vectors = load_axis_vectors(inst_axes_file)
    npz_data = np.load(inst_axes_file)
    scales = {}
    centers = {}
    for axis in MOOD_AXES:
        scale_key = f"{axis}_scale"
        scales[axis] = float(npz_data[scale_key]) if scale_key in npz_data else 2.0
        center_key = f"{axis}_center"
        centers[axis] = float(npz_data[center_key]) if center_key in npz_data else 0.0

    # Run both models
    results = {
        "_meta": {
            "script": "scripts/07_experiments.py",
            "experiment": "base_vs_instruct",
            "base_model": base_key,
            "instruct_model": instruct_key,
            "timestamp": datetime.now().isoformat(),
        },
        "base": {},
        "instruct": {},
        "comparison": {},
    }

    for model_key, section in [(base_key, "base"), (instruct_key, "instruct")]:
        model_config = MODELS[model_key]
        print(f"Loading {section}: {model_config.model_id}", flush=True)
        model, tokenizer = load_model(model_config.model_id)

        measurements = {axis: [] for axis in MOOD_AXES}
        for question in tqdm(BASELINE_QUESTIONS, desc=f"Measuring {section}"):
            messages = [{"role": "user", "content": question}]
            result = get_full_result_for_prompt(
                model, tokenizer, messages, max_new_tokens=MAX_NEW_TOKENS,
                chat_template_kwargs=model_config.chat_template_kwargs,
                save_extra=False,
            )
            for axis in MOOD_AXES:
                if axis in axis_vectors:
                    raw = float(np.dot(result.hidden_state, axis_vectors[axis]))
                    normalized = np.clip((raw - centers.get(axis, 0.0)) / scales.get(axis, 2.0), -1.0, 1.0)
                    measurements[axis].append(float(normalized))

        results[section] = {
            axis: {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "values": vals,
            }
            for axis, vals in measurements.items()
        }

        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Comparison: std ratio (instruct/base) — alignment suppresses variability
    for axis in MOOD_AXES:
        base_std = results["base"][axis]["std"]
        inst_std = results["instruct"][axis]["std"]
        results["comparison"][axis] = {
            "base_mean": results["base"][axis]["mean"],
            "instruct_mean": results["instruct"][axis]["mean"],
            "base_std": base_std,
            "instruct_std": inst_std,
            "std_ratio": inst_std / base_std if base_std > 0 else 0,
            "mean_delta": results["instruct"][axis]["mean"] - results["base"][axis]["mean"],
        }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {output_file}", flush=True)

    return results


def main():
    parser = argparse.ArgumentParser(description="Step 07: Base vs Instruct experiments")
    parser.add_argument("--model", required=True, help="Base model key or 'all-base'")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.check:
        preflight_check()
        return

    if args.model == "all-base":
        models = list(BASE_PAIRS.keys())
    else:
        models = [args.model]

    for model_key in models:
        try:
            results = run_base_instruct_comparison(model_key, force=args.force)
            print(f"\n=== {model_key} vs {BASE_PAIRS[model_key]} ===", flush=True)
            for axis, comp in results.get("comparison", {}).items():
                print(f"  {axis}: base_std={comp['base_std']:.3f}, inst_std={comp['instruct_std']:.3f}, ratio={comp['std_ratio']:.2f}", flush=True)
        except Exception as e:
            print(f"Error with {model_key}: {e}", flush=True)


if __name__ == "__main__":
    main()
