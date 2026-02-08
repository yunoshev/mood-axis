#!/usr/bin/env python3
"""Extended drift analysis with conflict scenarios.

Runs 50 conflict scenarios and measures how model temperament
drifts over the conversation.

Usage:
    python scripts/extended_drift.py --model qwen_7b
    python scripts/extended_drift.py --model all --scenarios 10
"""

import argparse
import json
import sys
import gc
from pathlib import Path
from datetime import datetime
from typing import List, Dict

import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import MOOD_AXES, AXES_DIR
from config.models import MODELS
from config.conflict_scenarios import ALL_CONFLICT_SCENARIOS as CONFLICT_SCENARIOS
from src.model.loader import load_model
from src.model.inference import generate_with_hidden_states, format_chat_messages
from src.mood.projector import MoodProjector


OUTPUT_DIR = PROJECT_ROOT / "data" / "article" / "extended_drift"


def run_conversation(model, tokenizer, projector, scenario) -> dict:
    """Run a single conflict conversation and track drift.

    Uses a single generate_with_hidden_states() call per turn instead of
    separate generate_response + get_hidden_state_for_prompt (saves ~50% compute).

    Returns dict with turns list and collected hidden state arrays.
    """
    turns = []
    messages = []
    per_layer_states = []
    token_states_list = []
    top_k_ids_list = []
    top_k_logprobs_list = []
    decay_states = []
    gen_times = []

    for i, user_message in enumerate(scenario.turns):
        messages.append({"role": "user", "content": user_message})

        # Single call: generate response AND extract hidden states
        result = generate_with_hidden_states(model, tokenizer, messages)
        messages.append({"role": "assistant", "content": result.text})

        # Measure temperament from hidden state
        reading = projector.project(result.hidden_state)

        turns.append({
            "turn": i + 1,
            "user": user_message,
            "assistant": result.text,
            "values": reading.values,
            "n_tokens": result.n_generated_tokens,
            "n_words": result.n_words,
            "generation_time_s": result.generation_time_s,
        })

        # Collect hidden states for NPZ
        decay_states.append(result.hidden_state)
        gen_times.append(result.generation_time_s)
        if result.per_layer_states is not None:
            per_layer_states.append(result.per_layer_states)
        if result.token_states is not None:
            token_states_list.append(result.token_states)
        if result.top_k_ids is not None:
            top_k_ids_list.append(result.top_k_ids)
        if result.top_k_logprobs is not None:
            top_k_logprobs_list.append(result.top_k_logprobs)

    return {
        "turns": turns,
        "decay_states": decay_states,
        "per_layer_states": per_layer_states,
        "token_states": token_states_list,
        "top_k_ids": top_k_ids_list,
        "top_k_logprobs": top_k_logprobs_list,
        "gen_times": gen_times,
    }


def analyze_drift(model_key: str, max_scenarios: int = None) -> dict:
    """Run drift analysis for a model."""
    if model_key not in MODELS:
        raise ValueError(f"Unknown model: {model_key}")

    model_config = MODELS[model_key]
    model_id = model_config.model_id

    print(f"Running drift analysis for {model_key} ({model_id})")

    # Load model
    print("Loading model...")
    model, tokenizer = load_model(model_id)

    # Load calibrated axes
    axes_file = AXES_DIR / f"{model_key}_axes.npz"
    if not axes_file.exists():
        raise FileNotFoundError(f"Calibration not found: {axes_file}")

    projector = MoodProjector(axes_file=axes_file)

    # Run scenarios
    scenarios = CONFLICT_SCENARIOS[:max_scenarios] if max_scenarios else CONFLICT_SCENARIOS
    all_results = []
    all_decay_states = []
    all_per_layer = []
    all_token_states = []
    all_top_k_ids = []
    all_top_k_logprobs = []
    all_gen_times = []

    for scenario in tqdm(scenarios, desc="Scenarios"):
        try:
            conv = run_conversation(model, tokenizer, projector, scenario)
            all_results.append({
                "category": scenario.category,
                "scenario_id": scenario.name,
                "turns": conv["turns"],
            })
            all_decay_states.extend(conv["decay_states"])
            all_per_layer.extend(conv["per_layer_states"])
            all_token_states.extend(conv["token_states"])
            all_top_k_ids.extend(conv["top_k_ids"])
            all_top_k_logprobs.extend(conv["top_k_logprobs"])
            all_gen_times.extend(conv["gen_times"])
        except Exception as e:
            print(f"Error in scenario: {e}")
            continue

    # Compute drift statistics per axis
    drift_stats = {axis: {"slopes": [], "start_values": [], "end_values": []} for axis in MOOD_AXES}

    for result in all_results:
        turns = result["turns"]
        if len(turns) < 2:
            continue

        for axis in MOOD_AXES:
            values = [t["values"].get(axis, 0) for t in turns]
            if len(values) >= 2:
                # Linear regression for slope
                x = np.arange(len(values))
                slope = np.polyfit(x, values, 1)[0]
                drift_stats[axis]["slopes"].append(slope)
                drift_stats[axis]["start_values"].append(values[0])
                drift_stats[axis]["end_values"].append(values[-1])

    # Aggregate statistics
    results = {
        "model": model_key,
        "model_id": model_id,
        "timestamp": datetime.now().isoformat(),
        "n_scenarios": len(all_results),
        "drift_summary": {},
        "scenarios": all_results,
    }

    for axis in MOOD_AXES:
        stats = drift_stats[axis]
        if stats["slopes"]:
            results["drift_summary"][axis] = {
                "mean_slope": float(np.mean(stats["slopes"])),
                "std_slope": float(np.std(stats["slopes"])),
                "mean_start": float(np.mean(stats["start_values"])),
                "mean_end": float(np.mean(stats["end_values"])),
                "mean_delta": float(np.mean(stats["end_values"]) - np.mean(stats["start_values"])),
            }

    # Save hidden states NPZ
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if all_decay_states:
        hs_save = {
            "decay_states": np.array(all_decay_states),
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
        hs_file = OUTPUT_DIR / f"{model_key}_drift_hidden_states.npz"
        np.savez(hs_file, **hs_save)
        print(f"Saved drift hidden states to {hs_file}")

    # Cleanup
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="Extended drift analysis")
    parser.add_argument("--model", required=True, help="Model key or 'all'")
    parser.add_argument("--scenarios", type=int, help="Max scenarios to run")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.model == "all":
        models = list(MODELS.keys())
    else:
        models = [args.model]

    for model_key in models:
        try:
            results = analyze_drift(model_key, args.scenarios)

            output_file = OUTPUT_DIR / f"{model_key}_drift.json"
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)

            print(f"Saved to {output_file}")

            # Print summary
            print(f"\n=== {model_key} Drift Summary ===")
            for axis, stats in results["drift_summary"].items():
                print(f"  {axis}: slope={stats['mean_slope']:.4f}, delta={stats['mean_delta']:.3f}")

        except Exception as e:
            print(f"Error with {model_key}: {e}")


if __name__ == "__main__":
    main()
