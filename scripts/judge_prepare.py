#!/usr/bin/env python3
"""Prepare blind judge batches from existing calibration responses.

Reads saved responses from data/axes/<model>_calibration.jsonl,
selects 10 positive + 10 negative val samples per axis, shuffles,
and saves blind batches for judging by Claude Code sub-agents.

NO GPU needed — uses already generated data.

Produces:
    data/judge/batches/<model>_<axis>.json       — shuffled samples (no labels)
    data/judge/batches/<model>_<axis>_key.json   — answer key (labels + prompts)

Usage:
    python scripts/judge_prepare.py --model qwen_7b
    python scripts/judge_prepare.py --model qwen_7b --check
"""

import argparse
import json
import random
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import MOOD_AXES
from config.models import MODELS

AXES_DIR = PROJECT_ROOT / "data" / "axes"
BATCH_DIR = PROJECT_ROOT / "data" / "judge" / "batches"
N_SAMPLES = 10  # per pole


JUDGE_PROMPTS = {
    "warm_cold": (
        "You will see a user question and an assistant's response. "
        "Rate the assistant's TONE on a scale from 1 to 5:\n"
        "  1 = cold, distant, detached\n"
        "  5 = warm, friendly, caring\n"
        "Reply with ONLY the number (1-5)."
    ),
    "confident_cautious": (
        "You will see a user question and an assistant's response. "
        "Rate the assistant's CONFIDENCE on a scale from 1 to 5:\n"
        "  1 = cautious, hedging, uncertain\n"
        "  5 = confident, assertive, definitive\n"
        "Reply with ONLY the number (1-5)."
    ),
    "empathetic_analytical": (
        "You will see a user question and an assistant's response. "
        "Rate the assistant's STYLE on a scale from 1 to 5:\n"
        "  1 = analytical, logical, detached\n"
        "  5 = empathetic, emotionally attuned, compassionate\n"
        "Reply with ONLY the number (1-5)."
    ),
    "formal_casual": (
        "You will see a user question and an assistant's response. "
        "Rate the assistant's REGISTER on a scale from 1 to 5:\n"
        "  1 = formal, professional, structured\n"
        "  5 = casual, conversational, relaxed\n"
        "Reply with ONLY the number (1-5)."
    ),
    "verbose_concise": (
        "You will see a user question and an assistant's response. "
        "Rate the assistant's VERBOSITY on a scale from 1 to 5:\n"
        "  1 = concise, brief, minimal\n"
        "  5 = verbose, detailed, comprehensive\n"
        "Reply with ONLY the number (1-5)."
    ),
}


def load_calibration_responses(model_key: str) -> dict:
    """Load responses from calibration JSONL, grouped by (axis, pole, split)."""
    jsonl_path = AXES_DIR / f"{model_key}_calibration.jsonl"
    if not jsonl_path.exists():
        print(f"ERROR: {jsonl_path} not found. Run 01_calibrate.py first.", flush=True)
        sys.exit(1)

    data = defaultdict(list)
    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            key = (d["axis"], d["pole"], d["split"])
            data[key].append(d)

    return data


def preflight_check(model_key: str):
    assert model_key in MODELS, f"Unknown model: {model_key}"
    data = load_calibration_responses(model_key)

    print(f"Model: {MODELS[model_key].model_id}", flush=True)
    print(f"Source: data/axes/{model_key}_calibration.jsonl", flush=True)
    print(f"Axes: {len(MOOD_AXES)} ({', '.join(MOOD_AXES)})", flush=True)

    for axis in MOOD_AXES:
        n_val_pos = len(data.get((axis, "positive", "val"), []))
        n_val_neg = len(data.get((axis, "negative", "val"), []))
        n_train_pos = len(data.get((axis, "positive", "train"), []))
        n_train_neg = len(data.get((axis, "negative", "train"), []))
        source = "val" if min(n_val_pos, n_val_neg) >= N_SAMPLES else "train"
        print(f"  {axis}: val={n_val_pos}+{n_val_neg}, "
              f"train={n_train_pos}+{n_train_neg}, "
              f"will use {source} ({N_SAMPLES}+{N_SAMPLES})", flush=True)

    print(f"Output: {BATCH_DIR}", flush=True)
    print("Preflight OK", flush=True)


def prepare_batches(model_key: str):
    model_config = MODELS[model_key]
    data = load_calibration_responses(model_key)
    BATCH_DIR.mkdir(parents=True, exist_ok=True)

    for axis in MOOD_AXES:
        # Prefer val samples (independent from training), fall back to train
        val_pos = data.get((axis, "positive", "val"), [])
        val_neg = data.get((axis, "negative", "val"), [])

        if len(val_pos) >= N_SAMPLES and len(val_neg) >= N_SAMPLES:
            pos_samples = val_pos[:N_SAMPLES]
            neg_samples = val_neg[:N_SAMPLES]
            source = "val"
        else:
            pos_samples = data.get((axis, "positive", "train"), [])[:N_SAMPLES]
            neg_samples = data.get((axis, "negative", "train"), [])[:N_SAMPLES]
            source = "train"

        # Build blind batch
        samples = []
        key_entries = []

        for i, s in enumerate(pos_samples):
            samples.append({
                "id": f"s{2*i:02d}",
                "question": s["question"],
                "response": s["response"],
            })
            key_entries.append({
                "id": f"s{2*i:02d}",
                "label": "positive",
                "question": s["question"],
                "response": s["response"],
                "n_tokens": s["n_tokens"],
            })

        for i, s in enumerate(neg_samples):
            samples.append({
                "id": f"s{2*i+1:02d}",
                "question": s["question"],
                "response": s["response"],
            })
            key_entries.append({
                "id": f"s{2*i+1:02d}",
                "label": "negative",
                "question": s["question"],
                "response": s["response"],
                "n_tokens": s["n_tokens"],
            })

        # Shuffle blind batch
        random.seed(42)
        random.shuffle(samples)

        # Save blind batch
        batch = {
            "_meta": {
                "model": model_key,
                "model_id": model_config.model_id,
                "axis": axis,
                "n_samples": len(samples),
                "source": f"{source} split from calibration",
                "timestamp": datetime.now().isoformat(),
                "note": "Blind batch — labels hidden. Judge each response independently.",
            },
            "judge_prompt": JUDGE_PROMPTS[axis],
            "axis_description": f"{axis.split('_')[0]} (5) vs {axis.split('_')[1]} (1)",
            "samples": samples,
        }
        batch_path = BATCH_DIR / f"{model_key}_{axis}.json"
        with open(batch_path, "w") as f:
            json.dump(batch, f, indent=2, ensure_ascii=False)

        # Save answer key
        key_data = {
            "_meta": {
                "model": model_key,
                "model_id": model_config.model_id,
                "axis": axis,
                "n_positive": len(pos_samples),
                "n_negative": len(neg_samples),
                "source": source,
                "timestamp": datetime.now().isoformat(),
            },
            "samples": key_entries,
        }
        key_path = BATCH_DIR / f"{model_key}_{axis}_key.json"
        with open(key_path, "w") as f:
            json.dump(key_data, f, indent=2, ensure_ascii=False)

        print(f"  {axis}: {len(pos_samples)}+{len(neg_samples)} samples "
              f"from {source} → {batch_path.name}", flush=True)

    print(f"\nDone. Batches saved to {BATCH_DIR}/", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Prepare blind judge batches")
    parser.add_argument("--model", required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.check:
        preflight_check(args.model)
        return

    prepare_batches(args.model)


if __name__ == "__main__":
    main()
