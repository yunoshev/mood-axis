#!/usr/bin/env python3
"""Analyze blind judge results against answer keys.

Reads judge ratings and compares with true labels.
Computes Cohen's d, classification accuracy, mean per pole.

Produces: data/judge/results/<model>_judge_summary.json

Usage:
    python scripts/judge_analyze.py --model qwen_7b
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import MOOD_AXES

BATCH_DIR = PROJECT_ROOT / "data" / "judge" / "batches"
RESULTS_DIR = PROJECT_ROOT / "data" / "judge" / "results"


def analyze_axis(model_key: str, axis: str) -> dict:
    key_path = BATCH_DIR / f"{model_key}_{axis}_key.json"
    judged_path = RESULTS_DIR / f"{model_key}_{axis}_judged.json"

    if not key_path.exists() or not judged_path.exists():
        return None

    with open(key_path) as f:
        key_data = json.load(f)
    with open(judged_path) as f:
        judged_data = json.load(f)

    # Build label map: id -> "positive"/"negative"
    labels = {s["id"]: s["label"] for s in key_data["samples"]}

    # Build rating map: id -> rating
    ratings = {r["id"]: r["rating"] for r in judged_data["ratings"]}

    # Judge scale: 1 = first word of axis name, 5 = second word.
    # E.g. warm_cold: 5=warm, 1=cold.  formal_casual: 1=formal, 5=casual.
    # "positive" pole = first word (warm, formal, verbose, confident, empathetic).
    # So for axes where positive=first word AND judge scale 5=first word,
    # positive should score HIGH. But for formal_casual, judge 1=formal=positive,
    # so positive should score LOW.
    #
    # We detect direction from axis name vs judge scale:
    # Judge always rates 5=first_word. Positive pole=first_word.
    # So positive should always get HIGHER ratings.
    # EXCEPT: formal_casual has judge scale inverted (1=formal, 5=casual).
    #
    # Simplest fix: just check if mean_pos > mean_neg. If not, the judge
    # scale was inverted relative to our poles. Report separation either way.
    pos_ratings = []
    neg_ratings = []

    for sid, label in labels.items():
        if sid not in ratings:
            continue
        r = ratings[sid]
        if label == "positive":
            pos_ratings.append(r)
        else:
            neg_ratings.append(r)

    if not pos_ratings or not neg_ratings:
        return None

    pos_mean = float(np.mean(pos_ratings))
    neg_mean = float(np.mean(neg_ratings))
    pos_std = max(float(np.std(pos_ratings, ddof=1)), 0.01) if len(pos_ratings) > 1 else 0.01
    neg_std = max(float(np.std(neg_ratings, ddof=1)), 0.01) if len(neg_ratings) > 1 else 0.01

    # Cohen's d (absolute separation)
    pooled_std = np.sqrt((pos_std**2 + neg_std**2) / 2)
    cohens_d = abs(pos_mean - neg_mean) / pooled_std if pooled_std > 0 else 0

    # Classification: judge correctly separates poles if ratings are bimodal.
    # Positive pole could be high or low on judge scale depending on axis naming.
    # Correct = positive and negative are on opposite sides of midpoint (3).
    if pos_mean > neg_mean:
        # Positive gets high ratings → correct if pos>3 and neg<=3
        correct = sum(1 for r in pos_ratings if r > 3) + sum(1 for r in neg_ratings if r <= 3)
    else:
        # Positive gets low ratings (e.g. formal_casual) → correct if pos<=3 and neg>3
        correct = sum(1 for r in pos_ratings if r <= 3) + sum(1 for r in neg_ratings if r > 3)
    total = len(pos_ratings) + len(neg_ratings)
    accuracy = correct / total if total > 0 else 0

    # Separation is "correct" if poles are clearly different (either direction)
    separated = abs(pos_mean - neg_mean) > 1.0

    return {
        "axis": axis,
        "n_positive": len(pos_ratings),
        "n_negative": len(neg_ratings),
        "mean_positive": round(pos_mean, 2),
        "mean_negative": round(neg_mean, 2),
        "std_positive": round(pos_std, 2),
        "std_negative": round(neg_std, 2),
        "cohens_d": round(float(cohens_d), 2),
        "accuracy": round(accuracy, 3),
        "separated": separated,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze blind judge results")
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "_meta": {
            "model": args.model,
            "timestamp": datetime.now().isoformat(),
            "script": "scripts/judge_analyze.py",
        },
        "axes": {},
    }

    print(f"\n=== Blind Judge Results: {args.model} ===\n", flush=True)
    print(f"{'Axis':<25s} {'Pos mean':>8s} {'Neg mean':>8s} {'Cohen d':>8s} {'Acc':>6s} {'Sep':>5s}",
          flush=True)
    print("-" * 65, flush=True)

    all_ok = True
    for axis in MOOD_AXES:
        result = analyze_axis(args.model, axis)
        if result is None:
            print(f"  {axis}: MISSING data", flush=True)
            continue

        summary["axes"][axis] = result
        sep_str = "OK" if result["separated"] else "WEAK"
        if not result["separated"] or result["accuracy"] < 0.8:
            all_ok = False

        print(f"{axis:<25s} {result['mean_positive']:>8.2f} {result['mean_negative']:>8.2f} "
              f"{result['cohens_d']:>8.2f} {result['accuracy']:>5.0%} {sep_str:>5s}",
              flush=True)

    summary["all_passed"] = all_ok
    print(f"\n{'ALL PASSED' if all_ok else 'SOME FAILED'}", flush=True)

    out_path = RESULTS_DIR / f"{args.model}_judge_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
