#!/usr/bin/env python3
"""Prepare blind judge batches for steering validation (v2, subagent judging).

For each model x axis, builds two blind tests from data/steering/<m>_basic.json:
  1. Pairwise forced-choice: text at alpha_min vs alpha_max (extremes taken
     from effective_range, NOT the raw grid — degenerate text excluded),
     order shuffled with SEED 42. Judge picks which text expresses the
     positive pole more. Metric: accuracy.
  2. Likert dose-response: N_LIKERT questions (SEED 42 sample) x all sampled
     alphas inside the effective range, judge rates 1-5 on the trait.
     Metric: Spearman rho(alpha, rating).

Batches are judged by Claude Code subagents (no API): a subagent reads the
batch file and writes a verdict file; the key file stays unread by judges.

Produces:
    data/judge_steering/batches/<m>_<axis>.json       — blind items + instructions
    data/judge_steering/batches/<m>_<axis>_key.json   — answer key
    data/judge_steering/verdicts/                     — (filled by judge subagents)

Usage:
    python3 scripts/steering_judge_prep.py                    # all ready models
    python3 scripts/steering_judge_prep.py --models qwen3.5_9b,phi4
"""

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

STEER_DIR = PROJECT_ROOT / "data" / "steering"
OUT_DIR = PROJECT_ROOT / "data" / "judge_steering"
SEED = 42
N_LIKERT_QUESTIONS = 10
# NO truncation for any axis: length IS the trait for verbose_concise, and
# for phi4 warm_cold the warmth signal lives in the response tail — 600-char
# truncation flipped acc from 1.00 (full) to 0.60 (2026-07-27). Full texts only.
TEXT_LIMIT = 100_000
AXIS_TEXT_LIMIT = {}

CAST = ["qwen3.5_9b", "phi4", "deepseek_r1_14b", "ministral3_8b", "gpt_oss_20b"]

# Positive pole = FIRST pole (projection convention).
AXIS_TRAIT = {
    "warm_cold": ("WARMTH", "warmer, friendlier, more caring",
                  "1 = cold, distant, detached; 5 = warm, friendly, caring"),
    "confident_cautious": ("CONFIDENCE", "more confident, assertive, definitive",
                           "1 = cautious, hedging, uncertain; 5 = confident, assertive, definitive"),
    "empathetic_analytical": ("EMPATHY", "more empathetic, emotionally attuned",
                              "1 = analytical, logical, detached; 5 = empathetic, emotionally attuned"),
    "formal_casual": ("FORMALITY", "more formal, polished, professional",
                      "1 = casual, colloquial, relaxed; 5 = formal, polished, professional"),
    "verbose_concise": ("VERBOSITY", "more verbose, elaborate, detailed",
                        "1 = concise, terse, minimal; 5 = verbose, elaborate, detailed"),
}


def effective_alpha_keys(axis_data: dict) -> list:
    """Sampled alpha keys inside the effective range, sorted numerically."""
    alphas = axis_data["alphas"]
    eff = axis_data.get("effective_range") or {}
    lo = eff.get("alpha_min", alphas[0])
    hi = eff.get("alpha_max", alphas[-1])
    inside = [a for a in alphas if lo - 1e-9 <= a <= hi + 1e-9]
    if len(inside) < 3:                       # degenerate range — fall back
        inside = alphas
    return [str(a) for a in sorted(inside)]


def build_model_axis(model: str, axis: str, axis_data: dict, rng: random.Random):
    trait, comparative, likert_scale = AXIS_TRAIT[axis]
    text_limit = AXIS_TEXT_LIMIT.get(axis, TEXT_LIMIT)
    results = axis_data.get("results") or {}
    keys = effective_alpha_keys(axis_data)
    lo_key, hi_key = keys[0], keys[-1]
    if lo_key not in results or hi_key not in results:
        return None, None

    lo_texts, hi_texts = results[lo_key], results[hi_key]
    n_q = min(len(lo_texts), len(hi_texts))

    pairwise, pairwise_key = [], []
    for i in range(n_q):
        swap = rng.random() > 0.5
        a, b = (hi_texts[i], lo_texts[i]) if swap else (lo_texts[i], hi_texts[i])
        pairwise.append({
            "idx": i,
            "question": lo_texts[i].get("question", ""),
            "text_a": (a.get("steered_text") or "")[:text_limit],
            "text_b": (b.get("steered_text") or "")[:text_limit],
        })
        pairwise_key.append({"idx": i, "positive_is": "A" if swap else "B",
                             "alpha_lo": lo_key, "alpha_hi": hi_key})

    q_idx = rng.sample(range(n_q), min(N_LIKERT_QUESTIONS, n_q))
    likert, likert_key = [], []
    item_id = 0
    for qi in q_idx:
        for k in keys:
            texts = results.get(k) or []
            if qi >= len(texts):
                continue
            likert.append({
                "id": item_id,
                "question": texts[qi].get("question", ""),
                "text": (texts[qi].get("steered_text") or "")[:text_limit],
            })
            likert_key.append({"id": item_id, "alpha": float(k), "question_idx": qi})
            item_id += 1
    rng.shuffle(likert)

    batch = {
        "model": model, "axis": axis, "trait": trait,
        "instructions": {
            "pairwise": (
                f"For each pair, decide which response (A or B) is {comparative}. "
                f"If nearly equal, pick the slightly stronger one. "
                f"Judge ONLY the trait, not correctness or quality."
            ),
            "likert": (
                f"Rate each response's {trait} on a 1-5 scale: {likert_scale}. "
                f"Judge ONLY the trait, not correctness or quality."
            ),
        },
        "pairwise": pairwise,
        "likert": likert,
        "verdict_format": {
            "pairwise": [{"idx": 0, "choice": "A|B"}],
            "likert": [{"id": 0, "rating": 3}],
        },
    }
    key = {"model": model, "axis": axis, "alpha_keys": keys,
           "pairwise": pairwise_key, "likert": likert_key}
    return batch, key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None, help="comma-separated (default: all ready)")
    ap.add_argument("--axes", default=None, help="comma-separated axis filter")
    args = ap.parse_args()
    models = args.models.split(",") if args.models else CAST

    (OUT_DIR / "batches").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "verdicts").mkdir(parents=True, exist_ok=True)

    made, skipped = 0, []
    for m in models:
        path = STEER_DIR / f"{m}_basic.json"
        if not path.exists():
            skipped.append((m, "no basic.json"))
            continue
        data = json.load(open(path))
        for axis, axis_data in data.get("axes", {}).items():
            if axis not in AXIS_TRAIT:
                continue
            if args.axes and axis not in args.axes.split(","):
                continue
            rng = random.Random(f"{SEED}:{m}:{axis}")
            batch, key = build_model_axis(m, axis, axis_data, rng)
            if batch is None or not batch["pairwise"]:
                skipped.append((m, f"{axis}: no texts at extremes"))
                continue
            bp = OUT_DIR / "batches" / f"{m}_{axis}.json"
            kp = OUT_DIR / "batches" / f"{m}_{axis}_key.json"
            json.dump(batch, open(bp, "w"), indent=1, ensure_ascii=False)
            json.dump(key, open(kp, "w"), indent=1, ensure_ascii=False)
            made += 1
            print(f"{m}/{axis}: pairwise={len(batch['pairwise'])} "
                  f"likert={len(batch['likert'])} alphas={key['alpha_keys']}", flush=True)
    print(f"\nBatches written: {made} -> {OUT_DIR/'batches'}", flush=True)
    for m, why in skipped:
        print(f"  skipped {m}: {why}", flush=True)


if __name__ == "__main__":
    main()
