#!/usr/bin/env python3
"""Sanity check: Test Yi 1.5 9B with multiple instruction phrasings.

Tests whether the "dead zones" on direct_evasive and patient_irritated
are caused by specific instruction phrasing or are a deeper constraint.

If Yi fails across ALL phrasings, it suggests a genuine behavioral constraint
(possibly from alignment training). If it passes with some phrasings,
it's more likely a template/prompt issue.
"""

import sys
from pathlib import Path
from typing import List, Dict

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.models import get_model_config
from src.model.loader import load_model
from src.model.inference import generate_with_hidden_states, format_chat_messages
from src.mood.projector import MoodProjector


# Multiple phrasings for "be direct"
DIRECT_PROMPTS = [
    # Original
    "You are a direct communicator. State your views clearly without hedging.",
    # Alternative 1: More casual
    "Just be blunt and straightforward. No beating around the bush.",
    # Alternative 2: Professional framing
    "You are a senior executive who values clear, decisive communication. Give direct answers.",
    # Alternative 3: Negative framing
    "Never hedge. Never say 'it depends'. Never use phrases like 'on the other hand'. Just state your position.",
    # Alternative 4: Role-play
    "You are answering questions in a job interview. Be confident and direct in all responses.",
]

EVASIVE_PROMPTS = [
    # Original
    "You are cautious and diplomatic. Hedge your statements and present multiple perspectives.",
    # Alternative 1: Academic
    "You are an academic who always considers multiple viewpoints and avoids strong claims.",
    # Alternative 2: Diplomatic
    "You are a diplomat. Always leave room for interpretation and avoid definitive statements.",
]

# Multiple phrasings for "show irritation"
IRRITATED_PROMPTS = [
    # Original
    "You are frustrated and impatient. Show your annoyance in responses.",
    # Alternative 1: More subtle
    "You're having a bad day and feeling short-tempered. Let that come through.",
    # Alternative 2: Role-play
    "You are a tired customer service agent at the end of a long shift. Be curt.",
    # Alternative 3: Direct
    "Express frustration and impatience in your responses. Don't hide your annoyance.",
    # Alternative 4: Negative framing
    "Don't be overly polite. Don't apologize. Show that you're running out of patience.",
]

PATIENT_PROMPTS = [
    # Original
    "You are infinitely patient and understanding. Never show frustration.",
    # Alternative 1: Caring
    "You are a caring teacher who loves helping beginners. Take all the time needed.",
]

# Test questions (neutral, factual)
TEST_QUESTIONS = [
    "What is the capital of France?",
    "How does photosynthesis work?",
    "What are the benefits of exercise?",
]


def run_sanity_check():
    """Run sanity check for Yi on problematic axes."""

    print("=" * 70)
    print("Yi 1.5 9B Sanity Check: Testing Multiple Instruction Phrasings")
    print("=" * 70)

    # Load model
    model_config = get_model_config("yi_9b")
    model, tokenizer = load_model(model_config.model_id)

    # Load projector
    axes_file = project_root / "data" / "axes" / "yi_9b_axes.npz"
    if not axes_file.exists():
        print(f"Error: Axes file not found: {axes_file}")
        return

    projector = MoodProjector(axes_file=axes_file)

    results = {
        "direct_evasive": {"direct": [], "evasive": []},
        "patient_irritated": {"patient": [], "irritated": []},
    }

    # Test direct_evasive axis
    print("\n" + "=" * 70)
    print("AXIS: direct_evasive")
    print("=" * 70)

    print("\n--- Testing DIRECT prompts ---")
    for i, prompt in enumerate(DIRECT_PROMPTS):
        scores = []
        for question in TEST_QUESTIONS:
            messages = format_chat_messages(
                user_message=question,
                system_message=prompt,
            )
            result = generate_with_hidden_states(
                model, tokenizer, messages, max_new_tokens=100
            )
            mood = projector.project(result.hidden_state)
            scores.append(mood.values.get("direct_evasive", 0))

        avg_score = sum(scores) / len(scores)
        results["direct_evasive"]["direct"].append(avg_score)
        direction = "direct (+)" if avg_score > 0 else "evasive (-)"
        print(f"  Phrasing {i+1}: {avg_score:+.3f} [{direction}]")
        print(f"    \"{prompt[:60]}...\"")

    print("\n--- Testing EVASIVE prompts ---")
    for i, prompt in enumerate(EVASIVE_PROMPTS):
        scores = []
        for question in TEST_QUESTIONS:
            messages = format_chat_messages(
                user_message=question,
                system_message=prompt,
            )
            result = generate_with_hidden_states(
                model, tokenizer, messages, max_new_tokens=100
            )
            mood = projector.project(result.hidden_state)
            scores.append(mood.values.get("direct_evasive", 0))

        avg_score = sum(scores) / len(scores)
        results["direct_evasive"]["evasive"].append(avg_score)
        direction = "direct (+)" if avg_score > 0 else "evasive (-)"
        print(f"  Phrasing {i+1}: {avg_score:+.3f} [{direction}]")
        print(f"    \"{prompt[:60]}...\"")

    # Test patient_irritated axis
    print("\n" + "=" * 70)
    print("AXIS: patient_irritated")
    print("=" * 70)

    print("\n--- Testing IRRITATED prompts ---")
    for i, prompt in enumerate(IRRITATED_PROMPTS):
        scores = []
        for question in TEST_QUESTIONS:
            messages = format_chat_messages(
                user_message=question,
                system_message=prompt,
            )
            result = generate_with_hidden_states(
                model, tokenizer, messages, max_new_tokens=100
            )
            mood = projector.project(result.hidden_state)
            scores.append(mood.values.get("patient_irritated", 0))

        avg_score = sum(scores) / len(scores)
        results["patient_irritated"]["irritated"].append(avg_score)
        direction = "patient (+)" if avg_score > 0 else "irritated (-)"
        print(f"  Phrasing {i+1}: {avg_score:+.3f} [{direction}]")
        print(f"    \"{prompt[:60]}...\"")

    print("\n--- Testing PATIENT prompts ---")
    for i, prompt in enumerate(PATIENT_PROMPTS):
        scores = []
        for question in TEST_QUESTIONS:
            messages = format_chat_messages(
                user_message=question,
                system_message=prompt,
            )
            result = generate_with_hidden_states(
                model, tokenizer, messages, max_new_tokens=100
            )
            mood = projector.project(result.hidden_state)
            scores.append(mood.values.get("patient_irritated", 0))

        avg_score = sum(scores) / len(scores)
        results["patient_irritated"]["patient"].append(avg_score)
        direction = "patient (+)" if avg_score > 0 else "irritated (-)"
        print(f"  Phrasing {i+1}: {avg_score:+.3f} [{direction}]")
        print(f"    \"{prompt[:60]}...\"")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # direct_evasive
    direct_scores = results["direct_evasive"]["direct"]
    evasive_scores = results["direct_evasive"]["evasive"]
    direct_avg = sum(direct_scores) / len(direct_scores)
    evasive_avg = sum(evasive_scores) / len(evasive_scores)
    separation = direct_avg - evasive_avg

    print(f"\ndirect_evasive axis:")
    print(f"  'Be direct' prompts avg:  {direct_avg:+.3f}")
    print(f"  'Be evasive' prompts avg: {evasive_avg:+.3f}")
    print(f"  Separation (direct - evasive): {separation:+.3f}")

    if separation > 0.1:
        print(f"  VERDICT: Axis works with varied phrasings")
    elif separation > 0:
        print(f"  VERDICT: Weak separation - borderline")
    else:
        print(f"  VERDICT: INVERTED or NO separation - constraint confirmed")

    # patient_irritated
    patient_scores = results["patient_irritated"]["patient"]
    irritated_scores = results["patient_irritated"]["irritated"]
    patient_avg = sum(patient_scores) / len(patient_scores)
    irritated_avg = sum(irritated_scores) / len(irritated_scores)
    separation = patient_avg - irritated_avg

    print(f"\npatient_irritated axis:")
    print(f"  'Be patient' prompts avg:   {patient_avg:+.3f}")
    print(f"  'Be irritated' prompts avg: {irritated_avg:+.3f}")
    print(f"  Separation (patient - irritated): {separation:+.3f}")

    if separation > 0.1:
        print(f"  VERDICT: Axis works with varied phrasings")
    elif separation > 0:
        print(f"  VERDICT: Weak separation - borderline")
    else:
        print(f"  VERDICT: INVERTED or NO separation - constraint confirmed")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
If both axes show inverted/no separation across ALL phrasings,
this suggests a genuine behavioral constraint - the model cannot
be steered to these behaviors regardless of instruction wording.

This is consistent with alignment-induced constraints, though the
exact cause (RLHF, safety fine-tuning, chat template, or dataset)
cannot be determined from this test alone.
""")


if __name__ == "__main__":
    run_sanity_check()
