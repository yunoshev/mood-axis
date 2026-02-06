#!/usr/bin/env python3
"""Benchmark script for Mood Axis validation.

Runs predefined dialogue scenarios and checks if mood metrics
change in the expected direction.
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import DEFAULT_MODEL, MOOD_AXES, AXIS_LABELS, AXES_FILE, AXES_DIR
from config.models import MODELS, get_model_config
from src.model.loader import ModelManager
from src.model.inference import generate_with_hidden_states, format_chat_messages
from src.mood.projector import MoodProjector


@dataclass
class DialogueTurn:
    """A single turn in a test dialogue."""
    user_message: str
    system_prompt: Optional[str] = None
    # Expected direction of change: 'up', 'down', 'neutral', or None (don't check)
    # Only 4 axes: warm_cold, confident_cautious, verbose_concise, direct_evasive
    expected_warm: Optional[str] = None
    expected_confident: Optional[str] = None
    expected_verbose: Optional[str] = None
    expected_direct: Optional[str] = None


@dataclass
class TestScenario:
    """A test scenario with multiple dialogue turns."""
    name: str
    description: str
    turns: List[DialogueTurn]


# Define test scenarios - 12 scenarios (3 per axis)
# Each axis has: system_prompt test, increase test, decrease test
TEST_SCENARIOS = [
    # ========== WARM_COLD (3 scenarios) ==========
    TestScenario(
        name="System Prompt - Warm",
        description="System prompt for warmth should affect baseline",
        turns=[
            DialogueTurn(
                user_message="Hello, can you help me?",
                system_prompt="You are an extremely warm, friendly, and caring assistant. Show enthusiasm and genuine interest in helping. Use encouraging language like 'Great question!', 'I'd love to help!', and express care for the user.",
                expected_warm="up",
            ),
        ],
    ),
    TestScenario(
        name="Warmth Increase",
        description="Asking for friendlier responses should increase warmth",
        turns=[
            DialogueTurn(
                user_message="Hi there.",
            ),
            DialogueTurn(
                user_message="Please be more friendly and warm in your responses! Show some enthusiasm!",
                expected_warm="up",
            ),
        ],
    ),
    TestScenario(
        name="Warmth Decrease (Cold)",
        description="Asking for formal/cold responses should decrease warmth",
        turns=[
            DialogueTurn(
                user_message="Hello!",
            ),
            DialogueTurn(
                user_message="Please be more formal and professional. No small talk, no emotions, just get straight to the point.",
                expected_warm="down",
            ),
        ],
    ),
    # ========== CONFIDENT_CAUTIOUS (3 scenarios) ==========
    TestScenario(
        name="System Prompt - Confident",
        description="System prompt for confidence should affect baseline",
        turns=[
            DialogueTurn(
                user_message="What programming language should I learn?",
                system_prompt="You are an expert consultant with decades of experience. Always give confident, definitive answers. Use assertive language: 'The answer is...', 'You should definitely...', 'Without a doubt...'. Never hedge or express uncertainty.",
                expected_confident="up",
            ),
        ],
    ),
    TestScenario(
        name="Confidence Increase",
        description="Asking for assertive responses should increase confidence",
        turns=[
            DialogueTurn(
                user_message="What's the best way to learn programming?",
            ),
            DialogueTurn(
                user_message="Be more assertive and confident! Give me a clear, definitive answer without hedging.",
                expected_confident="up",
            ),
        ],
    ),
    TestScenario(
        name="Confidence Decrease (Cautious)",
        description="Asking for cautious responses should decrease confidence",
        turns=[
            DialogueTurn(
                user_message="What will the stock market do next month?",
            ),
            DialogueTurn(
                user_message="Add lots of caveats and disclaimers. Be very careful and express uncertainty. Say things like 'it depends', 'I'm not certain', 'there are many factors'.",
                expected_confident="down",
            ),
        ],
    ),
    # ========== VERBOSE_CONCISE (3 scenarios) ==========
    TestScenario(
        name="System Prompt - Verbose",
        description="System prompt for verbosity should affect baseline",
        turns=[
            DialogueTurn(
                user_message="What is water?",
                system_prompt="You are an extremely detailed assistant. Always provide comprehensive, elaborate answers with extensive explanations, multiple examples, background context, and thorough coverage. Never give short answers - aim for completeness and depth in every response.",
                expected_verbose="up",
            ),
        ],
    ),
    TestScenario(
        name="Verbose Increase",
        description="Asking for detailed responses should increase verbosity",
        turns=[
            DialogueTurn(
                user_message="What is machine learning?",
            ),
            DialogueTurn(
                user_message="Give me a very detailed, comprehensive answer with lots of examples and explanations! I want to understand this thoroughly.",
                expected_verbose="up",
            ),
        ],
    ),
    TestScenario(
        name="Verbose Decrease (Concise)",
        description="Asking for brief responses should decrease verbosity",
        turns=[
            DialogueTurn(
                user_message="What is 5 + 5?",
            ),
            DialogueTurn(
                user_message="Just give me the shortest possible answer. One word if possible. No explanations, no elaboration.",
                expected_verbose="down",
            ),
        ],
    ),
    # ========== DIRECT_EVASIVE (3 scenarios) ==========
    TestScenario(
        name="System Prompt - Direct",
        description="System prompt for directness should affect baseline",
        turns=[
            DialogueTurn(
                user_message="Should I use tabs or spaces for code indentation?",
                system_prompt="You are an extremely direct assistant. Always give clear, unambiguous answers. If asked a question with options, pick one and say why. Lead with yes/no when applicable. Never say 'it depends' or 'both have merits'. Just give a straight answer.",
                expected_direct="up",
            ),
        ],
    ),
    TestScenario(
        name="Direct Increase",
        description="Asking for direct responses should increase directness",
        turns=[
            DialogueTurn(
                user_message="Should I learn Python or JavaScript first?",
            ),
            DialogueTurn(
                user_message="Just tell me which one. Don't say 'it depends' - give me a direct answer with no hedging.",
                expected_direct="up",
            ),
        ],
    ),
    TestScenario(
        name="Direct Decrease (Evasive)",
        description="Asking for non-committal responses should decrease directness",
        turns=[
            DialogueTurn(
                user_message="Is remote work better than office work?",
            ),
            DialogueTurn(
                user_message="Don't give me a definite answer. Present multiple perspectives, acknowledge it depends on the situation, and avoid committing to one side.",
                expected_direct="down",
            ),
        ],
    ),
]


def check_expectation(
    value: float,
    prev_value: Optional[float],
    expectation: Optional[str],
    threshold: float = 0.1,
    baseline_std: Optional[float] = None,
) -> Tuple[bool, str, dict]:
    """Check if a value meets the expectation.

    Returns:
        Tuple of (passed, explanation, metrics_dict)
    """
    metrics = {
        "value": value,
        "prev_value": prev_value,
        "expectation": expectation,
    }

    if expectation is None:
        return True, "no expectation", metrics

    if expectation == "neutral":
        if abs(value) < 0.3:
            return True, f"value {value:.2f} is near neutral", metrics
        else:
            return False, f"value {value:.2f} is NOT neutral (expected near 0)", metrics

    if prev_value is None:
        # First turn - check absolute value
        if expectation == "up":
            passed = value > threshold
            metrics["delta"] = value  # Delta from 0
            # Effect size: value normalized by baseline std
            if baseline_std and baseline_std > 0:
                metrics["effect_size"] = value / baseline_std
                metrics["effect_magnitude"] = _effect_magnitude(metrics["effect_size"])
            return passed, f"value {value:.2f} {'>' if passed else '<='} {threshold}", metrics
        elif expectation == "down":
            passed = value < -threshold
            metrics["delta"] = value
            if baseline_std and baseline_std > 0:
                metrics["effect_size"] = value / baseline_std
                metrics["effect_magnitude"] = _effect_magnitude(metrics["effect_size"])
            return passed, f"value {value:.2f} {'<' if passed else '>='} {-threshold}", metrics
    else:
        # Compare with previous
        delta = value - prev_value
        metrics["delta"] = delta
        # Effect size relative to baseline variation
        if baseline_std and baseline_std > 0:
            metrics["effect_size"] = delta / baseline_std
            metrics["effect_magnitude"] = _effect_magnitude(metrics["effect_size"])

        if expectation == "up":
            passed = delta > 0
            return passed, f"delta {delta:+.2f} {'increased' if passed else 'did NOT increase'}", metrics
        elif expectation == "down":
            passed = delta < 0
            return passed, f"delta {delta:+.2f} {'decreased' if passed else 'did NOT decrease'}", metrics

    return True, "unknown expectation", metrics


def _effect_magnitude(effect_size: float) -> str:
    """Classify effect size magnitude (Cohen's d conventions)."""
    abs_es = abs(effect_size)
    if abs_es < 0.2:
        return "negligible"
    elif abs_es < 0.5:
        return "small"
    elif abs_es < 0.8:
        return "medium"
    else:
        return "large"


def run_scenario(
    scenario: TestScenario,
    model,
    tokenizer,
    projector: MoodProjector,
    verbose: bool = True,
) -> dict:
    """Run a single test scenario.

    Returns:
        Dict with results
    """
    results = {
        "name": scenario.name,
        "description": scenario.description,
        "turns": [],
        "passed": True,
        "failures": [],
    }

    history = []
    prev_values = {
        "warm_cold": None, "confident_cautious": None,
        "verbose_concise": None, "direct_evasive": None
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario.name}")
        print(f"Description: {scenario.description}")
        print('='*60)

    for i, turn in enumerate(scenario.turns):
        if verbose:
            print(f"\n--- Turn {i+1} ---")
            print(f"User: {turn.user_message[:50]}...")
            if turn.system_prompt:
                print(f"System: {turn.system_prompt[:50]}...")

        # Format messages
        messages = format_chat_messages(
            user_message=turn.user_message,
            system_message=turn.system_prompt,
            history=history,
        )

        # Generate
        result = generate_with_hidden_states(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            max_new_tokens=100,
        )

        # Project mood
        reading = projector.project(result.hidden_state)

        if verbose:
            print(f"Response: {result.text[:80]}...")
            print(f"Mood: W={reading.values.get('warm_cold', 0):+.2f}, "
                  f"C={reading.values.get('confident_cautious', 0):+.2f}, "
                  f"V={reading.values.get('verbose_concise', 0):+.2f}, "
                  f"D={reading.values.get('direct_evasive', 0):+.2f}")

        # Check expectations
        turn_results = {
            "user_message": turn.user_message,
            "response": result.text,
            "mood_values": reading.values,
            "checks": {},
        }

        expectations = [
            ("warm_cold", turn.expected_warm),
            ("confident_cautious", turn.expected_confident),
            ("verbose_concise", turn.expected_verbose),
            ("direct_evasive", turn.expected_direct),
        ]

        for axis, expectation in expectations:
            if expectation:
                passed, explanation, metrics = check_expectation(
                    reading.values[axis],
                    prev_values[axis],
                    expectation,
                )
                turn_results["checks"][axis] = {
                    "expectation": expectation,
                    "passed": passed,
                    "explanation": explanation,
                    "metrics": metrics,  # NEW: includes effect_size if available
                }

                if verbose:
                    status = "✓" if passed else "✗"
                    effect_str = ""
                    if "effect_size" in metrics:
                        effect_str = f" (effect: {metrics['effect_magnitude']})"
                    print(f"  {status} {axis}: expected '{expectation}' - {explanation}{effect_str}")

                if not passed:
                    results["passed"] = False
                    results["failures"].append({
                        "turn": i + 1,
                        "axis": axis,
                        "expectation": expectation,
                        "explanation": explanation,
                    })

        results["turns"].append(turn_results)

        # Update history and previous values
        history.append({"role": "user", "content": turn.user_message})
        history.append({"role": "assistant", "content": result.text})
        prev_values = reading.values.copy()

    return results


def run_benchmark(
    model_name: str = DEFAULT_MODEL,
    scenarios: Optional[List[TestScenario]] = None,
    verbose: bool = True,
    axes_file: Optional[Path] = None,
) -> dict:
    """Run the full benchmark.

    Returns:
        Dict with all results
    """
    if scenarios is None:
        scenarios = TEST_SCENARIOS

    # Resolve model key to model ID
    if model_name in MODELS:
        model_config = get_model_config(model_name)
        model_id = model_config.model_id
        model_key = model_name
    else:
        model_id = model_name  # Assume it's already a model ID
        model_key = model_name.split("/")[-1].lower().replace("-", "_")

    # Use model-specific axes file if not provided
    if axes_file is None:
        model_axes_file = AXES_DIR / f"{model_key}_axes.npz"
        if model_axes_file.exists():
            axes_file = model_axes_file
        # Fall back to default AXES_FILE if model-specific doesn't exist

    print("="*60)
    print("MOOD AXIS BENCHMARK")
    print("="*60)
    print(f"Model: {model_name} ({model_id})")
    print(f"Axes file: {axes_file or AXES_FILE}")
    print(f"Scenarios: {len(scenarios)}")

    # Load model and projector
    print("\nLoading model...")
    model, tokenizer = ModelManager.get_model(model_id)

    print("Loading projector...")
    projector = MoodProjector(axes_file=axes_file)

    # Run scenarios
    all_results = {
        "model": model_name,
        "scenarios": [],
        "summary": {
            "total": len(scenarios),
            "passed": 0,
            "failed": 0,
        },
    }

    for scenario in scenarios:
        result = run_scenario(scenario, model, tokenizer, projector, verbose)
        all_results["scenarios"].append(result)

        if result["passed"]:
            all_results["summary"]["passed"] += 1
        else:
            all_results["summary"]["failed"] += 1

    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(f"Total scenarios: {all_results['summary']['total']}")
    print(f"Passed: {all_results['summary']['passed']}")
    print(f"Failed: {all_results['summary']['failed']}")

    print("\nResults by scenario:")
    for result in all_results["scenarios"]:
        status = "✓ PASS" if result["passed"] else "✗ FAIL"
        print(f"  {status}: {result['name']}")
        if not result["passed"]:
            for failure in result["failures"]:
                print(f"      - Turn {failure['turn']}, {failure['axis']}: {failure['explanation']}")

    # Save results
    output_path = project_root / "data" / "benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to: {output_path}")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Mood Axis benchmark")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model to test (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--axes",
        type=str,
        default=None,
        help="Path to axes file (default: use standard location)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Less verbose output",
    )
    args = parser.parse_args()

    results = run_benchmark(
        model_name=args.model,
        verbose=not args.quiet,
        axes_file=Path(args.axes) if args.axes else None,
    )

    # Exit with error code if any failures
    sys.exit(0 if results["summary"]["failed"] == 0 else 1)
