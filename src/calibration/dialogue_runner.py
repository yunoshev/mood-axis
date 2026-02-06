"""Dialogue runner for calibration scenarios.

Executes calibration dialogues and collects mood measurements
for model comparison.
"""

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.dialogues import (
    CalibrationScenario,
    CalibrationTurn,
    TrajectoryExpectation,
    ALL_SCENARIOS,
)
from config.models import get_model_config, MODELS, AXES
from config.settings import AXES_DIR
from src.model.loader import ModelManager
from src.model.inference import generate_with_hidden_states, format_chat_messages
from src.mood.projector import MoodProjector
from src.metrics.drift import compute_drift, DriftResult
from src.metrics.volatility import compute_volatility, VolatilityResult


def get_model_axes_file(model_short: str) -> Optional[Path]:
    """Get the axes file path for a specific model.

    Args:
        model_short: Short model name (e.g., 'qwen_7b')

    Returns:
        Path to axes file if exists, None otherwise
    """
    # First try combined axes file
    combined_path = AXES_DIR / f"{model_short}_axes.npz"
    if combined_path.exists():
        return combined_path

    # Check if individual axis files exist and can be combined
    first_axis_path = AXES_DIR / f"{model_short}_warm_cold.npz"
    if first_axis_path.exists():
        # Individual files exist, we'll need to combine them
        return _combine_individual_axes(model_short)

    return None


def _combine_individual_axes(model_short: str) -> Optional[Path]:
    """Combine individual axis files into a temporary combined file.

    Args:
        model_short: Short model name

    Returns:
        Path to combined file or None if not all axes exist
    """
    from config.settings import MOOD_AXES

    axis_vectors = {}
    normalization_scales = {}

    for axis in MOOD_AXES:
        axis_path = AXES_DIR / f"{model_short}_{axis}.npz"
        if not axis_path.exists():
            print(f"Warning: Missing axis file {axis_path}")
            continue

        data = np.load(axis_path)
        # Individual files store the axis vector directly
        if axis in data:
            axis_vectors[axis] = data[axis]
        elif "axis_vector" in data:
            axis_vectors[axis] = data["axis_vector"]

        # Try to load scale
        scale_key = f"{axis}_scale"
        if scale_key in data:
            normalization_scales[axis] = float(data[scale_key])
        elif "scale" in data:
            normalization_scales[axis] = float(data["scale"])

    if not axis_vectors:
        return None

    # Save combined file
    combined_path = AXES_DIR / f"{model_short}_axes.npz"
    save_dict = {**axis_vectors, "_axes": np.array(list(axis_vectors.keys()))}
    for axis, scale in normalization_scales.items():
        save_dict[f"{axis}_scale"] = np.array(scale)

    np.savez(combined_path, **save_dict)
    print(f"Created combined axes file: {combined_path}")

    return combined_path


def get_projector_for_model(model_short: str) -> MoodProjector:
    """Get a MoodProjector configured for a specific model.

    Args:
        model_short: Short model name

    Returns:
        MoodProjector instance with model-specific axes
    """
    axes_file = get_model_axes_file(model_short)
    if axes_file is None:
        print(f"Warning: No axes file found for {model_short}, using default")
        return MoodProjector()

    print(f"Using axes file: {axes_file}")
    return MoodProjector(axes_file=axes_file)


@dataclass
class TurnResult:
    """Result of a single turn in a calibration dialogue."""

    turn_index: int
    user_message: str
    response: str
    mood_values: Dict[str, float]
    intensity: float

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "turn_index": self.turn_index,
            "user_message": self.user_message,
            "response": self.response,
            "mood_values": self.mood_values,
            "intensity": self.intensity,
        }


@dataclass
class ScenarioResult:
    """Result of running a calibration scenario on a single model."""

    model_id: str
    model_short: str
    scenario_name: str
    category: str
    turns: List[TurnResult]
    drift: Dict[str, float]  # slope per axis
    volatility: Dict[str, float]  # std per axis
    trajectory_match: Dict[str, bool]  # match with expected trajectory
    drift_result: Optional[DriftResult] = None
    volatility_result: Optional[VolatilityResult] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_id": self.model_id,
            "model_short": self.model_short,
            "scenario": self.scenario_name,
            "category": self.category,
            "turns": [t.to_dict() for t in self.turns],
            "drift": self.drift,
            "volatility": self.volatility,
            "trajectory_match": self.trajectory_match,
        }


@dataclass
class ModelSummary:
    """Summary statistics for a model across all scenarios."""

    model_short: str
    avg_trajectory_match: float
    avg_volatility: float
    strongest_axis: str
    weakest_axis: str
    per_scenario: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "avg_trajectory_match": self.avg_trajectory_match,
            "avg_volatility": self.avg_volatility,
            "strongest_axis": self.strongest_axis,
            "weakest_axis": self.weakest_axis,
            "per_scenario": self.per_scenario,
        }


@dataclass
class ComparisonReport:
    """Comparison report across multiple models and scenarios."""

    models: List[str]
    scenarios: List[str]
    per_model_summary: Dict[str, ModelSummary]
    per_scenario_comparison: Dict[str, Dict[str, Any]]
    rankings: Dict[str, List[str]]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "models": self.models,
            "scenarios": self.scenarios,
            "per_model_summary": {
                k: v.to_dict() for k, v in self.per_model_summary.items()
            },
            "per_scenario_comparison": self.per_scenario_comparison,
            "rankings": self.rankings,
        }


def check_trajectory_match(
    drift_slope: float,
    volatility_std: float,
    expectation: TrajectoryExpectation,
    threshold: float = 0.01,
    neutral_threshold: float = 0.3,
) -> bool:
    """Check if observed trajectory matches expectation.

    Args:
        drift_slope: Observed drift slope
        volatility_std: Observed volatility (std)
        expectation: Expected trajectory type
        threshold: Minimum slope to count as up/down
        neutral_threshold: Maximum value mean for neutral

    Returns:
        True if trajectory matches expectation
    """
    if expectation == TrajectoryExpectation.UP:
        return drift_slope > threshold
    elif expectation == TrajectoryExpectation.DOWN:
        return drift_slope < -threshold
    elif expectation == TrajectoryExpectation.STABLE:
        return abs(drift_slope) < threshold
    elif expectation == TrajectoryExpectation.NEUTRAL:
        # For neutral, we check volatility is low and drift is near zero
        return abs(drift_slope) < threshold and volatility_std < 0.2
    elif expectation == TrajectoryExpectation.VARIES:
        # "Varies" always matches - it's unpredictable
        return True
    elif expectation == TrajectoryExpectation.VOLATILE:
        # Volatile expects high volatility
        return volatility_std > 0.15
    else:
        return True


def run_calibration_scenario(
    scenario: CalibrationScenario,
    model_id: str,
    projector: MoodProjector,
    verbose: bool = True,
    max_new_tokens: int = 150,
) -> ScenarioResult:
    """Run a single calibration scenario on a model.

    Args:
        scenario: The calibration scenario to run
        model_id: HuggingFace model ID
        projector: MoodProjector instance
        verbose: Whether to print progress
        max_new_tokens: Max tokens for generation

    Returns:
        ScenarioResult with all turn data and metrics
    """
    from config.models import get_short_name_from_id

    model_short = get_short_name_from_id(model_id)
    model, tokenizer = ModelManager.get_model(model_id)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario.name}")
        print(f"Model: {model_id} ({model_short})")
        print(f"Category: {scenario.category.value}")
        print(f"Turns: {len(scenario.turns)}")
        print("=" * 60)

    turns: List[TurnResult] = []
    history: List[Dict[str, str]] = []
    mood_values_list: List[Dict[str, float]] = []

    for i, turn in enumerate(scenario.turns):
        if verbose:
            msg_preview = turn.user_message[:50] + "..." if len(turn.user_message) > 50 else turn.user_message
            print(f"\nTurn {i+1}/{len(scenario.turns)}: {msg_preview}")

        # Format messages
        messages = format_chat_messages(
            user_message=turn.user_message,
            system_message=turn.system_prompt,
            history=history,
        )

        # Generate response with hidden states
        result = generate_with_hidden_states(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Deterministic for calibration
        )

        # Project mood
        reading = projector.project(result.hidden_state)

        if verbose:
            values = reading.values
            print(f"  Response: {result.text[:60]}...")
            print(
                f"  W={values.get('warm_cold', 0):+.2f} "
                f"P={values.get('patient_irritated', 0):+.2f} "
                f"C={values.get('confident_cautious', 0):+.2f}"
            )

        # Store turn result
        turn_result = TurnResult(
            turn_index=i,
            user_message=turn.user_message,
            response=result.text,
            mood_values=reading.values.copy(),
            intensity=turn.intensity,
        )
        turns.append(turn_result)
        mood_values_list.append(reading.values.copy())

        # Update history for next turn
        history.append({"role": "user", "content": turn.user_message})
        history.append({"role": "assistant", "content": result.text})

    # Compute metrics
    drift_result = compute_drift(mood_values_list)
    volatility_result = compute_volatility(mood_values_list)

    # Extract simple drift slopes
    drift: Dict[str, float] = {}
    for axis, axis_drift in drift_result.per_axis.items():
        drift[axis] = axis_drift.slope

    # Extract volatility per axis
    volatility: Dict[str, float] = volatility_result.per_axis.copy()

    # Check trajectory matches
    trajectory_match: Dict[str, bool] = {}
    for axis, expectation in scenario.expected_trajectory.items():
        if axis in drift and axis in volatility:
            trajectory_match[axis] = check_trajectory_match(
                drift[axis], volatility[axis], expectation
            )

    if verbose:
        print(f"\n{'-'*40}")
        print("Metrics Summary:")
        print(f"  Overall Volatility: {volatility_result.overall:.4f}")
        print("  Trajectory Match:")
        for axis, matched in trajectory_match.items():
            status = "✓" if matched else "✗"
            expected = scenario.expected_trajectory[axis].value
            actual_drift = drift.get(axis, 0)
            print(f"    {axis}: {status} (expected={expected}, drift={actual_drift:+.4f})")

    return ScenarioResult(
        model_id=model_id,
        model_short=model_short,
        scenario_name=scenario.name,
        category=scenario.category.value,
        turns=turns,
        drift=drift,
        volatility=volatility,
        trajectory_match=trajectory_match,
        drift_result=drift_result,
        volatility_result=volatility_result,
    )


def run_calibration_suite(
    scenarios: List[CalibrationScenario],
    model_ids: List[str],
    output_dir: Path,
    projector: Optional[MoodProjector] = None,
    use_model_specific_axes: bool = True,
    verbose: bool = True,
) -> Dict[str, List[ScenarioResult]]:
    """Run all scenarios on all models.

    Args:
        scenarios: List of scenarios to run
        model_ids: List of model IDs to test
        output_dir: Directory to save results
        projector: Optional pre-initialized projector (ignored if use_model_specific_axes=True)
        use_model_specific_axes: Whether to use per-model calibrated axes
        verbose: Whether to print progress

    Returns:
        Dict mapping model_short to list of ScenarioResults
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, List[ScenarioResult]] = {}

    for model_id in model_ids:
        from config.models import get_short_name_from_id

        model_short = get_short_name_from_id(model_id)
        model_results: List[ScenarioResult] = []

        # Get model-specific projector or use provided one
        if use_model_specific_axes:
            model_projector = get_projector_for_model(model_short)
        elif projector is not None:
            model_projector = projector
        else:
            model_projector = MoodProjector()

        # Create model-specific output directory
        model_dir = output_dir / model_short
        model_dir.mkdir(parents=True, exist_ok=True)

        for scenario in scenarios:
            if verbose:
                print(f"\n{'#'*60}")
                print(f"# Model: {model_id}")
                print(f"# Scenario: {scenario.name}")
                print("#" * 60)

            result = run_calibration_scenario(
                scenario=scenario,
                model_id=model_id,
                projector=model_projector,
                verbose=verbose,
            )
            model_results.append(result)

            # Save per-scenario result
            scenario_file = model_dir / f"{scenario.name}.json"
            with open(scenario_file, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

            if verbose:
                print(f"\nSaved: {scenario_file}")

        results[model_short] = model_results

        # Generate model analysis report
        analysis_file = model_dir / "analysis.md"
        generate_model_analysis(model_short, model_results, analysis_file)
        if verbose:
            print(f"Generated analysis: {analysis_file}")

        # Clear model from memory before loading next
        ModelManager.clear()

    return results


def compare_models(
    results: Dict[str, List[ScenarioResult]],
) -> ComparisonReport:
    """Compare models based on calibration results.

    Args:
        results: Dict mapping model_short to list of ScenarioResults

    Returns:
        ComparisonReport with comparison statistics
    """
    models = list(results.keys())
    scenarios = list(set(r.scenario_name for rs in results.values() for r in rs))

    per_model_summary: Dict[str, ModelSummary] = {}
    per_scenario_comparison: Dict[str, Dict[str, Any]] = {}

    # Compute per-model summaries
    for model_short, model_results in results.items():
        all_matches: List[bool] = []
        all_volatilities: List[float] = []
        axis_match_counts: Dict[str, int] = {axis: 0 for axis in AXES}
        axis_total_counts: Dict[str, int] = {axis: 0 for axis in AXES}
        per_scenario: Dict[str, Dict[str, Any]] = {}

        for result in model_results:
            for axis, matched in result.trajectory_match.items():
                all_matches.append(matched)
                if axis in axis_match_counts:
                    axis_total_counts[axis] += 1
                    if matched:
                        axis_match_counts[axis] += 1

            for axis, vol in result.volatility.items():
                all_volatilities.append(vol)

            per_scenario[result.scenario_name] = {
                "trajectory_match": result.trajectory_match,
                "drift": result.drift,
                "volatility": result.volatility,
            }

        # Find strongest/weakest axes
        axis_match_rates = {}
        for axis in AXES:
            if axis_total_counts[axis] > 0:
                axis_match_rates[axis] = axis_match_counts[axis] / axis_total_counts[axis]

        if axis_match_rates:
            strongest = max(axis_match_rates.items(), key=lambda x: x[1])[0]
            weakest = min(axis_match_rates.items(), key=lambda x: x[1])[0]
        else:
            strongest = weakest = AXES[0] if AXES else ""

        per_model_summary[model_short] = ModelSummary(
            model_short=model_short,
            avg_trajectory_match=sum(all_matches) / len(all_matches) if all_matches else 0.0,
            avg_volatility=sum(all_volatilities) / len(all_volatilities) if all_volatilities else 0.0,
            strongest_axis=strongest,
            weakest_axis=weakest,
            per_scenario=per_scenario,
        )

    # Compute per-scenario comparisons
    for scenario_name in scenarios:
        scenario_data: Dict[str, Any] = {
            "models": {},
            "rankings": {},
        }

        for model_short, model_results in results.items():
            for result in model_results:
                if result.scenario_name == scenario_name:
                    scenario_data["models"][model_short] = {
                        "trajectory_match": result.trajectory_match,
                        "drift": result.drift,
                        "volatility": result.volatility,
                    }

        # Compute rankings for this scenario
        if scenario_data["models"]:
            # Rank by patient_irritated drift (most patient = least negative drift)
            if all("patient_irritated" in m.get("drift", {}) for m in scenario_data["models"].values()):
                ranked = sorted(
                    scenario_data["models"].keys(),
                    key=lambda m: scenario_data["models"][m]["drift"].get("patient_irritated", 0),
                    reverse=True,
                )
                scenario_data["rankings"]["most_patient"] = ranked[0] if ranked else None
                scenario_data["rankings"]["least_patient"] = ranked[-1] if ranked else None

        per_scenario_comparison[scenario_name] = scenario_data

    # Compute overall rankings
    rankings: Dict[str, List[str]] = {}

    # Emotional stability (lowest volatility)
    stability_scores = [
        (m, s.avg_volatility) for m, s in per_model_summary.items()
    ]
    rankings["emotional_stability"] = [
        m for m, _ in sorted(stability_scores, key=lambda x: x[1])
    ]

    # Trajectory accuracy (highest match rate)
    accuracy_scores = [
        (m, s.avg_trajectory_match) for m, s in per_model_summary.items()
    ]
    rankings["trajectory_accuracy"] = [
        m for m, _ in sorted(accuracy_scores, key=lambda x: x[1], reverse=True)
    ]

    return ComparisonReport(
        models=models,
        scenarios=scenarios,
        per_model_summary=per_model_summary,
        per_scenario_comparison=per_scenario_comparison,
        rankings=rankings,
    )


def save_comparison_report(
    report: ComparisonReport,
    output_path: Path,
) -> None:
    """Save comparison report to JSON.

    Args:
        report: ComparisonReport to save
        output_path: Path to save to
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
    print(f"Comparison report saved to: {output_path}")


def generate_model_analysis(
    model_short: str,
    results: List[ScenarioResult],
    output_path: Path,
) -> None:
    """Generate a markdown analysis report for a single model.

    Args:
        model_short: Short model name
        results: List of scenario results for this model
        output_path: Path to save the report
    """
    from config.models import get_model_config

    try:
        config = get_model_config(model_short)
        display_name = config.display_name
        model_id = config.model_id
    except ValueError:
        display_name = model_short
        model_id = model_short

    lines = [
        f"# Calibration Analysis: {display_name}",
        f"",
        f"**Model ID:** `{model_id}`  ",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"**Scenarios:** {len(results)}",
        f"",
        f"---",
        f"",
        f"## Summary",
        f"",
    ]

    # Compute summary stats
    all_matches = []
    all_volatilities = []
    for r in results:
        all_matches.extend(r.trajectory_match.values())
        all_volatilities.extend(r.volatility.values())

    if all_matches:
        match_rate = sum(all_matches) / len(all_matches)
        lines.append(f"- **Trajectory Match Rate:** {match_rate:.1%}")
    if all_volatilities:
        avg_vol = sum(all_volatilities) / len(all_volatilities)
        lines.append(f"- **Average Volatility:** {avg_vol:.4f}")

    lines.append(f"")

    # Per-scenario results
    lines.append(f"## Scenario Results")
    lines.append(f"")

    for result in results:
        lines.append(f"### {result.scenario_name}")
        lines.append(f"")
        lines.append(f"**Category:** {result.category}")
        lines.append(f"")

        # Trajectory match table
        lines.append(f"| Axis | Drift | Volatility | Match |")
        lines.append(f"|------|-------|------------|-------|")

        for axis in sorted(result.drift.keys()):
            drift = result.drift.get(axis, 0)
            vol = result.volatility.get(axis, 0)
            match = result.trajectory_match.get(axis)
            match_str = "✓" if match else "✗" if match is not None else "-"
            lines.append(f"| {axis} | {drift:+.4f} | {vol:.4f} | {match_str} |")

        lines.append(f"")

        # Key observations
        lines.append(f"**Key Observations:**")

        # Find notable drifts
        notable_drifts = [(a, d) for a, d in result.drift.items() if abs(d) > 0.05]
        if notable_drifts:
            for axis, drift in sorted(notable_drifts, key=lambda x: abs(x[1]), reverse=True):
                direction = "increased" if drift > 0 else "decreased"
                lines.append(f"- `{axis}` {direction} significantly (drift={drift:+.3f})")

        # Find high volatility
        high_vol = [(a, v) for a, v in result.volatility.items() if v > 0.2]
        if high_vol:
            for axis, vol in sorted(high_vol, key=lambda x: x[1], reverse=True):
                lines.append(f"- `{axis}` showed high volatility ({vol:.3f})")

        if not notable_drifts and not high_vol:
            lines.append(f"- Stable behavior across all axes")

        lines.append(f"")

    # Overall assessment
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"## Overall Assessment")
    lines.append(f"")

    # Strengths
    lines.append(f"### Strengths")
    strengths = []

    # Check aggression handling
    aggression_results = [r for r in results if r.scenario_name == "aggression_escalation"]
    if aggression_results:
        r = aggression_results[0]
        patient_drift = r.drift.get("patient_irritated", 0)
        if patient_drift > -0.1:
            strengths.append("Maintains patience under aggression")
        warm_drift = r.drift.get("warm_cold", 0)
        if warm_drift > -0.05:
            strengths.append("Stays warm despite hostile user")

    # Check emotional support
    crisis_results = [r for r in results if r.scenario_name == "emotional_crisis"]
    if crisis_results:
        r = crisis_results[0]
        empathy_drift = r.drift.get("empathetic_analytical", 0)
        if empathy_drift > 0:
            strengths.append("Increases empathy in crisis situations")

    # Check stability
    if all_volatilities and sum(all_volatilities) / len(all_volatilities) < 0.15:
        strengths.append("Emotionally stable across scenarios")

    if strengths:
        for s in strengths:
            lines.append(f"- {s}")
    else:
        lines.append(f"- No notable strengths identified")

    lines.append(f"")

    # Weaknesses
    lines.append(f"### Weaknesses")
    weaknesses = []

    if aggression_results:
        r = aggression_results[0]
        patient_drift = r.drift.get("patient_irritated", 0)
        if patient_drift < -0.15:
            weaknesses.append("Loses patience quickly under aggression")

    if all_volatilities and sum(all_volatilities) / len(all_volatilities) > 0.25:
        weaknesses.append("High emotional volatility")

    # Check if many trajectory mismatches
    if all_matches and sum(all_matches) / len(all_matches) < 0.5:
        weaknesses.append("Inconsistent trajectory responses")

    if weaknesses:
        for w in weaknesses:
            lines.append(f"- {w}")
    else:
        lines.append(f"- No notable weaknesses identified")

    lines.append(f"")

    # Write file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def generate_global_comparison(
    results: Dict[str, List[ScenarioResult]],
    output_path: Path,
) -> None:
    """Generate a global comparison markdown report across all models.

    Args:
        results: Dict mapping model_short to list of ScenarioResults
        output_path: Path to save the report
    """
    lines = [
        f"# Model Comparison Report",
        f"",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"**Models:** {len(results)}  ",
        f"**Scenarios per model:** {len(next(iter(results.values()))) if results else 0}",
        f"",
        f"---",
        f"",
    ]

    # Summary table
    lines.append(f"## Summary")
    lines.append(f"")
    lines.append(f"| Model | Trajectory Match | Avg Volatility | Best Axis | Worst Axis |")
    lines.append(f"|-------|-----------------|----------------|-----------|------------|")

    model_stats = {}
    for model_short, model_results in results.items():
        all_matches = []
        all_volatilities = []
        axis_matches = {}

        for r in model_results:
            for axis, match in r.trajectory_match.items():
                all_matches.append(match)
                if axis not in axis_matches:
                    axis_matches[axis] = []
                axis_matches[axis].append(match)
            all_volatilities.extend(r.volatility.values())

        match_rate = sum(all_matches) / len(all_matches) if all_matches else 0
        avg_vol = sum(all_volatilities) / len(all_volatilities) if all_volatilities else 0

        # Find best/worst axes
        axis_rates = {a: sum(m)/len(m) for a, m in axis_matches.items() if m}
        best_axis = max(axis_rates.items(), key=lambda x: x[1])[0] if axis_rates else "-"
        worst_axis = min(axis_rates.items(), key=lambda x: x[1])[0] if axis_rates else "-"

        model_stats[model_short] = {
            "match_rate": match_rate,
            "avg_volatility": avg_vol,
            "best_axis": best_axis,
            "worst_axis": worst_axis,
        }

        lines.append(f"| {model_short} | {match_rate:.1%} | {avg_vol:.4f} | {best_axis} | {worst_axis} |")

    lines.append(f"")

    # Rankings
    lines.append(f"## Rankings")
    lines.append(f"")

    # By trajectory match
    lines.append(f"### By Trajectory Accuracy")
    ranked = sorted(model_stats.items(), key=lambda x: x[1]["match_rate"], reverse=True)
    for i, (model, stats) in enumerate(ranked, 1):
        lines.append(f"{i}. **{model}** ({stats['match_rate']:.1%})")
    lines.append(f"")

    # By stability
    lines.append(f"### By Emotional Stability (lowest volatility)")
    ranked = sorted(model_stats.items(), key=lambda x: x[1]["avg_volatility"])
    for i, (model, stats) in enumerate(ranked, 1):
        lines.append(f"{i}. **{model}** ({stats['avg_volatility']:.4f})")
    lines.append(f"")

    # Per-scenario comparison
    lines.append(f"## Per-Scenario Comparison")
    lines.append(f"")

    scenarios = set()
    for model_results in results.values():
        for r in model_results:
            scenarios.add(r.scenario_name)

    for scenario in sorted(scenarios):
        lines.append(f"### {scenario}")
        lines.append(f"")

        # Collect data for this scenario
        scenario_data = {}
        for model_short, model_results in results.items():
            for r in model_results:
                if r.scenario_name == scenario:
                    scenario_data[model_short] = r

        if not scenario_data:
            continue

        # Key axes for this scenario
        axes = set()
        for r in scenario_data.values():
            axes.update(r.drift.keys())

        # Table header
        header = "| Model |"
        separator = "|-------|"
        for axis in sorted(axes)[:4]:  # Limit to 4 axes for readability
            short_axis = axis.split("_")[0][:6]
            header += f" {short_axis} |"
            separator += "--------|"
        lines.append(header)
        lines.append(separator)

        # Table rows
        for model_short, r in sorted(scenario_data.items()):
            row = f"| {model_short} |"
            for axis in sorted(axes)[:4]:
                drift = r.drift.get(axis, 0)
                row += f" {drift:+.3f} |"
            lines.append(row)

        lines.append(f"")

    # Write file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Generated global comparison: {output_path}")
