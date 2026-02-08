#!/usr/bin/env python3
"""Multi-seed benchmark replication for Mood Axis.

Runs the benchmark K times with different seeds per model,
then aggregates results with bootstrap CI and ICC reliability.

Usage:
    python scripts/replicate_benchmark.py --model qwen_7b --runs 5
    python scripts/replicate_benchmark.py --models article --runs 5
    python scripts/replicate_benchmark.py --models article --aggregate-only
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.benchmark import run_benchmark, TEST_SCENARIOS
from src.metrics.statistics import (
    bootstrap_ci,
    compute_icc,
    icc_interpretation,
    holm_bonferroni_correction,
)

ARTICLE_MODELS = ["qwen_7b", "mistral_7b", "deepseek_7b", "llama_8b", "yi_9b", "gemma_9b"]
DEFAULT_SEEDS = [42, 123, 456, 789, 1024]
OUTPUT_DIR = project_root / "data" / "article" / "benchmark_replication"


def run_replication(
    model_name: str,
    seeds: List[int],
    verbose: bool = False,
) -> dict:
    """Run benchmark K times with different seeds for one model.

    Returns:
        Dict with per-run results and raw data for aggregation.
    """
    runs = []
    for i, seed in enumerate(seeds):
        print(f"\n  Run {i+1}/{len(seeds)} (seed={seed})...")
        t0 = time.time()

        result = run_benchmark(
            model_name=model_name,
            verbose=verbose,
            seed=seed,
        )

        elapsed = time.time() - t0
        run_data = {
            "seed": seed,
            "elapsed_sec": round(elapsed, 1),
            "passed": result["summary"]["passed"],
            "failed": result["summary"]["failed"],
            "total": result["summary"]["total"],
            "accuracy": result["summary"]["passed"] / result["summary"]["total"],
            "scenarios": {},
        }

        # Extract per-scenario pass/fail and mood values
        for scenario_result in result["scenarios"]:
            name = scenario_result["name"]
            run_data["scenarios"][name] = {
                "passed": scenario_result["passed"],
                "mood_values": [],
                "checks": [],
            }
            for turn in scenario_result["turns"]:
                run_data["scenarios"][name]["mood_values"].append(turn["mood_values"])
                run_data["scenarios"][name]["checks"].append(turn.get("checks", {}))

        runs.append(run_data)
        print(f"    accuracy: {run_data['accuracy']:.1%} ({run_data['passed']}/{run_data['total']}) in {elapsed:.1f}s")

    return {
        "model": model_name,
        "n_runs": len(seeds),
        "seeds": seeds,
        "runs": runs,
    }


def aggregate_results(replication_data: dict) -> dict:
    """Aggregate multi-run results with bootstrap CI and ICC.

    Args:
        replication_data: Output from run_replication()

    Returns:
        Dict with aggregated statistics
    """
    runs = replication_data["runs"]
    n_runs = len(runs)

    # 1. Overall accuracy CI
    accuracies = [r["accuracy"] for r in runs]
    acc_ci = bootstrap_ci(accuracies, n_bootstrap=10000, random_state=42)

    # 2. Per-scenario pass rate and sign stability
    scenario_names = list(runs[0]["scenarios"].keys())
    scenario_stats = {}

    for name in scenario_names:
        passes = [1 if runs[i]["scenarios"][name]["passed"] else 0 for i in range(n_runs)]
        pass_rate = sum(passes) / n_runs

        scenario_stats[name] = {
            "pass_rate": pass_rate,
            "passes": passes,
            "n_runs": n_runs,
            "consistent": pass_rate == 0.0 or pass_rate == 1.0,
        }

    # 3. Per-scenario mood value ICC (test-retest reliability)
    # For each scenario, collect per-axis values across runs
    icc_results = {}
    for name in scenario_names:
        icc_results[name] = {}
        # Get the last turn's mood values from each run
        for axis in ["warm_cold", "confident_cautious", "verbose_concise", "direct_evasive",
                      "patient_irritated", "proactive_reluctant", "empathetic_analytical", "formal_casual"]:
            measurements = []
            for run in runs:
                scenario = run["scenarios"][name]
                # Use last turn's mood values
                if scenario["mood_values"]:
                    last_turn = scenario["mood_values"][-1]
                    if axis in last_turn:
                        measurements.append(last_turn[axis])

            if len(measurements) == n_runs:
                icc_results[name][axis] = {
                    "values": measurements,
                    "mean": float(np.mean(measurements)),
                    "std": float(np.std(measurements)),
                }

    # Compute ICC across scenarios for each axis (subjects=scenarios, raters=runs)
    axis_icc = {}
    for axis in ["warm_cold", "confident_cautious", "verbose_concise", "direct_evasive",
                  "patient_irritated", "proactive_reluctant", "empathetic_analytical", "formal_casual"]:
        # Build matrix: rows=scenarios, cols=runs
        measurement_matrix = []
        valid = True
        for name in scenario_names:
            if axis in icc_results.get(name, {}):
                measurement_matrix.append(icc_results[name][axis]["values"])
            else:
                valid = False
                break

        if valid and len(measurement_matrix) >= 2:
            # measurement_matrix is [n_scenarios][n_runs], transpose for ICC
            icc_val = compute_icc([list(row) for row in zip(*measurement_matrix)])
            axis_icc[axis] = {
                "icc": icc_val,
                "interpretation": icc_interpretation(icc_val),
            }

    # 4. Summary
    n_consistent = sum(1 for s in scenario_stats.values() if s["consistent"])
    n_scenarios = len(scenario_stats)

    return {
        "model": replication_data["model"],
        "n_runs": n_runs,
        "seeds": replication_data["seeds"],
        "accuracy": {
            "mean": acc_ci.mean,
            "std": acc_ci.std,
            "ci_lower": acc_ci.ci_lower,
            "ci_upper": acc_ci.ci_upper,
            "per_run": accuracies,
        },
        "scenario_stability": {
            "n_consistent": n_consistent,
            "n_total": n_scenarios,
            "consistency_rate": n_consistent / n_scenarios if n_scenarios else 0,
            "per_scenario": scenario_stats,
        },
        "axis_icc": axis_icc,
        "per_scenario_values": icc_results,
    }


def print_model_summary(agg: dict):
    """Print aggregated results for one model."""
    print(f"\n{'='*70}")
    print(f"  {agg['model']}  ({agg['n_runs']} runs, seeds: {agg['seeds']})")
    print(f"{'='*70}")

    # Accuracy
    acc = agg["accuracy"]
    print(f"\n  Accuracy: {acc['mean']:.1%} +/- {acc['std']:.1%}  "
          f"[{acc['ci_lower']:.1%}, {acc['ci_upper']:.1%}]")
    print(f"  Per-run:  {', '.join(f'{a:.0%}' for a in acc['per_run'])}")

    # Scenario stability
    stab = agg["scenario_stability"]
    print(f"\n  Scenario consistency: {stab['n_consistent']}/{stab['n_total']} "
          f"({stab['consistency_rate']:.0%} always pass or always fail)")

    print(f"\n  {'Scenario':<35} {'Pass rate':>10} {'Consistent':>12}")
    print(f"  {'-'*57}")
    for name, info in stab["per_scenario"].items():
        cons = "yes" if info["consistent"] else "NO"
        print(f"  {name:<35} {info['pass_rate']:>8.0%}   {cons:>10}")

    # ICC
    if agg["axis_icc"]:
        print(f"\n  Axis ICC (test-retest reliability across scenarios):")
        print(f"  {'Axis':<25} {'ICC':>6} {'Interpretation':<25}")
        print(f"  {'-'*56}")
        for axis, info in sorted(agg["axis_icc"].items()):
            print(f"  {axis:<25} {info['icc']:>6.3f} {info['interpretation']}")


def main():
    parser = argparse.ArgumentParser(description="Multi-seed benchmark replication")
    parser.add_argument("--model", help="Single model key")
    parser.add_argument("--models", default="article", help="'article' for 6 main models")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs per model (default: 5)")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seeds (default: 42,123,456,789,1024)")
    parser.add_argument("--verbose", action="store_true", help="Verbose per-scenario output")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Only aggregate existing results (no GPU needed)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine seeds
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    else:
        seeds = DEFAULT_SEEDS[:args.runs]

    # Determine models
    if args.model:
        models = [args.model]
    elif args.models == "article":
        models = ARTICLE_MODELS
    else:
        models = [args.models]

    if args.aggregate_only:
        # Load existing results and aggregate
        print(f"Aggregating existing results from {output_dir}...")
        all_agg = {}
        for model_key in models:
            raw_path = output_dir / f"{model_key}_replication.json"
            if not raw_path.exists():
                print(f"  Skipping {model_key}: {raw_path} not found")
                continue
            with open(raw_path) as f:
                replication_data = json.load(f)
            agg = aggregate_results(replication_data)
            all_agg[model_key] = agg
            print_model_summary(agg)
    else:
        # Run replication
        print(f"Running benchmark replication: {len(models)} models x {len(seeds)} seeds")
        print(f"Seeds: {seeds}")

        all_agg = {}
        for model_key in models:
            print(f"\n{'#'*70}")
            print(f"  Model: {model_key}")
            print(f"{'#'*70}")

            replication_data = run_replication(model_key, seeds, verbose=args.verbose)

            # Save raw results
            raw_path = output_dir / f"{model_key}_replication.json"
            with open(raw_path, "w") as f:
                json.dump(replication_data, f, indent=2, ensure_ascii=False)
            print(f"  Raw results saved to {raw_path}")

            # Aggregate
            agg = aggregate_results(replication_data)
            all_agg[model_key] = agg
            print_model_summary(agg)

    # Save aggregated summary
    summary_path = output_dir / "replication_summary.json"

    # Make JSON-serializable
    serializable = {}
    for model_key, agg in all_agg.items():
        serializable[model_key] = {
            "n_runs": agg["n_runs"],
            "seeds": agg["seeds"],
            "accuracy_mean": agg["accuracy"]["mean"],
            "accuracy_std": agg["accuracy"]["std"],
            "accuracy_ci_lower": agg["accuracy"]["ci_lower"],
            "accuracy_ci_upper": agg["accuracy"]["ci_upper"],
            "consistency_rate": agg["scenario_stability"]["consistency_rate"],
            "axis_icc": {
                axis: info["icc"]
                for axis, info in agg.get("axis_icc", {}).items()
            },
        }

    with open(summary_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    # Cross-model summary table
    if len(all_agg) > 1:
        print(f"\n{'='*70}")
        print(f"  CROSS-MODEL SUMMARY")
        print(f"{'='*70}")
        print(f"  {'Model':<15} {'Accuracy':>18} {'Consistency':>12} {'Mean ICC':>10}")
        print(f"  {'-'*55}")

        for model_key, agg in all_agg.items():
            acc = agg["accuracy"]
            cons = agg["scenario_stability"]["consistency_rate"]
            icc_vals = [v["icc"] for v in agg.get("axis_icc", {}).values()]
            mean_icc = np.mean(icc_vals) if icc_vals else float("nan")
            print(f"  {model_key:<15} {acc['mean']:.0%} [{acc['ci_lower']:.0%}-{acc['ci_upper']:.0%}]"
                  f"   {cons:>10.0%}   {mean_icc:>8.3f}")


if __name__ == "__main__":
    main()
