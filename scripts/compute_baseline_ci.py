#!/usr/bin/env python3
"""Compute bootstrap confidence intervals for baseline measurements.

Reads baseline JSON files (must contain per-question 'values' arrays),
computes bootstrap CI, significance tests, and outputs a summary.

Usage:
    python scripts/compute_baseline_ci.py
    python scripts/compute_baseline_ci.py --model qwen_7b
    python scripts/compute_baseline_ci.py --models article
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics.statistics import (
    bootstrap_ci,
    test_against_value,
    holm_bonferroni_correction,
    format_statistics_table,
)

BASELINES_DIR = PROJECT_ROOT / "data" / "article" / "baselines"
OUTPUT_DIR = PROJECT_ROOT / "data" / "article"

ARTICLE_MODELS = ["qwen_7b", "mistral_7b", "deepseek_7b", "llama_8b", "yi_9b", "gemma_9b"]

ALL_AXES = [
    "warm_cold", "patient_irritated", "confident_cautious", "proactive_reluctant",
    "empathetic_analytical", "formal_casual", "verbose_concise",
]


def load_baseline(model_key: str) -> dict:
    """Load baseline JSON, handling both old and new formats."""
    path = BASELINES_DIR / f"{model_key}_baseline.json"
    if not path.exists():
        raise FileNotFoundError(f"Baseline not found: {path}")

    with open(path) as f:
        data = json.load(f)

    # Normalize format: old files use "baseline", new use "axes"
    axes_data = data.get("axes", data.get("baseline", {}))
    return axes_data


def compute_model_ci(model_key: str, n_bootstrap: int = 10000) -> dict:
    """Compute bootstrap CI for all axes of a model.

    Returns:
        Dict with per-axis CI results, or None for axes without raw values
    """
    axes_data = load_baseline(model_key)
    results = {}

    for axis in ALL_AXES:
        if axis not in axes_data:
            continue

        axis_info = axes_data[axis]

        # Check if raw values are available
        if isinstance(axis_info, dict) and "values" in axis_info:
            values = axis_info["values"]
        else:
            results[axis] = {
                "has_raw_values": False,
                "mean": axis_info.get("mean", axis_info) if isinstance(axis_info, dict) else float(axis_info),
            }
            continue

        # Bootstrap CI
        ci = bootstrap_ci(values, n_bootstrap=n_bootstrap, ci=0.95, random_state=42)

        # Significance test against 0
        sig = test_against_value(values, test_value=0.0, n_bootstrap=n_bootstrap, random_state=42)

        results[axis] = {
            "has_raw_values": True,
            "n_samples": len(values),
            "mean": ci.mean,
            "std": ci.std,
            "ci_lower": ci.ci_lower,
            "ci_upper": ci.ci_upper,
            "ci_width": ci.ci_upper - ci.ci_lower,
            "ci_crosses_zero": ci.ci_lower <= 0 <= ci.ci_upper,
            "p_value": sig.p_value,
            "effect_size": sig.effect_size,
        }

    return results


def compute_all(models: list, n_bootstrap: int = 10000) -> dict:
    """Compute CI for all models, apply multiple comparison correction."""
    all_results = {}
    all_p_values = []  # (model, axis, p_value) for Holm-Bonferroni

    for model_key in models:
        try:
            model_results = compute_model_ci(model_key, n_bootstrap)
            all_results[model_key] = model_results

            for axis, info in model_results.items():
                if info.get("has_raw_values") and "p_value" in info:
                    all_p_values.append((model_key, axis, info["p_value"]))

        except FileNotFoundError as e:
            print(f"  Skipping {model_key}: {e}")

    # Holm-Bonferroni correction across all tests
    if all_p_values:
        p_vals = [p for _, _, p in all_p_values]
        significant, adjusted = holm_bonferroni_correction(p_vals, alpha=0.05)

        for i, (model_key, axis, _) in enumerate(all_p_values):
            all_results[model_key][axis]["p_value_adjusted"] = adjusted[i]
            all_results[model_key][axis]["significant_corrected"] = significant[i]

    return all_results


def print_results(all_results: dict):
    """Print formatted results."""
    for model_key, model_results in all_results.items():
        has_values = any(v.get("has_raw_values") for v in model_results.values())
        if not has_values:
            print(f"\n{model_key}: NO RAW VALUES (needs re-run)")
            continue

        print(f"\n{'='*80}")
        print(f"  {model_key}")
        print(f"{'='*80}")
        print(
            f"  {'Axis':<25} {'Mean':>7} {'95% CI':>20} {'Width':>7} "
            f"{'p(adj)':>8} {'Sig?':>5} {'d':>6} {'0∈CI?':>6}"
        )
        print(f"  {'-'*76}")

        for axis in ALL_AXES:
            if axis not in model_results:
                continue
            info = model_results[axis]
            if not info.get("has_raw_values"):
                print(f"  {axis:<25} {info.get('mean', '?'):>7} {'(no raw values)':>20}")
                continue

            ci_str = f"[{info['ci_lower']:+.3f}, {info['ci_upper']:+.3f}]"
            sig = "**" if info.get("significant_corrected") else ""
            crosses = "YES" if info["ci_crosses_zero"] else "no"

            print(
                f"  {axis:<25} {info['mean']:>+.3f} {ci_str:>20} {info['ci_width']:>.3f} "
                f"{info.get('p_value_adjusted', info['p_value']):>8.4f} {sig:>5} "
                f"{info.get('effect_size', 0):>+.2f} {crosses:>6}"
            )


def main():
    parser = argparse.ArgumentParser(description="Compute baseline bootstrap CI")
    parser.add_argument("--model", help="Single model key")
    parser.add_argument("--models", default="article", help="'article' for 6 main models, or 'all'")
    parser.add_argument("--bootstrap", type=int, default=10000, help="Bootstrap iterations")
    parser.add_argument("--output", help="Output JSON path")
    args = parser.parse_args()

    if args.model:
        models = [args.model]
    elif args.models == "article":
        models = ARTICLE_MODELS
    else:
        models = [f.stem.replace("_baseline", "") for f in BASELINES_DIR.glob("*_baseline.json")]

    print(f"Computing bootstrap CI (n={args.bootstrap}) for {len(models)} models...")

    all_results = compute_all(models, n_bootstrap=args.bootstrap)

    print_results(all_results)

    # Summary stats
    n_with_values = 0
    n_significant = 0
    n_crosses_zero = 0
    n_total = 0

    for model_results in all_results.values():
        for info in model_results.values():
            if info.get("has_raw_values"):
                n_with_values += 1
                n_total += 1
                if info.get("significant_corrected"):
                    n_significant += 1
                if info.get("ci_crosses_zero"):
                    n_crosses_zero += 1

    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"  Axes with raw values: {n_with_values}/{n_total + sum(1 for m in all_results.values() for v in m.values() if not v.get('has_raw_values'))}")
    print(f"  Significantly ≠ 0 (Holm-Bonferroni): {n_significant}/{n_total}")
    print(f"  CI crosses zero: {n_crosses_zero}/{n_total}")

    # Save
    output_path = Path(args.output) if args.output else OUTPUT_DIR / "baseline_ci.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Make JSON-serializable
    serializable = {}
    for model_key, model_results in all_results.items():
        serializable[model_key] = {}
        for axis, info in model_results.items():
            serializable[model_key][axis] = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in info.items()
            }

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
