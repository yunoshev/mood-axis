#!/usr/bin/env python3
"""Run the full Mood Axis pipeline.

This script runs the complete pipeline:
1. Calibrate axis vectors (or skip if already done)
2. Collect baseline temperament
3. Run validation benchmarks
4. Run drift analysis
5. Generate visualizations

Usage:
    # Single model
    python scripts/run_pipeline.py --model qwen_7b

    # Model set
    python scripts/run_pipeline.py --model-set small
    python scripts/run_pipeline.py --model-set quick  # Single model for testing

    # All models in active set
    python scripts/run_pipeline.py --model all

    # With options
    python scripts/run_pipeline.py --model-set small --skip-calibration
    python scripts/run_pipeline.py --model qwen_1.5b --skip-drift

Model Sets:
    article  - 7-9B models (qwen_7b, mistral_7b, deepseek_7b, llama_8b, yi_9b)
    small    - 1-2B models (qwen_1.5b, smollm_1.7b, llama_1b)
    quick    - Single model for quick testing (qwen_1.5b)
    all      - All registered models
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.models import MODELS, MODEL_SETS, get_active_models, set_model_set, get_model_config
from config.settings import AXES_DIR, MOOD_AXES

import numpy as np


def run_verify(model_key: str, check: str) -> bool:
    """Run data verification for a specific check. Returns True if no FAILs."""
    result = subprocess.run(
        [sys.executable, "scripts/verify_data.py", "--model", model_key, "--check", check],
        cwd=PROJECT_ROOT
    )
    return result.returncode == 0


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode == 0


def check_calibration_complete(model_key: str) -> bool:
    """Check if ALL required axes are calibrated for a model."""
    axes_file = AXES_DIR / f"{model_key}_axes.npz"
    if not axes_file.exists():
        return False
    data = np.load(axes_file)
    existing = set(data["_axes"].tolist())
    required = set(MOOD_AXES)
    missing = required - existing
    if missing:
        print(f"  Missing axes: {sorted(missing)}")
        return False
    return True


def run_pipeline(model_key: str, skip_calibration: bool = False, skip_drift: bool = False):
    """Run the full pipeline for a model."""
    print(f"\n{'#'*60}")
    print(f"  PIPELINE: {model_key}")
    print(f"{'#'*60}")

    # Step 1: Calibration
    if skip_calibration:
        print("\n[SKIP] Calibration (--skip-calibration)")
    elif check_calibration_complete(model_key):
        print(f"\n[SKIP] Calibration (all {len(MOOD_AXES)} axes present: {AXES_DIR / f'{model_key}_axes.npz'})")
    else:
        success = run_command(
            [sys.executable, "scripts/calibrate_local.py", "--model", model_key],
            f"Step 1: Calibrating {model_key}"
        )
        if not success:
            print(f"[ERROR] Calibration failed for {model_key}")
            return False
        if not run_verify(model_key, "calibration"):
            print(f"[ERROR] Calibration verification failed for {model_key}")
            return False

    # Step 2: Baseline
    success = run_command(
        [sys.executable, "scripts/collect_baseline.py", "--model", model_key],
        f"Step 2: Collecting baseline for {model_key}"
    )
    if not success:
        print(f"[WARNING] Baseline collection failed for {model_key}")
    else:
        run_verify(model_key, "baseline")

    # Step 3: Benchmark
    success = run_command(
        [sys.executable, "scripts/benchmark.py", "--model", model_key],
        f"Step 3: Running benchmarks for {model_key}"
    )
    if not success:
        print(f"[WARNING] Benchmark failed for {model_key}")

    # Step 4: Drift analysis
    model_config = get_model_config(model_key)
    if model_config.is_base_model:
        print("\n[SKIP] Drift analysis (base model — no multi-turn support)")
    elif skip_drift:
        print("\n[SKIP] Drift analysis (--skip-drift)")
    else:
        success = run_command(
            [sys.executable, "scripts/extended_drift.py", "--model", model_key, "--scenarios", "20"],
            f"Step 4: Running drift analysis for {model_key}"
        )
        if not success:
            print(f"[WARNING] Drift analysis failed for {model_key}")
        else:
            run_verify(model_key, "drift")

    print(f"\n[DONE] Pipeline complete for {model_key}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run full Mood Axis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model Sets:
  article  - 7-9B models for the original article
  small    - 1-2B models for quick testing
  quick    - Single model (qwen_1.5b) for fastest testing
  all      - All registered models

Examples:
  python scripts/run_pipeline.py --model qwen_1.5b
  python scripts/run_pipeline.py --model-set small
  python scripts/run_pipeline.py --model-set quick --skip-drift
        """
    )
    parser.add_argument("--model", help="Model key (e.g., qwen_7b, qwen_1.5b) or 'all' for active set")
    parser.add_argument("--model-set", choices=MODEL_SETS.keys(), help="Use a predefined model set")
    parser.add_argument("--skip-calibration", action="store_true", help="Skip calibration step")
    parser.add_argument("--skip-drift", action="store_true", help="Skip drift analysis")
    parser.add_argument("--list-sets", action="store_true", help="List available model sets and exit")
    args = parser.parse_args()

    # List model sets
    if args.list_sets:
        from config.models import print_model_sets
        print_model_sets()
        return

    # Determine which models to run
    if args.model_set:
        set_model_set(args.model_set)
        models = MODEL_SETS[args.model_set]
        print(f"Using model set: {args.model_set}")
    elif args.model == "all":
        models = get_active_models()
        print(f"Using active model set")
    elif args.model:
        if args.model not in MODELS:
            print(f"Unknown model: {args.model}")
            print(f"Available: {list(MODELS.keys())}")
            sys.exit(1)
        models = [args.model]
    else:
        parser.print_help()
        print("\nError: Either --model or --model-set is required")
        sys.exit(1)

    print(f"Running pipeline for: {models}")

    for model_key in models:
        run_pipeline(model_key, args.skip_calibration, args.skip_drift)

    # Step 5: Generate visualizations
    print("\n" + "="*60)
    print("  Step 5: Generating visualizations")
    print("="*60)
    subprocess.run([sys.executable, "scripts/visualize_article.py"], cwd=PROJECT_ROOT)

    print("\n" + "#"*60)
    print("  PIPELINE COMPLETE")
    print("#"*60)


if __name__ == "__main__":
    main()
