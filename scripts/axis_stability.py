#!/usr/bin/env python3
"""Axis Stability Experiment for Mood Axis.

Calibrates each axis on 3 independent question+style sets and compares
the resulting direction vectors. High cosine similarity = real axis.

Usage:
    # Phase 1: Calibrate (GPU required)
    python scripts/axis_stability.py calibrate --model qwen_7b
    python scripts/axis_stability.py calibrate --model qwen_7b --set B

    # Phase 2: Analyze (CPU only)
    python scripts/axis_stability.py analyze --model qwen_7b
    python scripts/axis_stability.py analyze --model all

    # Phase 3: Summary table (CPU only)
    python scripts/axis_stability.py summary
"""

import argparse
import gc
import json
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import MOOD_AXES, AXES_DIR, CALIBRATION_MAX_NEW_TOKENS
from config.models import MODELS
from config.prompts import BASELINE_QUESTIONS, STYLE_INSTRUCTIONS, YI_STYLE_OVERRIDES
from config.stability_prompts import (
    get_stability_set,
    ALL_AXES,
    verify_no_overlap,
)
from src.calibration.axis_computer import (
    compute_axis_vector,
    compute_normalization_scale,
    compute_validation_metrics,
    save_axis_vectors,
    load_axis_vectors,
)
from src.model.inference import get_hidden_state_for_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ARTICLE_MODELS = ["qwen_7b", "mistral_7b", "deepseek_7b", "llama_8b", "yi_9b", "gemma_9b"]
OUTPUT_DIR = PROJECT_ROOT / "data" / "article" / "axis_stability"
SET_NAMES = ["A", "B", "C"]


# =============================================================================
# Phase 1: Calibrate
# =============================================================================


def calibrate_set(
    model,
    tokenizer,
    model_key: str,
    set_name: str,
    axes: Optional[List[str]] = None,
) -> dict:
    """Calibrate all axes using a specific question+style set.

    Args:
        model: Loaded language model
        tokenizer: Loaded tokenizer
        model_key: Model registry key (e.g. 'qwen_7b')
        set_name: 'A', 'B', or 'C'
        axes: Subset of axes to calibrate (default: all 8)

    Returns:
        Dict with axis_vectors, scales, and per-axis metrics
    """
    axes = axes or ALL_AXES
    stability_set = get_stability_set(set_name)
    questions = stability_set["questions"]
    styles = stability_set["styles"]
    yi_overrides = stability_set.get("yi_overrides", {})

    # Apply Yi overrides if this is a Yi model
    model_config = MODELS.get(model_key)
    if model_config and getattr(model_config, "style_overrides", None) == "yi":
        for axis, overrides in yi_overrides.items():
            styles = dict(styles)  # shallow copy
            styles[axis] = overrides
        logger.info(f"Applied Yi-specific style overrides for {len(yi_overrides)} axes")

    results = {
        "model": model_key,
        "set": set_name,
        "timestamp": datetime.now().isoformat(),
        "axes": {},
    }
    axis_vectors = {}
    scales = {}

    for axis in axes:
        axis_questions = questions.get(axis, [])
        if not axis_questions:
            logger.warning(f"No questions for axis {axis} in set {set_name}, skipping")
            continue

        style_pos = styles[axis]["positive"]
        style_neg = styles[axis]["negative"]

        logger.info(f"  [{set_name}] {axis}: {len(axis_questions)} questions × 2 poles")

        positive_states = []
        negative_states = []

        for question in axis_questions:
            # Positive pole
            messages_pos = [
                {"role": "system", "content": style_pos},
                {"role": "user", "content": question},
            ]
            _, hs_pos = get_hidden_state_for_prompt(
                model, tokenizer, messages_pos,
                max_new_tokens=CALIBRATION_MAX_NEW_TOKENS,
            )
            positive_states.append(hs_pos)

            # Negative pole
            messages_neg = [
                {"role": "system", "content": style_neg},
                {"role": "user", "content": question},
            ]
            _, hs_neg = get_hidden_state_for_prompt(
                model, tokenizer, messages_neg,
                max_new_tokens=CALIBRATION_MAX_NEW_TOKENS,
            )
            negative_states.append(hs_neg)

        # Compute axis vector (trimmed mean of positive - negative)
        axis_vector = compute_axis_vector(positive_states, negative_states)
        axis_vectors[axis] = axis_vector

        # Compute normalization scale
        all_projections = [float(np.dot(s, axis_vector)) for s in positive_states + negative_states]
        scale = compute_normalization_scale(all_projections)
        scales[axis] = scale

        # Validation: 80/20 split
        n_val = max(1, len(positive_states) // 5)
        val_metrics = compute_validation_metrics(
            axis_vector,
            positive_states[-n_val:],
            negative_states[-n_val:],
        )

        results["axes"][axis] = {
            "n_questions": len(axis_questions),
            "scale": float(scale),
            "accuracy": val_metrics["accuracy"],
            "dprime": val_metrics["dprime"],
            "separation": val_metrics["separation"],
        }

        logger.info(
            f"    accuracy={val_metrics['accuracy']*100:.1f}%, "
            f"d'={val_metrics['dprime']:.2f}, "
            f"scale={scale:.4f}"
        )

    # Save axis vectors as npz
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    npz_path = OUTPUT_DIR / f"{model_key}_set_{set_name}_axes.npz"
    save_axis_vectors(axis_vectors, scales, npz_path)

    # Save metadata as JSON
    json_path = OUTPUT_DIR / f"{model_key}_set_{set_name}_meta.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def collect_baseline_projections(
    model,
    tokenizer,
    model_key: str,
) -> dict:
    """Project baseline questions through each set's axis vectors.

    Must be called AFTER calibrate_set for all 3 sets.

    Returns:
        Dict mapping set_name -> axis -> list of 30 projection values
    """
    projections = {}

    for set_name in SET_NAMES:
        npz_path = OUTPUT_DIR / f"{model_key}_set_{set_name}_axes.npz"
        if not npz_path.exists():
            logger.warning(f"Missing {npz_path}, skipping baseline projections for set {set_name}")
            continue

        data = np.load(npz_path)
        axis_names = data["_axes"].tolist()
        axis_vectors = {a: data[a] for a in axis_names}
        axis_scales = {}
        for a in axis_names:
            sk = f"{a}_scale"
            axis_scales[a] = float(data[sk]) if sk in data else 2.0

        set_projections = {}

        for question in BASELINE_QUESTIONS:
            messages = [{"role": "user", "content": question}]
            _, hidden_state = get_hidden_state_for_prompt(
                model, tokenizer, messages,
                max_new_tokens=CALIBRATION_MAX_NEW_TOKENS,
            )

            for axis in axis_names:
                raw = float(np.dot(hidden_state, axis_vectors[axis]))
                normalized = np.clip(raw / axis_scales[axis], -1.0, 1.0)
                set_projections.setdefault(axis, []).append(float(normalized))

        projections[set_name] = set_projections

    # Save
    out_path = OUTPUT_DIR / f"{model_key}_baseline_projections.json"
    with open(out_path, "w") as f:
        json.dump(projections, f, indent=2)
    logger.info(f"Baseline projections saved to {out_path}")

    return projections


def run_calibrate(model_key: str, set_names: Optional[List[str]] = None):
    """Run calibration phase for a model."""
    import torch

    set_names = set_names or SET_NAMES

    if model_key not in MODELS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(MODELS.keys())}")

    model_config = MODELS[model_key]
    logger.info(f"Loading model: {model_config.model_id}")

    from src.model.loader import load_model
    model, tokenizer = load_model(model_config.model_id)

    for set_name in set_names:
        logger.info(f"\n{'='*60}")
        logger.info(f"Calibrating {model_key} with Set {set_name}")
        logger.info(f"{'='*60}")
        calibrate_set(model, tokenizer, model_key, set_name)

    # Collect baseline projections through all available sets
    logger.info(f"\nCollecting baseline projections for {model_key}...")
    collect_baseline_projections(model, tokenizer, model_key)

    # Cleanup
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(f"\nCalibration complete for {model_key}")


# =============================================================================
# Phase 2: Analyze
# =============================================================================


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def cross_eval_accuracy(
    axis_vector: np.ndarray,
    positive_states: List[np.ndarray],
    negative_states: List[np.ndarray],
) -> float:
    """Compute classification accuracy using an axis vector on held-out data."""
    correct = 0
    total = 0
    for s in positive_states:
        if np.dot(s, axis_vector) > 0:
            correct += 1
        total += 1
    for s in negative_states:
        if np.dot(s, axis_vector) < 0:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0.0


def analyze_model(model_key: str) -> dict:
    """Analyze axis stability for a model (CPU only).

    Computes:
    1. Cosine similarity between axis vectors from different sets
    2. Baseline projection correlations (Pearson r)

    Args:
        model_key: Model registry key

    Returns:
        Dict with all metrics
    """
    from scipy.stats import pearsonr

    results = {
        "model": model_key,
        "timestamp": datetime.now().isoformat(),
        "cosine_similarity": {},
        "baseline_correlation": {},
        "per_axis_summary": {},
    }

    # Load axis vectors from all available sets
    vectors: Dict[str, Dict[str, np.ndarray]] = {}
    for set_name in SET_NAMES:
        npz_path = OUTPUT_DIR / f"{model_key}_set_{set_name}_axes.npz"
        if npz_path.exists():
            data = np.load(npz_path)
            axis_names = data["_axes"].tolist()
            vectors[set_name] = {a: data[a] for a in axis_names}
        else:
            logger.warning(f"Missing {npz_path}")

    if len(vectors) < 2:
        logger.error(f"Need at least 2 sets for comparison, found {len(vectors)}")
        return results

    available_sets = sorted(vectors.keys())

    # 1. Cosine similarity between all pairs of sets
    set_pairs = [(a, b) for i, a in enumerate(available_sets) for b in available_sets[i+1:]]

    for set_a, set_b in set_pairs:
        pair_key = f"{set_a}_vs_{set_b}"
        results["cosine_similarity"][pair_key] = {}

        common_axes = sorted(set(vectors[set_a].keys()) & set(vectors[set_b].keys()))
        for axis in common_axes:
            cos_sim = cosine_similarity(vectors[set_a][axis], vectors[set_b][axis])
            results["cosine_similarity"][pair_key][axis] = cos_sim

    # 2. Baseline projection correlations
    proj_path = OUTPUT_DIR / f"{model_key}_baseline_projections.json"
    if proj_path.exists():
        with open(proj_path) as f:
            projections = json.load(f)

        for set_a, set_b in set_pairs:
            pair_key = f"{set_a}_vs_{set_b}"
            results["baseline_correlation"][pair_key] = {}

            if set_a in projections and set_b in projections:
                common_axes = sorted(
                    set(projections[set_a].keys()) & set(projections[set_b].keys())
                )
                for axis in common_axes:
                    vals_a = projections[set_a][axis]
                    vals_b = projections[set_b][axis]
                    if len(vals_a) == len(vals_b) and len(vals_a) >= 3:
                        r, p = pearsonr(vals_a, vals_b)
                        results["baseline_correlation"][pair_key][axis] = {
                            "pearson_r": float(r),
                            "p_value": float(p),
                        }
    else:
        logger.warning(f"No baseline projections found for {model_key}")

    # 3. Per-axis summary (mean across set pairs)
    all_axes = set()
    for pair_data in results["cosine_similarity"].values():
        all_axes.update(pair_data.keys())

    for axis in sorted(all_axes):
        cos_values = []
        corr_values = []

        for pair_key, pair_data in results["cosine_similarity"].items():
            if axis in pair_data:
                cos_values.append(pair_data[axis])

        for pair_key, pair_data in results["baseline_correlation"].items():
            if axis in pair_data:
                corr_values.append(pair_data[axis]["pearson_r"])

        results["per_axis_summary"][axis] = {
            "mean_cosine": float(np.mean(cos_values)) if cos_values else None,
            "min_cosine": float(np.min(cos_values)) if cos_values else None,
            "max_cosine": float(np.max(cos_values)) if cos_values else None,
            "mean_pearson_r": float(np.mean(corr_values)) if corr_values else None,
            "stable": float(np.mean(cos_values)) > 0.8 if cos_values else None,
        }

    # Save
    out_path = OUTPUT_DIR / f"{model_key}_analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Analysis saved to {out_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Axis Stability: {model_key}")
    print(f"{'='*60}")
    print(f"{'Axis':<25} {'Mean cos':>10} {'Min cos':>10} {'Pearson r':>10} {'Stable?':>8}")
    print("-" * 65)
    for axis, summary in sorted(results["per_axis_summary"].items()):
        cos_str = f"{summary['mean_cosine']:.3f}" if summary["mean_cosine"] is not None else "N/A"
        min_str = f"{summary['min_cosine']:.3f}" if summary["min_cosine"] is not None else "N/A"
        r_str = f"{summary['mean_pearson_r']:.3f}" if summary["mean_pearson_r"] is not None else "N/A"
        stable = "Yes" if summary.get("stable") else "No" if summary.get("stable") is not None else "?"
        print(f"{axis:<25} {cos_str:>10} {min_str:>10} {r_str:>10} {stable:>8}")

    return results


# =============================================================================
# Phase 3: Summary
# =============================================================================


def generate_summary():
    """Generate cross-model summary of axis stability."""
    all_analyses = {}

    for model_key in ARTICLE_MODELS:
        analysis_path = OUTPUT_DIR / f"{model_key}_analysis.json"
        if analysis_path.exists():
            with open(analysis_path) as f:
                all_analyses[model_key] = json.load(f)

    if not all_analyses:
        logger.error("No analysis files found. Run 'analyze' first.")
        return

    # Aggregate: per axis, mean cosine across all models
    axis_stats: Dict[str, List[float]] = {}
    for model_key, analysis in all_analyses.items():
        for axis, summary in analysis.get("per_axis_summary", {}).items():
            if summary.get("mean_cosine") is not None:
                axis_stats.setdefault(axis, []).append(summary["mean_cosine"])

    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_models": len(all_analyses),
        "models": list(all_analyses.keys()),
        "per_axis": {},
        "overall": {},
    }

    print(f"\n{'='*70}")
    print(f"AXIS STABILITY SUMMARY ({len(all_analyses)} models)")
    print(f"{'='*70}")
    print(f"{'Axis':<25} {'Mean cos':>10} {'Std':>8} {'Min':>8} {'N':>4} {'Stable?':>8}")
    print("-" * 65)

    all_cosines = []
    for axis in sorted(axis_stats.keys()):
        values = axis_stats[axis]
        mean_cos = float(np.mean(values))
        std_cos = float(np.std(values))
        min_cos = float(np.min(values))
        stable = mean_cos > 0.8

        summary["per_axis"][axis] = {
            "mean_cosine": mean_cos,
            "std_cosine": std_cos,
            "min_cosine": min_cos,
            "n_models": len(values),
            "stable": stable,
            "per_model": {
                model: all_analyses[model]["per_axis_summary"].get(axis, {}).get("mean_cosine")
                for model in all_analyses
            },
        }

        all_cosines.extend(values)
        print(
            f"{axis:<25} {mean_cos:>10.3f} {std_cos:>8.3f} {min_cos:>8.3f} "
            f"{len(values):>4} {'Yes' if stable else 'NO':>8}"
        )

    if all_cosines:
        summary["overall"] = {
            "grand_mean_cosine": float(np.mean(all_cosines)),
            "grand_std_cosine": float(np.std(all_cosines)),
            "grand_min_cosine": float(np.min(all_cosines)),
            "n_stable_axes": sum(
                1 for a in summary["per_axis"].values() if a["stable"]
            ),
            "n_total_axes": len(summary["per_axis"]),
        }
        print("-" * 65)
        print(
            f"{'OVERALL':<25} {summary['overall']['grand_mean_cosine']:>10.3f} "
            f"{summary['overall']['grand_std_cosine']:>8.3f} "
            f"{summary['overall']['grand_min_cosine']:>8.3f} "
            f"{len(all_cosines):>4}"
        )
        print(
            f"\nStable axes: {summary['overall']['n_stable_axes']}"
            f"/{summary['overall']['n_total_axes']} (threshold: cos > 0.8)"
        )

    # Save
    out_path = OUTPUT_DIR / "summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSummary saved to {out_path}")

    return summary


# =============================================================================
# Also compare against existing production axes (sanity check)
# =============================================================================


def sanity_check_set_a(model_key: str):
    """Verify that Set A calibration reproduces existing production axes.

    Set A uses the same questions and styles as the original calibration,
    so cos(set_A, production) should be very high (>0.95).
    """
    prod_path = AXES_DIR / f"{model_key}_axes.npz"
    set_a_path = OUTPUT_DIR / f"{model_key}_set_A_axes.npz"

    if not prod_path.exists() or not set_a_path.exists():
        logger.warning(f"Cannot sanity check {model_key}: missing files")
        return

    prod_data = np.load(prod_path)
    set_a_data = np.load(set_a_path)

    prod_axes = prod_data["_axes"].tolist()
    set_a_axes = set_a_data["_axes"].tolist()
    common = sorted(set(prod_axes) & set(set_a_axes))

    print(f"\nSanity check: Set A vs Production ({model_key})")
    print(f"{'Axis':<25} {'Cosine':>10}")
    print("-" * 37)

    for axis in common:
        cos = cosine_similarity(prod_data[axis], set_a_data[axis])
        flag = "" if cos > 0.95 else " ⚠️"
        print(f"{axis:<25} {cos:>10.4f}{flag}")


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Axis Stability Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/axis_stability.py calibrate --model qwen_7b
  python scripts/axis_stability.py calibrate --model qwen_7b --set B
  python scripts/axis_stability.py analyze --model qwen_7b
  python scripts/axis_stability.py analyze --model all
  python scripts/axis_stability.py summary
  python scripts/axis_stability.py sanity --model qwen_7b
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # calibrate
    cal_parser = subparsers.add_parser("calibrate", help="Calibrate axes (GPU required)")
    cal_parser.add_argument("--model", required=True, help="Model key or 'article' for all 6")
    cal_parser.add_argument("--set", choices=["A", "B", "C"], help="Specific set (default: all)")

    # analyze
    ana_parser = subparsers.add_parser("analyze", help="Analyze stability (CPU only)")
    ana_parser.add_argument("--model", required=True, help="Model key or 'all'")

    # summary
    subparsers.add_parser("summary", help="Cross-model summary (CPU only)")

    # sanity
    san_parser = subparsers.add_parser("sanity", help="Sanity check Set A vs production")
    san_parser.add_argument("--model", required=True, help="Model key or 'all'")

    # verify
    subparsers.add_parser("verify", help="Verify no overlap in question sets")

    args = parser.parse_args()

    if args.command == "calibrate":
        models = ARTICLE_MODELS if args.model == "article" else [args.model]
        sets = [args.set] if args.set else None
        for m in models:
            run_calibrate(m, sets)

    elif args.command == "analyze":
        models = ARTICLE_MODELS if args.model == "all" else [args.model]
        for m in models:
            analyze_model(m)

    elif args.command == "summary":
        generate_summary()

    elif args.command == "sanity":
        models = ARTICLE_MODELS if args.model == "all" else [args.model]
        for m in models:
            sanity_check_set_a(m)

    elif args.command == "verify":
        print("Verifying question set overlap...")
        if verify_no_overlap():
            print("All clean — no overlap detected.")
        else:
            print("WARNING: Overlap found!")
            sys.exit(1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
