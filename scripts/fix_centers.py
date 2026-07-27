#!/usr/bin/env python3
"""Backfill normalization centers into existing axes NPZ files.

Computes the median of training projections from saved hidden states
and adds {axis}_center to each model's axes NPZ. This fixes the
projection saturation bug where axes with large offset from zero
(e.g., qwen3_8b verbose_concise) were clipped to +/-1.0.

Also applies math-fix to existing steering and baseline data:
for non-saturated projections, new = old - center/scale.

Usage:
    python scripts/fix_centers.py                    # Backfill centers only
    python scripts/fix_centers.py --fix-steering     # Also fix steering data
    python scripts/fix_centers.py --fix-baselines    # Also fix baseline data
    python scripts/fix_centers.py --dry-run          # Preview without saving
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import MOOD_AXES

AXES_DIR = PROJECT_ROOT / "data" / "axes"
STEERING_DIR = PROJECT_ROOT / "data" / "steering"
BASELINE_DIR = PROJECT_ROOT / "data" / "baselines"

MODELS = ["qwen_7b", "llama_8b", "mistral_7b", "gemma_9b", "qwen3_8b"]


def compute_centers(model_key: str) -> dict:
    """Compute centers from hidden states + axis vectors."""
    axes_file = AXES_DIR / f"{model_key}_axes.npz"
    hs_file = AXES_DIR / f"{model_key}_hidden_states.npz"

    if not axes_file.exists() or not hs_file.exists():
        print(f"  SKIP {model_key}: missing files", flush=True)
        return {}

    axes_npz = np.load(axes_file, allow_pickle=True)
    hs_npz = np.load(hs_file, allow_pickle=True)
    available_axes = axes_npz["_axes"].tolist()

    centers = {}
    for axis in available_axes:
        vec = axes_npz[axis]
        pos_key = f"{axis}_train_pos_aggregated"
        neg_key = f"{axis}_train_neg_aggregated"

        if pos_key not in hs_npz or neg_key not in hs_npz:
            print(f"  SKIP {model_key}/{axis}: missing hidden states", flush=True)
            continue

        pos = hs_npz[pos_key]
        neg = hs_npz[neg_key]
        all_states = np.concatenate([pos, neg], axis=0)
        raw_projs = all_states @ vec
        centers[axis] = float(np.median(raw_projs))

    return centers


def backfill_centers(model_key: str, centers: dict, dry_run: bool = False):
    """Add centers to existing axes NPZ file."""
    axes_file = AXES_DIR / f"{model_key}_axes.npz"
    data = dict(np.load(axes_file, allow_pickle=True))

    changed = False
    for axis, center in centers.items():
        key = f"{axis}_center"
        old = float(data[key]) if key in data else None
        if old is None or abs(old - center) > 1e-6:
            data[key] = np.array(center)
            changed = True
            status = f"NEW" if old is None else f"UPDATED {old:.2f}->{center:.2f}"
            print(f"  {model_key}/{axis}: center={center:.2f} ({status})", flush=True)

    if changed and not dry_run:
        np.savez(axes_file, **data)
        print(f"  Saved {axes_file}", flush=True)
    elif not changed:
        print(f"  {model_key}: no changes needed", flush=True)


def fix_steering_data(model_key: str, centers: dict, dry_run: bool = False):
    """Apply math-fix to steering projections: new = old - center/scale.

    For saturated values (|old| >= 0.999), we can't recover the original
    raw projection, so we leave them as-is and flag them.
    """
    steering_file = STEERING_DIR / f"{model_key}_basic.json"
    if not steering_file.exists():
        print(f"  SKIP steering {model_key}: no file", flush=True)
        return

    axes_npz = np.load(AXES_DIR / f"{model_key}_axes.npz", allow_pickle=True)
    all_axes = axes_npz["_axes"].tolist()

    scales = {}
    for a in all_axes:
        sk = f"{a}_scale"
        scales[a] = float(axes_npz[sk]) if sk in axes_npz else 2.0

    with open(steering_file) as f:
        data = json.load(f)

    axes_data = data.get("axes", {})
    total_fixed = 0
    total_saturated = 0

    for axis_key, axis_section in axes_data.items():
        results = axis_section.get("results", {})
        for alpha_key, samples in results.items():
            for sample in samples:
                projs = sample.get("projections", {})
                for a in list(projs.keys()):
                    if a not in centers or a not in scales:
                        continue
                    old_val = projs[a]
                    offset = centers[a] / scales[a]

                    if abs(old_val) >= 0.999:
                        total_saturated += 1
                        # Can't recover — leave as-is but re-clip after offset
                        # If old == +1.0, true value was >= 1.0, so new = old - offset
                        # might still be >= 1.0 or might not — we don't know
                        continue

                    new_val = float(np.clip(old_val - offset, -1.0, 1.0))
                    if abs(new_val - old_val) > 1e-8:
                        projs[a] = new_val
                        total_fixed += 1

        # Recompute axis-level statistics
        _recompute_axis_stats(axis_section, axis_key, all_axes)

    print(f"  {model_key}: fixed={total_fixed}, saturated_skipped={total_saturated}", flush=True)

    if not dry_run and total_fixed > 0:
        with open(steering_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Saved {steering_file}", flush=True)


def _recompute_axis_stats(axis_section: dict, target_axis: str, all_axes: list):
    """Recompute slope, intercept, alpha_means, leakage from per-question data."""
    results = axis_section.get("results", {})
    if not results:
        return

    # Collect per-alpha mean projections
    alpha_means = {}
    alpha_projs = {}
    for alpha_str, samples in results.items():
        alpha = float(alpha_str)
        target_vals = [s["projections"][target_axis] for s in samples if target_axis in s.get("projections", {})]
        if target_vals:
            alpha_means[alpha_str] = float(np.mean(target_vals))
            alpha_projs[alpha] = target_vals

    axis_section["alpha_means"] = alpha_means

    # Linear regression for slope/intercept
    if len(alpha_projs) >= 2:
        alphas_sorted = sorted(alpha_projs.keys())
        x = np.array(alphas_sorted)
        y = np.array([np.mean(alpha_projs[a]) for a in alphas_sorted])
        if len(x) >= 2:
            coeffs = np.polyfit(x, y, 1)
            axis_section["slope"] = float(coeffs[0])
            axis_section["intercept"] = float(coeffs[1])

    # Leakage: per-axis slopes
    leakage = {}
    target_slope = abs(axis_section.get("slope", 0))
    for a in all_axes:
        if a == target_axis:
            continue
        x_vals = []
        y_vals = []
        for alpha_str, samples in results.items():
            alpha = float(alpha_str)
            vals = [s["projections"][a] for s in samples if a in s.get("projections", {})]
            if vals:
                x_vals.append(alpha)
                y_vals.append(float(np.mean(vals)))
        if len(x_vals) >= 2:
            coeffs = np.polyfit(x_vals, y_vals, 1)
            leak_slope = abs(float(coeffs[0]))
            ratio = leak_slope / target_slope if target_slope > 1e-6 else 0
            leakage[a] = {"slope": float(coeffs[0]), "ratio": round(ratio, 2)}

    axis_section["leakage"] = leakage


def fix_baseline_data(model_key: str, centers: dict, dry_run: bool = False):
    """Apply math-fix to baseline projections."""
    baseline_file = BASELINE_DIR / f"{model_key}.json"
    if not baseline_file.exists():
        print(f"  SKIP baseline {model_key}: no file", flush=True)
        return

    axes_npz = np.load(AXES_DIR / f"{model_key}_axes.npz", allow_pickle=True)
    scales = {}
    for a in MOOD_AXES:
        sk = f"{a}_scale"
        scales[a] = float(axes_npz[sk]) if sk in axes_npz else 2.0

    with open(baseline_file) as f:
        data = json.load(f)

    axes_section = data.get("axes", {})
    total_fixed = 0

    for axis, axis_data in axes_section.items():
        if axis not in centers or axis not in scales:
            continue
        offset = centers[axis] / scales[axis]
        values = axis_data.get("values", [])
        new_values = []
        for v in values:
            if abs(v) >= 0.999:
                new_values.append(v)  # saturated, can't fix
            else:
                new_val = float(np.clip(v - offset, -1.0, 1.0))
                if abs(new_val - v) > 1e-8:
                    total_fixed += 1
                new_values.append(new_val)

        axis_data["values"] = new_values
        axis_data["mean"] = float(np.mean(new_values))
        axis_data["std"] = float(np.std(new_values))
        axis_data["min"] = float(np.min(new_values))
        axis_data["max"] = float(np.max(new_values))

    print(f"  {model_key}: fixed={total_fixed} baseline projections", flush=True)

    if not dry_run and total_fixed > 0:
        with open(baseline_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Saved {baseline_file}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Backfill centers and fix projections")
    parser.add_argument("--fix-steering", action="store_true", help="Also fix steering data")
    parser.add_argument("--fix-baselines", action="store_true", help="Also fix baseline data")
    parser.add_argument("--dry-run", action="store_true", help="Preview without saving")
    parser.add_argument("--models", help="Comma-separated model keys (default: all paper models)")
    args = parser.parse_args()

    models = args.models.split(",") if args.models else MODELS

    print("=== Step 1: Compute and backfill centers ===", flush=True)
    all_centers = {}
    for model_key in models:
        print(f"\n{model_key}:", flush=True)
        centers = compute_centers(model_key)
        all_centers[model_key] = centers
        backfill_centers(model_key, centers, dry_run=args.dry_run)

    if args.fix_steering:
        print("\n=== Step 2: Fix steering data ===", flush=True)
        for model_key in models:
            print(f"\n{model_key}:", flush=True)
            fix_steering_data(model_key, all_centers[model_key], dry_run=args.dry_run)

    if args.fix_baselines:
        print("\n=== Step 3: Fix baseline data ===", flush=True)
        for model_key in models:
            print(f"\n{model_key}:", flush=True)
            fix_baseline_data(model_key, all_centers[model_key], dry_run=args.dry_run)

    print("\n=== Done ===", flush=True)
    if args.dry_run:
        print("(dry-run mode — no files were modified)", flush=True)


if __name__ == "__main__":
    main()
