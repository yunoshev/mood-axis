#!/usr/bin/env python3
"""Step 06: Cross-model analysis (CPU only).

Produces: data/analysis/*.json
Depends on: 02_baseline, 05_drift (reads all model data)

Computes:
- PCA dimensionality analysis
- Cross-axis correlations
- Dead zone detection (RLHF suppression)
- Cross-model fingerprint comparison

Usage:
    python scripts/06_analysis.py
    python scripts/06_analysis.py --check
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
from config.models import MODELS, MODEL_SETS

BASELINES_DIR = PROJECT_ROOT / "data" / "baselines"
DRIFT_DIR = PROJECT_ROOT / "data" / "drift"
OUTPUT_DIR = PROJECT_ROOT / "data" / "analysis"


def preflight_check():
    baselines = list(BASELINES_DIR.glob("*.json"))
    drifts = list(DRIFT_DIR.glob("*.json"))
    print(f"Baselines found: {len(baselines)}", flush=True)
    print(f"Drift files found: {len(drifts)}", flush=True)
    print(f"Output: {OUTPUT_DIR}", flush=True)
    print("Preflight OK", flush=True)


def load_all_baselines() -> dict:
    """Load all baseline JSON files."""
    baselines = {}
    for f in sorted(BASELINES_DIR.glob("*.json")):
        model_key = f.stem
        data = json.load(open(f))
        baselines[model_key] = data
    return baselines


def load_all_drift() -> dict:
    """Load all drift JSON files."""
    drifts = {}
    for f in sorted(DRIFT_DIR.glob("*.json")):
        model_key = f.stem
        data = json.load(open(f))
        drifts[model_key] = data
    return drifts


def compute_pca_analysis(baselines: dict) -> dict:
    """PCA dimensionality analysis across models."""
    results = {}

    for model_key, data in baselines.items():
        axes_data = data.get("axes", {})
        means = []
        for axis in MOOD_AXES:
            if axis in axes_data:
                means.append(axes_data[axis]["mean"])

        if len(means) < 2:
            continue

        # Get all per-question values
        all_values = []
        for axis in MOOD_AXES:
            if axis in axes_data and "values" in axes_data[axis]:
                all_values.append(axes_data[axis]["values"])

        if not all_values:
            continue

        # Build matrix: questions x axes
        matrix = np.array(all_values).T  # (n_questions, n_axes)

        # PCA via SVD
        centered = matrix - matrix.mean(axis=0)
        if centered.shape[0] < 2:
            continue

        _, s, _ = np.linalg.svd(centered, full_matrices=False)
        explained = (s ** 2) / (s ** 2).sum()

        # Effective dimensionality (participation ratio)
        eff_dim = (np.sum(explained) ** 2) / np.sum(explained ** 2)

        results[model_key] = {
            "pc1_variance": float(explained[0]),
            "pc2_variance": float(explained[1]) if len(explained) > 1 else 0,
            "effective_dimensionality": float(eff_dim),
            "explained_variance": [float(e) for e in explained],
        }

    return results


def compute_dead_zones(baselines: dict) -> dict:
    """Detect dead zones (axes where model is stuck near 0)."""
    results = {}
    dead_threshold = 0.15  # |mean| < threshold = dead zone

    for model_key, data in baselines.items():
        axes_data = data.get("axes", {})
        dead_axes = []
        alive_axes = []

        for axis in MOOD_AXES:
            if axis not in axes_data:
                continue
            mean = abs(axes_data[axis]["mean"])
            std = axes_data[axis]["std"]

            if mean < dead_threshold and std < 0.2:
                dead_axes.append(axis)
            else:
                alive_axes.append(axis)

        results[model_key] = {
            "n_dead": len(dead_axes),
            "n_alive": len(alive_axes),
            "dead_axes": dead_axes,
            "alive_axes": alive_axes,
        }

    return results


def compute_cross_axis_correlations(baselines: dict) -> dict:
    """Compute correlation matrix between axes for each model."""
    results = {}

    for model_key, data in baselines.items():
        axes_data = data.get("axes", {})
        all_values = {}
        for axis in MOOD_AXES:
            if axis in axes_data and "values" in axes_data[axis]:
                all_values[axis] = axes_data[axis]["values"]

        if len(all_values) < 2:
            continue

        axes_list = sorted(all_values.keys())
        n = len(axes_list)
        corr_matrix = np.zeros((n, n))

        for i, a1 in enumerate(axes_list):
            for j, a2 in enumerate(axes_list):
                v1, v2 = np.array(all_values[a1]), np.array(all_values[a2])
                min_len = min(len(v1), len(v2))
                if min_len >= 3:
                    corr_matrix[i, j] = np.corrcoef(v1[:min_len], v2[:min_len])[0, 1]

        # Max absolute off-diagonal correlation
        mask = ~np.eye(n, dtype=bool)
        max_abs_corr = float(np.max(np.abs(corr_matrix[mask]))) if n > 1 else 0

        results[model_key] = {
            "axes": axes_list,
            "correlation_matrix": corr_matrix.tolist(),
            "max_abs_correlation": max_abs_corr,
        }

    return results


def run_analysis(force: bool = False):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    baselines = load_all_baselines()
    drifts = load_all_drift()

    if not baselines:
        print("No baseline data found. Run 02_baseline first.", flush=True)
        return

    meta = {
        "_meta": {
            "script": "scripts/06_analysis.py",
            "timestamp": datetime.now().isoformat(),
            "models_baseline": list(baselines.keys()),
            "models_drift": list(drifts.keys()),
        }
    }

    # PCA
    pca = compute_pca_analysis(baselines)
    pca["_meta"] = meta["_meta"]
    with open(OUTPUT_DIR / "pca_dimensionality.json", "w") as f:
        json.dump(pca, f, indent=2)
    print(f"PCA analysis: {len(pca) - 1} models", flush=True)

    # Dead zones
    dead = compute_dead_zones(baselines)
    dead["_meta"] = meta["_meta"]
    with open(OUTPUT_DIR / "dead_zones.json", "w") as f:
        json.dump(dead, f, indent=2)
    print(f"Dead zone analysis: {len(dead) - 1} models", flush=True)

    # Correlations
    corr = compute_cross_axis_correlations(baselines)
    corr["_meta"] = meta["_meta"]
    with open(OUTPUT_DIR / "cross_axis_correlations.json", "w") as f:
        json.dump(corr, f, indent=2)
    print(f"Cross-axis correlations: {len(corr) - 1} models", flush=True)

    # Drift summary (if available)
    if drifts:
        drift_summary = {"_meta": meta["_meta"]}
        for model_key, data in drifts.items():
            ds = data.get("drift_summary", {})
            if not ds:
                # New format: conflict_drift with different field names
                cd = data.get("conflict_drift", {})
                if cd:
                    ds = {}
                    for axis, vals in cd.items():
                        if not isinstance(vals, dict):
                            continue
                        ds[axis] = {
                            "mean_slope": vals.get("mean_conflict_slope", 0),
                            "std_slope": vals.get("std_conflict_slope", 0),
                            "mean_start": vals.get("mean_start", 0),
                            "mean_end": vals.get("mean_end_conflict", 0),
                            "mean_delta": vals.get("mean_end_conflict", 0) - vals.get("mean_start", 0),
                        }
            drift_summary[model_key] = ds
        with open(OUTPUT_DIR / "drift_summary.json", "w") as f:
            json.dump(drift_summary, f, indent=2)
        print(f"Drift summary: {len(drift_summary) - 1} models", flush=True)

    # Fingerprint comparison
    fingerprints = {"_meta": meta["_meta"]}
    for model_key, data in baselines.items():
        axes_data = data.get("axes", {})
        fingerprints[model_key] = {axis: axes_data[axis]["mean"] for axis in MOOD_AXES if axis in axes_data}
    with open(OUTPUT_DIR / "fingerprints.json", "w") as f:
        json.dump(fingerprints, f, indent=2)
    print(f"Fingerprints: {len(fingerprints) - 1} models", flush=True)

    print(f"\nAll analysis saved to {OUTPUT_DIR}/", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Step 06: Cross-model analysis")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.check:
        preflight_check()
        return

    run_analysis(force=args.force)


if __name__ == "__main__":
    main()
