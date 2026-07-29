#!/usr/bin/env python3
"""Cross-model steering analysis (CPU only).

Produces: data/analysis/steering_summary.json
Depends on: steering_basic.py (data/steering/<model>_basic.json)
            steering_judge.py (data/steering/judge_results.json)

Computes:
- Per-model steerability metrics (total effect, slopes, leakage, saturation)
- Cross-model rankings and cluster assignment
- Per-axis steerability heatmap data
- Auto-generated key findings

Usage:
    python scripts/steering_analyze.py
    python scripts/steering_analyze.py --check
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
from config.models import MODELS

STEERING_DIR = PROJECT_ROOT / "data" / "steering"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"
JUDGE_DIR = PROJECT_ROOT / "data" / "judge_steering"
DEAD_ZONES_FILE = ANALYSIS_DIR / "dead_zones.json"

# Paper models: core-5 roster (4 dense + 1 MoE). Steering data is the merged
# monolithic <model>_basic.json (zero-truncation v3trunc rerun).
PAPER_MODELS = ["qwen3.5_9b", "phi4", "deepseek_r1_14b", "ministral3_8b", "gpt_oss_20b"]

# Cluster thresholds (sum of total_effect across all axes)
CLUSTER_STEERABLE = 1.0
CLUSTER_MODERATE = 0.1


def preflight_check():
    """Show available data and exit."""
    print("=== Steering Analysis Preflight ===", flush=True)
    print(f"Steering dir: {STEERING_DIR}", flush=True)
    print(f"Output dir:   {ANALYSIS_DIR}", flush=True)
    print(f"MOOD_AXES:    {MOOD_AXES}", flush=True)
    print(f"Paper models: {PAPER_MODELS}", flush=True)
    print(flush=True)

    for model in PAPER_MODELS:
        path = STEERING_DIR / f"{model}_basic.json"
        backup = STEERING_DIR / f"{model}_basic_backup.json"
        if path.exists():
            data = json.load(open(path))
            axes = list(data.get("axes", {}).keys())
            print(f"  {model}: OK ({len(axes)} axes: {axes})", flush=True)
        elif backup.exists():
            data = json.load(open(backup))
            axes = list(data.get("axes", {}).keys())
            print(f"  {model}: BACKUP ({len(axes)} axes: {axes})", flush=True)
        else:
            print(f"  {model}: MISSING", flush=True)

    judge_path = JUDGE_DIR / "judge_results.json"
    if judge_path.exists():
        jdata = json.load(open(judge_path))
        print(f"\nJudge results: {list(jdata.keys())}", flush=True)
    else:
        print("\nJudge results: MISSING", flush=True)

    print("\nPreflight OK", flush=True)


def load_steering_data() -> dict:
    """Load steering data for all paper models.

    Falls back to _backup.json if _basic.json doesn't exist.
    Returns {model_key: parsed_json}.
    """
    data = {}
    for model in PAPER_MODELS:
        path = STEERING_DIR / f"{model}_basic.json"
        if not path.exists():
            # Try backup
            path = STEERING_DIR / f"{model}_basic_backup.json"
        if not path.exists():
            print(f"  WARNING: No steering data for {model}, skipping", flush=True)
            continue
        with open(path) as f:
            data[model] = json.load(f)
    return data


def load_judge_results() -> dict | None:
    """Load judge results (v2 flat schema) and adapt to a per-model dict.

    The v2 file (data/judge_steering/judge_results.json) stores a flat
    `results` list of {model, axis, pairwise_accuracy, likert_spearman_rho,
    pairwise_n, ...}. Aggregate to
    {model: {pairwise_accuracy, likert_rho_alpha, likert_rho_proj, per_axis}}
    restricted to the current core roster.
    """
    path = JUDGE_DIR / "judge_results.json"
    if not path.exists():
        return None
    with open(path) as f:
        raw = json.load(f)
    rows = raw.get("results", [])
    by_model: dict = {}
    for r in rows:
        m = r.get("model")
        if m not in PAPER_MODELS:
            continue
        by_model.setdefault(m, []).append(r)
    adapted: dict = {}
    for m, rs in by_model.items():
        pw_num = sum((r.get("pairwise_accuracy") or 0.0) * (r.get("pairwise_n") or 0) for r in rs)
        pw_den = sum((r.get("pairwise_n") or 0) for r in rs)
        pw = (pw_num / pw_den) if pw_den else None
        rhos = [r.get("likert_spearman_rho") for r in rs if r.get("likert_spearman_rho") is not None]
        rho = float(np.mean(rhos)) if rhos else None
        adapted[m] = {
            "pairwise_accuracy": pw,
            "likert_rho_alpha": rho,
            "likert_rho_proj": None,
            "per_axis": {
                r["axis"]: {
                    "pairwise_accuracy": r.get("pairwise_accuracy"),
                    "likert_rho": r.get("likert_spearman_rho"),
                }
                for r in rs
            },
        }
    return adapted or None


def load_dead_zones() -> dict | None:
    """Load dead zone analysis if available."""
    if not DEAD_ZONES_FILE.exists():
        return None
    with open(DEAD_ZONES_FILE) as f:
        return json.load(f)


def compute_axis_metrics(axis_data: dict, axis_name: str) -> dict:
    """Compute metrics for a single axis within a model.

    Args:
        axis_data: axis-level dict from the steering JSON
        axis_name: name of the steered axis (e.g. "warm_cold")

    Returns dict with slope, total_effect, effective_range, max_leakage, saturation_pct.
    """
    slope = axis_data.get("slope", 0.0)

    # Effective range and total effect
    eff = axis_data.get("effective_range", {})
    alpha_min = eff.get("alpha_min", 0.0)
    alpha_max = eff.get("alpha_max", 0.0)
    total_effect = eff.get("total_effect", 0.0)
    effective_slope = eff.get("effective_slope", 0.0)

    # If effective_range is missing or incomplete, compute total_effect from alpha_means
    if total_effect == 0.0 and "alpha_means" in axis_data:
        alpha_means = axis_data["alpha_means"]
        if alpha_means:
            vals = list(alpha_means.values())
            # alpha_means values are floats (mean projection on the steered axis)
            total_effect = max(vals) - min(vals)

    # Max leakage ratio
    leakage = axis_data.get("leakage", {})
    max_leakage = 0.0
    max_leakage_axis = None
    for other_axis, leak_info in leakage.items():
        if isinstance(leak_info, dict):
            ratio = abs(leak_info.get("ratio", 0.0))
        else:
            # In case leakage is stored as a plain float (ratio)
            ratio = abs(leak_info)
        if ratio > max_leakage:
            max_leakage = ratio
            max_leakage_axis = other_axis

    # Mean leakage ratio (average across all non-target axes)
    leakage_ratios = []
    for other_axis, leak_info in leakage.items():
        if isinstance(leak_info, dict):
            leakage_ratios.append(abs(leak_info.get("ratio", 0.0)))
        else:
            leakage_ratios.append(abs(leak_info))
    mean_leakage = float(np.mean(leakage_ratios)) if leakage_ratios else 0.0

    # Saturation: fraction of projections at |proj| >= 0.999
    n_saturated = 0
    n_total = 0
    results = axis_data.get("results", {})
    for alpha_key, result_list in results.items():
        for r in result_list:
            proj = r.get("projections", {}).get(axis_name, 0.0)
            n_total += 1
            if abs(proj) >= 0.999:
                n_saturated += 1
    saturation_pct = (n_saturated / n_total * 100) if n_total > 0 else 0.0

    return {
        "slope": round(float(slope), 6),
        "effective_slope": round(float(effective_slope), 6),
        "total_effect": round(float(total_effect), 4),
        "effective_range": {
            "alpha_min": float(alpha_min),
            "alpha_max": float(alpha_max),
        },
        "max_leakage": round(float(max_leakage), 4),
        "max_leakage_axis": max_leakage_axis,
        "mean_leakage": round(float(mean_leakage), 4),
        "saturation_pct": round(float(saturation_pct), 1),
        "n_samples": n_total,
    }


def compute_per_model(steering_data: dict, judge_data: dict | None) -> dict:
    """Compute per-model steering metrics.

    Returns {model_key: {total_effect, cluster, axes, judge}}.
    """
    per_model = {}

    for model_key, model_data in steering_data.items():
        axes_data = model_data.get("axes", {})
        axis_metrics = {}
        total_effect_sum = 0.0
        leakage_ratios_all = []

        for axis_name in MOOD_AXES:
            if axis_name not in axes_data:
                continue
            metrics = compute_axis_metrics(axes_data[axis_name], axis_name)
            axis_metrics[axis_name] = metrics
            total_effect_sum += abs(metrics["total_effect"])
            leakage_ratios_all.append(metrics["mean_leakage"])

        # Cluster assignment
        if total_effect_sum >= CLUSTER_STEERABLE:
            cluster = "steerable"
        elif total_effect_sum >= CLUSTER_MODERATE:
            cluster = "moderate"
        else:
            cluster = "resistant"

        # Judge results
        judge = None
        if judge_data and model_key in judge_data:
            jm = judge_data[model_key]
            judge = {
                "pairwise_accuracy": jm.get("pairwise_accuracy"),
                "likert_rho_alpha": jm.get("likert_rho_alpha"),
                "likert_rho_proj": jm.get("likert_rho_proj"),
            }

        per_model[model_key] = {
            "total_effect": round(total_effect_sum, 4),
            "cluster": cluster,
            "n_axes": len(axis_metrics),
            "mean_leakage": round(float(np.mean(leakage_ratios_all)), 4) if leakage_ratios_all else 0.0,
            "axes": axis_metrics,
            "judge": judge,
        }

    return per_model


def compute_rankings(per_model: dict) -> list:
    """Rank models by total_effect descending."""
    ranked = sorted(per_model.items(), key=lambda x: x[1]["total_effect"], reverse=True)
    return [
        {
            "model": model_key,
            "display_name": MODELS[model_key].display_name if model_key in MODELS else model_key,
            "total_effect": data["total_effect"],
            "cluster": data["cluster"],
            "n_axes": data["n_axes"],
        }
        for model_key, data in ranked
    ]


def compute_axis_heatmap(per_model: dict) -> dict:
    """Model x axis -> total_effect heatmap."""
    heatmap = {}
    for model_key, data in per_model.items():
        heatmap[model_key] = {}
        for axis_name in MOOD_AXES:
            if axis_name in data["axes"]:
                heatmap[model_key][axis_name] = data["axes"][axis_name]["total_effect"]
            else:
                heatmap[model_key][axis_name] = None
    return heatmap


def compute_findings(per_model: dict, dead_zones: dict | None) -> dict:
    """Auto-generate key findings."""
    findings = {}

    if not per_model:
        return findings

    # Most/least steerable model
    sorted_models = sorted(per_model.items(), key=lambda x: x[1]["total_effect"], reverse=True)
    most = sorted_models[0]
    least = sorted_models[-1]
    findings["most_steerable"] = {
        "model": most[0],
        "display_name": MODELS[most[0]].display_name if most[0] in MODELS else most[0],
        "total_effect": most[1]["total_effect"],
        "cluster": most[1]["cluster"],
    }
    findings["least_steerable"] = {
        "model": least[0],
        "display_name": MODELS[least[0]].display_name if least[0] in MODELS else least[0],
        "total_effect": least[1]["total_effect"],
        "cluster": least[1]["cluster"],
    }

    # Most/least responsive axis (mean total_effect across models)
    axis_effects = {a: [] for a in MOOD_AXES}
    for model_key, data in per_model.items():
        for axis_name in MOOD_AXES:
            if axis_name in data["axes"]:
                axis_effects[axis_name].append(abs(data["axes"][axis_name]["total_effect"]))

    axis_means = {}
    for axis_name, effects in axis_effects.items():
        if effects:
            axis_means[axis_name] = float(np.mean(effects))

    if axis_means:
        most_axis = max(axis_means, key=axis_means.get)
        least_axis = min(axis_means, key=axis_means.get)
        findings["most_responsive_axis"] = {
            "axis": most_axis,
            "mean_effect": round(axis_means[most_axis], 4),
        }
        findings["least_responsive_axis"] = {
            "axis": least_axis,
            "mean_effect": round(axis_means[least_axis], 4),
        }

    # Dead zone vs steerability correlation
    if dead_zones:
        models_with_both = []
        for model_key in per_model:
            if model_key in dead_zones:
                dz = dead_zones[model_key]
                # Count dead zones only among current MOOD_AXES
                dead_in_mood = [a for a in dz.get("dead_axes", []) if a in MOOD_AXES]
                models_with_both.append({
                    "model": model_key,
                    "n_dead": len(dead_in_mood),
                    "total_effect": per_model[model_key]["total_effect"],
                })

        if len(models_with_both) >= 3:
            from scipy.stats import spearmanr
            n_dead = [m["n_dead"] for m in models_with_both]
            effects = [m["total_effect"] for m in models_with_both]
            rho, p_value = spearmanr(n_dead, effects)
            findings["dead_zone_correlation"] = {
                "spearman_rho": round(float(rho), 4),
                "p_value": round(float(p_value), 4),
                "n_models": len(models_with_both),
                "detail": [
                    {"model": m["model"], "n_dead": m["n_dead"], "total_effect": m["total_effect"]}
                    for m in models_with_both
                ],
            }

    return findings


def print_summary(per_model: dict, rankings: list, axis_heatmap: dict, findings: dict):
    """Print formatted summary to stdout."""
    print("\n" + "=" * 70, flush=True)
    print("  STEERING ANALYSIS SUMMARY", flush=True)
    print("=" * 70, flush=True)

    # Rankings table
    print("\n--- Model Rankings (by total effect) ---\n", flush=True)
    print(f"  {'Model':<20} {'Total Effect':>12} {'Cluster':>12} {'Axes':>6}", flush=True)
    print(f"  {'-'*52}", flush=True)
    for r in rankings:
        print(f"  {r['display_name']:<20} {r['total_effect']:>12.4f} {r['cluster']:>12} {r['n_axes']:>6}", flush=True)

    # Per-axis heatmap
    print("\n--- Steerability Heatmap (total effect per axis) ---\n", flush=True)
    axes_present = [a for a in MOOD_AXES if any(axis_heatmap.get(m, {}).get(a) is not None for m in per_model)]
    header = f"  {'Model':<20}" + "".join(f" {a[:12]:>12}" for a in axes_present)
    print(header, flush=True)
    print(f"  {'-'*(20 + 13*len(axes_present))}", flush=True)

    for model_key in sorted(per_model.keys(), key=lambda m: per_model[m]["total_effect"], reverse=True):
        display = MODELS[model_key].display_name if model_key in MODELS else model_key
        row = f"  {display:<20}"
        for axis in axes_present:
            val = axis_heatmap.get(model_key, {}).get(axis)
            if val is not None:
                row += f" {val:>+12.4f}"
            else:
                row += f" {'N/A':>12}"
        print(row, flush=True)

    # Per-model detail
    print("\n--- Per-Model Detail ---\n", flush=True)
    for model_key in sorted(per_model.keys(), key=lambda m: per_model[m]["total_effect"], reverse=True):
        data = per_model[model_key]
        display = MODELS[model_key].display_name if model_key in MODELS else model_key
        print(f"  {display} (total={data['total_effect']:.4f}, cluster={data['cluster']})", flush=True)

        for axis_name in MOOD_AXES:
            if axis_name not in data["axes"]:
                continue
            am = data["axes"][axis_name]
            eff_range = f"[{am['effective_range']['alpha_min']:+.0f}, {am['effective_range']['alpha_max']:+.0f}]"
            print(
                f"    {axis_name:<25} effect={am['total_effect']:>+8.4f}  "
                f"slope={am['slope']:>+9.6f}  range={eff_range:>10}  "
                f"leak={am['max_leakage']:>5.2f}  sat={am['saturation_pct']:>5.1f}%",
                flush=True,
            )

        if data["judge"]:
            j = data["judge"]
            pw = j.get("pairwise_accuracy")
            rho = j.get("likert_rho_alpha")
            pw_str = f"{pw:.1%}" if pw is not None else "N/A"
            rho_str = f"{rho:.3f}" if rho is not None else "N/A"
            print(f"    Judge: pairwise={pw_str}, likert_rho={rho_str}", flush=True)
        print(flush=True)

    # Key findings
    print("--- Key Findings ---\n", flush=True)
    if "most_steerable" in findings:
        ms = findings["most_steerable"]
        print(f"  Most steerable:     {ms['display_name']} (total_effect={ms['total_effect']:.4f}, {ms['cluster']})", flush=True)
    if "least_steerable" in findings:
        ls = findings["least_steerable"]
        print(f"  Least steerable:    {ls['display_name']} (total_effect={ls['total_effect']:.4f}, {ls['cluster']})", flush=True)
    if "most_responsive_axis" in findings:
        ma = findings["most_responsive_axis"]
        print(f"  Most responsive:    {ma['axis']} (mean_effect={ma['mean_effect']:.4f})", flush=True)
    if "least_responsive_axis" in findings:
        la = findings["least_responsive_axis"]
        print(f"  Least responsive:   {la['axis']} (mean_effect={la['mean_effect']:.4f})", flush=True)
    if "dead_zone_correlation" in findings:
        dzc = findings["dead_zone_correlation"]
        print(f"  Dead zone vs steerability: rho={dzc['spearman_rho']:.3f}, p={dzc['p_value']:.3f} (n={dzc['n_models']})", flush=True)
        for d in dzc["detail"]:
            name = MODELS[d["model"]].display_name if d["model"] in MODELS else d["model"]
            print(f"    {name:<20} dead={d['n_dead']}, effect={d['total_effect']:.4f}", flush=True)

    print(flush=True)


def run_analysis():
    """Main analysis pipeline."""
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading steering data...", flush=True)
    steering_data = load_steering_data()
    if not steering_data:
        print("ERROR: No steering data found. Run steering_basic.py first.", flush=True)
        return

    print(f"  Loaded {len(steering_data)} models: {list(steering_data.keys())}", flush=True)

    print("Loading judge results...", flush=True)
    judge_data = load_judge_results()
    if judge_data:
        print(f"  Judge models: {list(judge_data.keys())}", flush=True)
    else:
        print("  No judge results found (optional)", flush=True)

    print("Loading dead zone analysis...", flush=True)
    dead_zones = load_dead_zones()
    if dead_zones:
        dz_models = [k for k in dead_zones if k != "_meta"]
        print(f"  Dead zones for {len(dz_models)} models", flush=True)
    else:
        print("  No dead zone data found (optional)", flush=True)

    # Compute
    print("\nComputing per-model metrics...", flush=True)
    per_model = compute_per_model(steering_data, judge_data)

    print("Computing rankings...", flush=True)
    rankings = compute_rankings(per_model)

    print("Computing axis heatmap...", flush=True)
    axis_heatmap = compute_axis_heatmap(per_model)

    print("Computing findings...", flush=True)
    findings = compute_findings(per_model, dead_zones)

    # Collect axes actually present in the data
    axes_present = sorted(set(
        axis for m in per_model.values() for axis in m["axes"]
    ))

    # Build output
    output = {
        "_meta": {
            "script": "scripts/steering_analyze.py",
            "timestamp": datetime.now().isoformat(),
            "models": list(per_model.keys()),
            "axes": axes_present,
            "cluster_thresholds": {
                "steerable": f">= {CLUSTER_STEERABLE}",
                "moderate": f">= {CLUSTER_MODERATE}",
                "resistant": f"< {CLUSTER_MODERATE}",
            },
        },
        "per_model": per_model,
        "rankings": rankings,
        "axis_heatmap": axis_heatmap,
        "findings": findings,
    }

    output_path = ANALYSIS_DIR / "steering_summary.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)

    # Print formatted summary
    print_summary(per_model, rankings, axis_heatmap, findings)

    print(f"Done. Output: {output_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Cross-model steering analysis")
    parser.add_argument("--check", action="store_true", help="Preflight check only")
    args = parser.parse_args()

    if args.check:
        preflight_check()
        return

    run_analysis()


if __name__ == "__main__":
    main()
