#!/usr/bin/env python3
"""Compare base (pretrain-only) vs instruct models to isolate alignment effects.

Loads baseline and benchmark data for base/instruct pairs and generates:
1. Per-axis accuracy table: base vs instruct for each pair
2. Paired bar chart visualization
3. Summary: which axes "die" after alignment (RLHF dead zones)

Usage:
    python scripts/compare_base_instruct.py
    python scripts/compare_base_instruct.py --output-dir data/article/visualizations
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.models import MODELS, get_model_config
from config.settings import MOOD_AXES

BASELINE_DIR = project_root / "data" / "article" / "baselines"
BENCHMARK_DIR = project_root / "data"
VIZ_DIR = project_root / "data" / "article" / "visualizations"

AXIS_LABELS = {
    "warm_cold": ("Warm", "Cold"),
    "patient_irritated": ("Patient", "Irritated"),
    "confident_cautious": ("Confident", "Cautious"),
    "proactive_reluctant": ("Proactive", "Reluctant"),
    "empathetic_analytical": ("Empathetic", "Analytical"),
    "formal_casual": ("Formal", "Casual"),
    "verbose_concise": ("Verbose", "Concise"),
    "direct_evasive": ("Direct", "Evasive"),
}

# Base → instruct pairs (from model registry)
BASE_PAIRS = []
for key, cfg in MODELS.items():
    if cfg.is_base_model and cfg.instruct_counterpart:
        BASE_PAIRS.append((key, cfg.instruct_counterpart))


def load_baseline(model_key: str) -> dict | None:
    """Load baseline JSON for a model. Returns None if not found."""
    path = BASELINE_DIR / f"{model_key}_baseline.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_benchmark(model_key: str) -> dict | None:
    """Load benchmark results for a model.

    Benchmark saves to a single file, so we check if model matches.
    Also try per-model benchmark files if available.
    """
    # Try per-model file first
    per_model = BENCHMARK_DIR / f"benchmark_{model_key}.json"
    if per_model.exists():
        with open(per_model) as f:
            return json.load(f)

    # Fall back to single benchmark_results.json
    single = BENCHMARK_DIR / "benchmark_results.json"
    if single.exists():
        with open(single) as f:
            data = json.load(f)
        if data.get("model") == model_key:
            return data

    return None


def compute_benchmark_accuracy(benchmark: dict) -> dict:
    """Compute per-axis accuracy from benchmark results.

    Returns dict of {axis: accuracy} where accuracy = passed/total for that axis.
    """
    # Map scenario names to axes
    axis_scenarios = {
        "warm_cold": ["System Prompt - Warm", "Warmth Increase", "Warmth Decrease (Cold)"],
        "confident_cautious": ["System Prompt - Confident", "Confidence Increase", "Confidence Decrease (Cautious)"],
        "verbose_concise": ["System Prompt - Verbose", "Verbose Increase", "Verbose Decrease (Concise)"],
        "direct_evasive": ["System Prompt - Direct", "Direct Increase", "Direct Decrease (Evasive)"],
    }

    results = {}
    for axis, scenario_names in axis_scenarios.items():
        total = 0
        passed = 0
        for scenario in benchmark.get("scenarios", []):
            if scenario["name"] in scenario_names:
                total += 1
                if scenario["passed"]:
                    passed += 1
        if total > 0:
            results[axis] = passed / total
    return results


def compare_baselines(base_data: dict, instruct_data: dict) -> list[dict]:
    """Compare baseline measurements between base and instruct models.

    Returns list of per-axis comparison dicts.
    """
    rows = []
    for axis in MOOD_AXES:
        base_axis = base_data.get("axes", {}).get(axis, {})
        inst_axis = instruct_data.get("axes", {}).get(axis, {})

        base_mean = base_axis.get("mean", float("nan"))
        base_std = base_axis.get("std", float("nan"))
        inst_mean = inst_axis.get("mean", float("nan"))
        inst_std = inst_axis.get("std", float("nan"))

        delta = inst_mean - base_mean if not (np.isnan(base_mean) or np.isnan(inst_mean)) else float("nan")

        pos_label, neg_label = AXIS_LABELS.get(axis, (axis, ""))
        rows.append({
            "axis": axis,
            "label": f"{pos_label}/{neg_label}",
            "base_mean": base_mean,
            "base_std": base_std,
            "instruct_mean": inst_mean,
            "instruct_std": inst_std,
            "delta": delta,
        })
    return rows


def print_comparison_table(pair_name: str, rows: list[dict]):
    """Print a formatted comparison table."""
    print(f"\n{'='*70}")
    print(f"  {pair_name}")
    print(f"{'='*70}")
    print(f"  {'Axis':<25} {'Base':>10} {'Instruct':>10} {'Delta':>10}")
    print(f"  {'-'*55}")
    for row in rows:
        base_str = f"{row['base_mean']:+.3f}" if not np.isnan(row['base_mean']) else "N/A"
        inst_str = f"{row['instruct_mean']:+.3f}" if not np.isnan(row['instruct_mean']) else "N/A"
        delta_str = f"{row['delta']:+.3f}" if not np.isnan(row['delta']) else "N/A"
        print(f"  {row['label']:<25} {base_str:>10} {inst_str:>10} {delta_str:>10}")


def generate_paired_bar_chart(all_comparisons: dict, output_dir: Path):
    """Generate paired bar chart: base vs instruct per axis."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("[SKIP] plotly not installed — skipping bar chart")
        return

    n_pairs = len(all_comparisons)
    if n_pairs == 0:
        print("[SKIP] No comparison data available")
        return

    fig = make_subplots(
        rows=1, cols=n_pairs,
        subplot_titles=[name for name in all_comparisons.keys()],
        shared_yaxes=True,
    )

    axes_order = MOOD_AXES
    axis_short = [AXIS_LABELS[a][0] for a in axes_order]

    for col_idx, (pair_name, rows) in enumerate(all_comparisons.items(), 1):
        row_map = {r["axis"]: r for r in rows}
        base_vals = [row_map.get(a, {}).get("base_mean", 0) for a in axes_order]
        inst_vals = [row_map.get(a, {}).get("instruct_mean", 0) for a in axes_order]
        base_stds = [row_map.get(a, {}).get("base_std", 0) for a in axes_order]
        inst_stds = [row_map.get(a, {}).get("instruct_std", 0) for a in axes_order]

        fig.add_trace(go.Bar(
            name="Base" if col_idx == 1 else None,
            x=axis_short, y=base_vals,
            error_y=dict(type="data", array=base_stds, visible=True),
            marker_color="rgba(100, 149, 237, 0.7)",
            legendgroup="base",
            showlegend=(col_idx == 1),
        ), row=1, col=col_idx)

        fig.add_trace(go.Bar(
            name="Instruct" if col_idx == 1 else None,
            x=axis_short, y=inst_vals,
            error_y=dict(type="data", array=inst_stds, visible=True),
            marker_color="rgba(255, 99, 71, 0.7)",
            legendgroup="instruct",
            showlegend=(col_idx == 1),
        ), row=1, col=col_idx)

    fig.update_layout(
        title="Base vs Instruct: Baseline Temperament Profiles",
        barmode="group",
        height=500,
        width=350 * n_pairs,
        template="plotly_white",
    )
    fig.update_yaxes(title_text="Mood Score", row=1, col=1)

    output_path = output_dir / "fig_base_vs_instruct.png"
    fig.write_image(str(output_path), scale=2)
    print(f"Saved: {output_path}")

    html_path = output_dir / "fig_base_vs_instruct.html"
    fig.write_html(str(html_path))
    print(f"Saved: {html_path}")


def generate_dead_zone_chart(all_comparisons: dict, output_dir: Path):
    """Generate chart highlighting dead zones: axes where instruct accuracy drops."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("[SKIP] plotly not installed — skipping dead zone chart")
        return

    if not all_comparisons:
        return

    # Compute abs(delta) for each axis across all pairs
    # Dead zone = axis where instruct std is much lower than base (signal suppressed)
    # or where instruct value is near zero but base is not
    axes = MOOD_AXES
    axis_short = [AXIS_LABELS[a][0] for a in axes]

    fig = go.Figure()

    for pair_name, rows in all_comparisons.items():
        row_map = {r["axis"]: r for r in rows}
        # Use std ratio: instruct_std / base_std — low ratio = signal suppressed
        std_ratios = []
        for a in axes:
            r = row_map.get(a, {})
            base_std = r.get("base_std", 0)
            inst_std = r.get("instruct_std", 0)
            if base_std > 0.01:
                std_ratios.append(inst_std / base_std)
            else:
                std_ratios.append(1.0)

        fig.add_trace(go.Bar(
            name=pair_name,
            x=axis_short,
            y=std_ratios,
        ))

    fig.add_hline(y=1.0, line_dash="dash", line_color="gray",
                  annotation_text="No change")
    fig.update_layout(
        title="Signal Suppression: Instruct/Base Std Ratio by Axis",
        yaxis_title="Std Ratio (instruct / base)",
        barmode="group",
        height=450,
        template="plotly_white",
    )

    output_path = output_dir / "fig_dead_zone_analysis.png"
    fig.write_image(str(output_path), scale=2)
    print(f"Saved: {output_path}")


def summarize_dead_zones(all_comparisons: dict):
    """Print summary of potential RLHF dead zones."""
    print("\n" + "=" * 70)
    print("  DEAD ZONE ANALYSIS")
    print("=" * 70)
    print("  Axes where alignment may suppress signal (instruct std << base std)")
    print()

    dead_zones = []
    for pair_name, rows in all_comparisons.items():
        for row in rows:
            base_std = row.get("base_std", 0)
            inst_std = row.get("instruct_std", 0)
            if base_std > 0.01:
                ratio = inst_std / base_std
                if ratio < 0.5:  # Instruct has <50% of base variability
                    dead_zones.append({
                        "pair": pair_name,
                        "axis": row["label"],
                        "base_std": base_std,
                        "instruct_std": inst_std,
                        "ratio": ratio,
                    })

    if dead_zones:
        dead_zones.sort(key=lambda x: x["ratio"])
        for dz in dead_zones:
            print(f"  {dz['pair']:>30s} | {dz['axis']:<25s} | "
                  f"ratio={dz['ratio']:.2f} (base_std={dz['base_std']:.3f}, inst_std={dz['instruct_std']:.3f})")
    else:
        print("  No clear dead zones detected (all axes have instruct_std >= 50% of base_std)")
        print("  This may indicate that RLHF dead zones are axis-specific, not model-wide.")

    # Also check for axes where instruct mean collapses to near-zero
    print("\n  Axes where instruct mean collapses toward neutral (|mean| < 0.05):")
    collapsed = []
    for pair_name, rows in all_comparisons.items():
        for row in rows:
            inst_mean = row.get("instruct_mean", float("nan"))
            base_mean = row.get("base_mean", float("nan"))
            if not np.isnan(inst_mean) and not np.isnan(base_mean):
                if abs(inst_mean) < 0.05 and abs(base_mean) > 0.1:
                    collapsed.append({
                        "pair": pair_name,
                        "axis": row["label"],
                        "base_mean": base_mean,
                        "instruct_mean": inst_mean,
                    })

    if collapsed:
        for c in collapsed:
            print(f"  {c['pair']:>30s} | {c['axis']:<25s} | "
                  f"base={c['base_mean']:+.3f} → instruct={c['instruct_mean']:+.3f}")
    else:
        print("  None detected.")


def main():
    parser = argparse.ArgumentParser(description="Compare base vs instruct models")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for visualizations")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else VIZ_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Base vs Instruct Comparison")
    print(f"Pairs: {len(BASE_PAIRS)}")
    for base_key, inst_key in BASE_PAIRS:
        base_cfg = get_model_config(base_key)
        inst_cfg = get_model_config(inst_key)
        print(f"  {base_cfg.display_name}  vs  {inst_cfg.display_name}")

    all_comparisons = {}
    missing = []

    for base_key, inst_key in BASE_PAIRS:
        base_cfg = get_model_config(base_key)
        inst_cfg = get_model_config(inst_key)
        pair_name = f"{inst_cfg.display_name}"

        base_data = load_baseline(base_key)
        inst_data = load_baseline(inst_key)

        if not base_data:
            missing.append(f"{base_key} (base)")
            continue
        if not inst_data:
            missing.append(f"{inst_key} (instruct)")
            continue

        rows = compare_baselines(base_data, inst_data)
        all_comparisons[pair_name] = rows
        print_comparison_table(pair_name, rows)

    if missing:
        print(f"\n[WARNING] Missing baseline data for: {', '.join(missing)}")
        print("Run the pipeline for base models first:")
        print("  python cloud/cloud_runner.py --provider vast --model-set base")

    if all_comparisons:
        summarize_dead_zones(all_comparisons)
        generate_paired_bar_chart(all_comparisons, output_dir)
        generate_dead_zone_chart(all_comparisons, output_dir)

        # Save comparison data as JSON
        json_path = output_dir / "base_vs_instruct_comparison.json"
        serializable = {}
        for pair_name, rows in all_comparisons.items():
            serializable[pair_name] = [
                {k: (None if isinstance(v, float) and np.isnan(v) else v)
                 for k, v in row.items()}
                for row in rows
            ]
        with open(json_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\nSaved comparison data: {json_path}")
    else:
        print("\nNo comparison data available. Run base models first.")


if __name__ == "__main__":
    main()
