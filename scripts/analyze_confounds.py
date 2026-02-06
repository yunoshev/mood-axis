#!/usr/bin/env python3
"""Analyze potential confounds in Mood Axis measurements.

Computes:
1. Correlation matrix between axes (are axes independent?)
2. Correlation of axis values with response length (length confound)
3. Generates correlation heatmap visualization

Usage:
    python scripts/analyze_confounds.py [--baselines-dir PATH] [--plot]
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def load_baseline_data(baselines_dir: Path) -> Tuple[List[str], Dict[str, List[Dict]]]:
    """Load all baseline data from JSON files.

    Returns:
        Tuple of (axes list, dict mapping model -> list of per-sample readings)
    """
    all_data = {}
    all_axes_sets = []

    for json_file in sorted(baselines_dir.glob("*_baseline.json")):
        with open(json_file) as f:
            data = json.load(f)

        model = data["model"]
        # Handle both formats: older files use "baseline", newer use "axes"
        baseline = data.get("baseline") or data.get("axes")
        if baseline is None:
            print(f"WARNING: skipping {json_file.name} (no 'baseline' or 'axes' key)")
            continue

        all_axes_sets.append(set(baseline.keys()))
        all_data[model] = baseline

    # Use axes common to all models
    if all_axes_sets:
        common_axes = sorted(set.intersection(*all_axes_sets))
    else:
        common_axes = []

    return common_axes, all_data


def compute_axis_correlation_from_aggregates(
    all_data: Dict[str, Dict],
    axes: List[str],
) -> np.ndarray:
    """Compute correlation matrix from aggregate baseline data.

    Uses mean values across models as proxy (limited but informative).
    """
    n_axes = len(axes)
    n_models = len(all_data)

    # Build matrix: rows = models, cols = axes
    matrix = np.zeros((n_models, n_axes))

    for i, (model, baseline) in enumerate(all_data.items()):
        for j, axis in enumerate(axes):
            matrix[i, j] = baseline[axis]["mean"]

    # Compute correlation matrix
    corr_matrix = np.corrcoef(matrix.T)

    return corr_matrix


def print_correlation_matrix(corr_matrix: np.ndarray, axes: List[str]) -> str:
    """Format correlation matrix as readable table."""
    lines = []

    # Header
    header = "           " + "  ".join(f"{a[:8]:>8}" for a in axes)
    lines.append(header)
    lines.append("-" * len(header))

    # Rows
    for i, axis in enumerate(axes):
        row_values = []
        for j in range(len(axes)):
            val = corr_matrix[i, j]
            if i == j:
                row_values.append("   1.00 ")
            elif abs(val) > 0.7:
                row_values.append(f"  {val:+.2f}*")  # Flag high correlation
            else:
                row_values.append(f"  {val:+.2f} ")

        lines.append(f"{axis[:10]:<10} " + "".join(row_values))

    lines.append("")
    lines.append("* = |r| > 0.7 (potential confound)")

    return "\n".join(lines)


def analyze_variance_by_model(
    all_data: Dict[str, Dict],
    axes: List[str],
) -> Dict[str, Dict]:
    """Analyze within-model variance for each axis."""
    results = {}

    for axis in axes:
        stds = []
        means = []
        for model, baseline in all_data.items():
            stds.append(baseline[axis]["std"])
            means.append(baseline[axis]["mean"])

        results[axis] = {
            "mean_of_means": np.mean(means),
            "std_of_means": np.std(means),  # Between-model variance
            "mean_within_std": np.mean(stds),  # Within-model variance (avg)
            "between_within_ratio": np.std(means) / np.mean(stds) if np.mean(stds) > 0 else 0,
        }

    return results


def print_variance_analysis(variance: Dict[str, Dict]) -> str:
    """Format variance analysis as readable table."""
    lines = []
    lines.append("Axis             Mean    Between-σ  Within-σ   Ratio")
    lines.append("-" * 55)

    for axis, stats in variance.items():
        ratio_flag = "*" if stats["between_within_ratio"] > 1.0 else " "
        lines.append(
            f"{axis:<16} {stats['mean_of_means']:+.3f}   "
            f"{stats['std_of_means']:.3f}      "
            f"{stats['mean_within_std']:.3f}      "
            f"{stats['between_within_ratio']:.2f}{ratio_flag}"
        )

    lines.append("")
    lines.append("Ratio = between-model σ / within-model σ")
    lines.append("* = ratio > 1.0 (good discrimination between models)")

    return "\n".join(lines)


def generate_report(baselines_dir: Path, axes_dir: Path = None) -> str:
    """Generate full confound analysis report."""
    if axes_dir is None:
        axes_dir = baselines_dir.parent.parent / "axes"
    axes, all_data = load_baseline_data(baselines_dir)

    if not all_data:
        return "ERROR: No baseline data found"

    lines = []
    lines.append("=" * 60)
    lines.append("MOOD AXIS CONFOUND ANALYSIS")
    lines.append("=" * 60)
    lines.append(f"Models analyzed: {len(all_data)}")
    lines.append(f"Axes: {len(axes)}")
    lines.append(f"Models: {', '.join(all_data.keys())}")
    lines.append("")

    # Correlation matrix
    lines.append("=" * 60)
    lines.append("1. AXIS CORRELATION MATRIX (from model means)")
    lines.append("=" * 60)
    lines.append("")

    corr_matrix = compute_axis_correlation_from_aggregates(all_data, axes)
    lines.append(print_correlation_matrix(corr_matrix, axes))
    lines.append("")

    # Interpretation
    high_corr = []
    for i in range(len(axes)):
        for j in range(i + 1, len(axes)):
            if abs(corr_matrix[i, j]) > 0.7:
                high_corr.append((axes[i], axes[j], corr_matrix[i, j]))

    if high_corr:
        lines.append("⚠️  High correlations detected:")
        for a1, a2, r in high_corr:
            lines.append(f"   {a1} ↔ {a2}: r = {r:.2f}")
        lines.append("")
    else:
        lines.append("✓ No strong inter-axis correlations (|r| > 0.7)")
        lines.append("")

    # Variance analysis
    lines.append("=" * 60)
    lines.append("2. VARIANCE ANALYSIS")
    lines.append("=" * 60)
    lines.append("")

    variance = analyze_variance_by_model(all_data, axes)
    lines.append(print_variance_analysis(variance))
    lines.append("")

    # Summary
    lines.append("=" * 60)
    lines.append("3. SUMMARY FOR ARTICLE")
    lines.append("=" * 60)
    lines.append("")

    # Generate summary text
    n_high_corr = len(high_corr)
    good_discrimination = sum(1 for v in variance.values() if v["between_within_ratio"] > 1.0)

    if n_high_corr == 0:
        lines.append("✓ Axes show low inter-correlation (all |r| < 0.7)")
    else:
        lines.append(f"⚠️  {n_high_corr} axis pairs show high correlation (|r| > 0.7)")

    lines.append(f"✓ {good_discrimination}/{len(axes)} axes discriminate well between models")
    lines.append("")

    # Suggested text for article
    lines.append("Suggested disclaimer text:")
    lines.append("-" * 40)
    if n_high_corr == 0:
        lines.append('"Cross-axis correlations are low (all |r| < 0.7), suggesting')
        lines.append('axes capture relatively independent behavioral dimensions."')
    else:
        corr_pairs = ", ".join(f"{a1}↔{a2}" for a1, a2, _ in high_corr)
        lines.append(f'"Some axes show correlation ({corr_pairs}),')
        lines.append('suggesting partial overlap in measured behaviors."')

    # Cosine similarity of axis vectors
    lines.append("")
    cosine_report = compute_axis_cosine_similarity(axes_dir, list(all_data.keys()))
    lines.append(cosine_report)

    return "\n".join(lines)


def load_axis_vectors_for_model(axes_dir: Path, model_short: str) -> Dict[str, np.ndarray]:
    """Load axis vectors from per-model NPZ file."""
    path = axes_dir / f"{model_short}_axes.npz"
    if not path.exists():
        return {}
    data = np.load(path)
    axes_list = data["_axes"].tolist()
    return {axis: data[axis] for axis in axes_list}


def compute_axis_cosine_similarity(axes_dir: Path, models: List[str]) -> str:
    """Compute cosine similarity between all axis vector pairs per model.

    This answers: "are the axes measuring different directions in hidden state space?"
    Even if baseline means are correlated, low cosine similarity means axes point
    in different directions -- models just happen to behave similarly on them.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("AXIS VECTOR COSINE SIMILARITY")
    lines.append("=" * 60)
    lines.append("(Do axis vectors point in different directions?)")
    lines.append("")

    all_cosines = {}

    for model in models:
        vectors = load_axis_vectors_for_model(axes_dir, model)
        if not vectors:
            continue

        axes = sorted(vectors.keys())
        n = len(axes)
        cosine_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                vi = vectors[axes[i]]
                vj = vectors[axes[j]]
                cos_sim = np.dot(vi, vj) / (np.linalg.norm(vi) * np.linalg.norm(vj))
                cosine_matrix[i, j] = cos_sim

        lines.append(f"--- {model} (dim={len(list(vectors.values())[0])}) ---")

        # Header
        header = "              " + "  ".join(f"{a[:8]:>8}" for a in axes)
        lines.append(header)

        for i, axis in enumerate(axes):
            row = []
            for j in range(n):
                v = cosine_matrix[i, j]
                if i == j:
                    row.append("   1.00 ")
                elif abs(v) > 0.5:
                    row.append(f"  {v:+.2f}*")
                else:
                    row.append(f"  {v:+.2f} ")
            lines.append(f"{axis[:13]:<13} " + "".join(row))

        # Summary for this model
        off_diag = []
        for i in range(n):
            for j in range(i+1, n):
                off_diag.append(abs(cosine_matrix[i, j]))
        mean_cos = np.mean(off_diag)
        max_cos = np.max(off_diag)
        lines.append(f"  Mean |cos|: {mean_cos:.3f}, Max |cos|: {max_cos:.3f}")
        lines.append("")

        all_cosines[model] = {"mean": mean_cos, "max": max_cos}

    # Cross-model summary
    if all_cosines:
        lines.append("--- Summary across models ---")
        lines.append(f"{'Model':<15}  {'Mean |cos|':>10}  {'Max |cos|':>10}")
        lines.append("-" * 40)
        for model, stats in all_cosines.items():
            lines.append(f"{model:<15}  {stats['mean']:>10.3f}  {stats['max']:>10.3f}")
        overall_mean = np.mean([s["mean"] for s in all_cosines.values()])
        lines.append(f"{'OVERALL':<15}  {overall_mean:>10.3f}")
        lines.append("")
        if overall_mean < 0.3:
            lines.append("Interpretation: Axes are nearly orthogonal in hidden state space.")
            lines.append("High baseline correlations reflect model behavior, not axis overlap.")
        elif overall_mean < 0.5:
            lines.append("Interpretation: Axes have moderate overlap in hidden state space.")
        else:
            lines.append("Interpretation: Axes have substantial overlap -- consider merging.")

    return "\n".join(lines)


def save_report(report: str, output_path: Path) -> None:
    """Save report to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)
    print(f"Report saved to: {output_path}")


def create_correlation_heatmap(
    corr_matrix: np.ndarray,
    axes: List[str],
    output_path: Path,
) -> None:
    """Create and save correlation matrix heatmap."""
    if not HAS_PLOTLY:
        print("Plotly not available, skipping heatmap generation")
        return

    # Shorten axis names for display
    short_names = [a.replace("_", " ").title()[:12] for a in axes]

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=short_names,
        y=short_names,
        colorscale=[
            [0.0, "#2563EB"],   # Strong negative (blue)
            [0.3, "#93C5FD"],   # Weak negative (light blue)
            [0.5, "#F3F4F6"],   # Zero (gray)
            [0.7, "#FCA5A5"],   # Weak positive (light red)
            [1.0, "#DC2626"],   # Strong positive (red)
        ],
        zmin=-1,
        zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in corr_matrix],
        texttemplate="%{text}",
        textfont={"size": 11},
        hovertemplate="<b>%{x}</b> ↔ <b>%{y}</b><br>r = %{z:.2f}<extra></extra>",
    ))

    fig.update_layout(
        title={
            "text": "Axis Correlation Matrix",
            "x": 0.5,
            "font": {"size": 18},
        },
        width=700,
        height=600,
        xaxis={"tickangle": 45},
        yaxis={"autorange": "reversed"},
        margin={"l": 100, "r": 50, "t": 80, "b": 100},
    )

    # Add annotation for interpretation
    fig.add_annotation(
        x=0.5,
        y=-0.15,
        xref="paper",
        yref="paper",
        text="Red = positive correlation, Blue = negative correlation",
        showarrow=False,
        font={"size": 11, "color": "gray"},
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(output_path))
    print(f"Heatmap saved to: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze Mood Axis confounds")
    parser.add_argument(
        "--baselines-dir",
        type=Path,
        default=project_root / "data" / "article" / "baselines",
        help="Directory with baseline JSON files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file for report (default: print to stdout)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate correlation heatmap visualization",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=project_root / "data" / "article" / "visualizations" / "fig5_correlation_matrix.png",
        help="Output path for heatmap image",
    )
    args = parser.parse_args()

    # Load data and compute correlations
    axes, all_data = load_baseline_data(args.baselines_dir)

    if not all_data:
        print("ERROR: No baseline data found")
        sys.exit(1)

    corr_matrix = compute_axis_correlation_from_aggregates(all_data, axes)

    # Generate report
    report = generate_report(args.baselines_dir)
    print(report)

    if args.output:
        save_report(report, args.output)

    # Generate heatmap if requested
    if args.plot:
        create_correlation_heatmap(corr_matrix, axes, args.plot_output)
