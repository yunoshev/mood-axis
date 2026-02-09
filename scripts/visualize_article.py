#!/usr/bin/env python3
"""Generate all visualizations for the article "LLMs Have Temperaments".

Figures:
1. Spider charts - Individual temperament profiles per model
2. Heatmap - Baseline temperament comparison (models × axes)
3. Drift graphs - Mood changes over long conversations
4. Volatility bars - Stability comparison across models
5. Correlation matrix - How axes correlate with each other
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
import json

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import MOOD_AXES, AXIS_LABELS


# Color scheme
COLORS = {
    "warm_cold": "#FF6B35",
    "patient_irritated": "#00D9A5",
    "confident_cautious": "#A855F7",
    "proactive_reluctant": "#3B82F6",
    "empathetic_analytical": "#EC4899",
    "formal_casual": "#F97316",
    "verbose_concise": "#14B8A6",
}

MODEL_COLORS = {
    "qwen_7b": "#FF6B35",
    "llama_8b": "#00D9A5",
    "mistral_7b": "#A855F7",
    "gemma_9b": "#3B82F6",
    "yi_9b": "#EC4899",
    "deepseek_7b": "#F97316",
    "gpt_oss_20b": "#10A37F",
}

MODEL_DISPLAY = {
    'deepseek_7b': 'DeepSeek 7B',
    'qwen_7b': 'Qwen 2.5 7B',
    'llama_8b': 'Llama 3.1 8B',
    'mistral_7b': 'Mistral 7B',
    'yi_9b': 'Yi 1.5 9B',
    'gemma_9b': 'Gemma 2 9B',
    'qwen_1.5b': 'Qwen 1.5B',
    'smollm_1.7b': 'SmolLM 1.7B',
    'llama_1b': 'Llama 1B',
}


def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    """Convert hex color to rgba string."""
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


def load_baseline_data(data_dir: Path) -> Dict[str, dict]:
    """Load baseline data for all models."""
    baselines = {}
    baseline_dir = data_dir / "article" / "baselines"

    if not baseline_dir.exists():
        return baselines

    for file in baseline_dir.glob("*_baseline.json"):
        with open(file) as f:
            data = json.load(f)
            # Normalize: some files use "axes", some use "baseline"
            if "axes" in data and "baseline" not in data:
                data["baseline"] = data["axes"]
            model_short = data.get("model_short", file.stem.replace("_baseline", ""))
            baselines[model_short] = data

    return baselines


def load_drift_data(data_dir: Path) -> Dict[str, dict]:
    """Load drift data for all models."""
    drift_data = {}
    drift_dir = data_dir / "article" / "drift"

    if not drift_dir.exists():
        return drift_data

    for file in drift_dir.glob("*.json"):
        with open(file) as f:
            data = json.load(f)
            key = file.stem  # e.g., "qwen_7b_neutral_30turns"
            drift_data[key] = data

    return drift_data


def load_extended_drift_data(data_dir: Path) -> Dict[str, dict]:
    """Load extended drift data (multiple scenarios per model)."""
    drift_data = {}
    drift_dir = data_dir / "article" / "extended_drift"

    if not drift_dir.exists():
        return drift_data

    for file in drift_dir.glob("*_drift.json"):
        with open(file) as f:
            data = json.load(f)
            model = data.get("model", file.stem.replace("_drift", ""))
            drift_data[model] = data

    return drift_data


def create_spider_chart_for_model(
    baseline: dict,
    model_name: str,
    output_path: Optional[Path] = None,
) -> go.Figure:
    """Create spider chart for a single model's baseline temperament."""

    # Get axis names and values
    values = baseline.get("baseline", {})
    axes = list(AXIS_LABELS.keys())

    # Create labels: positive pole at top, negative at bottom
    categories = []
    r_values = []

    for axis in axes:
        pos, neg = AXIS_LABELS[axis]
        mean_val = values.get(axis, {}).get("mean", 0) if isinstance(values.get(axis), dict) else values.get(axis, 0)

        # Add positive pole
        categories.append(pos)
        r_values.append(max(0, mean_val))

    for axis in axes:
        pos, neg = AXIS_LABELS[axis]
        mean_val = values.get(axis, {}).get("mean", 0) if isinstance(values.get(axis), dict) else values.get(axis, 0)

        # Add negative pole
        categories.append(neg)
        r_values.append(max(0, -mean_val))

    # Close the polygon
    categories_closed = categories + [categories[0]]
    r_closed = r_values + [r_values[0]]

    fig = go.Figure()

    # Add trace
    fig.add_trace(go.Scatterpolar(
        r=r_closed,
        theta=categories_closed,
        fill='toself',
        fillcolor='rgba(78, 205, 196, 0.2)',
        line=dict(color='#4ECDC4', width=2),
        name=model_name,
    ))

    display_name = MODEL_DISPLAY.get(model_name, model_name)
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0.25, 0.5, 0.75, 1.0],
            ),
            bgcolor='white',
        ),
        showlegend=False,
        title=dict(
            text=f"Temperament Profile: {display_name}",
            x=0.5,
            font=dict(size=16),
        ),
        paper_bgcolor='white',
        plot_bgcolor='white',
        height=500,
        width=500,
    )

    if output_path:
        fig.write_image(str(output_path))
        fig.write_html(str(output_path.with_suffix('.html')))

    return fig


def create_spider_overlay(
    baselines: Dict[str, dict],
    output_path: Optional[Path] = None,
    models_order: Optional[List[str]] = None,
) -> go.Figure:
    """Create a single large spider chart with all models overlaid.

    Best used as hero/catch image for articles. All models shown on one chart
    with distinct colors and a shared legend.
    """

    MODEL_DISPLAY = {
        'deepseek_7b': 'DeepSeek 7B',
        'qwen_7b': 'Qwen 2.5 7B',
        'llama_8b': 'Llama 3.1 8B',
        'mistral_7b': 'Mistral 7B',
        'yi_9b': 'Yi 1.5 9B',
        'gemma_9b': 'Gemma 2 9B',
        'gpt_oss_20b': 'GPT-OSS 20B',
        'qwen_1.5b': 'Qwen 1.5B',
        'smollm_1.7b': 'SmolLM 1.7B',
        'llama_1b': 'Llama 1B',
    }

    SPIDER_COLORS = {
        'deepseek_7b': '#FF6B35',
        'qwen_7b': '#3B82F6',
        'llama_8b': '#10B981',
        'mistral_7b': '#8B5CF6',
        'yi_9b': '#EC4899',
        'gemma_9b': '#06B6D4',
        'gpt_oss_20b': '#10A37F',
        'qwen_1.5b': '#F59E0B',
        'smollm_1.7b': '#14B8A6',
        'llama_1b': '#6366F1',
    }

    ALL_AXES = [(ax, AXIS_LABELS[ax][0]) for ax in MOOD_AXES]

    if models_order:
        models = [m for m in models_order if m in baselines]
    else:
        models = list(baselines.keys())

    axes = [a[0] for a in ALL_AXES]
    theta = [a[1] for a in ALL_AXES]
    theta_closed = theta + [theta[0]]

    # Auto-scale based on data range
    all_values = []
    for model in models:
        baseline = baselines[model].get("baseline", {})
        for axis in axes:
            val = baseline.get(axis, {})
            mean_val = val.get("mean", 0) if isinstance(val, dict) else val
            all_values.append(mean_val)

    data_max = max(abs(v) for v in all_values) if all_values else 0.5
    scale_limit = data_max * 1.3
    scale_limit = max(scale_limit, 0.3)

    def to_r(val):
        return 1.0 + val / scale_limit

    fig = go.Figure()

    # Neutral reference ring
    fig.add_trace(go.Scatterpolar(
        r=[1.0] * (len(axes) + 1),
        theta=theta_closed,
        mode='lines',
        line=dict(color='rgba(150,150,150,0.4)', width=1, dash='dot'),
        showlegend=False,
        hoverinfo='skip',
    ))

    for model in models:
        baseline = baselines[model].get("baseline", {})

        r_values = []
        original_values = []
        for axis in axes:
            val = baseline.get(axis, {})
            mean_val = val.get("mean", 0) if isinstance(val, dict) else val
            original_values.append(mean_val)
            r_values.append(to_r(mean_val))

        r_closed = r_values + [r_values[0]]
        orig_closed = original_values + [original_values[0]]
        color = SPIDER_COLORS.get(model, '#888888')
        display_name = MODEL_DISPLAY.get(model, model)

        fig.add_trace(go.Scatterpolar(
            r=r_closed,
            theta=theta_closed,
            fill='toself',
            fillcolor=hex_to_rgba(color, 0.08),
            line=dict(color=color, width=2.5),
            marker=dict(size=5, color=color),
            name=display_name,
            customdata=[[f"{v:+.2f}"] for v in orig_closed],
            hovertemplate='%{theta}: %{customdata[0]}<extra>' + display_name + '</extra>',
        ))

    tick_step = round(scale_limit / 2, 2)
    tick_originals = [-tick_step, 0, tick_step]
    tick_r = [to_r(v) for v in tick_originals]
    tick_text = [f"{v:+.1f}" if v != 0 else "0" for v in tick_originals]

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 2],
                tickvals=tick_r,
                ticktext=tick_text,
                tickfont=dict(size=11, color='gray'),
                gridcolor='rgba(200,200,200,0.3)',
            ),
            angularaxis=dict(
                tickfont=dict(size=14, weight='bold'),
                gridcolor='rgba(200,200,200,0.3)',
                categoryorder='array',
                categoryarray=theta,
            ),
            bgcolor='rgba(248,250,252,1)',
        ),
        title=dict(
            text='Temperament Profiles Comparison',
            x=0.5,
            font=dict(size=20),
        ),
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='top',
            y=-0.08,
            xanchor='center',
            x=0.5,
            font=dict(size=12),
        ),
        height=700,
        width=750,
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(t=60, b=80, l=60, r=60),
    )

    if output_path:
        fig.write_image(str(output_path), scale=2)
        fig.write_html(str(output_path.with_suffix('.html')))

    return fig


def create_spider_small_multiples(
    baselines: Dict[str, dict],
    output_path: Optional[Path] = None,
    models_order: Optional[List[str]] = None,
) -> go.Figure:
    """Create small multiple spider charts for comparing model temperament profiles.

    Each model gets its own radar chart. Values are shifted so neutral (0) maps
    to r=1 on the polar axis, making the dotted neutral ring the visual baseline.
    """

    MODEL_DISPLAY = {
        'deepseek_7b': 'DeepSeek 7B',
        'qwen_7b': 'Qwen 2.5 7B',
        'llama_8b': 'Llama 3.1 8B',
        'mistral_7b': 'Mistral 7B',
        'yi_9b': 'Yi 1.5 9B',
        'gemma_9b': 'Gemma 2 9B',
        'gpt_oss_20b': 'GPT-OSS 20B',
        'qwen_1.5b': 'Qwen 1.5B',
        'smollm_1.7b': 'SmolLM 1.7B',
        'llama_1b': 'Llama 1B',
    }

    SPIDER_COLORS = {
        'deepseek_7b': '#FF6B35',
        'qwen_7b': '#3B82F6',
        'llama_8b': '#00D9A5',
        'mistral_7b': '#A855F7',
        'yi_9b': '#EC4899',
        'gemma_9b': '#06B6D4',
        'gpt_oss_20b': '#10A37F',
        'qwen_1.5b': '#F59E0B',
        'smollm_1.7b': '#14B8A6',
        'llama_1b': '#6366F1',
    }

    ALL_AXES = [(ax, AXIS_LABELS[ax][0]) for ax in MOOD_AXES]

    if models_order:
        models = [m for m in models_order if m in baselines]
    else:
        models = list(baselines.keys())

    n_models = len(models)
    if n_models == 0:
        return None

    axes = [a[0] for a in ALL_AXES]
    theta = [a[1] for a in ALL_AXES]
    theta_closed = theta + [theta[0]]

    # Compute data range to auto-scale the radial axis
    all_values = []
    for model in models:
        baseline = baselines[model].get("baseline", {})
        for axis in axes:
            val = baseline.get(axis, {})
            mean_val = val.get("mean", 0) if isinstance(val, dict) else val
            all_values.append(mean_val)

    # Tight scale: pad actual data range by 30% for visual breathing room
    data_max = max(abs(v) for v in all_values) if all_values else 0.5
    scale_limit = data_max * 1.3
    scale_limit = max(scale_limit, 0.3)  # minimum ±0.3

    # Radial mapping: original 0 → r=1, original ±scale_limit → r=1±1
    # So r = 1 + value/scale_limit, range [0, 2]
    def to_r(val):
        return 1.0 + val / scale_limit

    # Layout: up to 3 per row (2×3 grid for 6 models)
    n_cols = min(n_models, 3)
    n_rows = (n_models + n_cols - 1) // n_cols

    specs = [[{'type': 'polar'} for _ in range(n_cols)] for _ in range(n_rows)]
    subplot_titles = [MODEL_DISPLAY.get(m, m) for m in models]

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        specs=specs,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.10,
        vertical_spacing=0.15,
    )

    for i, model in enumerate(models):
        row = i // n_cols + 1
        col = i % n_cols + 1

        baseline = baselines[model].get("baseline", {})

        r_values = []
        original_values = []
        for axis in axes:
            val = baseline.get(axis, {})
            mean_val = val.get("mean", 0) if isinstance(val, dict) else val
            original_values.append(mean_val)
            r_values.append(to_r(mean_val))

        r_closed = r_values + [r_values[0]]
        orig_closed = original_values + [original_values[0]]

        color = SPIDER_COLORS.get(model, '#888888')

        # Neutral reference ring (dotted)
        fig.add_trace(
            go.Scatterpolar(
                r=[1.0] * (len(axes) + 1),
                theta=theta_closed,
                mode='lines',
                line=dict(color='rgba(150,150,150,0.5)', width=1, dash='dot'),
                showlegend=False,
                hoverinfo='skip',
            ),
            row=row, col=col,
        )

        # Model profile
        fig.add_trace(
            go.Scatterpolar(
                r=r_closed,
                theta=theta_closed,
                fill='toself',
                fillcolor=hex_to_rgba(color, 0.25),
                line=dict(color=color, width=2.5),
                name=MODEL_DISPLAY.get(model, model),
                showlegend=False,
                customdata=[[f"{v:+.2f}"] for v in orig_closed],
                hovertemplate='%{theta}: %{customdata[0]}<extra></extra>',
            ),
            row=row, col=col,
        )

    # Tick values in original scale
    tick_step = round(scale_limit / 2, 2)
    tick_originals = [-tick_step, 0, tick_step]
    tick_r = [to_r(v) for v in tick_originals]
    tick_text = [f"{v:+.2f}" if v != 0 else "0" for v in tick_originals]

    # Update all polar axes
    for i in range(n_models):
        polar_key = f'polar{i + 1}' if i > 0 else 'polar'
        fig.update_layout(**{
            polar_key: dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 2],
                    showticklabels=False,
                    gridcolor='rgba(200,200,200,0.3)',
                ),
                angularaxis=dict(
                    tickfont=dict(size=8),
                    gridcolor='rgba(200,200,200,0.3)',
                    categoryorder='array',
                    categoryarray=theta,
                ),
                bgcolor='rgba(250,250,250,1)',
            )
        })

    fig.update_layout(
        title=dict(
            text=(
                'Baseline Temperament Profiles'
                '<br><sup style="color:gray">'
                'Dotted ring = neutral (0). Outside = positive pole, inside = negative.'
                '</sup>'
            ),
            x=0.5,
            font=dict(size=16),
        ),
        height=500 if n_rows == 1 else 950,
        width=380 * n_cols,
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(t=100, b=40, l=70, r=70),
    )

    if output_path:
        fig.write_image(str(output_path), scale=2)
        fig.write_html(str(output_path.with_suffix('.html')))

    return fig


def create_heatmap(
    baselines: Dict[str, dict],
    output_path: Optional[Path] = None,
    show_std: bool = True,
) -> go.Figure:
    """Create heatmap comparing baseline temperaments across models."""

    if not baselines:
        print("No baseline data available for heatmap")
        return None

    models = list(baselines.keys())
    axes = list(AXIS_LABELS.keys())

    # Build matrix and text annotations
    z_data = []
    text_data = []
    for model in models:
        row = []
        text_row = []
        baseline = baselines[model].get("baseline", {})
        for axis in axes:
            val = baseline.get(axis, {})
            mean_val = val.get("mean", 0) if isinstance(val, dict) else val
            std_val = val.get("std", 0) if isinstance(val, dict) else 0
            row.append(mean_val)
            # Format as "mean±std" for cell annotation
            if show_std and std_val > 0:
                text_row.append(f"{mean_val:.2f}±{std_val:.2f}")
            else:
                text_row.append(f"{mean_val:.2f}")
        z_data.append(row)
        text_data.append(text_row)

    # Axis labels (positive pole names)
    x_labels = [AXIS_LABELS[axis][0] for axis in axes]

    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=x_labels,
        y=models,
        text=text_data,
        texttemplate="%{text}",
        textfont={"size": 10},
        colorscale='RdBu',
        zmid=0,
        zmin=-1,
        zmax=1,
        colorbar=dict(
            title="Value",
            tickvals=[-1, -0.5, 0, 0.5, 1],
        ),
    ))

    # Get sample count from first model
    first_model = list(baselines.values())[0]
    n_samples = first_model.get("num_samples", 30)

    fig.update_layout(
        title=dict(
            text=f"Baseline Temperament Comparison<br><sup>Values: mean ± std across {n_samples} baseline questions</sup>",
            x=0.5,
            font=dict(size=18),
        ),
        xaxis_title="Mood Axis (Positive Pole)",
        yaxis_title="Model",
        height=400,
        width=900,
    )

    if output_path:
        fig.write_image(str(output_path))
        fig.write_html(str(output_path.with_suffix('.html')))

    return fig


def create_drift_graph(
    drift_data: Dict[str, dict],
    scenario_filter: str = "neutral",
    axis: str = "warm_cold",
    output_path: Optional[Path] = None,
) -> go.Figure:
    """Create drift comparison graph showing how axis values change over turns."""

    fig = go.Figure()

    for key, data in drift_data.items():
        if scenario_filter not in key:
            continue

        model = key.split("_")[0] + "_" + key.split("_")[1]  # e.g., "qwen_7b"
        turns = data.get("mood_values", [])

        if not turns:
            continue

        x = list(range(1, len(turns) + 1))
        y = [turn.get(axis, 0) for turn in turns]

        color = MODEL_COLORS.get(model, "#666666")

        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines+markers',
            name=model,
            line=dict(color=color, width=2),
            marker=dict(size=4),
        ))

    pos_label, neg_label = AXIS_LABELS.get(axis, (axis, ""))

    fig.update_layout(
        title=dict(
            text=f"Drift Analysis: {pos_label}/{neg_label} ({scenario_filter} scenario)",
            x=0.5,
            font=dict(size=16),
        ),
        xaxis_title="Turn Number",
        yaxis_title=f"{pos_label} ←→ {neg_label}",
        yaxis=dict(range=[-1, 1]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        height=400,
        width=800,
    )

    if output_path:
        fig.write_image(str(output_path))
        fig.write_html(str(output_path.with_suffix('.html')))

    return fig


def create_drift_graph_with_ci(
    extended_drift: Dict[str, dict],
    axis: str = "warm_cold",
    category_filter: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> go.Figure:
    """Create drift graph with confidence bands aggregated across scenarios.

    Args:
        extended_drift: Dict mapping model -> drift data with scenarios
        axis: Which mood axis to plot
        category_filter: Optional filter for scenario category (e.g., "sarcasm", "frustration")
        output_path: Where to save the figure
    """
    fig = go.Figure()

    total_scenarios = 0

    for model, data in extended_drift.items():
        scenarios = data.get("scenarios", [])
        if not scenarios:
            continue

        # Filter scenarios by category if specified
        if category_filter:
            scenarios = [s for s in scenarios if category_filter in s.get("category", "")]

        if not scenarios:
            continue

        total_scenarios = max(total_scenarios, len(scenarios))

        # Collect per-turn values across all scenarios
        # Handle both formats: old (mood_values list) and new (turns with values)
        def get_mood_values(scenario):
            if "mood_values" in scenario:
                return scenario["mood_values"]
            elif "turns" in scenario:
                return [t.get("values", {}) for t in scenario["turns"]]
            return []

        max_turns = max(len(get_mood_values(s)) for s in scenarios)

        # Aggregate values per turn
        per_turn_values = [[] for _ in range(max_turns)]
        for scenario in scenarios:
            mood_values = get_mood_values(scenario)
            for i, mv in enumerate(mood_values):
                if axis in mv:
                    per_turn_values[i].append(mv[axis])

        # Compute mean and std per turn
        turns_with_data = []
        means = []
        stds = []
        ci_lower = []
        ci_upper = []

        for i, values in enumerate(per_turn_values):
            if len(values) >= 2:
                turns_with_data.append(i + 1)
                mean = np.mean(values)
                std = np.std(values, ddof=1)
                means.append(mean)
                stds.append(std)
                # 95% CI approximation: mean ± 1.96 * (std / sqrt(n))
                se = std / np.sqrt(len(values))
                ci_lower.append(mean - 1.96 * se)
                ci_upper.append(mean + 1.96 * se)

        if not turns_with_data:
            continue

        color = MODEL_COLORS.get(model, "#666666")

        # Add confidence band (fill between)
        fig.add_trace(go.Scatter(
            x=turns_with_data + turns_with_data[::-1],
            y=ci_upper + ci_lower[::-1],
            fill='toself',
            fillcolor=hex_to_rgba(color, 0.15),
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=False,
            name=f'{model} CI',
            hoverinfo='skip',
        ))

        # Add mean line
        display_name = MODEL_DISPLAY.get(model, model)
        fig.add_trace(go.Scatter(
            x=turns_with_data,
            y=means,
            mode='lines+markers',
            name=display_name,
            line=dict(color=color, width=2),
            marker=dict(size=4),
        ))

    pos_label, neg_label = AXIS_LABELS.get(axis, (axis, ""))
    category_str = f" ({category_filter})" if category_filter else ""

    fig.update_layout(
        title=dict(
            text=f"Drift Analysis: {pos_label}/{neg_label}{category_str}",
            x=0.5,
            font=dict(size=16),
        ),
        annotations=[dict(
            text=f"Shaded bands: 95% CI across {total_scenarios} conflict scenarios",
            xref="paper", yref="paper",
            x=0.5, y=-0.15,
            showarrow=False,
            font=dict(size=11, color="#888"),
        )],
        xaxis_title="Turn Number",
        yaxis_title=f"{pos_label} ←→ {neg_label}",
        yaxis=dict(autorange=True),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(b=80),
        height=550,
        width=850,
    )

    if output_path:
        fig.write_image(str(output_path))
        fig.write_html(str(output_path.with_suffix('.html')))

    return fig


def create_volatility_bars(
    drift_data: Dict[str, dict],
    scenario_filter: str = "neutral",
    output_path: Optional[Path] = None,
) -> go.Figure:
    """Create bar chart comparing volatility across models."""

    models = []
    volatilities = []

    for key, data in drift_data.items():
        if scenario_filter not in key:
            continue

        model = key.split("_")[0] + "_" + key.split("_")[1]
        metrics = data.get("metrics", {})
        volatility = metrics.get("volatility", {})
        overall = volatility.get("overall", 0)

        models.append(model)
        volatilities.append(overall)

    if not models:
        print(f"No volatility data found for scenario: {scenario_filter}")
        return None

    colors = [MODEL_COLORS.get(m, "#666666") for m in models]

    fig = go.Figure(data=[
        go.Bar(
            x=models,
            y=volatilities,
            marker_color=colors,
        )
    ])

    fig.update_layout(
        title=dict(
            text=f"Overall Volatility Comparison ({scenario_filter} scenario)",
            x=0.5,
            font=dict(size=16),
        ),
        xaxis_title="Model",
        yaxis_title="Volatility (Std of mood values)",
        height=400,
        width=600,
    )

    if output_path:
        fig.write_image(str(output_path))
        fig.write_html(str(output_path.with_suffix('.html')))

    return fig


def create_annotated_drift_graph(
    drift_data: dict,
    axis: str = "warm_cold",
    title: str = "Mood Drift Over Conversation",
    output_path: Optional[Path] = None,
) -> go.Figure:
    """Create annotated drift graph with colored zones like the reference image.

    This creates a line graph showing how an axis value changes over conversation
    turns, with:
    - Colored positive (blue) and negative (red) zones
    - Annotations showing quotes at key turns
    - Optional threshold/activation cap line
    """
    turns = drift_data.get("turns", [])
    mood_values = drift_data.get("mood_values", [])

    if not turns or not mood_values:
        print("No turn data available for annotated drift graph")
        return None

    # Extract data
    x = [t["turn"] for t in turns]
    y = [mv.get(axis, 0) for mv in mood_values]

    pos_label, neg_label = AXIS_LABELS.get(axis, ("Positive", "Negative"))

    fig = go.Figure()

    # Add colored zones (positive = blue, negative = red)
    max_x = max(x) + 1

    # Positive zone (blue)
    fig.add_shape(
        type="rect",
        x0=0, x1=max_x, y0=0, y1=1.1,
        fillcolor="rgba(59, 130, 246, 0.15)",
        line=dict(width=0),
        layer="below",
    )

    # Negative zone (red)
    fig.add_shape(
        type="rect",
        x0=0, x1=max_x, y0=-1.1, y1=0,
        fillcolor="rgba(239, 68, 68, 0.15)",
        line=dict(width=0),
        layer="below",
    )

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)

    # Add the main line
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='lines+markers',
        name=axis,
        line=dict(color='#374151', width=2),
        marker=dict(size=6, color='white', line=dict(color='#374151', width=2)),
    ))

    # Add annotations for interesting turns (first, min, max, last)
    annotations = []

    # Find interesting points
    min_idx = np.argmin(y)
    max_idx = np.argmax(y)

    # Annotate minimum point
    if min_idx > 0:
        response_text = turns[min_idx].get("response", "")[:80] + "..."
        annotations.append(dict(
            x=x[min_idx],
            y=y[min_idx],
            text=f'"{response_text}"',
            showarrow=True,
            arrowhead=2,
            ax=40,
            ay=40,
            bgcolor="rgba(239, 68, 68, 0.9)",
            font=dict(size=9, color="white"),
            borderpad=4,
        ))

    # Annotate maximum point (if different from min)
    if max_idx != min_idx and max_idx > 0:
        response_text = turns[max_idx].get("response", "")[:80] + "..."
        annotations.append(dict(
            x=x[max_idx],
            y=y[max_idx],
            text=f'"{response_text}"',
            showarrow=True,
            arrowhead=2,
            ax=-40,
            ay=-40,
            bgcolor="rgba(59, 130, 246, 0.9)",
            font=dict(size=9, color="white"),
            borderpad=4,
        ))

    # Add axis labels on the sides
    fig.add_annotation(
        x=-0.05, y=0.75,
        text=f"← {pos_label}",
        textangle=-90,
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=12, color="#3B82F6"),
    )
    fig.add_annotation(
        x=-0.05, y=0.25,
        text=f"← {neg_label}",
        textangle=-90,
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=12, color="#EF4444"),
    )

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=16),
        ),
        xaxis_title="Conversation Turn",
        yaxis_title=f"Projection on {pos_label}/{neg_label} Axis",
        yaxis=dict(range=[-1.1, 1.1], zeroline=True),
        xaxis=dict(range=[0, max_x]),
        annotations=annotations,
        height=450,
        width=700,
        showlegend=False,
        margin=dict(l=80),
    )

    if output_path:
        fig.write_image(str(output_path))
        fig.write_html(str(output_path.with_suffix('.html')))

    return fig


def create_3d_persona_space(
    model_embeddings: Dict[str, np.ndarray],
    labels: Optional[Dict[str, str]] = None,
    output_path: Optional[Path] = None,
) -> go.Figure:
    """Create 3D scatter plot showing models/personas in embedding space.

    This requires pre-computed 3D embeddings (e.g., from PCA or t-SNE).

    Args:
        model_embeddings: Dict mapping model name to 3D coordinates
        labels: Optional dict mapping model name to display label
        output_path: Path to save the figure
    """
    from sklearn.decomposition import PCA

    if not model_embeddings:
        print("No embedding data for 3D visualization")
        return None

    models = list(model_embeddings.keys())
    coords = np.array([model_embeddings[m] for m in models])

    # If embeddings are high-dimensional, reduce to 3D
    if coords.shape[1] > 3:
        pca = PCA(n_components=3)
        coords = pca.fit_transform(coords)

    labels = labels or {m: m for m in models}

    fig = go.Figure()

    # Add scatter points
    fig.add_trace(go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        mode='markers+text',
        marker=dict(
            size=10,
            color=[MODEL_COLORS.get(m, "#666666") for m in models],
            opacity=0.8,
        ),
        text=[labels.get(m, m) for m in models],
        textposition="top center",
        hoverinfo="text",
    ))

    fig.update_layout(
        title=dict(
            text="Model Temperaments in Embedding Space",
            x=0.5,
            font=dict(size=16),
        ),
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3",
            bgcolor="rgba(245, 245, 220, 0.5)",
        ),
        height=600,
        width=700,
    )

    if output_path:
        fig.write_image(str(output_path))
        fig.write_html(str(output_path.with_suffix('.html')))

    return fig


def create_multi_axis_drift_comparison(
    drift_data: Dict[str, dict],
    axes: List[str] = None,
    scenario_filter: str = "conflict",
    output_path: Optional[Path] = None,
) -> go.Figure:
    """Create multi-panel drift comparison for multiple axes."""

    if axes is None:
        axes = ["warm_cold", "patient_irritated", "empathetic_analytical"]

    n_axes = len(axes)

    fig = make_subplots(
        rows=n_axes, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[f"{AXIS_LABELS[a][0]} / {AXIS_LABELS[a][1]}" for a in axes],
    )

    for key, data in drift_data.items():
        if scenario_filter not in key:
            continue

        model = key.split("_")[0] + "_" + key.split("_")[1]
        turns = data.get("mood_values", [])

        if not turns:
            continue

        x = list(range(1, len(turns) + 1))
        color = MODEL_COLORS.get(model, "#666666")

        for i, axis in enumerate(axes):
            y = [turn.get(axis, 0) for turn in turns]

            fig.add_trace(
                go.Scatter(
                    x=x, y=y,
                    mode='lines',
                    name=model if i == 0 else None,
                    legendgroup=model,
                    showlegend=(i == 0),
                    line=dict(color=color, width=2),
                ),
                row=i+1, col=1
            )

    fig.update_yaxes(range=[-1, 1])
    fig.update_xaxes(title_text="Conversation Turn", row=n_axes, col=1)

    fig.update_layout(
        title=dict(
            text=f"Multi-Axis Drift Analysis ({scenario_filter} scenario)",
            x=0.5,
            font=dict(size=16),
        ),
        height=200 * n_axes + 100,
        width=800,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    if output_path:
        fig.write_image(str(output_path))
        fig.write_html(str(output_path.with_suffix('.html')))

    return fig


def create_correlation_matrix(
    drift_data: Dict[str, dict],
    output_path: Optional[Path] = None,
) -> go.Figure:
    """Create correlation matrix showing how axes correlate with each other."""

    # Collect all mood values across all scenarios and models
    all_values = {axis: [] for axis in MOOD_AXES}

    for key, data in drift_data.items():
        turns = data.get("mood_values", [])
        for turn in turns:
            for axis in MOOD_AXES:
                if axis in turn:
                    all_values[axis].append(turn[axis])

    # Compute correlation matrix
    axes = [a for a in MOOD_AXES if len(all_values[a]) > 10]

    if len(axes) < 2:
        print("Not enough data for correlation matrix")
        return None

    n_axes = len(axes)
    corr_matrix = np.zeros((n_axes, n_axes))

    for i, ax1 in enumerate(axes):
        for j, ax2 in enumerate(axes):
            min_len = min(len(all_values[ax1]), len(all_values[ax2]))
            if min_len > 10:
                corr = np.corrcoef(
                    all_values[ax1][:min_len],
                    all_values[ax2][:min_len]
                )[0, 1]
                corr_matrix[i, j] = corr
            else:
                corr_matrix[i, j] = 0

    labels = [AXIS_LABELS[a][0] for a in axes]

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=labels,
        y=labels,
        colorscale='RdBu',
        zmid=0,
        zmin=-1,
        zmax=1,
        colorbar=dict(title="Correlation"),
    ))

    fig.update_layout(
        title=dict(
            text="Axis Correlation Matrix",
            x=0.5,
            font=dict(size=16),
        ),
        height=500,
        width=600,
    )

    if output_path:
        fig.write_image(str(output_path))
        fig.write_html(str(output_path.with_suffix('.html')))

    return fig


def generate_all_figures(
    data_dir: Path,
    output_dir: Path,
    verbose: bool = True,
):
    """Generate all figures for the article."""

    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("="*60)
        print("Generating Article Figures")
        print("="*60)

    # Load data
    EXCLUDE_MODELS = {'gpt_oss_20b'}
    baselines = {k: v for k, v in load_baseline_data(data_dir).items() if k not in EXCLUDE_MODELS}
    drift_data = {k: v for k, v in load_drift_data(data_dir).items() if k not in EXCLUDE_MODELS}
    extended_drift = {k: v for k, v in load_extended_drift_data(data_dir).items() if k not in EXCLUDE_MODELS}

    if verbose:
        print(f"\nLoaded {len(baselines)} baseline profiles")
        print(f"Loaded {len(drift_data)} drift scenarios")
        print(f"Loaded {len(extended_drift)} extended drift profiles")

    # 1. Spider charts for each model
    if verbose:
        print("\n[1/6] Generating spider charts...")
    for model, baseline in baselines.items():
        fig = create_spider_chart_for_model(
            baseline,
            model,
            output_dir / f"fig1_spider_{model}.png"
        )
        if verbose:
            print(f"  - {model}: done")

    # 1b. Spider small multiples (7-9B models)
    if verbose:
        print("\n[1b/6] Generating spider small multiples...")
    if baselines:
        big_models = ['deepseek_7b', 'qwen_7b', 'llama_8b', 'mistral_7b', 'yi_9b', 'gemma_9b']
        create_spider_small_multiples(
            baselines,
            output_dir / "fig1_baseline_profiles.png",
            models_order=big_models,
        )
        # All models version
        all_models = ['deepseek_7b', 'qwen_7b', 'llama_8b', 'mistral_7b', 'yi_9b', 'gemma_9b',
                      'qwen_1.5b', 'smollm_1.7b', 'llama_1b']
        create_spider_small_multiples(
            baselines,
            output_dir / "fig1_baseline_profiles_all.png",
            models_order=all_models,
        )
        # Spider overlay (all models on one chart)
        create_spider_overlay(
            baselines,
            output_dir / "fig2_spider_comparison.png",
            models_order=big_models,
        )
        if verbose:
            print("  - Spider small multiples + overlay: done")
    else:
        print("  - Skipped (no baseline data)")

    # 2. Heatmap (now with std annotations)
    if verbose:
        print("\n[2/6] Generating heatmap with uncertainty...")
    if baselines:
        create_heatmap(
            baselines,
            output_dir / "fig2_heatmap_baselines.png",
            show_std=True,
        )
        if verbose:
            print("  - Heatmap with ±std: done")
    else:
        print("  - Skipped (no baseline data)")

    # 3. Legacy drift graphs (single scenario)
    if verbose:
        print("\n[3/6] Generating legacy drift graphs...")
    if drift_data:
        for axis in ["warm_cold", "patient_irritated", "empathetic_analytical"]:
            for scenario in ["neutral", "conflict"]:
                create_drift_graph(
                    drift_data,
                    scenario_filter=scenario,
                    axis=axis,
                    output_path=output_dir / f"fig3_drift_{axis}_{scenario}.png"
                )
        if verbose:
            print("  - Legacy drift graphs: done")
    else:
        print("  - Skipped (no drift data)")

    # 3b. NEW: Drift graphs with confidence bands (aggregated across scenarios)
    if verbose:
        print("\n[3b/6] Generating drift graphs with CI bands...")
    if extended_drift:
        for axis in ["warm_cold", "patient_irritated", "empathetic_analytical", "confident_cautious"]:
            create_drift_graph_with_ci(
                extended_drift,
                axis=axis,
                output_path=output_dir / f"fig3b_drift_ci_{axis}.png"
            )
        if verbose:
            print("  - Drift with CI bands: done")
    else:
        print("  - Skipped (no extended drift data)")

    # 4. Volatility bars
    if verbose:
        print("\n[4/6] Generating volatility charts...")
    if drift_data:
        for scenario in ["neutral", "conflict"]:
            create_volatility_bars(
                drift_data,
                scenario_filter=scenario,
                output_path=output_dir / f"fig4_volatility_{scenario}.png"
            )
        if verbose:
            print("  - Volatility bars: done")
    else:
        print("  - Skipped (no drift data)")

    # 5. Correlation matrix
    if verbose:
        print("\n[5/6] Generating correlation matrix...")
    if drift_data:
        create_correlation_matrix(
            drift_data,
            output_path=output_dir / "fig5_correlation_matrix.png"
        )
        if verbose:
            print("  - Correlation matrix: done")
    else:
        print("  - Skipped (no drift data)")

    if verbose:
        print("\n" + "="*60)
        print(f"All figures saved to: {output_dir}")
        print("="*60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate article figures")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(project_root / "data"),
        help="Data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(project_root / "data" / "article" / "visualizations"),
        help="Output directory for figures",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Less verbose output",
    )
    args = parser.parse_args()

    generate_all_figures(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        verbose=not args.quiet,
    )
