"""Beautiful spider/radar chart visualization for Mood Axis."""

import plotly.graph_objects as go
from typing import Dict, Optional, List
import numpy as np


# Color scheme - Neon on dark
COLORS = {
    # Axis colors (positive poles) - 4 axes
    "warm": "#FF6B35",
    "confident": "#A855F7",
    "verbose": "#14B8A6",
    "direct": "#8B5CF6",
    # Axis colors (negative poles) - 4 axes
    "cold": "#4ECDC4",
    "cautious": "#FBBF24",
    "concise": "#2DD4BF",
    "evasive": "#A78BFA",
    # Chart colors
    "fill_positive": "rgba(78, 205, 196, 0.25)",
    "fill_negative": "rgba(255, 107, 107, 0.2)",
    "line_main": "#4ECDC4",
    "line_glow": "rgba(78, 205, 196, 0.6)",
    "grid": "rgba(255, 255, 255, 0.08)",
    "grid_accent": "rgba(255, 255, 255, 0.15)",
    "bg": "rgba(15, 15, 25, 0.95)",
    "text": "rgba(255, 255, 255, 0.9)",
    "text_dim": "rgba(255, 255, 255, 0.5)",
}

# Axis configuration
AXES_CONFIG = [
    {"name": "Warm", "color": "#FF6B35", "angle": 90},
    {"name": "Patient", "color": "#00D9A5", "angle": 210},
    {"name": "Confident", "color": "#A855F7", "angle": 330},
]


def value_to_color(value: float, pos_color: str, neg_color: str) -> str:
    """Interpolate color based on value (-1 to +1)."""
    if value >= 0:
        alpha = min(1.0, 0.3 + abs(value) * 0.5)
        return pos_color.replace(")", f", {alpha})").replace("rgb", "rgba") if "rgba" not in pos_color else pos_color
    else:
        alpha = min(1.0, 0.3 + abs(value) * 0.5)
        return neg_color.replace(")", f", {alpha})").replace("rgb", "rgba") if "rgba" not in neg_color else neg_color


def create_spider_chart(
    values: Dict[str, float],
    previous_values: Optional[Dict[str, float]] = None,
    show_ghost: bool = True,
    animated: bool = True,
) -> go.Figure:
    """Create a beautiful spider/radar chart for mood visualization.

    Args:
        values: Current mood values for 4 bipolar axes
        previous_values: Previous mood values for ghost trail
        show_ghost: Whether to show previous values as ghost
        animated: Whether to enable smooth transitions

    Returns:
        Plotly Figure with the spider chart
    """
    # 8-axis radar: 4 positive poles + 4 negative poles
    categories = [
        'Warm', 'Confident', 'Verbose', 'Direct',
        'Cold', 'Cautious', 'Concise', 'Evasive'
    ]

    # Map 4 bipolar axes to 8 unipolar values
    def to_radar_values(v: Dict[str, float]) -> List[float]:
        warm = v.get('warm_cold', 0)
        confident = v.get('confident_cautious', 0)
        verbose = v.get('verbose_concise', 0)
        direct = v.get('direct_evasive', 0)
        return [
            max(0, warm),           # Warm (positive warm_cold)
            max(0, confident),      # Confident (positive confident_cautious)
            max(0, verbose),        # Verbose (positive verbose_concise)
            max(0, direct),         # Direct (positive direct_evasive)
            max(0, -warm),          # Cold (negative warm_cold)
            max(0, -confident),     # Cautious (negative confident_cautious)
            max(0, -verbose),       # Concise (negative verbose_concise)
            max(0, -direct),        # Evasive (negative direct_evasive)
        ]

    r_values = to_radar_values(values)

    # Determine dominant mood for fill color
    max_idx = np.argmax(r_values)
    axis_colors = [
        COLORS["warm"], COLORS["confident"], COLORS["verbose"], COLORS["direct"],
        COLORS["cold"], COLORS["cautious"], COLORS["concise"], COLORS["evasive"]
    ]
    dominant_color = axis_colors[max_idx]

    # Create fill gradient based on dominant mood
    fill_color = f"rgba({int(dominant_color[1:3], 16)}, {int(dominant_color[3:5], 16)}, {int(dominant_color[5:7], 16)}, 0.2)"
    line_color = dominant_color

    fig = go.Figure()

    # Layer 2: Ghost trace (previous values)
    if show_ghost and previous_values:
        r_prev = to_radar_values(previous_values)
        # Close the polygon
        r_prev_closed = r_prev + [r_prev[0]]
        categories_closed = categories + [categories[0]]

        fig.add_trace(go.Scatterpolar(
            r=r_prev_closed,
            theta=categories_closed,
            fill='toself',
            fillcolor="rgba(255, 255, 255, 0.03)",
            line=dict(
                color="rgba(255, 255, 255, 0.15)",
                width=1,
                dash="dot"
            ),
            name="Previous",
            hoverinfo="skip",
            showlegend=False,
        ))

    # Layer 3: Main mood shape with gradient effect
    r_closed = r_values + [r_values[0]]
    categories_closed = categories + [categories[0]]

    # Outer glow line
    fig.add_trace(go.Scatterpolar(
        r=[v * 1.02 for v in r_closed],  # Slightly larger for glow effect
        theta=categories_closed,
        mode='lines',
        line=dict(
            color=f"rgba({int(line_color[1:3], 16)}, {int(line_color[3:5], 16)}, {int(line_color[5:7], 16)}, 0.3)",
            width=8,
        ),
        hoverinfo='skip',
        showlegend=False,
    ))

    # Main shape
    fig.add_trace(go.Scatterpolar(
        r=r_closed,
        theta=categories_closed,
        fill='toself',
        fillcolor=fill_color,
        line=dict(
            color=line_color,
            width=3,
        ),
        name="Current Mood",
        hovertemplate="<b>%{theta}</b><br>%{r:.2f}<extra></extra>",
        showlegend=False,
    ))

    # Layer 4: Data points with individual colors
    for i, (cat, val) in enumerate(zip(categories, r_values)):
        if val > 0.05:  # Only show visible points
            fig.add_trace(go.Scatterpolar(
                r=[val],
                theta=[cat],
                mode='markers',
                marker=dict(
                    size=12,
                    color=axis_colors[i],
                    line=dict(color='white', width=2),
                    symbol='circle',
                ),
                hovertemplate=f"<b>{cat}</b><br>{{r:.2f}}<extra></extra>",
                showlegend=False,
            ))

    # Layout with dark theme
    fig.update_layout(
        polar=dict(
            bgcolor=COLORS["bg"],
            radialaxis=dict(
                visible=True,
                range=[0, 1.1],
                tickvals=[0.25, 0.5, 0.75, 1.0],
                ticktext=["", "0.5", "", "1.0"],
                gridcolor=COLORS["grid"],
                linecolor=COLORS["grid"],
                tickfont=dict(color=COLORS["text_dim"], size=9),
                tickangle=45,
            ),
            angularaxis=dict(
                type="category",
                categoryorder="array",
                categoryarray=categories,
                gridcolor=COLORS["grid_accent"],
                linecolor=COLORS["grid_accent"],
                tickfont=dict(
                    color=COLORS["text"],
                    size=13,
                    family="system-ui, -apple-system, sans-serif",
                ),
                direction="clockwise",
                rotation=90,  # Start from top
            ),
        ),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=70, r=70, t=50, b=50),
        height=380,
        font=dict(color=COLORS["text"]),
    )

    # Animation settings
    if animated:
        fig.update_layout(
            transition=dict(
                duration=500,
                easing="cubic-in-out",
            ),
        )

    return fig


def create_compact_spider(
    values: Dict[str, float],
    size: int = 200,
) -> go.Figure:
    """Create a compact version of the spider chart for smaller displays."""
    fig = create_spider_chart(values, show_ghost=False, animated=False)
    fig.update_layout(
        height=size,
        margin=dict(l=30, r=30, t=20, b=20),
        polar=dict(
            radialaxis=dict(
                tickfont=dict(size=7),
                tickvals=[0.5, 1.0],
                ticktext=["", "1"],
            ),
            angularaxis=dict(
                tickfont=dict(size=10),
            ),
        ),
    )
    return fig


def get_mood_glow_css() -> str:
    """Get CSS for glow effects around the spider chart."""
    return """
<style>
/* Container with animated gradient border */
.mood-spider-container {
    position: relative;
    padding: 4px;
    border-radius: 20px;
    background: linear-gradient(135deg,
        rgba(255, 107, 53, 0.3),
        rgba(78, 205, 196, 0.3),
        rgba(168, 85, 247, 0.3),
        rgba(0, 217, 165, 0.3)
    );
    background-size: 300% 300%;
    animation: gradient-shift 8s ease infinite;
}

@keyframes gradient-shift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Inner container */
.mood-spider-inner {
    background: rgba(15, 15, 25, 0.98);
    border-radius: 16px;
    padding: 10px;
}

/* Glow effect on the chart */
.mood-spider-container::before {
    content: '';
    position: absolute;
    top: -10px;
    left: -10px;
    right: -10px;
    bottom: -10px;
    background: inherit;
    border-radius: 30px;
    filter: blur(20px);
    opacity: 0.4;
    z-index: -1;
    animation: glow-pulse 3s ease-in-out infinite;
}

@keyframes glow-pulse {
    0%, 100% { opacity: 0.3; filter: blur(20px); }
    50% { opacity: 0.5; filter: blur(25px); }
}

/* Pulse animation when mood updates */
.mood-updated {
    animation: mood-pulse 0.6s ease-out;
}

@keyframes mood-pulse {
    0% { transform: scale(1); }
    30% { transform: scale(1.02); }
    100% { transform: scale(1); }
}

/* Metric cards below chart */
.mood-metric {
    display: inline-flex;
    align-items: center;
    padding: 6px 12px;
    margin: 4px;
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    font-size: 13px;
    transition: all 0.3s ease;
}

.mood-metric:hover {
    background: rgba(255, 255, 255, 0.1);
    border-color: rgba(255, 255, 255, 0.2);
}

.mood-metric .value {
    font-weight: 600;
    margin-left: 8px;
}

.mood-metric .delta {
    font-size: 11px;
    margin-left: 4px;
    opacity: 0.7;
}

.mood-metric .delta.positive { color: #00D9A5; }
.mood-metric .delta.negative { color: #FF6B6B; }

/* Axis labels with color indicators */
.axis-warm { border-left: 3px solid #FF6B35; }
.axis-confident { border-left: 3px solid #A855F7; }
.axis-verbose { border-left: 3px solid #14B8A6; }
.axis-direct { border-left: 3px solid #8B5CF6; }
</style>
"""


def create_metrics_html(
    values: Dict[str, float],
    deltas: Optional[Dict[str, float]] = None,
) -> str:
    """Create HTML for mood metrics display below the chart."""

    metrics = [
        ("Warm", "warm_cold", "#FF6B35", "axis-warm"),
        ("Confident", "confident_cautious", "#A855F7", "axis-confident"),
        ("Verbose", "verbose_concise", "#14B8A6", "axis-verbose"),
        ("Direct", "direct_evasive", "#8B5CF6", "axis-direct"),
    ]

    html_parts = ['<div style="text-align: center; margin-top: 10px;">']

    for label, key, color, css_class in metrics:
        value = values.get(key, 0)
        delta = deltas.get(key, 0) if deltas else 0

        # Format value and delta
        value_str = f"{value:+.2f}"

        if delta != 0:
            delta_str = f"({delta:+.2f})"
            delta_class = "positive" if delta > 0 else "negative"
        else:
            delta_str = "(=)"
            delta_class = ""

        html_parts.append(f'''
            <div class="mood-metric {css_class}">
                <span style="color: {color};">●</span>
                <span>{label}:</span>
                <span class="value">{value_str}</span>
                <span class="delta {delta_class}">{delta_str}</span>
            </div>
        ''')

    html_parts.append('</div>')

    return ''.join(html_parts)
