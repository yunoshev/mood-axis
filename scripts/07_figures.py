#!/usr/bin/env python3
"""07_figures.py -- publication figures (PNG 300dpi + PDF) for the TMLR paper.

Generates into paper/figures/:
  fig_fingerprints  heatmap of baseline means, 9 cohort models x 5 axes
  fig_drift         forest plot of the 14 composite-significant drift axes (Sec 5.2)
  fig_consistency   cross-set consistency (mean ICC) bar chart, 9 models
  fig_geometry      (a) PC1 share of pooled projections; (b) axis-vector cosines

Idempotent (overwrites outputs). Exits non-zero if any required data file is
missing. Reads only from data/; never touches paper sections.
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PAPER_ROOT = Path(__file__).resolve().parents[1]  # papers/mood-axis
DATA_DIR = PAPER_ROOT / "data"
OUT_DIR = PAPER_ROOT / "paper" / "figures"

# --------------------------------------------------------------------------
# Model registry (colors + display names) with hardcoded fallback
# --------------------------------------------------------------------------
_FALLBACK_META = {
    "qwen3.5_9b": ("Qwen3.5 9B", "#E9C46A"),
    "phi4": ("Phi-4 14B", "#7B68EE"),
    "deepseek_r1_14b": ("DeepSeek-R1 14B", "#00D9A5"),
    "ministral3_8b": ("Ministral 3 8B", "#7C3AED"),
    "gpt_oss_20b": ("GPT-OSS 20B", "#10A37F"),
    "qwen3.6_27b": ("Qwen3.6 27B", "#D97706"),
    "mistral_small_3.2_24b": ("Mistral Small 3.2 24B", "#5B21B6"),
    "qwen3.6_35b_a3b": ("Qwen3.6 35B-A3B (MoE)", "#B45309"),
    "gemma4_26b_a4b": ("Gemma 4 26B-A4B (MoE)", "#0F9D58"),
}

sys.path.insert(0, str(PAPER_ROOT))
try:
    from config.models import MODELS  # type: ignore

    MODEL_META = {k: (v.display_name, v.color) for k, v in MODELS.items()}
    # make sure every cohort model is present; fall back per-model if not
    for k, v in _FALLBACK_META.items():
        MODEL_META.setdefault(k, v)
except Exception as exc:  # pragma: no cover - config import is best-effort
    print(f"[warn] config.models import failed ({exc}); using fallback map")
    MODEL_META = dict(_FALLBACK_META)

try:
    from config.settings import AXIS_LABELS, MOOD_AXES  # type: ignore
except Exception as exc:  # pragma: no cover
    print(f"[warn] config.settings import failed ({exc}); using fallback axes")
    MOOD_AXES = [
        "warm_cold",
        "confident_cautious",
        "empathetic_analytical",
        "formal_casual",
        "verbose_concise",
    ]
    AXIS_LABELS = {
        "warm_cold": ("Warm", "Cold"),
        "confident_cautious": ("Confident", "Cautious"),
        "empathetic_analytical": ("Empathetic", "Analytical"),
        "formal_casual": ("Formal", "Casual"),
        "verbose_concise": ("Verbose", "Concise"),
    }

MAIN_CAST = ["qwen3.5_9b", "phi4", "deepseek_r1_14b", "ministral3_8b"]
GPT_OSS = ["gpt_oss_20b"]
OVER_20B = ["qwen3.6_27b", "mistral_small_3.2_24b", "qwen3.6_35b_a3b", "gemma4_26b_a4b"]
COHORT = MAIN_CAST + GPT_OSS + OVER_20B

AXES5 = [a for a in MOOD_AXES if a in AXIS_LABELS][:5]


def disp(model: str) -> str:
    return MODEL_META[model][0]


def color(model: str) -> str:
    return MODEL_META[model][1]


def axis_pair(axis: str) -> str:
    pos, neg = AXIS_LABELS[axis]
    return f"{pos} ↔ {neg}"  # Warm <-> Cold


# --------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#dcdcdc",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.8,
        "axes.axisbelow": True,
        "axes.edgecolor": "#555555",
        "axes.linewidth": 0.8,
        "xtick.color": "#555555",
        "ytick.color": "#555555",
        "xtick.labelcolor": "#222222",
        "ytick.labelcolor": "#222222",
        "text.color": "#222222",
        "axes.labelcolor": "#222222",
        "savefig.dpi": 300,
    }
)


def require(path: Path) -> Path:
    if not path.exists():
        print(f"ERROR: required data file missing: {path}", file=sys.stderr)
        sys.exit(1)
    return path


def load_json(path: Path) -> dict:
    with open(require(path)) as f:
        return json.load(f)


def save(fig, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / f"{name}.png"
    pdf = OUT_DIR / f"{name}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(
        f"[{name}] wrote {png.name} ({png.stat().st_size / 1024:.0f} KB), "
        f"{pdf.name} ({pdf.stat().st_size / 1024:.0f} KB)"
    )


# --------------------------------------------------------------------------
# Figure 1: baseline fingerprints heatmap
# --------------------------------------------------------------------------
def fig_fingerprints() -> None:
    mat = np.zeros((len(COHORT), len(AXES5)))
    for i, m in enumerate(COHORT):
        d = load_json(DATA_DIR / "baselines" / f"{m}.json")
        for j, ax_name in enumerate(AXES5):
            entry = d["axes"][ax_name]
            mat[i, j] = entry.get("mean", np.mean(entry["values"]))

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.grid(False)
    vlim = 0.5
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vlim, vmax=vlim, aspect="auto")

    ax.set_xticks(range(len(AXES5)))
    ax.set_xticklabels(
        [axis_pair(a).replace(" ↔ ", " ↔\n") for a in AXES5], fontsize=8
    )
    ax.set_yticks(range(len(COHORT)))
    ax.set_yticklabels([disp(m) for m in COHORT], fontsize=8)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # thin separators: main cast | GPT-OSS | >20B group
    for y in (len(MAIN_CAST) - 0.5, len(MAIN_CAST) + len(GPT_OSS) - 0.5):
        ax.axhline(y, color="#333333", lw=1.1)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(
                j,
                i,
                f"{v:+.2f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if abs(v) > 0.30 else "#1a1a1a",
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, ticks=[-0.5, -0.25, 0, 0.25, 0.5])
    cbar.set_label("Mean baseline projection (+ = first pole)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    cbar.outline.set_visible(False)

    fig.tight_layout()
    save(fig, "fig_fingerprints")


# --------------------------------------------------------------------------
# Figure 2: drift forest plot (Sec 5.2)
# --------------------------------------------------------------------------
def fig_drift() -> None:
    d = load_json(DATA_DIR / "analysis" / "drift_stats.json")
    models = d["models"]

    # composite-significant rows, cohort order + per-model JSON axis order
    # (matches the paper table: qwen3.5 ea/cc, phi4 wc, then the four >20B models)
    rows = []
    for m in COHORT:
        if m not in models:
            continue
        for ax_name, av in models[m]["axes"].items():
            if av.get("significant_drift"):
                rows.append((m, ax_name, av))
    if len(rows) != 14:
        print(f"[warn] expected 14 composite-significant rows, found {len(rows)}")

    small = [r for r in rows if r[0] in MAIN_CAST + GPT_OSS]
    large = [r for r in rows if r[0] in OVER_20B]

    # y layout: header, block rows, header, block rows (top to bottom)
    entries = [("header", "≤20B: de-escalation")]
    entries += [("row", r) for r in small]
    entries += [("header", ">20B: hardening")]
    entries += [("row", r) for r in large]

    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")

    ylabels, ypos = [], []
    y = 0
    block2_top = None
    for kind, payload in entries:
        if kind == "header":
            ylabels.append(payload)
            ypos.append(y)
            if payload.startswith(">"):
                block2_top = y
            y += 1
            continue
        m, ax_name, av = payload
        c = color(m)
        mean = av["conflict_slope"]["mean"]
        lo, hi = av["conflict_slope"]["ci_95"]
        ctrl = av["control_slope"]["mean"]
        ax.errorbar(
            mean,
            y,
            xerr=[[mean - lo], [hi - mean]],
            fmt="o",
            color=c,
            ecolor=c,
            elinewidth=1.6,
            capsize=2.5,
            capthick=1.4,
            ms=5.5,
            zorder=3,
        )
        ax.plot(
            ctrl,
            y,
            marker="D",
            ms=4.5,
            mfc="none",
            mec="#6f6f6f",
            mew=1.3,
            ls="none",
            zorder=2,
        )
        pos, neg = AXIS_LABELS[ax_name]
        ylabels.append(f"{disp(m)} · {pos.lower()}–{neg.lower()}")
        ypos.append(y)
        y += 1

    n_rows = y
    ax.set_ylim(n_rows - 0.5, -0.7)  # top-down
    ax.set_yticks(ypos)
    ax.set_yticklabels(ylabels)
    ax.tick_params(axis="y", length=0)

    # bold block headers + light band behind the >20B block
    for tick, (kind, _) in zip(ax.get_yticklabels(), entries):
        if kind == "header":
            tick.set_fontweight("bold")
            tick.set_fontsize(8.5)
    ax.axhspan(block2_top - 0.5, n_rows - 0.5, color="0.5", alpha=0.07, zorder=0)
    ax.axhline(block2_top - 0.5, color="#999999", lw=0.7, zorder=1)

    ax.axvline(0, color="#444444", lw=0.9, zorder=1)
    ax.set_xlabel("Axis-projection slope per turn")

    handles = [
        Line2D(
            [0], [0], marker="o", color="#333333", lw=1.6, ms=5.5,
            label="Conflict slope (95% CI)",
        ),
        Line2D(
            [0], [0], marker="D", mfc="none", mec="#6f6f6f", mew=1.3, ms=4.5,
            ls="none", label="Matched-control slope",
        ),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.95, edgecolor="#cccccc")

    fig.tight_layout()
    save(fig, "fig_drift")


# --------------------------------------------------------------------------
# Figure 3: cross-set consistency bars
# --------------------------------------------------------------------------
def fig_consistency() -> None:
    vals = []
    for m in COHORT:
        d = load_json(DATA_DIR / "stability" / f"{m}.json")
        vals.append(d["icc"]["mean_icc"])

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    # NB: set spine position BEFORE styling tick labels -- Spine.set_position()
    # resets the axis ticks and would wipe rotation/fontsize set earlier.
    ax.spines["bottom"].set_position(("data", -0.09))
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")

    x = np.arange(len(COHORT))
    for i, (m, v) in enumerate(zip(COHORT, vals)):
        hatch = "//" if m == "gpt_oss_20b" else None
        ax.bar(
            i,
            v,
            width=0.68,
            color=color(m),
            edgecolor="white",
            linewidth=0.8,
            hatch=hatch,
            zorder=3,
        )
        ax.text(i, v + 0.018, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)

    gpt_i = COHORT.index("gpt_oss_20b")
    ax.annotate(
        "format confound → §6",
        xy=(gpt_i, vals[gpt_i] + 0.055),
        xytext=(gpt_i, vals[gpt_i] + 0.22),
        ha="center",
        fontsize=7.5,
        color="#333333",
        arrowprops=dict(arrowstyle="-", color="#777777", lw=0.8),
    )

    ax.axhline(0, ls="--", color="#666666", lw=1.0, zorder=2)
    ax.text(
        -0.35, -0.045, "random-direction null ≈ 0",
        fontsize=7.5, color="#555555", va="top", ha="left",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([disp(m) for m in COHORT], rotation=32, ha="right", fontsize=7.5)
    ax.set_ylim(-0.09, 1.0)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("Cross-set axis consistency (mean ICC)")
    ax.tick_params(axis="x", length=0)

    fig.tight_layout()
    save(fig, "fig_consistency")


# --------------------------------------------------------------------------
# Figure 4: axis geometry (PC1 share + cosine matrix)
# --------------------------------------------------------------------------
def fig_geometry() -> None:
    d = load_json(DATA_DIR / "analysis" / "axis_geometry.json")
    models = d["models"]
    panel_a_models = [m for m in MAIN_CAST + GPT_OSS if m in models]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7.6, 3.4), gridspec_kw={"width_ratios": [1.1, 1.0]}
    )

    # --- (a) PC1 share of pooled projections
    ax1.grid(True, axis="y")
    ax1.grid(False, axis="x")
    for i, m in enumerate(panel_a_models):
        v = models[m]["cross_axis"]["pca_on_projections"]["pc1_share"]
        hatch = "//" if m == "gpt_oss_20b" else None
        ax1.bar(
            i, v, width=0.66, color=color(m), edgecolor="white",
            linewidth=0.8, hatch=hatch, zorder=3,
        )
        weight = "bold" if m == "gpt_oss_20b" else "normal"
        ax1.text(i, v + 0.015, f"{v:.2f}", ha="center", va="bottom",
                 fontsize=7.5, fontweight=weight)
    ax1.set_xticks(range(len(panel_a_models)))
    ax1.set_xticklabels([disp(m) for m in panel_a_models], rotation=32, ha="right", fontsize=7.5)
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel("PC1 share of variance")
    ax1.set_title("(a) PCA on pooled projections", fontsize=9)
    ax1.tick_params(axis="x", length=0)

    # --- (b) axis-vector cosine matrix, representative model
    rep = "qwen3.5_9b"
    ca = models[rep]["cross_axis"]
    order = ca["axes_order"]
    mat = np.array(ca["cosine_matrix"], dtype=float)
    labels = [AXIS_LABELS[a][0] for a in order]  # positive-pole shorthand

    ax2.grid(False)
    im = ax2.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
    ax2.set_xticks(range(len(order)))
    ax2.set_xticklabels(labels, rotation=38, ha="right", fontsize=7.5)
    ax2.set_yticks(range(len(order)))
    ax2.set_yticklabels(labels, fontsize=7.5)
    ax2.tick_params(length=0)
    for spine in ax2.spines.values():
        spine.set_visible(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax2.text(
                j, i, f"{v:+.2f}".replace("+1.00", "1.00"),
                ha="center", va="center", fontsize=7,
                color="white" if abs(v) > 0.60 else "#1a1a1a",
            )
    ax2.set_title(f"(b) Axis-vector cosines ({disp(rep)})", fontsize=9)
    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, ticks=[-1, -0.5, 0, 0.5, 1])
    cbar.ax.tick_params(labelsize=7)
    cbar.outline.set_visible(False)

    fig.tight_layout()
    save(fig, "fig_geometry")


def main() -> None:
    # fail fast on all required inputs
    for m in COHORT:
        require(DATA_DIR / "baselines" / f"{m}.json")
        require(DATA_DIR / "stability" / f"{m}.json")
    require(DATA_DIR / "analysis" / "drift_stats.json")
    require(DATA_DIR / "analysis" / "axis_geometry.json")

    fig_fingerprints()
    fig_drift()
    fig_consistency()
    fig_geometry()
    print(f"All figures written to {OUT_DIR}")


if __name__ == "__main__":
    main()
