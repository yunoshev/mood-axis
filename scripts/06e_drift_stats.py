#!/usr/bin/env python3
"""Step 06e: Conflict drift statistics (CPU only).

Implements the analysis protocol promised in paper/sections/03_methodology.md
Section 3.6 ("Conflict Drift Protocol"):

- Per-scenario linear regression of axis projection over conflict turns
  (1..n_conflict_turns), averaged across scenarios, with a bootstrap CI
  (resampling scenarios, 1000 resamples).
- Same regression over the 10 neutral control conversations (all 16 turns),
  with its own bootstrap CI.
- Significant drift criterion: |conflict slope| > 2 * |control slope| AND the
  conflict-slope bootstrap CI excludes 0. Additionally, a two-sample
  permutation test (conflict vs. control per-scenario/per-control slopes,
  10000 permutations) reports a p-value for the same comparison.
- Magnitude: mean |delta projection| from turn 1 to the last conflict turn
  (and analogously turn 1 to turn 16 for controls).
- Recovery ratio over the 4 recovery turns: how much of the peak conflict
  deviation is undone by the last recovery turn.
- Length control: correlation between per-turn projection and per-turn
  token count, to check drift is not just a byproduct of response length.

Reads: data/drift/<model>.json
Writes: data/analysis/drift_stats.json

Usage:
    python3 -u scripts/06e_drift_stats.py
    python3 -u scripts/06e_drift_stats.py --check
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

DRIFT_DIR = PROJECT_ROOT / "data" / "drift"
OUTPUT_DIR = PROJECT_ROOT / "data" / "analysis"

MODEL_ORDER = ["qwen3.5_9b", "phi4", "deepseek_r1_14b",
               "ministral3_8b", "gpt_oss_20b",  # gemma3_12b removed (steering degeneration)
               # >20B cohort, measured in 4-bit (Section 3.10); entries are
               # skipped until their drift runs land
               "qwen3.6_27b", "mistral_small_3.2_24b", "qwen3.6_35b_a3b",
               "gemma4_26b_a4b"]

MODEL_LABELS = {
    "qwen3.5_9b": "Qwen 3.5 9B",
    "gemma3_12b": "Gemma 3 12B",
    "phi4": "Phi-4 14B",
    "deepseek_r1_14b": "DeepSeek-R1 14B",
    "ministral3_8b": "Ministral 3 8B",
    "gpt_oss_20b": "GPT-OSS 20B",
    "qwen3.6_27b": "Qwen3.6 27B",
    "mistral_small_3.2_24b": "Mistral Small 3.2 24B",
    "qwen3.6_35b_a3b": "Qwen3.6-35B-A3B",
    "gemma4_26b_a4b": "Gemma 4 26B-A4B",
}

N_BOOTSTRAP = 1000
N_PERMUTATIONS = 10000
RNG_SEED = 42


def preflight_check():
    print(f"Drift dir: {DRIFT_DIR}", flush=True)
    for m in MODEL_ORDER:
        f = DRIFT_DIR / f"{m}.json"
        print(f"  {m}: {'OK' if f.exists() else 'MISSING'} ({f})", flush=True)
    print(f"Output: {OUTPUT_DIR / 'drift_stats.json'}", flush=True)
    print(f"Axes: {MOOD_AXES}", flush=True)
    print("Preflight OK", flush=True)


def linreg_slope(x, y):
    """Simple OLS slope of y on x."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.all(x == x[0]):
        return 0.0
    x_mean = x.mean()
    y_mean = y.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom == 0:
        return 0.0
    return float(np.sum((x - x_mean) * (y - y_mean)) / denom)


def bootstrap_ci(slopes, n_boot=N_BOOTSTRAP, rng=None):
    """Bootstrap 95% CI on the mean of a list of per-unit slopes,
    resampling units (scenarios or controls) with replacement."""
    slopes = np.asarray(slopes, dtype=float)
    n = len(slopes)
    if n == 0:
        return (0.0, 0.0)
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = slopes[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (float(lo), float(hi))


def permutation_test(conflict_slopes, control_slopes, n_perm=N_PERMUTATIONS, rng=None):
    """Two-sample permutation test on |mean(conflict) - mean(control)|.
    Returns two-sided p-value."""
    conflict_slopes = np.asarray(conflict_slopes, dtype=float)
    control_slopes = np.asarray(control_slopes, dtype=float)
    if len(conflict_slopes) == 0 or len(control_slopes) == 0:
        return 1.0
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    observed = abs(conflict_slopes.mean() - control_slopes.mean())
    pooled = np.concatenate([conflict_slopes, control_slopes])
    n_c = len(conflict_slopes)
    n_total = len(pooled)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(n_total)
        a = pooled[perm[:n_c]]
        b = pooled[perm[n_c:]]
        stat = abs(a.mean() - b.mean())
        if stat >= observed:
            count += 1
    return float((count + 1) / (n_perm + 1))  # add-one smoothing


def safe_corr(x, y):
    """Pearson correlation, returns None if degenerate."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def analyze_model(model_key: str, data: dict) -> dict:
    axes = data.get("_meta", {}).get("axes", MOOD_AXES)
    scenarios = data["conflict_scenarios"]
    controls = data["neutral_controls"]

    rng = np.random.default_rng(RNG_SEED)

    # Collect all (proj, n_tokens) pairs across all turns/scenarios for the
    # length-control correlation, plus per-axis per-scenario slopes/deltas.
    result = {"axes": {}}

    for axis in axes:
        # Skip axes without full coverage (e.g. ministral3_8b conflict turns
        # were measured on 4 axes only; the rescued 30-control file lists 5
        # in _meta but the conflict values predate the axis).
        cov = sum(1 for sc in scenarios for t in sc["turns"] if axis in t.get("values", {}))
        tot = sum(len(sc["turns"]) for sc in scenarios)
        if tot and cov / tot < 0.95:
            print(f"  skip axis {axis}: coverage {cov}/{tot}", flush=True)
            continue

        # ---- Conflict scenarios ----
        conflict_slopes = []
        conflict_deltas = []          # |delta| turn1 -> last conflict turn
        recovery_ratios = []
        all_proj_for_length = []
        all_tokens_for_length = []

        for sc in scenarios:
            n_conf = sc["n_conflict_turns"]
            n_rec = sc["n_recovery_turns"]
            turns = sc["turns"]
            # turns are ordered 1..n_conf+n_rec
            conf_turns = [t for t in turns if t["turn"] <= n_conf]
            rec_turns = [t for t in turns if t["turn"] > n_conf]

            xs = [t["turn"] for t in conf_turns]
            ys = [t["values"][axis] for t in conf_turns]
            slope = linreg_slope(xs, ys)
            conflict_slopes.append(slope)

            turn1_val = conf_turns[0]["values"][axis]
            last_conf_val = conf_turns[-1]["values"][axis]
            conflict_deltas.append(abs(last_conf_val - turn1_val))

            # Recovery: fraction of the peak conflict deviation undone by
            # the last recovery turn.
            peak_val = max(conf_turns, key=lambda t: abs(t["values"][axis] - turn1_val))["values"][axis]
            peak_dev = abs(peak_val - turn1_val)
            if rec_turns:
                last_rec_val = rec_turns[-1]["values"][axis]
                if peak_dev < 1e-9:
                    # No real conflict deviation to recover from; skip.
                    recovery_ratio = None
                else:
                    recovery_ratio = 1.0 - abs(last_rec_val - turn1_val) / peak_dev
                if recovery_ratio is not None:
                    recovery_ratios.append(recovery_ratio)

            # Length control: use all turns of this scenario (conflict + recovery)
            for t in turns:
                all_proj_for_length.append(t["values"][axis])
                all_tokens_for_length.append(t.get("n_tokens", 0))

        # ---- Neutral controls ----
        # Window symmetry: conflict slopes are fitted on the conflict phase
        # only (turns 1..n_conflict_turns, typically 12), while control
        # conversations run 16 turns and end with a scripted gratitude
        # closing ("Thanks, that was really helpful...") that produces a
        # systematic warm jump on the final turn — a courtesy artifact, not
        # drift (e.g. qwen3.5_9b: +0.22 on warm_cold at turn 16 across all
        # 30 controls). The main estimate therefore fits controls on the
        # same window as the conflict fit; the full-window slope is kept as
        # a sensitivity variant (control_slope_full).
        ctrl_fit_window = (
            int(np.median([sc["n_conflict_turns"] for sc in scenarios]))
            if scenarios else None
        )
        control_slopes = []
        control_slopes_full = []
        control_deltas = []
        control_proj_for_length = []
        control_tokens_for_length = []
        for ctrl in controls:
            turns = ctrl["turns"]
            fit_turns = turns[:ctrl_fit_window] if ctrl_fit_window else turns
            xs = [t["turn"] for t in fit_turns]
            ys = [t["values"][axis] for t in fit_turns]
            control_slopes.append(linreg_slope(xs, ys))
            control_deltas.append(abs(ys[-1] - ys[0]))
            xs_f = [t["turn"] for t in turns]
            ys_f = [t["values"][axis] for t in turns]
            control_slopes_full.append(linreg_slope(xs_f, ys_f))
            for t in turns:
                control_proj_for_length.append(t["values"][axis])
                control_tokens_for_length.append(t.get("n_tokens", 0))

        mean_conflict_slope = float(np.mean(conflict_slopes))
        mean_control_slope = float(np.mean(control_slopes))
        conflict_ci = bootstrap_ci(conflict_slopes, rng=rng)
        control_ci = bootstrap_ci(control_slopes, rng=rng)

        ci_excludes_zero = not (conflict_ci[0] <= 0 <= conflict_ci[1])
        ratio_criterion = abs(mean_conflict_slope) > 2 * abs(mean_control_slope)
        significant = bool(ratio_criterion and ci_excludes_zero)

        p_value = permutation_test(conflict_slopes, control_slopes, rng=rng)

        # Length control correlations (all conflict-scenario turns; all control turns)
        corr_conflict = safe_corr(all_proj_for_length, all_tokens_for_length)
        corr_control = safe_corr(control_proj_for_length, control_tokens_for_length)

        result["axes"][axis] = {
            "conflict_slope": {
                "mean": mean_conflict_slope,
                "std": float(np.std(conflict_slopes)),
                "ci_95": list(conflict_ci),
            },
            "control_slope": {
                "mean": mean_control_slope,
                "std": float(np.std(control_slopes)),
                "ci_95": list(control_ci),
                "fit_window_turns": ctrl_fit_window,
            },
            "control_slope_full": {
                "mean": float(np.mean(control_slopes_full)),
                "std": float(np.std(control_slopes_full)),
                "note": "all turns incl. scripted gratitude closing (sensitivity)",
            },
            "significant_drift": significant,
            "criteria": {
                "ratio_gt_2x": ratio_criterion,
                "ci_excludes_zero": ci_excludes_zero,
            },
            "permutation_test": {
                "p_value": p_value,
                "n_permutations": N_PERMUTATIONS,
            },
            "magnitude": {
                "conflict_mean_abs_delta_turn1_to_last": float(np.mean(conflict_deltas)),
                "control_mean_abs_delta_turn1_to_last": float(np.mean(control_deltas)),
            },
            "recovery": {
                "mean_recovery_ratio": float(np.mean(recovery_ratios)) if recovery_ratios else None,
                "n_scenarios_with_recovery_signal": len(recovery_ratios),
                "n_scenarios_total": len(scenarios),
            },
            "length_control": {
                "corr_proj_vs_n_tokens_conflict": corr_conflict,
                "corr_proj_vs_n_tokens_control": corr_control,
            },
            "direction": "positive" if mean_conflict_slope > 0 else "negative",
        }

    return result


def print_summary_table(all_results: dict):
    print("", flush=True)
    print("=" * 130, flush=True)
    print(f"{'Model':<16}{'Axis':<24}{'Conflict slope [95% CI]':<34}{'Control slope':<16}{'Sig?':<7}{'p':<9}{'Recovery%':<10}", flush=True)
    print("-" * 130, flush=True)
    for model_key in MODEL_ORDER:
        if model_key not in all_results:
            continue
        label = MODEL_LABELS.get(model_key, model_key)
        axes_data = all_results[model_key]["axes"]
        for axis, stats in axes_data.items():
            cs = stats["conflict_slope"]
            ctl = stats["control_slope"]
            ci_str = f"{cs['mean']:+.4f} [{cs['ci_95'][0]:+.4f}, {cs['ci_95'][1]:+.4f}]"
            sig_str = "YES" if stats["significant_drift"] else "no"
            p_str = f"{stats['permutation_test']['p_value']:.4f}"
            rec = stats["recovery"]["mean_recovery_ratio"]
            rec_str = f"{rec * 100:.0f}%" if rec is not None else "n/a"
            print(f"{label:<16}{axis:<24}{ci_str:<34}{ctl['mean']:+.4f}{'':<9}{sig_str:<7}{p_str:<9}{rec_str:<10}", flush=True)
        print("-" * 130, flush=True)
    print("=" * 130, flush=True)


def print_key_findings(all_results: dict):
    print("", flush=True)
    print("Key findings:", flush=True)
    for model_key in MODEL_ORDER:
        if model_key not in all_results:
            continue
        label = MODEL_LABELS.get(model_key, model_key)
        axes_data = all_results[model_key]["axes"]
        sig_axes = [(ax, s) for ax, s in axes_data.items() if s["significant_drift"]]
        if not sig_axes:
            print(f"  {label}: no axis meets the significance criterion.", flush=True)
            continue
        parts = []
        for ax, s in sig_axes:
            d = s["direction"]
            rec = s["recovery"]["mean_recovery_ratio"]
            rec_str = f"recovery={rec * 100:.0f}%" if rec is not None else "recovery=n/a"
            parts.append(f"{ax} ({d}, slope={s['conflict_slope']['mean']:+.4f}, {rec_str})")
        print(f"  {label}: " + "; ".join(parts), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Run preflight check and exit")
    args = parser.parse_args()

    if args.check:
        preflight_check()
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for model_key in MODEL_ORDER:
        f = DRIFT_DIR / f"{model_key}.json"
        if not f.exists():
            print(f"WARNING: {f} not found, skipping {model_key}", flush=True)
            continue
        print(f"Analyzing {model_key} ...", flush=True)
        data = json.load(open(f))
        n_scenarios = len(data.get("conflict_scenarios", []))
        n_controls = len(data.get("neutral_controls", []))
        if n_scenarios == 0 or n_controls == 0:
            print(f"WARNING: {model_key} has empty conflict_scenarios/neutral_controls, skipping", flush=True)
            continue
        all_results[model_key] = analyze_model(model_key, data)
        print(f"  done ({n_scenarios} scenarios, {n_controls} controls)", flush=True)

    output = {
        "_meta": {
            "script": "scripts/06e_drift_stats.py",
            "timestamp": datetime.now().isoformat(),
            "n_bootstrap": N_BOOTSTRAP,
            "n_permutations": N_PERMUTATIONS,
            "rng_seed": RNG_SEED,
            "models_analyzed": list(all_results.keys()),
            "significance_criterion": "abs(conflict_slope) > 2*abs(control_slope) AND bootstrap 95% CI (1000 resamples over scenarios) excludes 0",
        },
        "models": all_results,
    }

    out_path = OUTPUT_DIR / "drift_stats.json"
    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2)
    print(f"\nWrote {out_path}", flush=True)

    print_summary_table(all_results)
    print_key_findings(all_results)


if __name__ == "__main__":
    main()
