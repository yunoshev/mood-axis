#!/usr/bin/env python3
"""Step 06f: Instruction-following accuracy stats for Section 5.4a (CPU only).

IMPROVEMENT_PLAN.md Blocker 8: the McNemar test / Wilson CIs hardcoded into
paper/sections/05_results.md Section 5.4a were computed on an old 7-axis x
10-eval-question set (N=70, from the retired paper1_mood_axis/ 7-axis system).
The current canonical calibration corpus (data/calibration/<axis>.json) uses
5 axes x 15 eval questions = 75 (see paper_data.json total_eval_questions).
This script recomputes the same statistics on the CURRENT 75-question set
from data actually on disk — no model runs, CPU only.

Metric definition ("does the model follow the personality instruction?"):
for each of the 15 held-out eval questions per axis, the model generates a
response under the positive-pole instruction (e.g. "respond warmly") and
under the negative-pole instruction (e.g. "respond coldly"). Both responses'
hidden states are projected onto the model's OWN axis vector (the one fit
on that same model's 50 train questions, data/axes/<model>_axes.npz). The
item is classified "correct" iff proj(positive) > proj(negative) for that
specific question -- i.e. the instructed pole is on the expected side of
the model's own decision axis. This is an item-level (paired) refinement of
the existing "val_accuracy" already stored in data/axes/<model>_axes_meta.json;
that stored value is a pairwise AUC-style statistic (all 15x15 pos/neg pairs
compared, so n=225 "pairs" and not directly usable for a paired test against
a fixed set of items). Here we use only the PAIRED (same-question) comparison,
giving exactly 15 independent Bernoulli trials per axis, 75 per model total.

Pairing is by construction, not inferred: scripts/01_calibrate.py's
_build_eval_samples() iterates `for pole in [positive, negative]: for q in
eval_questions`, so val_pos_aggregated[i] and val_neg_aggregated[i] both
correspond to eval_questions[i] for a given axis, for every model (base and
instruct read the same data/calibration/<axis>.json, same order). This exact
pairing property is already relied on and documented by
scripts/06d_axis_geometry.py ("verified empirically against
scripts/01_calibrate.py's generation order").

Because both base and instruct models are scored against the SAME 75 items
(same axis, same eval question, same pole labels), base-vs-instruct accuracy
is a genuine matched-pairs comparison -> McNemar's test is applicable exactly
as the original (retired) analysis intended, just recomputed on real,
on-disk per-item outcomes instead of hand-written numbers.

Reads:
    data/axes/<model>_axes.npz            (axis_vector per axis)
    data/axes/<model>_hidden_states.npz   (val_pos_aggregated / val_neg_aggregated)
Writes:
    data/analysis/probe_stats_75q.json

Usage:
    python3 -u scripts/06f_probe_stats.py
    python3 -u scripts/06f_probe_stats.py --check
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy.stats import chi2 as chi2_dist
from scipy.stats import binom

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import MOOD_AXES

AXES_DIR = PROJECT_ROOT / "data" / "axes"
OUTPUT_DIR = PROJECT_ROOT / "data" / "analysis"

# The three base/instruct pairs discussed in paper/sections/05_results.md Sec 5.4a.
MODEL_PAIRS = {
    "llama_8b": "llama_8b_base",
    "qwen_7b": "qwen_7b_base",
    "mistral_7b": "mistral_7b_base",
}

DISPLAY_NAMES = {
    "llama_8b": "Llama-8B",
    "qwen_7b": "Qwen-7B",
    "mistral_7b": "Mistral-7B",
}

ALPHA = 0.05
N_MODELS_TESTED = 3
BONFERRONI_ALPHA = ALPHA / N_MODELS_TESTED

# Numbers currently hardcoded in paper/sections/05_results.md Sec 5.4a
# ("Statistical Details"), computed on the old 70-question (7-axis x 10)
# set. Kept here ONLY for the printed old-vs-new comparison -- never used
# in any computation.
OLD_PAPER_NUMBERS = {
    "llama_8b": {
        "n": 70, "base_correct": 60, "instruct_correct": 44,
        "base_acc": 0.86, "instruct_acc": 0.63,
        "base_ci": [0.75, 0.94], "instruct_ci": [0.50, 0.74],
        "mcnemar_chi2": 9.84, "mcnemar_p": 0.002, "cohens_h": 0.47,
    },
    "qwen_7b": {
        "n": 70, "base_correct": 62, "instruct_correct": 68,
        "base_acc": 0.88, "instruct_acc": 0.97,
        "base_ci": [0.77, 0.95], "instruct_ci": [0.88, 1.00],
        "mcnemar_chi2": 2.0, "mcnemar_p": 0.16, "cohens_h": None,
    },
    "mistral_7b": {
        "n": 70, "base_correct": 59, "instruct_correct": 57,
        "base_acc": 0.84, "instruct_acc": 0.81,
        "base_ci": [0.73, 0.92], "instruct_ci": [0.69, 0.89],
        "mcnemar_chi2": 0.25, "mcnemar_p": 0.62, "cohens_h": None,
    },
}


def preflight_check():
    print("Checking required files for scripts/06f_probe_stats.py ...", flush=True)
    ok = True
    for instruct_key, base_key in MODEL_PAIRS.items():
        for key in (instruct_key, base_key):
            axes_f = AXES_DIR / f"{key}_axes.npz"
            hs_f = AXES_DIR / f"{key}_hidden_states.npz"
            for f in (axes_f, hs_f):
                status = "OK" if f.exists() else "MISSING"
                if not f.exists():
                    ok = False
                print(f"  {f.relative_to(PROJECT_ROOT)}: {status}", flush=True)
    for axis in MOOD_AXES:
        cal_f = PROJECT_ROOT / "data" / "calibration" / f"{axis}.json"
        n_eval = len(json.load(open(cal_f)).get("eval_questions", [])) if cal_f.exists() else 0
        print(f"  {cal_f.relative_to(PROJECT_ROOT)}: eval_questions={n_eval}", flush=True)
    print("Preflight " + ("OK" if ok else "FAILED"), flush=True)
    return ok


def wilson_ci(correct: int, n: int, alpha: float = ALPHA):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    p = correct / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = (z * np.sqrt((p * (1 - p) + z ** 2 / (4 * n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def cohens_h(p1: float, p2: float) -> float:
    return float(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2)))


def mcnemar_test(base_correct: np.ndarray, instruct_correct: np.ndarray):
    """Paired McNemar test between two binary outcome arrays of equal length.

    Returns both the classic continuity-corrected chi-square statistic (as
    used in the original hardcoded paper numbers) and an exact binomial
    p-value (more reliable when discordant-pair counts are small, as is
    expected at n=75 with high accuracies).
    """
    assert base_correct.shape == instruct_correct.shape
    # b = base correct, instruct wrong; c = base wrong, instruct correct
    b = int(np.sum((base_correct == 1) & (instruct_correct == 0)))
    c = int(np.sum((base_correct == 0) & (instruct_correct == 1)))
    n_discordant = b + c

    if n_discordant == 0:
        chi2_stat, chi2_p = 0.0, 1.0
    else:
        chi2_stat = (abs(b - c) - 1) ** 2 / n_discordant
        chi2_p = float(1 - chi2_dist.cdf(chi2_stat, df=1))

    if n_discordant == 0:
        exact_p = 1.0
    else:
        k = min(b, c)
        exact_p = float(2 * binom.cdf(k, n_discordant, 0.5))
        exact_p = min(1.0, exact_p)

    return {
        "b_base_correct_instruct_wrong": b,
        "c_base_wrong_instruct_correct": c,
        "n_discordant": n_discordant,
        "chi2_continuity_corrected": float(chi2_stat),
        "chi2_p": chi2_p,
        "exact_binomial_p": exact_p,
    }


def load_item_correctness(model_key: str) -> dict:
    """Per-axis, per-item (paired) correctness array for one model.

    correct_i = 1 iff proj(positive response to eval_question[i]) >
    proj(negative response to eval_question[i]), using the axis vector
    fit on this SAME model's own train split (data/axes/<model>_axes.npz).

    Axes missing from hidden_states.npz are skipped (with a warning), same
    convention as scripts/06g_cross_calibration.py's "axes_missing" handling
    -- e.g. mistral_7b_hidden_states.npz has no confident_cautious raw
    states even though mistral_7b_axes.npz lists a vector for it.
    """
    axes_npz = np.load(AXES_DIR / f"{model_key}_axes.npz", allow_pickle=True)
    hs_npz = np.load(AXES_DIR / f"{model_key}_hidden_states.npz", allow_pickle=True)

    per_axis = {}
    for axis in MOOD_AXES:
        if axis not in axes_npz:
            print(f"  WARNING: {model_key}: axis vector '{axis}' missing from axes.npz, skipping", flush=True)
            continue
        axis_vector = axes_npz[axis]

        pos_key = f"{axis}_val_pos_aggregated"
        neg_key = f"{axis}_val_neg_aggregated"
        if pos_key not in hs_npz or neg_key not in hs_npz:
            print(f"  WARNING: {model_key}: {pos_key}/{neg_key} missing from hidden_states.npz, skipping axis '{axis}'", flush=True)
            continue

        pos_states = hs_npz[pos_key]
        neg_states = hs_npz[neg_key]
        if pos_states.shape[0] != neg_states.shape[0]:
            raise ValueError(
                f"{model_key}/{axis}: val_pos ({pos_states.shape[0]}) and "
                f"val_neg ({neg_states.shape[0]}) sample counts differ -- "
                f"pairing assumption broken, cannot compute paired accuracy."
            )

        pos_proj = pos_states @ axis_vector
        neg_proj = neg_states @ axis_vector
        correct = (pos_proj > neg_proj).astype(int)
        per_axis[axis] = {
            "n": int(correct.shape[0]),
            "correct": correct,
        }

    return per_axis


def analyze_pair(instruct_key: str, base_key: str) -> dict:
    instruct_by_axis = load_item_correctness(instruct_key)
    base_by_axis = load_item_correctness(base_key)

    axes_used = [a for a in MOOD_AXES if a in instruct_by_axis and a in base_by_axis]

    base_all = np.concatenate([base_by_axis[a]["correct"] for a in axes_used])
    instruct_all = np.concatenate([instruct_by_axis[a]["correct"] for a in axes_used])
    n_total = int(base_all.shape[0])

    base_n_correct = int(base_all.sum())
    instruct_n_correct = int(instruct_all.sum())
    base_acc = base_n_correct / n_total
    instruct_acc = instruct_n_correct / n_total

    base_ci = wilson_ci(base_n_correct, n_total)
    instruct_ci = wilson_ci(instruct_n_correct, n_total)

    mcnemar = mcnemar_test(base_all, instruct_all)
    h = cohens_h(base_acc, instruct_acc)

    per_axis_breakdown = {}
    for a in axes_used:
        bc = int(base_by_axis[a]["correct"].sum())
        ic = int(instruct_by_axis[a]["correct"].sum())
        n = int(base_by_axis[a]["n"])
        per_axis_breakdown[a] = {
            "n": n,
            "base_correct": bc, "base_acc": bc / n,
            "instruct_correct": ic, "instruct_acc": ic / n,
        }

    return {
        "instruct_model": instruct_key,
        "base_model": base_key,
        "n_axes": len(axes_used),
        "n_total": n_total,
        "base": {
            "n_correct": base_n_correct, "n_total": n_total, "accuracy": base_acc,
            "wilson_ci_95": list(base_ci),
        },
        "instruct": {
            "n_correct": instruct_n_correct, "n_total": n_total, "accuracy": instruct_acc,
            "wilson_ci_95": list(instruct_ci),
        },
        "delta_pp": (instruct_acc - base_acc) * 100,
        "mcnemar": mcnemar,
        "cohens_h": h,
        "bonferroni_alpha": BONFERRONI_ALPHA,
        "significant_after_bonferroni": mcnemar["exact_binomial_p"] < BONFERRONI_ALPHA,
        "per_axis": per_axis_breakdown,
    }


def print_comparison(model_key: str, new: dict):
    old = OLD_PAPER_NUMBERS[model_key]
    name = DISPLAY_NAMES[model_key]
    print(f"\n=== {name} ===", flush=True)
    print(f"  OLD (N={old['n']}, 7 axes x 10 q): "
          f"base {old['base_correct']}/{old['n']}={old['base_acc']*100:.0f}% "
          f"[{old['base_ci'][0]*100:.0f}%,{old['base_ci'][1]*100:.0f}%], "
          f"instruct {old['instruct_correct']}/{old['n']}={old['instruct_acc']*100:.0f}% "
          f"[{old['instruct_ci'][0]*100:.0f}%,{old['instruct_ci'][1]*100:.0f}%], "
          f"McNemar chi2={old['mcnemar_chi2']}, p={old['mcnemar_p']}", flush=True)
    print(f"  NEW (N={new['n_total']}, 5 axes x 15 q): "
          f"base {new['base']['n_correct']}/{new['n_total']}={new['base']['accuracy']*100:.1f}% "
          f"[{new['base']['wilson_ci_95'][0]*100:.0f}%,{new['base']['wilson_ci_95'][1]*100:.0f}%], "
          f"instruct {new['instruct']['n_correct']}/{new['n_total']}={new['instruct']['accuracy']*100:.1f}% "
          f"[{new['instruct']['wilson_ci_95'][0]*100:.0f}%,{new['instruct']['wilson_ci_95'][1]*100:.0f}%], "
          f"McNemar chi2={new['mcnemar']['chi2_continuity_corrected']:.2f}, "
          f"exact p={new['mcnemar']['exact_binomial_p']:.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Step 06f: probe/instruction-accuracy stats (75q)")
    parser.add_argument("--check", action="store_true", help="Preflight check and exit")
    args = parser.parse_args()

    if args.check:
        ok = preflight_check()
        sys.exit(0 if ok else 1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for instruct_key, base_key in MODEL_PAIRS.items():
        print(f"Analyzing {instruct_key} vs {base_key} ...", flush=True)
        results[instruct_key] = analyze_pair(instruct_key, base_key)
        print_comparison(instruct_key, results[instruct_key])

    output = {
        "_meta": {
            "script": "scripts/06f_probe_stats.py",
            "timestamp": datetime.now().isoformat(),
            "n_axes": len(MOOD_AXES),
            "n_eval_questions_per_axis": 15,
            "n_total_per_model": len(MOOD_AXES) * 15,
            "bonferroni_alpha": BONFERRONI_ALPHA,
            "n_models_tested": N_MODELS_TESTED,
            "metric_definition": (
                "Item-level paired instruction-following accuracy: for eval "
                "question i on axis a, correct iff proj(hidden_state(positive "
                "instruction, q_i)) > proj(hidden_state(negative instruction, "
                "q_i)) on the model's OWN axis vector (fit on its own 50 train "
                "questions). 15 items/axis x 5 axes = 75 items/model. McNemar "
                "test pairs base vs instruct on the same 75 items (same axis, "
                "same question, same pole labels)."
            ),
            "old_paper_numbers_for_reference": OLD_PAPER_NUMBERS,
            "note": (
                "old_paper_numbers_for_reference reflects the retired 7-axis "
                "x 10-eval-question (N=70) methodology hardcoded in "
                "paper/sections/05_results.md Sec 5.4a; kept here only for "
                "comparison, not used in any computation above."
            ),
        },
        "pairs": results,
    }

    out_path = OUTPUT_DIR / "probe_stats_75q.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
