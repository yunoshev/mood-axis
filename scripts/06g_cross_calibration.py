#!/usr/bin/env python3
"""Step 06g: Cross-calibration base <-> instruct (Blocker 4b, CPU only).

Addresses the construct-validity objection that axes calibrated on
instructed-model responses might just measure instruction-following rather
than a content-level "mood" direction. If an axis vector estimated on a
BASE model's hidden states still separates the corresponding INSTRUCT
model's held-out validation states (and vice versa), that is evidence the
axis encodes something about the underlying representation shared by both
models, not something specific to RLHF/instruction-tuning.

Method (per base<->instruct pair, per axis present in both models'
data/axes/<model>_hidden_states.npz):

  1. axis_vector(model) = compute_axis_vector(train_pos_aggregated,
     train_neg_aggregated) -- the exact contrastive trimmed-mean-diff
     construction used by scripts/01_calibrate.py / src/calibration/
     axis_computer.py (TRIM_PROPORTION=0.1, unit-normalized). Computed
     directly from the saved hidden states (not read from *_axes.npz,
     which can go stale relative to the states file -- see note below).
  2. base->instruct: axis_vector(base) applied (dot product) to the
     INSTRUCT model's val_pos/val_neg states -> pairwise accuracy + d'.
  3. instruct->base: axis_vector(instruct) applied to the BASE model's
     val_pos/val_neg states -> pairwise accuracy + d'.
  4. within-model baselines: axis_vector(base) on base's own val states,
     axis_vector(instruct) on instruct's own val states (should reproduce
     the val_accuracy/val_dprime in *_axes_meta.json as a sanity check).
  5. cosine(axis_vector(base), axis_vector(instruct)).

Accuracy/d' use the exact same formulas as
src/calibration/axis_computer.py::compute_validation_metrics /
compute_dprime (pairwise proj(pos_i) > proj(neg_j) fraction; pooled-std
Cohen's d).

Pairs (base model, instruct model) are taken from config/models.py
(is_base_model=True, instruct_counterpart=...), restricted to pairs where
BOTH *_hidden_states.npz files exist. Hidden dimensions are asserted equal
(same architecture family); a mismatch is logged and the pair is skipped.
Per-axis intersection is based on which axes actually have complete
train+val pos+neg aggregated keys in BOTH models' hidden_states.npz --
*_axes.npz can list an axis that no longer has backing raw states in
hidden_states.npz (observed for mistral_7b: confident_cautious is in
mistral_7b_axes.npz but absent from mistral_7b_hidden_states.npz), in
which case that axis is skipped for that pair with a WARNING.

Interpretation: if cross-calibration accuracy is comparable to within-model
accuracy, axes encode axis-relevant content shared across base/instruct
representations rather than an instruction-following artifact. If
cross-calibration collapses towards chance (0.5) while within-model
accuracy stays high, the circularity objection stands for that axis/model.

Reads:  data/axes/<model>_hidden_states.npz (train/val pos/neg aggregated
        states), config/models.py (base<->instruct pairing + hidden_dim)
Writes: data/analysis/cross_calibration.json

Usage:
    python3 -u scripts/06g_cross_calibration.py
    python3 -u scripts/06g_cross_calibration.py --check
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import trim_mean

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import MOOD_AXES
from config.models import MODELS

AXES_DIR = PROJECT_ROOT / "data" / "axes"
OUTPUT_DIR = PROJECT_ROOT / "data" / "analysis"

TRIM_PROPORTION = 0.1  # Matches src/calibration/axis_computer.py

PAIR_LABELS = {
    "qwen_7b": "Qwen 2.5 7B",
    "llama_8b": "Llama 3.1 8B",
    "mistral_7b": "Mistral 7B v0.3",
    "gemma_9b": "Gemma 2 9B",
    "ministral3_8b": "Ministral 3 8B",
}


def discover_pairs():
    """Find (base_key, instruct_key) pairs where both hidden_states.npz exist."""
    pairs = []
    for key, cfg in MODELS.items():
        if not cfg.is_base_model or not cfg.instruct_counterpart:
            continue
        instr_key = cfg.instruct_counterpart
        if instr_key not in MODELS:
            continue
        base_hs = AXES_DIR / f"{key}_hidden_states.npz"
        instr_hs = AXES_DIR / f"{instr_key}_hidden_states.npz"
        if base_hs.exists() and instr_hs.exists():
            pairs.append((key, instr_key))
    return pairs


def preflight_check():
    pairs = discover_pairs()
    print(f"Axes dir: {AXES_DIR}", flush=True)
    print(f"Discovered {len(pairs)} base<->instruct pairs with both hidden_states.npz present:", flush=True)
    for base_key, instr_key in pairs:
        base_cfg, instr_cfg = MODELS[base_key], MODELS[instr_key]
        dim_ok = base_cfg.hidden_dim == instr_cfg.hidden_dim
        print(f"  {base_key} (dim={base_cfg.hidden_dim}) <-> {instr_key} (dim={instr_cfg.hidden_dim}) "
              f"{'OK' if dim_ok else 'DIM MISMATCH -- will skip'}", flush=True)
    print(f"Output: {OUTPUT_DIR / 'cross_calibration.json'}", flush=True)
    print(f"Axes: {MOOD_AXES}", flush=True)
    print("Preflight OK", flush=True)


def load_pole_states(hs_npz, axis, split, pole):
    key = f"{axis}_{split}_{pole}_aggregated"
    if key not in hs_npz.files:
        return None
    return hs_npz[key].astype(np.float64)


def axes_available(hs_npz):
    """Axes with complete train+val pos+neg aggregated keys."""
    avail = []
    for axis in MOOD_AXES:
        ok = all(
            f"{axis}_{split}_{pole}_aggregated" in hs_npz.files
            for split in ("train", "val") for pole in ("pos", "neg")
        )
        if ok:
            avail.append(axis)
    return avail


def compute_axis_vector(pos_states: np.ndarray, neg_states: np.ndarray) -> np.ndarray:
    """Exact reproduction of src/calibration/axis_computer.py::compute_axis_vector."""
    pos_mean = trim_mean(pos_states, proportiontocut=TRIM_PROPORTION, axis=0)
    neg_mean = trim_mean(neg_states, proportiontocut=TRIM_PROPORTION, axis=0)
    direction = pos_mean - neg_mean
    norm = np.linalg.norm(direction)
    if norm > 0:
        direction = direction / norm
    return direction


def compute_dprime(pos: np.ndarray, neg: np.ndarray) -> float:
    """Exact reproduction of src/calibration/axis_computer.py::compute_dprime."""
    n_pos, n_neg = len(pos), len(neg)
    mean_diff = float(np.mean(pos) - np.mean(neg))
    var_pos = np.var(pos, ddof=1) if n_pos > 1 else 0.0
    var_neg = np.var(neg, ddof=1) if n_neg > 1 else 0.0
    if n_pos + n_neg > 2:
        pooled_var = ((n_pos - 1) * var_pos + (n_neg - 1) * var_neg) / (n_pos + n_neg - 2)
        pooled_std = np.sqrt(pooled_var)
    else:
        pooled_std = 1.0
    if pooled_std < 1e-10:
        return 0.0
    return float(mean_diff / pooled_std)


def apply_axis(axis_vec: np.ndarray, val_pos: np.ndarray, val_neg: np.ndarray) -> dict:
    """Pairwise accuracy + d' of axis_vec applied to (val_pos, val_neg) states.
    Exact reproduction of compute_validation_metrics's accuracy/dprime."""
    pos_proj = val_pos @ axis_vec
    neg_proj = val_neg @ axis_vec
    correct = int(np.sum(pos_proj[:, None] > neg_proj[None, :]))
    total = len(pos_proj) * len(neg_proj)
    accuracy = correct / total if total > 0 else None
    dprime = compute_dprime(pos_proj, neg_proj)
    return {
        "accuracy": round(float(accuracy), 4) if accuracy is not None else None,
        "dprime": round(dprime, 4),
        "n_pos": int(len(pos_proj)),
        "n_neg": int(len(neg_proj)),
    }


def analyze_pair(base_key: str, instr_key: str) -> dict:
    base_hs = np.load(AXES_DIR / f"{base_key}_hidden_states.npz")
    instr_hs = np.load(AXES_DIR / f"{instr_key}_hidden_states.npz")

    base_axes = set(axes_available(base_hs))
    instr_axes = set(axes_available(instr_hs))
    common = [a for a in MOOD_AXES if a in base_axes and a in instr_axes]
    missing_base = [a for a in MOOD_AXES if a not in base_axes]
    missing_instr = [a for a in MOOD_AXES if a not in instr_axes]
    if missing_base:
        print(f"  WARNING: {base_key}: axes missing complete states: {missing_base}", flush=True)
    if missing_instr:
        print(f"  WARNING: {instr_key}: axes missing complete states: {missing_instr}", flush=True)

    per_axis = {}
    for axis in common:
        base_train_pos = load_pole_states(base_hs, axis, "train", "pos")
        base_train_neg = load_pole_states(base_hs, axis, "train", "neg")
        base_val_pos = load_pole_states(base_hs, axis, "val", "pos")
        base_val_neg = load_pole_states(base_hs, axis, "val", "neg")
        instr_train_pos = load_pole_states(instr_hs, axis, "train", "pos")
        instr_train_neg = load_pole_states(instr_hs, axis, "train", "neg")
        instr_val_pos = load_pole_states(instr_hs, axis, "val", "pos")
        instr_val_neg = load_pole_states(instr_hs, axis, "val", "neg")

        base_vec = compute_axis_vector(base_train_pos, base_train_neg)
        instr_vec = compute_axis_vector(instr_train_pos, instr_train_neg)

        cos = float(np.dot(base_vec, instr_vec))  # both unit-norm

        per_axis[axis] = {
            "base_within": apply_axis(base_vec, base_val_pos, base_val_neg),
            "instruct_within": apply_axis(instr_vec, instr_val_pos, instr_val_neg),
            "base_to_instruct": apply_axis(base_vec, instr_val_pos, instr_val_neg),
            "instruct_to_base": apply_axis(instr_vec, base_val_pos, base_val_neg),
            "cosine_base_vs_instruct_axis": round(cos, 4),
        }

    return {
        "axes": per_axis,
        "axes_common": common,
        "axes_missing_base": missing_base,
        "axes_missing_instruct": missing_instr,
    }


def print_summary_table(all_results: dict):
    print("", flush=True)
    print("=" * 148, flush=True)
    print(f"{'Pair':<20}{'Axis':<24}{'base_within':>12}{'instr_within':>13}"
          f"{'base->instr':>13}{'instr->base':>13}{'cos(b,i)':>10}", flush=True)
    print("-" * 148, flush=True)
    for pair_key, res in all_results.items():
        label = PAIR_LABELS.get(pair_key.split("__")[0], pair_key)
        for axis, stats in res["axes"].items():
            bw = stats["base_within"]["accuracy"]
            iw = stats["instruct_within"]["accuracy"]
            bi = stats["base_to_instruct"]["accuracy"]
            ib = stats["instruct_to_base"]["accuracy"]
            cos = stats["cosine_base_vs_instruct_axis"]
            fmt = lambda v: f"{v*100:.1f}%" if v is not None else "n/a"
            print(f"{label:<20}{axis:<24}{fmt(bw):>12}{fmt(iw):>13}{fmt(bi):>13}{fmt(ib):>13}{cos:>10.3f}", flush=True)
        print("-" * 148, flush=True)
    print("=" * 148, flush=True)


def print_key_findings(all_results: dict):
    print("", flush=True)
    print("Key findings:", flush=True)
    all_within, all_cross, all_cos = [], [], []
    for pair_key, res in all_results.items():
        for axis, stats in res["axes"].items():
            bw = stats["base_within"]["accuracy"]
            iw = stats["instruct_within"]["accuracy"]
            bi = stats["base_to_instruct"]["accuracy"]
            ib = stats["instruct_to_base"]["accuracy"]
            cos = stats["cosine_base_vs_instruct_axis"]
            if bw is not None:
                all_within.append(bw)
            if iw is not None:
                all_within.append(iw)
            if bi is not None:
                all_cross.append(bi)
            if ib is not None:
                all_cross.append(ib)
            all_cos.append(cos)
    if all_within and all_cross:
        print(f"  Overall mean within-model accuracy:  {np.mean(all_within)*100:.1f}% "
              f"(n={len(all_within)} axis/direction observations)", flush=True)
        print(f"  Overall mean cross-model accuracy:   {np.mean(all_cross)*100:.1f}% "
              f"(n={len(all_cross)} axis/direction observations)", flush=True)
        print(f"  Retention ratio (cross/within):      {np.mean(all_cross)/np.mean(all_within):.3f}", flush=True)
    if all_cos:
        print(f"  Mean cosine(base axis, instruct axis) across all pairs/axes: {np.mean(all_cos):.3f} "
              f"(min={np.min(all_cos):.3f}, max={np.max(all_cos):.3f})", flush=True)

    for pair_key, res in all_results.items():
        label = PAIR_LABELS.get(pair_key.split("__")[0], pair_key)
        worst = None
        for axis, stats in res["axes"].items():
            bi = stats["base_to_instruct"]["accuracy"]
            ib = stats["instruct_to_base"]["accuracy"]
            for direction, val in (("base->instruct", bi), ("instruct->base", ib)):
                if val is not None and (worst is None or val < worst[1]):
                    worst = (f"{axis} ({direction})", val)
        if worst:
            print(f"  {label}: weakest cross-calibration = {worst[0]} at {worst[1]*100:.1f}%", flush=True)
        if res["axes_missing_base"] or res["axes_missing_instruct"]:
            print(f"  {label}: skipped axes -- base missing {res['axes_missing_base']}, "
                  f"instruct missing {res['axes_missing_instruct']}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Run preflight check and exit")
    args = parser.parse_args()

    if args.check:
        preflight_check()
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pairs = discover_pairs()
    all_results = {}
    skipped_dim_mismatch = []
    for base_key, instr_key in pairs:
        base_cfg, instr_cfg = MODELS[base_key], MODELS[instr_key]
        if base_cfg.hidden_dim != instr_cfg.hidden_dim:
            print(f"SKIP {base_key}<->{instr_key}: hidden_dim mismatch "
                  f"({base_cfg.hidden_dim} vs {instr_cfg.hidden_dim})", flush=True)
            skipped_dim_mismatch.append([base_key, instr_key])
            continue
        pair_key = f"{base_key}__{instr_key}"
        print(f"Analyzing {base_key} <-> {instr_key} ...", flush=True)
        res = analyze_pair(base_key, instr_key)
        all_results[pair_key] = res
        print(f"  done ({len(res['axes'])} common axes)", flush=True)

    output = {
        "_meta": {
            "script": "scripts/06g_cross_calibration.py",
            "timestamp": datetime.now().isoformat(),
            "pairs_analyzed": list(all_results.keys()),
            "pairs_skipped_dim_mismatch": skipped_dim_mismatch,
            "axes": MOOD_AXES,
            "trim_proportion": TRIM_PROPORTION,
            "notes": (
                "axis_vector = compute_axis_vector(train_pos_aggregated, train_neg_aggregated) "
                "(trimmed-mean-diff, 10% trim, unit-normalized), reproducing scripts/01_calibrate.py "
                "exactly, computed directly from data/axes/<model>_hidden_states.npz rather than "
                "*_axes.npz (which can be stale relative to the states file -- e.g. mistral_7b_axes.npz "
                "lists confident_cautious but mistral_7b_hidden_states.npz no longer has backing raw "
                "states for that axis; such axes are skipped per-pair, see axes_missing_base/instruct). "
                "accuracy/dprime use the exact pairwise formulas from "
                "src/calibration/axis_computer.py::compute_validation_metrics (val split, independent "
                "of train, zero question overlap). base_within/instruct_within reproduce the "
                "val_accuracy/val_dprime already stored in *_axes_meta.json as a self-consistency check. "
                "base_to_instruct/instruct_to_base are the actual cross-calibration metrics: axis vector "
                "estimated on one model's train states, applied to the OTHER model's val states."
            ),
        },
        "pairs": all_results,
    }

    out_path = OUTPUT_DIR / "cross_calibration.json"
    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2)
    print(f"\nWrote {out_path}", flush=True)

    print_summary_table(all_results)
    print_key_findings(all_results)


if __name__ == "__main__":
    main()
