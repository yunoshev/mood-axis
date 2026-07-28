#!/usr/bin/env python3
"""Step 06h: Permutation null for cross-set axis consistency (Blocker 6, CPU only).

Section 5.3 currently asserts a permutation null "approx 0" for cross-set
axis consistency (the pairwise cosine between axis vectors independently
re-derived on 3 disjoint question+style sets, scripts/04_stability.py's
"cosine_similarity" / per_axis_summary.mean_cosine) without ever computing
it. This script computes it.

Real value (per model, per axis): mean pairwise cosine of the axis vector
across the 3 stability sets (A, B, C) -- i.e. mean of cosine(A,B),
cosine(A,C), cosine(B,C) using data/stability/<model>_set_{A,B,C}_axes.npz.
This reproduces scripts/04_stability.py::analyze_model's
per_axis_summary[axis]["mean_cosine"] exactly.

Null distribution: the paper's null hypothesis is "if pos/neg pole
assignment carried no true axis signal, would two independently-estimated
mean-diff vectors align by chance?". The literal protocol would permute
pos/neg labels *within each of the 3 stability sets' own per-sample hidden
states* and recompute. That data does not exist on disk any more:
scripts/04_stability.py::calibrate_set deletes each set's raw per-sample
states.npz after computing the axis vector (`partial_states_path.unlink()`,
"final data is in npz + meta + texts"). Checked explicitly for this run:
of the 6 main-cast models x 3 sets = 18 (model, set) cells, only one
leftover states.npz survives (deepseek_r1_14b_set_C, from an interrupted
run before cleanup) -- not enough to build a real cross-set null for any
model in the required set.

Substitute pool (documented, not literal stability-set data): each model's
OWN main-calibration hidden states, data/axes/<model>_hidden_states.npz,
pooled train+val pos/neg pairs (n_pairs = n_train + n_val, same paired
per-question diffs diff_i = pos_i - neg_i used by scripts/06d_axis_geometry.py).
These are real per-sample activations from the same model and the same axis
definition, just from the main 65-question calibration set rather than the
3 stability question sets. Per permutation: draw two INDEPENDENT Rademacher
sign vectors s1, s2 (each pair's pos/neg pole re-randomized with prob 0.5,
matching 06d's label_null_test convention), form null_vec_k = mean(s_k *
diff_i), and take cosine(null_vec_1, null_vec_2). Repeating N=1000 times
gives the null distribution of pairwise cosine between two independently
mislabeled mean-diff vectors drawn from real activations of the same model
-- exactly the quantity needed to check whether the real cross-set cosine
is far from chance. z-score = (real_mean_cosine - null_mean) / null_std.

Reads:  data/stability/<model>_set_{A,B,C}_axes.npz (real cross-set axis
        vectors), data/axes/<model>_hidden_states.npz (null-generating pool)
Writes: data/analysis/consistency_null.json

Usage:
    python3 -u scripts/06h_consistency_null.py
    python3 -u scripts/06h_consistency_null.py --check
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import MOOD_AXES

AXES_DIR = PROJECT_ROOT / "data" / "axes"
STABILITY_DIR = PROJECT_ROOT / "data" / "stability"
OUTPUT_DIR = PROJECT_ROOT / "data" / "analysis"

MODEL_ORDER = [
    "qwen3.5_9b",
    "phi4",
    "deepseek_r1_14b",
    "ministral3_8b",
    "gpt_oss_20b",
]  # gemma3_12b removed (steering degeneration)
MODEL_LABELS = {
    "qwen3.5_9b": "Qwen 3.5 9B",
    "gemma3_12b": "Gemma 3 12B",
    "phi4": "Phi-4 14B",
    "deepseek_r1_14b": "DeepSeek-R1 14B",
    "ministral3_8b": "Ministral 3 8B",
    "gpt_oss_20b": "GPT-OSS 20B",
}

SET_NAMES = ("A", "B", "C")
N_PERMUTATIONS = 1000
RNG_SEED = 42
TRAIN_VAL_SPLITS = ("train", "val")


def preflight_check():
    print(f"Stability dir: {STABILITY_DIR}", flush=True)
    print(f"Axes dir (null pool): {AXES_DIR}", flush=True)
    for m in MODEL_ORDER:
        set_status = []
        for s in SET_NAMES:
            f = STABILITY_DIR / f"{m}_set_{s}_axes.npz"
            set_status.append(f"{s}={'OK' if f.exists() else 'MISSING'}")
        hs = AXES_DIR / f"{m}_hidden_states.npz"
        print(f"  {m}: {', '.join(set_status)}, null_pool={'OK' if hs.exists() else 'MISSING'}", flush=True)
    print(f"Output: {OUTPUT_DIR / 'consistency_null.json'}", flush=True)
    print(f"Axes: {MOOD_AXES}", flush=True)
    print(f"Permutations: {N_PERMUTATIONS}, seed: {RNG_SEED}", flush=True)
    print("Preflight OK", flush=True)


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def load_real_cross_set_cosines(model_key: str) -> dict:
    """Load the 3 stability-set axis vectors and compute real pairwise
    cosine per axis (reproduces scripts/04_stability.py's per_axis_summary)."""
    vectors = {}
    for s in SET_NAMES:
        f = STABILITY_DIR / f"{model_key}_set_{s}_axes.npz"
        if not f.exists():
            continue
        data = np.load(f)
        axis_names = data["_axes"].tolist()
        vectors[s] = {a: data[a] for a in axis_names}

    if len(vectors) < 2:
        return {}

    available_sets = sorted(vectors.keys())
    set_pairs = [(a, b) for i, a in enumerate(available_sets) for b in available_sets[i + 1:]]

    per_axis = {}
    for axis in MOOD_AXES:
        pair_cosines = {}
        for set_a, set_b in set_pairs:
            if axis in vectors[set_a] and axis in vectors[set_b]:
                pair_cosines[f"{set_a}_vs_{set_b}"] = cosine_similarity(
                    vectors[set_a][axis], vectors[set_b][axis]
                )
        if pair_cosines:
            per_axis[axis] = {
                "pairwise": {k: round(v, 4) for k, v in pair_cosines.items()},
                "mean_cosine": round(float(np.mean(list(pair_cosines.values()))), 4),
                "n_pairs": len(pair_cosines),
            }
    return per_axis


def load_pair_diffs(hs_npz, axis):
    """Paired pos-neg diffs pooled over train+val, shape (n_pairs, d).
    Same construction as scripts/06d_axis_geometry.py::load_pair_diffs."""
    pos_parts, neg_parts = [], []
    for split in TRAIN_VAL_SPLITS:
        kp = f"{axis}_{split}_pos_aggregated"
        kn = f"{axis}_{split}_neg_aggregated"
        if kp not in hs_npz.files or kn not in hs_npz.files:
            continue
        pos_parts.append(hs_npz[kp].astype(np.float64))
        neg_parts.append(hs_npz[kn].astype(np.float64))
    if not pos_parts:
        return None
    pos = np.vstack(pos_parts)
    neg = np.vstack(neg_parts)
    if pos.shape[0] != neg.shape[0]:
        n = min(pos.shape[0], neg.shape[0])
        pos, neg = pos[:n], neg[:n]
    return pos - neg


def permutation_null_cosines(diffs: np.ndarray, rng, n_perm=N_PERMUTATIONS) -> np.ndarray:
    """N independent draws of cosine(null_vec_1, null_vec_2), where each
    null_vec_k is the mean of Rademacher-sign-flipped per-pair diffs
    (independent sign draw per k) -- the null "two independently mislabeled
    mean-diff vectors from the same real activation pool" cosine."""
    n_pairs = diffs.shape[0]
    null_cos = np.empty(n_perm)
    for i in range(n_perm):
        s1 = rng.choice([-1.0, 1.0], size=n_pairs)
        s2 = rng.choice([-1.0, 1.0], size=n_pairs)
        v1 = (diffs * s1[:, None]).mean(axis=0)
        v2 = (diffs * s2[:, None]).mean(axis=0)
        null_cos[i] = cosine_similarity(v1, v2)
    return null_cos


def analyze_model(model_key: str, rng, n_perm=N_PERMUTATIONS) -> dict:
    real = load_real_cross_set_cosines(model_key)
    if not real:
        print(f"  WARNING: {model_key}: fewer than 2 stability sets found, skipping", flush=True)
        return None

    hs_path = AXES_DIR / f"{model_key}_hidden_states.npz"
    if not hs_path.exists():
        print(f"  WARNING: {model_key}: {hs_path} not found, skipping null computation", flush=True)
        return None
    hs_npz = np.load(hs_path)

    per_axis = {}
    for axis in MOOD_AXES:
        if axis not in real:
            continue
        diffs = load_pair_diffs(hs_npz, axis)
        if diffs is None or diffs.shape[0] < 4:
            print(f"  WARNING: {model_key}: axis '{axis}' has too few pair diffs for null, skipping axis", flush=True)
            continue

        null_cos = permutation_null_cosines(diffs, rng, n_perm=n_perm)
        null_mean = float(null_cos.mean())
        null_std = float(null_cos.std())
        null_p95 = float(np.percentile(null_cos, 95))
        null_p2p5 = float(np.percentile(null_cos, 2.5))
        real_val = real[axis]["mean_cosine"]
        z = float((real_val - null_mean) / null_std) if null_std > 1e-12 else float("inf")

        per_axis[axis] = {
            "real_pairwise": real[axis]["pairwise"],
            "real_mean_cosine": real_val,
            "null_mean": round(null_mean, 4),
            "null_std": round(null_std, 4),
            "null_p2_5": round(null_p2p5, 4),
            "null_p95": round(null_p95, 4),
            "z_score": round(z, 2) if np.isfinite(z) else z,
            "n_permutations": n_perm,
            "n_pairs_null_pool": int(diffs.shape[0]),
        }

    return per_axis


def print_summary_table(all_results: dict):
    print("", flush=True)
    print("=" * 128, flush=True)
    print(f"{'Model':<18}{'Axis':<24}{'real_cos':>10}{'null_mean':>11}{'null_p95':>10}{'z_score':>10}", flush=True)
    print("-" * 128, flush=True)
    for model_key in MODEL_ORDER:
        if model_key not in all_results:
            continue
        label = MODEL_LABELS.get(model_key, model_key)
        for axis, stats in all_results[model_key].items():
            z = stats["z_score"]
            z_s = f"{z:.1f}" if isinstance(z, (int, float)) and np.isfinite(z) else str(z)
            print(f"{label:<18}{axis:<24}{stats['real_mean_cosine']:>10.3f}"
                  f"{stats['null_mean']:>11.4f}{stats['null_p95']:>10.4f}{z_s:>10}", flush=True)
        print("-" * 128, flush=True)
    print("=" * 128, flush=True)


def print_key_findings(all_results: dict):
    print("", flush=True)
    print("Key findings:", flush=True)
    all_real, all_null_mean, all_z = [], [], []
    for model_key, axes_data in all_results.items():
        for axis, stats in axes_data.items():
            all_real.append(stats["real_mean_cosine"])
            all_null_mean.append(stats["null_mean"])
            if isinstance(stats["z_score"], (int, float)) and np.isfinite(stats["z_score"]):
                all_z.append(stats["z_score"])
    if all_real:
        print(f"  Overall mean real cross-set cosine: {np.mean(all_real):.3f} "
              f"(n={len(all_real)} model/axis cells)", flush=True)
        print(f"  Overall mean null cosine:           {np.mean(all_null_mean):.4f} "
              f"(paper's 'null approx 0' claim: {'supported' if abs(np.mean(all_null_mean)) < 0.05 else 'NOT supported as-is'})", flush=True)
    if all_z:
        print(f"  Overall mean z-score: {np.mean(all_z):.1f} (min={np.min(all_z):.1f})", flush=True)

    for model_key in MODEL_ORDER:
        if model_key not in all_results:
            continue
        label = MODEL_LABELS.get(model_key, model_key)
        weak = [(a, s) for a, s in all_results[model_key].items()
                if isinstance(s["z_score"], (int, float)) and np.isfinite(s["z_score"]) and s["z_score"] < 5.0]
        if weak:
            parts = ", ".join(f"{a} (z={s['z_score']:.1f}, cos={s['real_mean_cosine']:.2f})" for a, s in weak)
            print(f"  {label}: weak separation from null (z<5): {parts}", flush=True)
        else:
            zs = [s["z_score"] for s in all_results[model_key].values()
                  if isinstance(s["z_score"], (int, float)) and np.isfinite(s["z_score"])]
            print(f"  {label}: all axes well-separated from null (min z={min(zs):.1f})", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Run preflight check and exit")
    parser.add_argument("--n-permutations", type=int, default=N_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    args = parser.parse_args()

    if args.check:
        preflight_check()
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    all_results = {}
    for model_key in MODEL_ORDER:
        print(f"Analyzing {model_key} ...", flush=True)
        res = analyze_model(model_key, rng, n_perm=args.n_permutations)
        if res is not None:
            all_results[model_key] = res
            print(f"  done ({len(res)} axes)", flush=True)

    output = {
        "_meta": {
            "script": "scripts/06h_consistency_null.py",
            "timestamp": datetime.now().isoformat(),
            "n_permutations": args.n_permutations,
            "rng_seed": args.seed,
            "models_analyzed": list(all_results.keys()),
            "axes": MOOD_AXES,
            "real_metric": (
                "Mean pairwise cosine of the axis vector across the 3 independent "
                "stability question+style sets (A,B,C), reproducing "
                "scripts/04_stability.py's per_axis_summary[axis].mean_cosine exactly "
                "from data/stability/<model>_set_{A,B,C}_axes.npz."
            ),
            "null_method": (
                "IMPORTANT DATA-AVAILABILITY NOTE: the literal protocol (permute pos/neg "
                "labels within each of the 3 stability sets' own per-sample hidden states, "
                "recompute mean-diff vectors, cosine between them) could not be run as "
                "specified because scripts/04_stability.py deletes each set's raw "
                "per-sample states.npz after computing the axis vector (only "
                "data/stability/deepseek_r1_14b_set_C_states.npz survives, from an "
                "interrupted run -- not enough for a real cross-set null on any model). "
                "Substitute: for each model/axis, 1000 permutations draw two INDEPENDENT "
                "Rademacher sign vectors over that model's own main-calibration paired "
                "pos/neg diffs (data/axes/<model>_hidden_states.npz, train+val pooled, "
                "n_pairs = n_train+n_val, same construction as scripts/06d_axis_geometry.py), "
                "form null_vec_k = mean(sign_k * diff_i), and take "
                "cosine(null_vec_1, null_vec_2). This uses real per-sample activations from "
                "the same model/axis (not fabricated), just a different question set (the "
                "main 65-question calibration set rather than the 3 stability sets)."
            ),
        },
        "models": all_results,
    }

    out_path = OUTPUT_DIR / "consistency_null.json"
    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2)
    print(f"\nWrote {out_path}", flush=True)

    print_summary_table(all_results)
    print_key_findings(all_results)


if __name__ == "__main__":
    main()
