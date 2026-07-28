#!/usr/bin/env python3
"""Step 06d: Axis geometry — concept-cone rank, steerability coherence,
label-null test, cross-axis structure (CPU only, numpy/scipy only).

Four checks per (model, axis), computed on the per-pair mean-difference
vectors diff_i = pos_i - neg_i (paired by construction: pos/neg samples for
a given train/val index share the same calibration question, verified
empirically against scripts/01_calibrate.py's generation order). Train and
val splits are pooled (n_pairs = n_train + n_val, using the *_aggregated
hidden-state keys — same decay-weighted aggregation used to build the axes
themselves, confirmed to reproduce data/axes/<model>_axes.npz via a trimmed
mean-diff on the train split, see scripts/06c_whitened_svd.py's TRIM=0.1
convention) to maximize sample size for the SVD / permutation tests below.

1. Effective rank / concept-cone check (D25, Wollschläger 2502.17420):
   SVD on the *centered* per-pair diff matrix (diff_i minus their mean, i.e.
   the residual variation around the common axis direction). Effective rank
   = participation ratio (sum(sigma))^2 / sum(sigma^2); pc1_variance_share
   = sigma_1^2 / sum(sigma^2). effective_rank close to 1 => diffs collapse
   onto (the residual of) a single extra direction; higher => broader cone.

2. Steerability predictor (Braun 2505.22637): mean pairwise cosine between
   *individual* (non-centered) diff_i vectors ("direction coherence" — how
   consistently individual examples agree on a direction) plus the fraction
   of diff_i with a negative projection onto the final saved axis vector
   (data/axes/<model>_axes.npz), i.e. examples where the pair actually
   points the "wrong way" relative to the aggregate axis.

3. Template/label-null test: 200 sign-flip permutations (each pair
   independently pos/neg-swapped with prob 0.5, i.e. Rademacher-weighted
   resampling) build a null distribution of ||mean(sign * diff_i)||; the
   z-score of the *true* (unflipped) mean-diff norm against this null shows
   the axis is not an artifact of the template's residual bias.

4. Cross-axis structure per model: cosine matrix between the 5 saved axis
   vectors; PCA on those 5 vectors (variance in PC1/PC2); PCA on the pooled
   per-sample projections of ALL calibration activations (across all 5
   axes' train+val pos/neg pools) onto the 5 axis directions, to see how
   correlated the axes are when applied to a common, diverse activation
   pool (not just their own calibration set).

Reads:  data/axes/<model>_hidden_states.npz, data/axes/<model>_axes.npz
Writes: data/analysis/axis_geometry.json

Usage:
    python3 -u scripts/06d_axis_geometry.py
    python3 -u scripts/06d_axis_geometry.py --check
    python3 -u scripts/06d_axis_geometry.py --models qwen3.5_9b,gemma3_12b
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

N_PERMUTATIONS = 200
RNG_SEED = 42
TRAIN_VAL_SPLITS = ("train", "val")
POLES = ("pos", "neg")


def preflight_check(models):
    print(f"Axes dir: {AXES_DIR}", flush=True)
    for m in models:
        hs = AXES_DIR / f"{m}_hidden_states.npz"
        ax = AXES_DIR / f"{m}_axes.npz"
        print(f"  {m}: hidden_states={'OK' if hs.exists() else 'MISSING'}, "
              f"axes={'OK' if ax.exists() else 'MISSING'}", flush=True)
    print(f"Output: {OUTPUT_DIR / 'axis_geometry.json'}", flush=True)
    print(f"Axes: {MOOD_AXES}", flush=True)
    print(f"Permutations: {N_PERMUTATIONS}, seed: {RNG_SEED}", flush=True)
    print("Preflight OK", flush=True)


def load_pair_diffs(hs_npz, axis):
    """Paired pos-neg diffs pooled over train+val, shape (n_pairs, d).
    Returns None if the axis is missing from this model's hidden states."""
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


def effective_rank_metrics(diffs):
    """Concept-cone check (D25, Wollschläger 2502.17420): SVD on the
    centered per-pair diff matrix. Returns participation-ratio effective
    rank and the variance share carried by the first component."""
    X = diffs - diffs.mean(axis=0, keepdims=True)
    try:
        s = np.linalg.svd(X, compute_uv=False)
    except np.linalg.LinAlgError:
        return None
    s = s[s > 1e-10]
    if len(s) == 0:
        return {"effective_rank": 1.0, "pc1_variance_share": 1.0, "n_components": 0}
    var = s ** 2
    total = var.sum()
    eff_rank = float((s.sum() ** 2) / total)
    pc1_share = float(var[0] / total)
    n_comp = int(len(s))
    return {
        "effective_rank": round(eff_rank, 3),
        # eff_rank capped at n_comp (=n_pairs-1 after centering); since n_pairs
        # is identical (65) for every model/axis here, raw effective_rank is
        # comparable across the table as-is, but the normalized version below
        # makes the "fraction of the available rank actually used" explicit
        # (see the large caveat in the module docstring / final report: with
        # n_pairs=65 << hidden_dim, ANY isotropic per-example noise inflates
        # effective_rank toward n_comp regardless of true concept dimensionality,
        # so absolute values are not directly comparable to the naive [1,5] prior
        # — only the relative ranking across models/axes is informative here).
        "effective_rank_normalized": round(eff_rank / n_comp, 4) if n_comp > 0 else None,
        "pc1_variance_share": round(pc1_share, 4),
        "n_components": n_comp,
    }


def steerability_coherence(diffs, axis_vec):
    """Braun 2505.22637-style predictor: mean pairwise cosine between
    individual (raw) diff_i vectors, plus the fraction of diff_i whose
    projection onto the final saved axis vector is negative."""
    norms = np.linalg.norm(diffs, axis=1)
    valid = norms > 1e-10
    unit = diffs[valid] / norms[valid, None]
    n = unit.shape[0]
    if n < 2:
        return {"mean_pairwise_cosine": None, "frac_negative_projection": None, "n_pairs": int(diffs.shape[0])}
    cos_mat = unit @ unit.T
    iu = np.triu_indices(n, k=1)
    mean_cos = float(cos_mat[iu].mean())

    proj = diffs @ axis_vec  # axis_vec is unit-norm
    frac_neg = float((proj < 0).mean())
    return {
        "mean_pairwise_cosine": round(mean_cos, 4),
        "frac_negative_projection": round(frac_neg, 4),
        "n_pairs": int(diffs.shape[0]),
    }


def label_null_test(diffs, rng, n_perm=N_PERMUTATIONS):
    """Sign-flip (pos/neg relabeling) permutation null for the mean-diff
    norm. z-score of the true (unflipped) axis norm against the null
    distribution obtained by randomly swapping pos/neg per pair."""
    n_pairs = diffs.shape[0]
    real_norm = float(np.linalg.norm(diffs.mean(axis=0)))
    null_norms = np.empty(n_perm)
    for i in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=n_pairs)
        null_norms[i] = np.linalg.norm((diffs * signs[:, None]).mean(axis=0))
    null_mean = float(null_norms.mean())
    null_std = float(null_norms.std())
    z = float((real_norm - null_mean) / null_std) if null_std > 1e-12 else float("inf")
    return {
        "real_norm": round(real_norm, 4),
        "null_mean": round(null_mean, 4),
        "null_std": round(null_std, 4),
        "z_score": round(z, 2) if np.isfinite(z) else z,
        "n_permutations": n_perm,
    }


def cross_axis_structure(hs_npz, axis_vecs):
    """Cosine matrix + PCA on the saved axis vectors themselves, and PCA on
    pooled per-sample projections of all calibration activations (across
    all present axes' train+val pos/neg pools) onto the axis directions."""
    axes_present = [a for a in MOOD_AXES if a in axis_vecs]
    if len(axes_present) < 2:
        return {"axes_order": axes_present}

    V = np.stack([axis_vecs[a] for a in axes_present])  # (k, d), unit rows
    cos_mat = V @ V.T
    k = len(axes_present)
    max_offdiag = 0.0
    for i in range(k):
        for j in range(k):
            if i != j:
                max_offdiag = max(max_offdiag, abs(float(cos_mat[i, j])))

    out = {
        "axes_order": axes_present,
        "cosine_matrix": [[round(float(x), 4) for x in row] for row in cos_mat],
        "max_abs_offdiag_cosine": round(max_offdiag, 4),
    }

    # PCA on the k axis vectors themselves (center across axes, SVD).
    Vc = V - V.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(Vc, compute_uv=False)
    sv = sv[sv > 1e-10]
    if len(sv) > 0:
        var = sv ** 2
        total = var.sum()
        ratios = (var / total).tolist()
        out["pca_on_axis_vectors"] = {
            "explained_variance_ratio": [round(r, 4) for r in ratios],
            "pc1_share": round(ratios[0], 4),
            "pc2_share": round(ratios[1], 4) if len(ratios) > 1 else None,
        }
    else:
        out["pca_on_axis_vectors"] = None

    # PCA on pooled per-sample projections: stack all axes' own train+val
    # pos/neg activation pools into one diverse activation set, project
    # every activation onto all k axis directions, then PCA the (n, k)
    # projection matrix.
    pooled = []
    for a in axes_present:
        for split in TRAIN_VAL_SPLITS:
            for pole in POLES:
                key = f"{a}_{split}_{pole}_aggregated"
                if key in hs_npz.files:
                    pooled.append(hs_npz[key].astype(np.float64))
    if pooled:
        X_all = np.vstack(pooled)  # (n_total, d)
        P = X_all @ V.T  # (n_total, k) projections onto each axis
        Pc = P - P.mean(axis=0, keepdims=True)
        svp = np.linalg.svd(Pc, compute_uv=False)
        svp = svp[svp > 1e-10]
        if len(svp) > 0:
            varp = svp ** 2
            totalp = varp.sum()
            ratiosp = (varp / totalp).tolist()
            out["pca_on_projections"] = {
                "n_samples": int(X_all.shape[0]),
                "explained_variance_ratio": [round(r, 4) for r in ratiosp],
                "pc1_share": round(ratiosp[0], 4),
                "pc2_share": round(ratiosp[1], 4) if len(ratiosp) > 1 else None,
            }
        else:
            out["pca_on_projections"] = None
    else:
        out["pca_on_projections"] = None

    return out


def analyze_model(model_key, rng, n_perm=N_PERMUTATIONS):
    hs_path = AXES_DIR / f"{model_key}_hidden_states.npz"
    ax_path = AXES_DIR / f"{model_key}_axes.npz"
    if not hs_path.exists():
        print(f"  WARNING: {hs_path} not found, skipping {model_key}", flush=True)
        return None
    if not ax_path.exists():
        print(f"  WARNING: {ax_path} not found, skipping {model_key}", flush=True)
        return None

    hs_npz = np.load(hs_path)
    axes_npz = np.load(ax_path)

    axis_vecs = {}
    per_axis = {}
    for axis in MOOD_AXES:
        if axis not in axes_npz.files:
            print(f"  WARNING: {model_key}: axis '{axis}' missing from axes.npz, skipping axis", flush=True)
            continue
        diffs = load_pair_diffs(hs_npz, axis)
        if diffs is None or diffs.shape[0] < 4:
            print(f"  WARNING: {model_key}: axis '{axis}' has too few pair diffs, skipping axis", flush=True)
            continue

        axis_vec = axes_npz[axis].astype(np.float64)
        v_norm = np.linalg.norm(axis_vec)
        if v_norm > 1e-10:
            axis_vec = axis_vec / v_norm
        axis_vecs[axis] = axis_vec

        per_axis[axis] = {
            "concept_cone": effective_rank_metrics(diffs),
            "steerability": steerability_coherence(diffs, axis_vec),
            "label_null": label_null_test(diffs, rng, n_perm=n_perm),
        }

    cross = cross_axis_structure(hs_npz, axis_vecs)
    return {"axes": per_axis, "cross_axis": cross}


def print_summary_table(all_results):
    print("", flush=True)
    print("=" * 128, flush=True)
    print(f"{'Model':<18}{'Axis':<24}{'eff_rank':>9}{'rank_norm':>10}{'pc1_var%':>10}"
          f"{'coherence':>11}{'frac_neg':>10}{'null_z':>9}", flush=True)
    print("-" * 128, flush=True)
    for model_key in MODEL_ORDER:
        if model_key not in all_results:
            continue
        label = MODEL_LABELS.get(model_key, model_key)
        axes_data = all_results[model_key]["axes"]
        for axis, stats in axes_data.items():
            cc = stats["concept_cone"] or {}
            st = stats["steerability"]
            nz = stats["label_null"]
            eff = cc.get("effective_rank")
            effn = cc.get("effective_rank_normalized")
            pc1 = cc.get("pc1_variance_share")
            coh = st.get("mean_pairwise_cosine")
            fneg = st.get("frac_negative_projection")
            z = nz.get("z_score")
            eff_s = f"{eff:.2f}" if eff is not None else "n/a"
            effn_s = f"{effn:.3f}" if effn is not None else "n/a"
            pc1_s = f"{pc1*100:.1f}" if pc1 is not None else "n/a"
            coh_s = f"{coh:.3f}" if coh is not None else "n/a"
            fneg_s = f"{fneg*100:.1f}%" if fneg is not None else "n/a"
            z_s = f"{z:.1f}" if isinstance(z, (int, float)) and np.isfinite(z) else str(z)
            print(f"{label:<18}{axis:<24}{eff_s:>9}{effn_s:>10}{pc1_s:>10}{coh_s:>11}{fneg_s:>10}{z_s:>9}", flush=True)
        cross = all_results[model_key]["cross_axis"]
        max_cos = cross.get("max_abs_offdiag_cosine")
        pca_v = (cross.get("pca_on_axis_vectors") or {}).get("pc1_share")
        pca_p = (cross.get("pca_on_projections") or {}).get("pc1_share")
        max_cos_s = f"{max_cos:.3f}" if max_cos is not None else "n/a"
        pca_v_s = f"{pca_v*100:.1f}%" if pca_v is not None else "n/a"
        pca_p_s = f"{pca_p*100:.1f}%" if pca_p is not None else "n/a"
        print(f"  cross-axis: max|cos|={max_cos_s}  PC1(axis-vecs)={pca_v_s}  "
              f"PC1(pooled-proj)={pca_p_s}", flush=True)
        print("-" * 128, flush=True)
    print("=" * 128, flush=True)


def print_key_findings(all_results):
    print("", flush=True)
    print("Key findings:", flush=True)
    print("  NOTE: with n_pairs=65 << hidden_dim, raw effective_rank saturates toward", flush=True)
    print("  n_pairs-1=64 for ANY noisy axis (verified on synthetic rank-1+noise data), so", flush=True)
    print("  every axis below reads as numerically 'cone-like' in absolute terms. The", flush=True)
    print("  informative signal is the RELATIVE ranking (effective_rank_normalized) plus", flush=True)
    print("  coherence / frac_negative_projection, which are scale-free.", flush=True)

    # Pool normalized rank across all models/axes to flag relative outliers
    # (values well below the typical range across this model set).
    all_norms = [
        (s["concept_cone"] or {}).get("effective_rank_normalized")
        for res in all_results.values() for s in res["axes"].values()
        if (s["concept_cone"] or {}).get("effective_rank_normalized") is not None
    ]
    rank_norm_cutoff = 0.75 if not all_norms else min(0.75, float(np.percentile(all_norms, 25)))

    for model_key in MODEL_ORDER:
        if model_key not in all_results:
            continue
        label = MODEL_LABELS.get(model_key, model_key)
        axes_data = all_results[model_key]["axes"]
        low_rank_norm = [a for a, s in axes_data.items()
                          if (s["concept_cone"] or {}).get("effective_rank_normalized", 1.0) < rank_norm_cutoff]
        low_coherence = [a for a, s in axes_data.items()
                          if (s["steerability"].get("mean_pairwise_cosine") or 1.0) < 0.3]
        nonzero_frac_neg = [a for a, s in axes_data.items()
                             if (s["steerability"].get("frac_negative_projection") or 0.0) > 0.0]
        weak_null = [a for a, s in axes_data.items()
                     if isinstance(s["label_null"].get("z_score"), (int, float))
                     and np.isfinite(s["label_null"]["z_score"])
                     and s["label_null"]["z_score"] < 3.0]
        parts = []
        if low_rank_norm:
            parts.append(f"low rank_norm (more concentrated residual, <{rank_norm_cutoff:.2f}): {', '.join(low_rank_norm)}")
        if low_coherence:
            parts.append(f"low coherence (<0.3): {', '.join(low_coherence)}")
        if nonzero_frac_neg:
            parts.append(f"some pairs flip sign vs axis: {', '.join(nonzero_frac_neg)}")
        if weak_null:
            parts.append(f"weak label-null z (<3): {', '.join(weak_null)}")
        if not parts:
            parts.append("unremarkable: typical rank_norm, coherence>=0.3, all pairs same-signed, z>>3")
        print(f"  {label}: " + "; ".join(parts), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Run preflight check and exit")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated subset of MODEL_ORDER to run (default: all)")
    parser.add_argument("--n-permutations", type=int, default=N_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    args = parser.parse_args()

    models = MODEL_ORDER
    if args.models:
        requested = [m.strip() for m in args.models.split(",") if m.strip()]
        unknown = [m for m in requested if m not in MODEL_ORDER]
        if unknown:
            print(f"WARNING: unknown models requested (not in MODEL_ORDER): {unknown}", flush=True)
        models = [m for m in requested if m in MODEL_ORDER]

    if args.check:
        preflight_check(models)
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    all_results = {}
    for model_key in models:
        print(f"Analyzing {model_key} ...", flush=True)
        res = analyze_model(model_key, rng, n_perm=args.n_permutations)
        if res is not None:
            all_results[model_key] = res
            print(f"  done ({len(res['axes'])} axes)", flush=True)

    output = {
        "_meta": {
            "script": "scripts/06d_axis_geometry.py",
            "timestamp": datetime.now().isoformat(),
            "n_permutations": args.n_permutations,
            "rng_seed": args.seed,
            "models_analyzed": list(all_results.keys()),
            "axes": MOOD_AXES,
            "notes": (
                "concept_cone: SVD on centered per-pair (pos_i-neg_i) diffs, "
                "effective_rank=participation ratio (sum(sigma))^2/sum(sigma^2); "
                "steerability: mean pairwise cosine between raw per-pair diffs "
                "(direction coherence) + frac of pairs with negative projection "
                "on the saved axis vector; label_null: 200 pos/neg sign-flip "
                "permutations, z-score of the true mean-diff norm vs the null "
                "norm distribution; cross_axis: cosine matrix + PCA on the 5 "
                "axis vectors + PCA on pooled per-sample projections."
            ),
        },
        "models": all_results,
    }

    out_path = OUTPUT_DIR / "axis_geometry.json"
    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2)
    print(f"\nWrote {out_path}", flush=True)

    print_summary_table(all_results)
    print_key_findings(all_results)


if __name__ == "__main__":
    main()
