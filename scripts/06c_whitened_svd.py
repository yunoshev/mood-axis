"""
06c_whitened_svd.py — Compare axis methods: mean-diff vs PCA vs WhitenedSVD vs LDA

WhitenedSVD insight from OBLITERATUS / Zou 2023:
  Accounting for the intrinsic variance structure of hidden states improves
  axis quality. Standard mean-diff ignores that some dimensions vary more
  "naturally" (even without mood stimuli). Whitening removes this bias.

Three methods compared per axis, per model:
  (0) Baseline   — current mean-diff in raw space (d' from axes_meta.json)
  (1) PCA+diff   — mean-diff in PCA-reduced space (top 40 PCs, no whitening)
  (2) WhitenedSVD — mean-diff in whitened PCA space (each PC scaled by 1/√λ)
  (3) LDA        — Fisher discriminant in PCA-reduced space (maximizes S_b/S_w)

Output: per-model table + summary verdict
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import json
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.stats import trim_mean

DATA_DIR = Path(__file__).parent.parent / "data" / "axes"
TRIM = 0.1
N_COMPONENTS = 40  # PCA components (safe with 100 train samples)

MODELS = ["qwen_7b", "gemma_9b", "mistral_7b", "llama_8b", "qwen3_8b"]


def compute_dprime(pos, neg):
    """Pooled Cohen's d."""
    pos, neg = np.array(pos), np.array(neg)
    n1, n2 = len(pos), len(neg)
    v1 = np.var(pos, ddof=1) if n1 > 1 else 1e-8
    v2 = np.var(neg, ddof=1) if n2 > 1 else 1e-8
    pooled = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    return float((np.mean(pos) - np.mean(neg)) / max(pooled, 1e-10))


def trim_mean_np(X, p=TRIM):
    """Trimmed mean along axis=0."""
    return trim_mean(X, proportiontocut=p, axis=0)


def method_meandiff_raw(pos_tr, neg_tr, pos_val, neg_val):
    """Current method: mean-diff in raw space."""
    axis = trim_mean_np(pos_tr) - trim_mean_np(neg_tr)
    norm = np.linalg.norm(axis)
    if norm < 1e-10:
        return 0.0
    axis /= norm
    return compute_dprime(pos_val @ axis, neg_val @ axis)


def method_pca_diff(pos_tr, neg_tr, pos_val, neg_val, whiten=False):
    """Mean-diff in PCA-reduced space (whiten=False) or WhitenedSVD (whiten=True)."""
    X_tr = np.vstack([pos_tr, neg_tr])
    n = min(N_COMPONENTS, X_tr.shape[0] - 1, X_tr.shape[1])
    pca = PCA(n_components=n, whiten=whiten)
    pca.fit(X_tr)

    pos_tr_p = pca.transform(pos_tr)
    neg_tr_p = pca.transform(neg_tr)
    pos_val_p = pca.transform(pos_val)
    neg_val_p = pca.transform(neg_val)

    axis = trim_mean_np(pos_tr_p) - trim_mean_np(neg_tr_p)
    norm = np.linalg.norm(axis)
    if norm < 1e-10:
        return 0.0
    axis /= norm
    return compute_dprime(pos_val_p @ axis, neg_val_p @ axis)


def method_lda(pos_tr, neg_tr, pos_val, neg_val):
    """Fisher LDA in PCA-reduced space."""
    X_tr = np.vstack([pos_tr, neg_tr])
    y_tr = np.array([1] * len(pos_tr) + [0] * len(neg_tr))

    n = min(N_COMPONENTS, X_tr.shape[0] - 1, X_tr.shape[1])
    pca = PCA(n_components=n, whiten=False)
    pca.fit(X_tr)

    Xp_tr = pca.transform(X_tr)
    Xp_val_pos = pca.transform(pos_val)
    Xp_val_neg = pca.transform(neg_val)

    lda = LDA(n_components=1)
    lda.fit(Xp_tr, y_tr)

    pos_proj = lda.transform(Xp_val_pos).ravel()
    neg_proj = lda.transform(Xp_val_neg).ravel()
    return compute_dprime(pos_proj, neg_proj)


def analyze_model(model_name):
    hs_path = DATA_DIR / f"{model_name}_hidden_states.npz"
    meta_path = DATA_DIR / f"{model_name}_axes_meta.json"

    if not hs_path.exists():
        print(f"  [SKIP] No hidden_states.npz")
        return

    with open(meta_path) as f:
        meta = json.load(f)

    axes = [k for k in meta if not k.startswith("_")]
    npz = np.load(hs_path)

    print(f"\n{'='*70}")
    print(f"Model: {model_name}  ({len(axes)} axes)")
    print(f"{'='*70}")
    print(f"  {'Axis':<28} {'Baseline':>9} {'PCA+diff':>9} {'Whitened':>9} {'LDA':>9}  {'Best':>6}")
    print(f"  {'-'*28} {'-'*9} {'-'*9} {'-'*9} {'-'*9}  {'-'*6}")

    results = {}
    for axis in axes:
        # Load train/val hidden states (aggregated = decay-weighted, same as calibration)
        try:
            pos_tr = npz[f"{axis}_train_pos_aggregated"].astype(np.float32)  # (50, d)
            neg_tr = npz[f"{axis}_train_neg_aggregated"].astype(np.float32)
            pos_val = npz[f"{axis}_val_pos_aggregated"].astype(np.float32)   # (15, d)
            neg_val = npz[f"{axis}_val_neg_aggregated"].astype(np.float32)
        except KeyError as e:
            print(f"  {axis:<28} MISSING KEY: {e}")
            continue

        d0 = meta[axis]["val_dprime"]      # baseline (from meta)
        d1 = method_pca_diff(pos_tr, neg_tr, pos_val, neg_val, whiten=False)
        d2 = method_pca_diff(pos_tr, neg_tr, pos_val, neg_val, whiten=True)
        d3 = method_lda(pos_tr, neg_tr, pos_val, neg_val)

        methods = {"Baseline": d0, "PCA+diff": d1, "Whitened": d2, "LDA": d3}
        best = max(methods, key=methods.get)
        results[axis] = methods

        def fmt(d, base=d0):
            delta = d - base
            sign = "+" if delta >= 0 else ""
            suffix = f"({sign}{delta:.1f})" if abs(delta) > 0.1 else ""
            return f"{d:6.2f}{suffix}"

        print(
            f"  {axis:<28} {d0:>9.2f} {fmt(d1):>12} {fmt(d2):>12} {fmt(d3):>12}  {best:>6}"
        )

    # Summary: mean d' per method
    if results:
        vals = {m: [results[a][m] for a in results] for m in ["Baseline", "PCA+diff", "Whitened", "LDA"]}
        print(f"\n  {'MEAN':<28} ", end="")
        for m in ["Baseline", "PCA+diff", "Whitened", "LDA"]:
            print(f" {np.mean(vals[m]):9.2f}", end="")
        best_mean = max(vals, key=lambda m: np.mean(vals[m]))
        print(f"  {best_mean:>6}")

    return results


def main():
    print("WhitenedSVD Analysis — Comparing Axis Computation Methods")
    print(f"PCA components: {N_COMPONENTS}, Trim: {TRIM*100:.0f}%")
    print(f"Val set: 15 pos + 15 neg per axis")

    all_results = {}
    for model in MODELS:
        print(f"\nAnalyzing {model}...", flush=True)
        res = analyze_model(model)
        if res:
            all_results[model] = res

    # Global summary
    print(f"\n\n{'='*70}")
    print("GLOBAL SUMMARY — Mean d' across all models × axes")
    print(f"{'='*70}")
    method_scores = {"Baseline": [], "PCA+diff": [], "Whitened": [], "LDA": []}
    for model, res in all_results.items():
        for axis, methods in res.items():
            for m, v in methods.items():
                if m in method_scores:
                    method_scores[m].append(v)

    for m, scores in method_scores.items():
        mn, mx = np.mean(scores), np.max(scores)
        print(f"  {m:<12}: mean={mn:.2f}  max={mx:.2f}  n={len(scores)}")

    best_method = max(method_scores, key=lambda m: np.mean(method_scores[m]))
    base_mean = np.mean(method_scores["Baseline"])
    best_mean = np.mean(method_scores[best_method])
    print(f"\n  Best method: {best_method} (mean d'={best_mean:.2f} vs baseline={base_mean:.2f}, Δ={best_mean-base_mean:+.2f})")

    # Verdict
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")
    whitened_mean = np.mean(method_scores["Whitened"])
    lda_mean = np.mean(method_scores["LDA"])
    if whitened_mean > base_mean + 0.5:
        print(f"  WhitenedSVD IMPROVES d' by +{whitened_mean-base_mean:.2f} on average.")
        print(f"  → D23 is worth implementing as improved calibration method.")
    elif whitened_mean > base_mean + 0.1:
        print(f"  WhitenedSVD gives marginal improvement (+{whitened_mean-base_mean:.2f}).")
        print(f"  → D23 useful but not critical.")
    else:
        delta = whitened_mean - base_mean
        print(f"  WhitenedSVD does NOT improve d' significantly (Δ={delta:+.2f}).")
        print(f"  → Current mean-diff is already near-optimal for this data.")

    if lda_mean > whitened_mean + 0.5:
        print(f"  LDA is better ({lda_mean:.2f} vs whitened {whitened_mean:.2f}) — consider full LDA.")


if __name__ == "__main__":
    main()
