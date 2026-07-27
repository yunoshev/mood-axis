"""
06b_alignment_imprint.py — Per-layer d' analysis and Alignment Imprint features

AlignmentImprint hypothesis (D24):
  Different alignment methods (RLHF, DPO, SFT, CAI) leave distinguishable
  geometric signatures in how well mood axes can be discriminated at each layer.

  E.g.:
  - RLHF (Llama): mid-layer peak, suppressed final layers (dead zones)
  - DPO (Qwen): monotonically increasing d', strong final layer
  - SFT (Mistral): flat profile, lower d' overall (less pronounced personality)

Features computed per (model, axis):
  peak_layer        — which layer has highest d'
  peak_norm         — peak_layer / n_layers (0=early, 1=late)
  peak_dprime       — d' at peak layer
  final_dprime      — d' at last layer
  final_pct_of_peak — final_dprime / peak_dprime (1.0 = final is peak)
  early_bias        — mean(d' in first third) / mean(d' in last third)
  gini              — Gini coef of d' distribution (0=uniform, 1=concentrated)
  smoothness        — mean cosine-sim between adjacent layer direction vectors
  effective_rank    — exp(entropy of normalized d') — diversity of discriminant signal

Output:
  Per-axis layer d' profile (ASCII sparkline)
  Per-model feature table
  Inter-model comparison for hypothesis D24
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "axes"
MODELS = ["qwen_7b", "gemma_9b", "mistral_7b", "llama_8b", "qwen3_8b"]

# Known alignment methods (approximate)
ALIGNMENT = {
    "qwen_7b":   "DPO+RLHF (Qwen2.5)",
    "gemma_9b":  "RLHF+SFT (Gemma2)",
    "mistral_7b": "SFT (Mistral-v0.3)",
    "llama_8b":  "RLHF (Llama3.1)",
    "qwen3_8b":  "DPO (Qwen3)",
}


def compute_dprime(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    n1, n2 = len(pos), len(neg)
    v1 = np.var(pos, ddof=1) if n1 > 1 else 1e-8
    v2 = np.var(neg, ddof=1) if n2 > 1 else 1e-8
    pooled = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    return float((np.mean(pos) - np.mean(neg)) / max(pooled, 1e-10))


def gini_coefficient(x):
    """Gini coefficient: 0=uniform, 1=concentrated."""
    x = np.abs(np.asarray(x, dtype=float))
    x = x[x > 0]
    if len(x) == 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * x) / (n * np.sum(x))) - (n + 1) / n)


def effective_rank(dprime_profile):
    """Effective rank via entropy of normalized d' distribution."""
    p = np.abs(dprime_profile)
    s = p.sum()
    if s < 1e-10:
        return 1.0
    p = p / s
    p = p[p > 1e-10]
    return float(np.exp(-np.sum(p * np.log(p))))


def layer_smoothness(pos_layers, neg_layers):
    """
    Mean cosine similarity between adjacent layer direction vectors.
    High = directions change slowly across layers (smooth gradient flow).
    """
    n_layers = pos_layers.shape[1]
    axes = []
    for l in range(n_layers):
        d = pos_layers[:, l, :].mean(0) - neg_layers[:, l, :].mean(0)
        norm = np.linalg.norm(d)
        axes.append(d / norm if norm > 1e-10 else d)

    sims = []
    for l in range(n_layers - 1):
        cos = np.dot(axes[l], axes[l + 1])
        sims.append(cos)
    return float(np.mean(sims)) if sims else 0.0


def sparkline(values, width=40):
    """ASCII sparkline for a list of values."""
    CHARS = "▁▂▃▄▅▆▇█"
    mn, mx = min(values), max(values)
    rng = mx - mn
    if rng < 1e-10:
        return CHARS[0] * min(len(values), width)
    step = len(values) / width
    result = ""
    for i in range(min(width, len(values))):
        idx = int(i * step)
        v = values[min(idx, len(values) - 1)]
        char_idx = int((v - mn) / rng * (len(CHARS) - 1))
        result += CHARS[char_idx]
    return result


def analyze_model(model_name):
    hs_path = DATA_DIR / f"{model_name}_hidden_states.npz"
    meta_path = DATA_DIR / f"{model_name}_axes_meta.json"

    if not hs_path.exists():
        print(f"  [SKIP] No hidden_states.npz")
        return None

    with open(meta_path) as f:
        meta = json.load(f)

    axes = [k for k in meta if not k.startswith("_")]
    npz = np.load(hs_path)
    model_n_layers = int(npz["_n_layers"])

    model_features = {}

    for axis in axes:
        try:
            pos_tr_l = npz[f"{axis}_train_pos_layers"].astype(np.float32)   # (50, L, d)
            neg_tr_l = npz[f"{axis}_train_neg_layers"].astype(np.float32)
            pos_val_l = npz[f"{axis}_val_pos_layers"].astype(np.float32)     # (15, L, d)
            neg_val_l = npz[f"{axis}_val_neg_layers"].astype(np.float32)
        except KeyError as e:
            print(f"  {axis}: MISSING KEY {e}")
            continue

        n_layers = pos_tr_l.shape[1]  # use actual array shape (may differ from _n_layers)
        # Compute d' at each layer
        layer_dp = []
        layer_axes = []
        for l in range(n_layers):
            pos_l = pos_tr_l[:, l, :]  # (50, d)
            neg_l = neg_tr_l[:, l, :]
            d_vec = pos_l.mean(0) - neg_l.mean(0)
            norm = np.linalg.norm(d_vec)
            if norm < 1e-10:
                layer_dp.append(0.0)
                layer_axes.append(np.zeros_like(d_vec))
                continue
            axis_l = d_vec / norm
            layer_axes.append(axis_l)

            # Val projections
            pos_vl = pos_val_l[:, l, :]  # (15, d)
            neg_vl = neg_val_l[:, l, :]
            dp = compute_dprime(pos_vl @ axis_l, neg_vl @ axis_l)
            layer_dp.append(dp)

        layer_dp = np.array(layer_dp)

        # Features
        peak_l = int(np.argmax(layer_dp))
        peak_dp = float(layer_dp[peak_l])
        final_dp = float(layer_dp[-1])
        third = n_layers // 3
        early_mean = float(np.mean(layer_dp[:third]))
        late_mean = float(np.mean(layer_dp[2 * third:]))
        early_bias = early_mean / max(late_mean, 1e-10)

        features = {
            "n_layers": n_layers,
            "layer_dp": layer_dp.tolist(),
            "baseline_dp": meta[axis]["val_dprime"],
            "peak_layer": peak_l,
            "peak_norm": peak_l / (n_layers - 1),
            "peak_dprime": peak_dp,
            "final_dprime": final_dp,
            "final_pct_of_peak": final_dp / max(peak_dp, 1e-10),
            "early_bias": early_bias,
            "gini": gini_coefficient(layer_dp),
            "smoothness": layer_smoothness(pos_tr_l, neg_tr_l),
            "effective_rank": effective_rank(layer_dp),
        }
        model_features[axis] = features

    return model_features


def print_model_report(model_name, features):
    n_layers = list(features.values())[0]["n_layers"]
    align = ALIGNMENT.get(model_name, "unknown")
    print(f"\n{'='*72}")
    print(f"Model: {model_name:<20} alignment={align}  n_layers={n_layers}")
    print(f"{'='*72}")

    print(f"  {'Axis':<25} {'Base d':>7} {'Peak d':>7} {'@L':>4} {'Final':>7} "
          f"{'F/P%':>6} {'E-bias':>7} {'Gini':>6} {'EfRnk':>6}")
    print(f"  {'-'*25} {'-'*7} {'-'*7} {'-'*4} {'-'*7} "
          f"{'-'*6} {'-'*7} {'-'*6} {'-'*6}")

    for axis, f in features.items():
        print(
            f"  {axis:<25} "
            f"{f['baseline_dp']:7.2f} "
            f"{f['peak_dprime']:7.2f} "
            f"{f['peak_layer']:4d} "
            f"{f['final_dprime']:7.2f} "
            f"{f['final_pct_of_peak']*100:6.0f}% "
            f"{f['early_bias']:7.2f} "
            f"{f['gini']:6.2f} "
            f"{f['effective_rank']:6.1f}"
        )

    # Sparklines
    print(f"\n  Layer d' profiles (layer 0 → {n_layers-1}):")
    for axis, f in features.items():
        spark = sparkline(f["layer_dp"], width=50)
        print(f"  {axis:<25} {spark}  peak@L{f['peak_layer']}")


def print_cross_model_summary(all_features):
    """Cross-model feature comparison table."""
    print(f"\n\n{'='*72}")
    print("CROSS-MODEL COMPARISON — Mean across axes")
    print(f"{'='*72}")
    print(f"  {'Model':<22} {'Align':<22} {'PeakNorm':>9} {'F/P%':>6} "
          f"{'EBias':>7} {'Gini':>6} {'Smooth':>7} {'EfRnk':>7}")
    print(f"  {'-'*22} {'-'*22} {'-'*9} {'-'*6} "
          f"{'-'*7} {'-'*6} {'-'*7} {'-'*7}")

    for model_name, features in all_features.items():
        if not features:
            continue
        fvals = list(features.values())
        pn  = np.mean([f["peak_norm"] for f in fvals])
        fp  = np.mean([f["final_pct_of_peak"] for f in fvals])
        eb  = np.mean([f["early_bias"] for f in fvals])
        gi  = np.mean([f["gini"] for f in fvals])
        sm  = np.mean([f["smoothness"] for f in fvals])
        er  = np.mean([f["effective_rank"] for f in fvals])
        align = ALIGNMENT.get(model_name, "?")[:20]
        print(
            f"  {model_name:<22} {align:<22} "
            f"{pn:9.2f} {fp*100:6.0f}% "
            f"{eb:7.2f} {gi:6.2f} {sm:7.3f} {er:7.1f}"
        )

    print(f"\nFeature meanings:")
    print(f"  PeakNorm   — where peak d' occurs (0=early layers, 1=last layer)")
    print(f"  F/P%       — final-layer d' as % of peak (100%=peak IS final layer)")
    print(f"  EBias      — early/late ratio (>1=discrimination builds early)")
    print(f"  Gini       — concentration of d' signal (0=uniform, 1=one layer)")
    print(f"  Smooth     — cross-layer direction cosine (1=stable, 0=rotating)")
    print(f"  EfRnk      — effective rank of layer profile (max=n_layers)")


def print_hypothesis_test(all_features):
    """Test D24 hypothesis: alignment type → distinctive layer d' pattern."""
    print(f"\n\n{'='*72}")
    print("HYPOTHESIS D24: Alignment Imprint in Layer d' Profiles")
    print(f"{'='*72}")

    # Collect per-model mean features
    model_stats = {}
    for model, features in all_features.items():
        if not features:
            continue
        fvals = list(features.values())
        model_stats[model] = {
            "peak_norm": np.mean([f["peak_norm"] for f in fvals]),
            "final_pct": np.mean([f["final_pct_of_peak"] for f in fvals]),
            "early_bias": np.mean([f["early_bias"] for f in fvals]),
            "gini": np.mean([f["gini"] for f in fvals]),
            "smoothness": np.mean([f["smoothness"] for f in fvals]),
        }

    # Find distinguishing features between models
    print("\n  Distinguishing patterns:")
    if "llama_8b" in model_stats and "qwen_7b" in model_stats:
        llama_fn = model_stats["llama_8b"]["final_pct"]
        qwen_fn  = model_stats["qwen_7b"]["final_pct"]
        print(f"\n  RLHF (Llama) vs DPO+RLHF (Qwen) — final layer retention:")
        print(f"    Llama_8b final/peak = {llama_fn*100:.0f}%")
        print(f"    Qwen_7b  final/peak = {qwen_fn*100:.0f}%")
        if qwen_fn > llama_fn + 0.1:
            print(f"    → Qwen retains more d' in final layers (DPO preserves structure)")
        elif llama_fn > qwen_fn + 0.1:
            print(f"    → Llama retains more d' in final layers")
        else:
            print(f"    → No significant difference in final layer retention")

    if "mistral_7b" in model_stats:
        mistral_smooth = model_stats["mistral_7b"]["smoothness"]
        others = [v["smoothness"] for k, v in model_stats.items() if k != "mistral_7b"]
        print(f"\n  SFT (Mistral) cross-layer smoothness: {mistral_smooth:.3f}")
        print(f"  Others mean smoothness: {np.mean(others):.3f}")
        if mistral_smooth > np.mean(others) + 0.05:
            print(f"  → Mistral SFT produces smoother (less volatile) representations")
        elif mistral_smooth < np.mean(others) - 0.05:
            print(f"  → Mistral SFT produces more rotating representations layer-to-layer")

    if "gemma_9b" in model_stats:
        gemma_pn = model_stats["gemma_9b"]["peak_norm"]
        others_pn = [v["peak_norm"] for k, v in model_stats.items() if k != "gemma_9b"]
        print(f"\n  RLHF+SFT (Gemma) peak location: layer {gemma_pn:.2f} (normalized)")
        print(f"  Others mean: {np.mean(others_pn):.2f}")

    print(f"\n  Verdict:")
    # Check if models cluster by alignment type in feature space
    dpo_models = ["qwen_7b", "qwen3_8b"]
    sft_models = ["mistral_7b"]
    rlhf_models = ["llama_8b", "gemma_9b"]

    for feat in ["peak_norm", "final_pct", "smoothness"]:
        dpo_vals = [model_stats[m][feat] for m in dpo_models if m in model_stats]
        rlhf_vals = [model_stats[m][feat] for m in rlhf_models if m in model_stats]
        sft_vals = [model_stats[m][feat] for m in sft_models if m in model_stats]
        if dpo_vals and rlhf_vals:
            sep = abs(np.mean(dpo_vals) - np.mean(rlhf_vals))
            print(f"  {feat:<15}: DPO={np.mean(dpo_vals):.3f}  RLHF={np.mean(rlhf_vals):.3f}  "
                  f"SFT={np.mean(sft_vals):.3f}  |DPO-RLHF|={sep:.3f}")


def main():
    print("AlignmentImprint Analysis — Per-Layer d' Profiles")
    print(f"Models: {MODELS}")

    all_features = {}
    for model in MODELS:
        print(f"\nProcessing {model}...", flush=True)
        features = analyze_model(model)
        if features:
            all_features[model] = features
            print_model_report(model, features)

    if len(all_features) >= 2:
        print_cross_model_summary(all_features)
        print_hypothesis_test(all_features)

    # Save features to JSON for later use
    out_path = DATA_DIR.parent / "analysis" / "alignment_imprint.json"
    out_path.parent.mkdir(exist_ok=True)

    # Make serializable
    save = {}
    for model, features in all_features.items():
        save[model] = {}
        for axis, f in features.items():
            d = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                 for k, v in f.items()}
            save[model][axis] = d

    import json
    with open(out_path, "w") as fp:
        json.dump(save, fp, indent=2)
    print(f"\n\nFeatures saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
