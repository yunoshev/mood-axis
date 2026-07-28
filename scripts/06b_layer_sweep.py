#!/usr/bin/env python3
"""Layer sweep ablation: axis quality (val d', val accuracy) per layer.

Uses per-layer last-token hidden states saved by 01_calibrate
(<model>_hidden_states.npz, keys <axis>_{train,val}_{pos,neg}_layers).
CPU-only, runs on the full v2 corpus. Outputs:
  data/analysis/layer_sweep/layer_sweep.json  — per model/axis/layer metrics
  data/analysis/layer_sweep/layer_sweep.png   — quality vs relative depth
  data/analysis/layer_sweep/aggregation_ablation.json — k / weights ablation
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
AXES_DIR = ROOT / "data" / "axes"
OUT_DIR = ROOT / "data" / "analysis" / "layer_sweep"
AXES = ["warm_cold", "confident_cautious", "empathetic_analytical",
        "formal_casual", "verbose_concise"]


def dprime_acc(axis_vec, val_pos, val_neg):
    """Project val states on axis, return (d', accuracy via midpoint)."""
    n = np.linalg.norm(axis_vec)
    if n == 0:
        return 0.0, 0.5
    v = axis_vec / n
    p = val_pos @ v
    q = val_neg @ v
    pooled = np.sqrt(0.5 * (p.var(ddof=1) + q.var(ddof=1))) + 1e-12
    d = float((p.mean() - q.mean()) / pooled)
    thr = 0.5 * (p.mean() + q.mean())
    acc = float((np.concatenate([p > thr, q <= thr])).mean())
    return d, acc


def sweep_model(npz_path):
    z = np.load(npz_path)
    res = {}
    for ax in AXES:
        try:
            tp = z[f"{ax}_train_pos_layers"].astype(np.float32)
            tn = z[f"{ax}_train_neg_layers"].astype(np.float32)
            vp = z[f"{ax}_val_pos_layers"].astype(np.float32)
            vn = z[f"{ax}_val_neg_layers"].astype(np.float32)
        except KeyError:
            continue
        n_rows = tp.shape[1]  # n_layers + 1 (row 0 = embedding)
        per_layer = []
        for L in range(n_rows):
            axis_vec = tp[:, L].mean(0) - tn[:, L].mean(0)
            d, acc = dprime_acc(axis_vec, vp[:, L], vn[:, L])
            per_layer.append({"layer": L, "dprime": round(d, 3),
                              "accuracy": round(acc, 3)})
        res[ax] = per_layer
    return res


def aggregation_ablation(npz_path):
    """k last layers, uniform vs linear-ramp weights."""
    z = np.load(npz_path)
    out = {}
    for ax in AXES:
        try:
            tp = z[f"{ax}_train_pos_layers"].astype(np.float32)
            tn = z[f"{ax}_train_neg_layers"].astype(np.float32)
            vp = z[f"{ax}_val_pos_layers"].astype(np.float32)
            vn = z[f"{ax}_val_neg_layers"].astype(np.float32)
        except KeyError:
            continue
        entry = {}
        n_rows = tp.shape[1]
        for k in (1, 2, 4, 8):
            if k > n_rows:
                continue
            for scheme in ("uniform", "ramp"):
                w = np.ones(k) if scheme == "uniform" else np.arange(1, k + 1, dtype=np.float64)
                w = (w / w.sum()).astype(np.float32)
                def agg(a):
                    return np.tensordot(a[:, -k:], w, axes=([1], [0]))
                axis_vec = agg(tp).mean(0) - agg(tn).mean(0)
                d, acc = dprime_acc(axis_vec, agg(vp), agg(vn))
                entry[f"k{k}_{scheme}"] = {"dprime": round(d, 3),
                                           "accuracy": round(acc, 3)}
        out[ax] = entry
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    models = sorted(p.name[:-len("_hidden_states.npz")]
                    for p in AXES_DIR.glob("*_hidden_states.npz")
                    if not p.name.startswith("."))
    only = sys.argv[1:] or None
    sweep, ablation = {}, {}
    for m in models:
        if only and m not in only:
            continue
        p = AXES_DIR / f"{m}_hidden_states.npz"
        print(f"[{m}]", flush=True)
        sweep[m] = sweep_model(p)
        ablation[m] = aggregation_ablation(p)
    json.dump(sweep, open(OUT_DIR / "layer_sweep.json", "w"), indent=1)
    json.dump(ablation, open(OUT_DIR / "aggregation_ablation.json", "w"), indent=1)

    # Figure: mean-over-axes accuracy and d' vs relative depth per model
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
        for m, res in sweep.items():
            if not res:
                continue
            n_rows = len(next(iter(res.values())))
            depth = np.arange(n_rows) / (n_rows - 1)
            acc = np.mean([[e["accuracy"] for e in res[a]] for a in res], axis=0)
            dpr = np.mean([[e["dprime"] for e in res[a]] for a in res], axis=0)
            axs[0].plot(depth, acc, alpha=0.7, label=m)
            axs[1].plot(depth, dpr, alpha=0.7)
        axs[0].set(xlabel="relative depth", ylabel="val accuracy (mean over axes)")
        axs[1].set(xlabel="relative depth", ylabel="val d' (mean over axes)")
        axs[0].axvspan(depth[-5] if len(depth) > 5 else 0.85, 1.0, color="gray", alpha=0.15)
        axs[0].legend(fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "layer_sweep.png", dpi=150)
        print("figure saved", flush=True)
    except Exception as e:
        print("figure skipped:", e, flush=True)
    print("done:", len(sweep), "models", flush=True)


if __name__ == "__main__":
    main()
