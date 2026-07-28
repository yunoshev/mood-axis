#!/usr/bin/env python3
"""06o: Layer / pooling sensitivity of axis directions (reviewer defense, pt 5).

The production pipeline aggregates the last HIDDEN_LAYERS_TO_USE=4 layers with
LAYER_WEIGHTS and a token-decay pooling. This audit asks how much the axes and
their held-out quality depend on those choices, using only the per-layer and
last-token states already archived in <model>_hidden_states.npz:

  * per-layer axis (diff of train means at layer L): held-out pair accuracy
    at L, and cos(axis_L, production axis) across depth;
  * last-token pooling: cos(axis_lt, production axis) + held-out pair acc.

CPU-only. Output: data/analysis/layer_pooling_sensitivity.json
"""
import json
import sys
from pathlib import Path

import numpy as np

PAPER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PAPER_DIR))
from config.settings import MOOD_AXES  # noqa: E402

AXES_DIR = PAPER_DIR / "data" / "axes"
OUT = PAPER_DIR / "data" / "analysis" / "layer_pooling_sensitivity.json"

MODELS = ["qwen3.5_9b", "gemma3_12b", "phi4", "deepseek_r1_14b",
          "ministral3_8b", "gpt_oss_20b"]


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _fin(x):
    """Zero out non-finite coords (rare bf16 activation overflows, e.g.
    Gemma 3 late layers: ~0.003% of coords hit +-inf). Dropping the coord
    from the dot products is the least-biased cheap treatment."""
    return np.where(np.isfinite(x), x, 0.0)


def _pair_acc(axis, vp, vn):
    """Fraction of held-out pairs whose positive projection exceeds negative."""
    return float(np.mean((vp @ axis) > (vn @ axis)))


def analyze(model):
    z = np.load(AXES_DIR / f"{model}_hidden_states.npz", allow_pickle=True)
    prod = np.load(AXES_DIR / f"{model}_axes.npz", allow_pickle=True)
    out = {}
    for ax in MOOD_AXES:
        need = [f"{ax}_{s}_{p}_{k}" for s in ("train", "val")
                for p in ("pos", "neg") for k in ("layers", "last_token")]
        if ax not in prod or any(k not in z.files for k in need):
            continue
        tp = _fin(z[f"{ax}_train_pos_layers"].astype(np.float64))
        tn = _fin(z[f"{ax}_train_neg_layers"].astype(np.float64))
        vp = _fin(z[f"{ax}_val_pos_layers"].astype(np.float64))
        vn = _fin(z[f"{ax}_val_neg_layers"].astype(np.float64))
        n_layers = tp.shape[1]
        v_prod = _unit(prod[ax].astype(np.float64))

        per_layer = []
        for L in range(n_layers):
            a = _unit(tp[:, L].mean(0) - tn[:, L].mean(0))
            per_layer.append({
                "layer": L,
                "val_pair_acc": _pair_acc(a, vp[:, L], vn[:, L]),
                "cos_with_production": float(np.dot(a, v_prod))
                if a.shape == v_prod.shape else None,
            })

        a_lt = _unit(_fin(z[f"{ax}_train_pos_last_token"].astype(np.float64)).mean(0)
                     - _fin(z[f"{ax}_train_neg_last_token"].astype(np.float64)).mean(0))
        lt = {
            "cos_with_production": float(np.dot(a_lt, v_prod))
            if a_lt.shape == v_prod.shape else None,
            "val_pair_acc": _pair_acc(
                a_lt, _fin(z[f"{ax}_val_pos_last_token"].astype(np.float64)),
                _fin(z[f"{ax}_val_neg_last_token"].astype(np.float64))),
        }

        accs = [p["val_pair_acc"] for p in per_layer]
        upper_half = per_layer[n_layers // 2:]
        out[ax] = {
            "n_layers": n_layers,
            "per_layer": per_layer,
            "last_token": lt,
            "summary": {
                "acc_at_production_pooling": None,  # by construction ~ val acc in axes meta
                "best_layer": int(np.argmax(accs)),
                "best_layer_acc": float(np.max(accs)),
                "n_layers_acc_ge_0.9": int(np.sum(np.asarray(accs) >= 0.9)),
                "upper_half_min_acc": float(min(p["val_pair_acc"] for p in upper_half)),
                "upper_half_min_cos_prod": float(min(
                    p["cos_with_production"] for p in upper_half
                    if p["cos_with_production"] is not None)),
            },
        }
    return out


def main():
    report = {"_meta": {
        "production_pooling": "weighted mix of last 4 layers + token decay",
        "note": "per-layer axis = unit diff of train means at that layer; "
                "val pair accuracy on held-out 15 question pairs",
    }, "models": {}}
    for m in MODELS:
        try:
            res = analyze(m)
        except FileNotFoundError as e:
            print(f"SKIP {m}: {e}", flush=True)
            continue
        report["models"][m] = res
        print(f"\n{m}:", flush=True)
        for ax, r in res.items():
            s = r["summary"]
            print(f"  {ax:24s} layers>=0.9acc: {s['n_layers_acc_ge_0.9']:2d}/"
                  f"{r['n_layers']}  upper-half min acc {s['upper_half_min_acc']:.2f} "
                  f"min cos(prod) {s['upper_half_min_cos_prod']:+.2f}  "
                  f"last_token acc {r['last_token']['val_pair_acc']:.2f} "
                  f"cos {r['last_token']['cos_with_production']:+.2f}", flush=True)

    OUT.write_text(json.dumps(report, indent=2))
    print(f"\nSaved {OUT}", flush=True)


if __name__ == "__main__":
    main()
