#!/usr/bin/env python3
"""Step 06j: Quantization robustness — do mood axes survive 4-bit weights?

Motivation (decided 2026-07-24): models >20B are measured in 4-bit
(mlx-community quants) as the deployment-realistic precision. To justify
comparing 4-bit giants against bf16 mid-size models, every main-cast model
gets a paired calibration in both precisions, and this script quantifies
how much the measured temperament machinery moves between them.

Pairs compared:
  * Main-cast twins: <m>_axes.npz (bf16) vs <m>_4bit_axes.npz — both under
    data/axes/, separate model keys, identical prompts/settings.
  * Giants: data/axes_bf16/<m>_axes.npz (bf16 run, restored from git
    752051e2^) vs data/axes/<m>_axes.npz (the current 4-bit run) — same
    model key, the bf16 copy is an archived snapshot.

Per (model, axis):
  1. axis_cos          — cosine between the bf16 and 4-bit axis directions
                         (unit-normalized). The headline number.
  2. val_accuracy      — held-out pole classification accuracy in each
                         precision (from *_axes_meta.json).
  3. val_dprime        — held-out d' in each precision.
  4. center_shift_bf16_scale — |center_4bit - center_bf16| / scale_bf16:
                         how far the neutral point moves, in bf16 units.
  5. scale_ratio       — scale_4bit / scale_bf16.

Per model (when both precisions have stability sets on disk, i.e. after
04_stability finishes for the 4-bit twin): mean cross-set axis consistency
(mean pairwise cos between per-set axis directions, sets A/B/C) in each
precision — does 4-bit degrade the *re-measurability* of the axes, not just
their direction?

Reads:  data/axes/<m>_axes.npz + meta, data/axes/<m>_4bit_axes.npz + meta,
        data/axes_bf16/<m>_axes.npz + meta,
        data/stability/<m>[_4bit]_set_<S>_axes.npz (optional)
Writes: data/analysis/quant_robustness.json

Usage:
    python3 -u scripts/06j_quant_robustness.py
    python3 -u scripts/06j_quant_robustness.py --check
"""

import argparse
import json
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import MOOD_AXES

AXES_DIR = PROJECT_ROOT / "data" / "axes"
AXES_BF16_DIR = PROJECT_ROOT / "data" / "axes_bf16"
STABILITY_DIR = PROJECT_ROOT / "data" / "stability"
OUT_FILE = PROJECT_ROOT / "data" / "analysis" / "quant_robustness.json"

# (label, bf16 npz stem, 4bit npz stem, stability stems bf16/4bit)
TWINS = ["qwen3.5_9b", "gemma3_12b", "phi4", "deepseek_r1_14b",
         "ministral3_8b"]
GIANTS = ["qwen3.6_27b", "mistral_small_3.2_24b", "qwen3.6_35b_a3b",
          "gemma4_26b_a4b"]
SET_NAMES = ["A", "B", "C"]


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _load_axes(npz_path):
    z = np.load(npz_path)
    return {a: np.asarray(z[a], dtype=np.float64) for a in MOOD_AXES
            if a in z.files}, z


def _load_norm(z, axis):
    c = float(z[f"{axis}_center"]) if f"{axis}_center" in z.files else None
    s = float(z[f"{axis}_scale"]) if f"{axis}_scale" in z.files else None
    return c, s


def _meta_axis(meta_path, axis):
    if not meta_path.exists():
        return {}
    meta = json.load(open(meta_path))
    return meta.get(axis, {})


def _consistency(stem):
    """Mean pairwise cos between per-set axis directions; None if sets absent."""
    sets = {}
    for s in SET_NAMES:
        p = STABILITY_DIR / f"{stem}_set_{s}_axes.npz"
        if p.exists():
            sets[s], _ = _load_axes(p)
    if len(sets) < 2:
        return None
    per_axis = {}
    for a in MOOD_AXES:
        cs = [float(np.dot(_unit(sets[x][a]), _unit(sets[y][a])))
              for x, y in combinations(sorted(sets), 2)
              if a in sets[x] and a in sets[y]]
        if cs:
            per_axis[a] = float(np.mean(cs))
    if not per_axis:
        return None
    return {"per_axis": per_axis,
            "mean": float(np.mean(list(per_axis.values()))),
            "n_sets": len(sets)}


def compare(label, bf16_stem, bf16_dir, q4_stem, q4_dir,
            stab_bf16_stem=None, stab_q4_stem=None):
    bf16_npz = bf16_dir / f"{bf16_stem}_axes.npz"
    q4_npz = q4_dir / f"{q4_stem}_axes.npz"
    if not bf16_npz.exists() or not q4_npz.exists():
        missing = [str(p) for p in (bf16_npz, q4_npz) if not p.exists()]
        print(f"  SKIP {label}: missing {missing}", flush=True)
        return None
    vecs16, z16 = _load_axes(bf16_npz)
    vecs4, z4 = _load_axes(q4_npz)
    m16 = bf16_dir / f"{bf16_stem}_axes_meta.json"
    m4 = q4_dir / f"{q4_stem}_axes_meta.json"

    axes_out = {}
    for a in MOOD_AXES:
        if a not in vecs16 or a not in vecs4:
            continue
        if vecs16[a].shape != vecs4[a].shape:
            print(f"  WARN {label}/{a}: dim mismatch "
                  f"{vecs16[a].shape} vs {vecs4[a].shape}", flush=True)
            continue
        c16, s16 = _load_norm(z16, a)
        c4, s4 = _load_norm(z4, a)
        meta16, meta4 = _meta_axis(m16, a), _meta_axis(m4, a)
        axes_out[a] = {
            "axis_cos": float(np.dot(_unit(vecs16[a]), _unit(vecs4[a]))),
            "val_accuracy_bf16": meta16.get("val_accuracy"),
            "val_accuracy_4bit": meta4.get("val_accuracy"),
            "val_dprime_bf16": meta16.get("val_dprime"),
            "val_dprime_4bit": meta4.get("val_dprime"),
            "center_shift_bf16_scale":
                (abs(c4 - c16) / s16) if (c16 is not None and c4 is not None
                                          and s16) else None,
            "scale_ratio": (s4 / s16) if (s16 and s4) else None,
        }
    if not axes_out:
        print(f"  SKIP {label}: no comparable axes", flush=True)
        return None

    cons16 = _consistency(stab_bf16_stem) if stab_bf16_stem else None
    cons4 = _consistency(stab_q4_stem) if stab_q4_stem else None
    cos_vals = [v["axis_cos"] for v in axes_out.values()]
    out = {
        "axes": axes_out,
        "mean_axis_cos": float(np.mean(cos_vals)),
        "min_axis_cos": float(np.min(cos_vals)),
        "min_axis": min(axes_out, key=lambda a: axes_out[a]["axis_cos"]),
        "consistency_bf16": cons16,
        "consistency_4bit": cons4,
    }
    print(f"  {label}: mean cos {out['mean_axis_cos']:.3f}, "
          f"min {out['min_axis_cos']:.3f} ({out['min_axis']})", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        for m in TWINS:
            print(m, (AXES_DIR / f"{m}_axes.npz").exists(),
                  (AXES_DIR / f"{m}_4bit_axes.npz").exists())
        for m in GIANTS:
            print(m, (AXES_BF16_DIR / f"{m}_axes.npz").exists(),
                  (AXES_DIR / f"{m}_axes.npz").exists())
        return

    results = {"twins": {}, "giants": {}}
    print("Main-cast twins (bf16 vs 4bit, separate keys):", flush=True)
    for m in TWINS:
        r = compare(m, m, AXES_DIR, f"{m}_4bit", AXES_DIR,
                    stab_bf16_stem=m, stab_q4_stem=f"{m}_4bit")
        if r:
            results["twins"][m] = r
    print("Giants (archived bf16 vs current 4bit run):", flush=True)
    for m in GIANTS:
        r = compare(m, m, AXES_BF16_DIR, m, AXES_DIR,
                    stab_bf16_stem=None, stab_q4_stem=m)
        if r:
            results["giants"][m] = r

    pooled = [v["axes"][a]["axis_cos"]
              for grp in results.values() for v in grp.values()
              for a in v["axes"]]
    summary = {
        "n_models": sum(len(g) for g in results.values()),
        "n_axis_pairs": len(pooled),
        "pooled_mean_axis_cos": float(np.mean(pooled)) if pooled else None,
        "pooled_min_axis_cos": float(np.min(pooled)) if pooled else None,
    }
    out = {
        "_meta": {
            "script": "scripts/06j_quant_robustness.py",
            "timestamp": datetime.now().isoformat(),
            "bf16_giant_source": "git 752051e2^ -> data/axes_bf16/",
            "axes": list(MOOD_AXES),
        },
        "summary": summary,
        **results,
    }
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT_FILE, "w"), indent=1)
    print(f"\nPooled: mean cos {summary['pooled_mean_axis_cos']:.3f}, "
          f"min {summary['pooled_min_axis_cos']:.3f} over "
          f"{summary['n_axis_pairs']} axis pairs / {summary['n_models']} models",
          flush=True)
    print(f"Saved {OUT_FILE}", flush=True)


if __name__ == "__main__":
    main()
