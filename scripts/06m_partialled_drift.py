#!/usr/bin/env python3
"""06m: Drift robustness on format-partialled axes (reviewer defense, pt a).

For every model with saved per-turn drift hidden states, remove the format
subspace (VC mean contrast + top-FORMAT_PCS PCs of VC-diff variation, same
construction as 06k) from each content axis, re-derive center/scale with the
calibration formulas (median / IQR of train projections), re-project every
drift turn, and re-run the 06e significance machinery. Output: per-model
raw-vs-partialled comparison of significant axes.

CPU-only, local data. verbose_concise itself IS the format probe, so it is
reported as-is (partialling it against itself would delete the axis).
"""
import json
import sys
import importlib.util
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PAPER_DIR))

from config.settings import MOOD_AXES  # noqa: E402

AXES_DIR = PAPER_DIR / "data" / "axes"
DRIFT_DIR = PAPER_DIR / "data" / "drift"
HS_DIR = DRIFT_DIR / "hidden_states"
OUT_PATH = PAPER_DIR / "data" / "analysis" / "format_partialled_drift.json"
RAW_STATS = PAPER_DIR / "data" / "analysis" / "drift_stats.json"

VC = "verbose_concise"
FORMAT_PCS = 3  # keep in sync with 06k
CONTENT_AXES = [a for a in MOOD_AXES if a != VC]


def _import_06e():
    spec = importlib.util.spec_from_file_location(
        "drift_stats_06e", SCRIPT_DIR / "06e_drift_stats.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _format_basis(z):
    """Orthonormal basis of the VC-diff format subspace (as in 06k)."""
    blocks = []
    for split in ("train", "val"):
        kp, kn = f"{VC}_{split}_pos_aggregated", f"{VC}_{split}_neg_aggregated"
        if kp not in z.files or kn not in z.files:
            return None
        blocks.append(z[kp].astype(np.float64) - z[kn].astype(np.float64))
    V = np.vstack(blocks)
    Vt = np.linalg.svd(V - V.mean(0), full_matrices=False)[2]
    P = np.vstack([_unit(V.mean(0))] + [Vt[i] for i in range(FORMAT_PCS)])
    Q, _ = np.linalg.qr(P.T)
    return Q


def _partialled_axes(model):
    """Per-axis (vector, center, scale) after format-subspace removal.

    center/scale re-derived exactly as calibration does (median / IQR of
    train pos+neg projections, scale floored at 0.1).
    """
    axes_npz = np.load(AXES_DIR / f"{model}_axes.npz", allow_pickle=True)
    z = np.load(AXES_DIR / f"{model}_hidden_states.npz", allow_pickle=True)
    Q = _format_basis(z)
    if Q is None:
        return None
    out = {}
    for axis in CONTENT_AXES:
        if axis not in axes_npz:
            continue
        v = axes_npz[axis].astype(np.float64)
        vp = _unit(v - Q @ (Q.T @ v))
        kp, kn = f"{axis}_train_pos_aggregated", f"{axis}_train_neg_aggregated"
        if kp not in z.files or kn not in z.files:
            continue
        train = np.vstack([z[kp], z[kn]]).astype(np.float64)
        proj = train @ vp
        q75, q25 = np.percentile(proj, [75, 25])
        out[axis] = {
            "vector": vp,
            "center": float(np.median(proj)),
            "scale": max(float(q75 - q25), 0.1),
            "cos_raw_partialled": float(np.dot(_unit(v), vp)),
        }
    return out


def _reproject(data, model, part):
    """Deep-copied drift data with partialled projections; drops units
    lacking hidden-state NPZs (coverage is reported, never silent)."""
    new = json.loads(json.dumps(data))
    hs_dir = HS_DIR / model
    covered, dropped = 0, 0

    def _do(entry, name):
        nonlocal covered, dropped
        p = hs_dir / f"{name}.npz"
        if not p.exists():
            dropped += 1
            return None
        z = np.load(p)
        for i, turn in enumerate(entry["turns"]):
            key = f"turn_{i}"
            if key not in z:
                return None  # partial npz: drop whole unit, keep pairing clean
            h = z[key].astype(np.float64)
            for axis, pa in part.items():
                val = (float(h @ pa["vector"]) - pa["center"]) / pa["scale"]
                turn["values"][axis] = float(np.clip(val, -1.0, 1.0))
        covered += 1
        return entry

    new["conflict_scenarios"] = [
        e for e in new["conflict_scenarios"]
        if _do(e, e["scenario_id"]) is not None]
    new["neutral_controls"] = [
        e for e in new["neutral_controls"] if _do(e, e["name"]) is not None]
    return new, covered, dropped


def main():
    mod = _import_06e()
    raw_stats = json.loads(RAW_STATS.read_text()) if RAW_STATS.exists() else {}
    raw_models = raw_stats.get("models", raw_stats)

    models = sorted(d.name for d in HS_DIR.iterdir()
                    if d.is_dir() and (DRIFT_DIR / f"{d.name}.json").exists()
                    and (AXES_DIR / f"{d.name}_hidden_states.npz").exists())
    print(f"Models with drift hidden states: {models}", flush=True)

    report = {"_meta": {
        "format_subspace": f"VC mean diff + top-{FORMAT_PCS} PCs (as 06k)",
        "note": "verbose_concise reported raw (it is the format probe itself)",
    }, "models": {}}

    for m in models:
        part = _partialled_axes(m)
        if not part:
            print(f"  SKIP {m}: no format basis", flush=True)
            continue
        data = json.loads((DRIFT_DIR / f"{m}.json").read_text())
        new, covered, dropped = _reproject(data, m, part)
        n_conf, n_ctrl = len(new["conflict_scenarios"]), len(new["neutral_controls"])
        print(f"\n{m}: coverage {covered} units ({dropped} dropped, "
              f"{n_conf} conflict + {n_ctrl} control kept)", flush=True)
        res = mod.analyze_model(m, new)

        raw_m = (raw_models.get(m) or {}).get("axes", {})
        comp = {}
        for axis in CONTENT_AXES:
            pa, ra = res["axes"].get(axis), raw_m.get(axis)
            if not pa:
                continue
            comp[axis] = {
                "cos_raw_partialled": part[axis]["cos_raw_partialled"],
                "raw_significant": ra.get("significant_drift") if ra else None,
                "raw_slope": (ra.get("conflict_slope") or {}).get("mean") if ra else None,
                "part_significant": pa["significant_drift"],
                "part_slope": (pa.get("conflict_slope") or {}).get("mean"),
            }
            flip = (ra is not None
                    and bool(ra.get("significant_drift")) != pa["significant_drift"])
            mark = "  <-- CHANGED" if flip else ""
            rs = ((ra.get("conflict_slope") or {}).get("mean") if ra else None) or float("nan")
            print(f"  {axis:24s} cos={comp[axis]['cos_raw_partialled']:.3f}  "
                  f"raw: {'YES' if ra and ra.get('significant_drift') else 'no '} "
                  f"{rs:+.4f}  part: "
                  f"{'YES' if pa['significant_drift'] else 'no '} "
                  f"{((pa.get('conflict_slope') or {}).get('mean') or 0):+.4f}{mark}", flush=True)
        report["models"][m] = {
            "coverage_units": covered, "dropped_units": dropped,
            "n_conflict_kept": n_conf, "n_controls_kept": n_ctrl,
            "axes": comp,
        }
        if n_ctrl == 0:
            report["models"][m]["warning"] = (
                "no control hidden states saved; the 2x-control criterion is "
                "degenerate here — compare slopes/CIs only, not significance")
            print(f"  WARNING {m}: 0 controls kept; significance flags "
                  f"not comparable (slopes only)", flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nSaved {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
