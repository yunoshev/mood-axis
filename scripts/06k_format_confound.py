#!/usr/bin/env python3
"""Step 06k: Format/length confound diagnosis (CPU only).

Motivated by the GPT-OSS 20B anomaly forensics (2026-07-24): on persona
calibration prompts GPT-OSS shifts its response FORMAT (length, markdown
document structure) far more than its content style, so contrastive axes
calibrated on those responses partly measure format, not temperament.
This script turns that forensic session into a reproducible, cohort-wide
diagnostic.

Per model and content axis (the four non-length axes):
  1. cos_vc            — cosine between the saved axis vector and the saved
                         verbose_concise axis vector. |cos| >~ 0.4 flags an
                         axis that substantially shares the length direction.
  2. r_proj_dlen       — correlation between per-pair diff projections on the
                         axis direction and the pos-neg response length gap.
  3. resid_pc1_share   — variance share of PC1 of the centered diffs;
     r_pc1_dlen        — its correlation with the length gap (is the
                         competing direction just length?).
  4. val_pairacc_raw / val_pairacc_fmt — paired val accuracy (pos > neg on
     the axis) before/after projecting out the format subspace.
  5. cross_set_cos_raw / cross_set_cos_fmt — mean pairwise cosine between
     per-set stability axis directions before/after the same projection
     (only when data/stability/<m>_set_<S>_axes.npz exist).

Format subspace: the mean verbose_concise contrastive diff plus the top
FORMAT_PCS principal components of its variation, orthonormalized. Built
from the model's own main calibration; stability sets are independent data,
so projecting them onto its orthogonal complement involves no leakage.

Pole length stats (mean tokens per pole, markdown-document share, refusal
count) are reported per axis as behavioral context.

Reads:  data/axes/<m>_hidden_states.npz, <m>_axes.npz, <m>_calibration.jsonl,
        data/stability/<m>_set_<S>_axes.npz (optional)
Writes: data/analysis/format_confound.json

Usage:
    python3 -u scripts/06k_format_confound.py
    python3 -u scripts/06k_format_confound.py --models gpt_oss_20b,deepseek_r1_14b
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

AXES_DIR = PROJECT_ROOT / "data" / "axes"
STABILITY_DIR = PROJECT_ROOT / "data" / "stability"
OUT_FILE = PROJECT_ROOT / "data" / "analysis" / "format_confound.json"

ALL_AXES = ["warm_cold", "confident_cautious", "empathetic_analytical",
            "formal_casual", "verbose_concise"]
CONTENT_AXES = ALL_AXES[:-1]
VC = "verbose_concise"
FORMAT_PCS = 3
SET_NAMES = ["A", "B", "C"]

MD_RE = re.compile(r"^#{1,4} |^\s*[-*] |^\s*\d+\. ", re.M)
REFUSAL_RE = re.compile(
    r"I can.t|I cannot|I.m sorry|I won.t|as an AI|I am unable", re.I)


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _load_jsonl_meta(path):
    """(axis, split, pole) -> ordered lists of lengths/questions + behavior."""
    lens, questions = defaultdict(list), defaultdict(list)
    beh = defaultdict(lambda: {"n": 0, "tok": 0, "md": 0, "refusal": 0})
    for line in open(path):
        r = json.loads(line)
        k = (r["axis"], r["split"], r["pole"])
        lens[k].append(r["n_tokens"])
        questions[k].append(r["question"])
        b = beh[(r["axis"], r["pole"])]
        b["n"] += 1
        b["tok"] += r["n_tokens"]
        b["md"] += bool(len(MD_RE.findall(r["response"])) >= 3)
        b["refusal"] += bool(REFUSAL_RE.search(r["response"][:400]))
    return lens, questions, beh


def _diffs_and_dlen(z, lens, questions, axis):
    """Stacked train+val contrastive diffs + per-pair length gaps.

    Returns None when pos/neg question order does not align (cannot pair).
    """
    diffs, dlen = [], []
    for split in ("train", "val"):
        kp = f"{axis}_{split}_pos_aggregated"
        kn = f"{axis}_{split}_neg_aggregated"
        if kp not in z.files or kn not in z.files:
            return None, None
        if (questions[(axis, split, "positive")]
                != questions[(axis, split, "negative")]):
            return None, None
        if len(questions[(axis, split, "positive")]) != z[kp].shape[0]:
            return None, None
        diffs.append(z[kp].astype(np.float64) - z[kn].astype(np.float64))
        dlen.extend(np.asarray(lens[(axis, split, "positive")], dtype=float)
                    - np.asarray(lens[(axis, split, "negative")], dtype=float))
    return np.vstack(diffs), np.asarray(dlen)


def _format_basis(z):
    """Orthonormal basis of the VC-diff format subspace, or None."""
    d, _ = None, None
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


def _remove(M, Q):
    return M - (M @ Q) @ Q.T


def _corr(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _cross_set_cos(model, axis, Q):
    vecs = {}
    for s in SET_NAMES:
        p = STABILITY_DIR / f"{model}_set_{s}_axes.npz"
        if p.exists():
            zs = np.load(p)
            if axis in zs.files:
                vecs[s] = np.asarray(zs[axis], dtype=np.float64)
    if len(vecs) < 2:
        return None, None
    raw, fmt = [], []
    for x, y in combinations(sorted(vecs), 2):
        raw.append(float(np.dot(_unit(vecs[x]), _unit(vecs[y]))))
        fmt.append(float(np.dot(_unit(_remove(vecs[x][None], Q)[0]),
                                _unit(_remove(vecs[y][None], Q)[0]))))
    return float(np.mean(raw)), float(np.mean(fmt))


def analyze_model(model):
    hs = AXES_DIR / f"{model}_hidden_states.npz"
    cal = AXES_DIR / f"{model}_calibration.jsonl"
    axf = AXES_DIR / f"{model}_axes.npz"
    if not (hs.exists() and cal.exists() and axf.exists()):
        return None
    z = np.load(hs)
    az = np.load(axf)
    if VC not in az.files:
        return None
    lens, questions, beh = _load_jsonl_meta(cal)
    Q = _format_basis(z)
    if Q is None:
        return None
    vc_dir = _unit(np.asarray(az[VC], dtype=np.float64))

    axes_out = {}
    for a in CONTENT_AXES:
        if a not in az.files:
            continue
        D, dlen = _diffs_and_dlen(z, lens, questions, a)
        if D is None:
            print(f"  {model}/{a}: pos/neg pairing broken, skipped", flush=True)
            continue
        axis_dir = _unit(D.mean(0))
        R = D - D.mean(0)
        S, Vt = np.linalg.svd(R, full_matrices=False)[1:]
        pc1 = Vt[0]
        # val paired accuracy, raw and format-removed (axis refit on
        # format-removed train diffs)
        pv = z[f"{a}_val_pos_aggregated"].astype(np.float64)
        nv = z[f"{a}_val_neg_aggregated"].astype(np.float64)
        Dtr = (z[f"{a}_train_pos_aggregated"].astype(np.float64)
               - z[f"{a}_train_neg_aggregated"].astype(np.float64))
        axis_tr = _unit(Dtr.mean(0))
        acc_raw = float(np.mean((pv @ axis_tr) > (nv @ axis_tr)))
        axis_fmt = _unit(_remove(Dtr, Q).mean(0))
        acc_fmt = float(np.mean((_remove(pv, Q) @ axis_fmt)
                                > (_remove(nv, Q) @ axis_fmt)))
        cs_raw, cs_fmt = _cross_set_cos(model, a, Q)
        bp, bn = beh[(a, "positive")], beh[(a, "negative")]
        axes_out[a] = {
            "cos_vc": float(np.dot(_unit(np.asarray(az[a], dtype=np.float64)),
                                   vc_dir)),
            "r_proj_dlen": _corr(D @ axis_dir, dlen),
            "resid_pc1_share": float(S[0] ** 2 / np.sum(S ** 2)),
            "r_pc1_dlen": _corr(R @ pc1, dlen),
            "val_pairacc_raw": acc_raw,
            "val_pairacc_fmt": acc_fmt,
            "cross_set_cos_raw": cs_raw,
            "cross_set_cos_fmt": cs_fmt,
            "mean_tokens_pos": bp["tok"] / max(bp["n"], 1),
            "mean_tokens_neg": bn["tok"] / max(bn["n"], 1),
            "md_share_pos": bp["md"] / max(bp["n"], 1),
            "md_share_neg": bn["md"] / max(bn["n"], 1),
            "n_refusals": bp["refusal"] + bn["refusal"],
        }
    if not axes_out:
        return None
    worst = max(axes_out, key=lambda a: abs(axes_out[a]["cos_vc"]))
    return {
        "axes": axes_out,
        "max_abs_cos_vc": abs(axes_out[worst]["cos_vc"]),
        "max_abs_cos_vc_axis": worst,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None,
                    help="comma-separated; default: every model with data")
    args = ap.parse_args()
    if args.models:
        models = args.models.split(",")
    else:
        models = sorted(p.name.replace("_hidden_states.npz", "")
                        for p in AXES_DIR.glob("*_hidden_states.npz"))
    results = {}
    for m in models:
        r = analyze_model(m)
        if r is None:
            continue
        results[m] = r
        print(f"{m:28} max|cos_vc| {r['max_abs_cos_vc']:.2f} "
              f"({r['max_abs_cos_vc_axis']})", flush=True)

    flagged = {m: r["max_abs_cos_vc"] for m, r in results.items()
               if r["max_abs_cos_vc"] >= 0.4}
    out = {
        "_meta": {
            "script": "scripts/06k_format_confound.py",
            "timestamp": datetime.now().isoformat(),
            "format_subspace": f"VC mean diff + top-{FORMAT_PCS} PCs of "
                               "VC diff variation (per model)",
            "flag_threshold_cos_vc": 0.4,
        },
        "summary": {
            "n_models": len(results),
            "flagged_models": dict(sorted(flagged.items(),
                                          key=lambda kv: -kv[1])),
        },
        "models": results,
    }
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT_FILE, "w"), indent=1)
    print(f"\nFlagged (max|cos_vc| >= 0.4): {len(flagged)}/{len(results)}: "
          f"{sorted(flagged, key=flagged.get, reverse=True)}", flush=True)
    print(f"Saved {OUT_FILE}", flush=True)


if __name__ == "__main__":
    main()
