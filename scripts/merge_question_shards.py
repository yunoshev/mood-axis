#!/usr/bin/env python3
"""Merge per-question-range shards of ONE axis into a single axis result.

steering_basic.py --question-range lets one axis be split across nodes (e.g.
3x10 questions instead of 1x30). Each shard file holds the SAME axis with the
SAME alphas but a disjoint subset of question_idx. This script unions the
per-alpha question lists back together (sorted by question_idx), recomputes the
effective-range analysis over the full set, and writes the combined axis into
the model's <model>_basic_ax_<axis>.json shard (ready for merge_steering_parts).

Usage:
    python3 scripts/merge_question_shards.py --model gemma3_12b --axis verbose_concise \
        --parts _ax_verbose_concise_L18_p1,_ax_verbose_concise_L18_p2,_ax_verbose_concise_L18_p3 \
        --out-suffix _ax_verbose_concise_L18
"""
import argparse
import json
from pathlib import Path

import steering_basic as sb

PROJECT_ROOT = Path(__file__).parent.parent
STEER_DIR = PROJECT_ROOT / "data" / "steering"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--axis", required=True)
    ap.add_argument("--parts", required=True, help="comma-separated out-suffixes")
    ap.add_argument("--out-suffix", required=True, help="suffix for merged shard")
    args = ap.parse_args()

    parts = args.parts.split(",")
    combined_results = {}   # alpha_key -> {question_idx: item}
    axis_meta = None
    for suf in parts:
        p = STEER_DIR / f"{args.model}_basic{suf}.json"
        d = json.load(open(p))
        ax = d["axes"][args.axis]
        if axis_meta is None:
            axis_meta = {k: v for k, v in ax.items() if k != "results"}
        for alpha_key, items in ax["results"].items():
            bucket = combined_results.setdefault(alpha_key, {})
            for it in items:
                bucket[it["question_idx"]] = it
        print(f"  {p.name}: {sum(len(v) for v in ax['results'].values())} texts", flush=True)

    # Rebuild results: per alpha, sorted by question_idx
    results = {ak: [bucket[q] for q in sorted(bucket)]
               for ak, bucket in combined_results.items()}
    n = sum(len(v) for v in results.values())
    per_alpha = {ak: len(v) for ak, v in results.items()}
    print(f"combined: {n} texts, per_alpha={list(per_alpha.values())}", flush=True)

    axis_data = dict(axis_meta)
    axis_data["results"] = results
    axis_data["alphas"] = sorted(float(k) for k in results)
    # Recompute alpha_means over the FULL question set (part1's copy only saw
    # its own question subset). Mean projection per alpha, per axis.
    import numpy as np
    alpha_means = {}
    for ak, items in results.items():
        axes_keys = items[0]["projections"].keys()
        alpha_means[ak] = {a: float(np.mean([it["projections"][a] for it in items]))
                           for a in axes_keys}
    axis_data["alpha_means"] = alpha_means
    # Recompute effective range/slope over the full question set.
    axis_data["effective_range"] = sb.compute_effective_range(axis_data, args.axis)

    out = {"axes": {args.axis: axis_data}}
    dst = STEER_DIR / f"{args.model}_basic{args.out_suffix}.json"
    json.dump(out, open(dst, "w"), indent=1)
    print(f"-> {dst.name} ({len(results)} alphas, {n} texts)", flush=True)


if __name__ == "__main__":
    main()
