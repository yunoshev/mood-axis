#!/usr/bin/env python3
"""Merge per-axis steering_basic shards into the canonical <model>_basic.json.

Shards are produced by steering_basic.py --axes <axis> --out-suffix _ax_<axis>
(one file per axis, so shards of one model can run on different nodes without
clobbering each other). This script unions their "axes" dicts, optionally
seeded with completed axes from a rescued monolith partial (--base).

Usage:
    python3 scripts/merge_steering_parts.py --model gemma3_12b \
        [--base data/steering/gemma3_12b_basic_partial.rescued.json]
"""

import argparse
import json
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "steering"
EXPECTED_AXES = ["warm_cold", "confident_cautious", "empathetic_analytical",
                 "formal_casual", "verbose_concise"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", default=None,
                    help="Rescued monolith partial whose completed axes seed the merge")
    ap.add_argument("--out", default=None, help="Output path (default data/steering/<m>_basic.json)")
    args = ap.parse_args()
    m = args.model

    axes, sources = {}, []
    if args.base:
        raw = json.load(open(args.base))
        base_axes = raw.get("axes") or {}
        for k, v in base_axes.items():
            if not k.startswith("_wip_"):
                axes[k] = v
        sources.append(f"base:{Path(args.base).name} ({sorted(base_axes)})")

    meta, config = None, None
    parts = sorted(p for p in DATA.glob(f"{m}_basic_ax_*.json")
                   if not p.name.endswith("_partial.json"))
    for p in parts:
        d = json.load(open(p))
        for k, v in d.get("axes", {}).items():
            axes[k] = v
        meta, config = d.get("_meta"), d.get("config")
        sources.append(p.name)

    missing = [a for a in EXPECTED_AXES if a not in axes]
    if meta is None:
        sys.exit(f"No shard files found: {DATA}/{m}_basic_ax_*.json")
    if missing:
        sys.exit(f"Incomplete: missing axes {missing} (have {sorted(axes)})")

    final = {
        "_meta": {**meta, "status": "merged", "merged_from": sources},
        "config": config,
        "axes": {a: axes[a] for a in EXPECTED_AXES},
    }
    out = Path(args.out) if args.out else DATA / f"{m}_basic.json"
    with open(out, "w") as f:
        json.dump(final, f, indent=2)
    n_texts = sum(
        sum(len(r) for r in (v.get("results") or {}).values())
        for v in final["axes"].values()
    )
    print(f"Merged {len(parts)} shards (+base: {bool(args.base)}) -> {out} "
          f"({len(final['axes'])} axes, {n_texts} texts)")


if __name__ == "__main__":
    main()
