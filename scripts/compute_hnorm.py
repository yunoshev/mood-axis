#!/usr/bin/env python3
"""Precompute per-axis/per-layer hidden-state norm medians for steering.

Steering normalizes alpha as a percentage of the typical hidden-state norm at
the target layer (see steering_basic.py). The raw source is the calibration
hidden_states npz (hundreds of MB per model); this script distills it into a
tiny <model>_hnorm.json so downstream steps (steering_basic, layer sweep) can
run without the heavy file. Values are bit-identical to computing directly
from the npz: median over train_pos samples of float32 L2 norms per layer.

Usage:
    python3 scripts/compute_hnorm.py --model qwen3.5_9b
    python3 scripts/compute_hnorm.py --all        # every *_hidden_states.npz
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
AXES_DIR = PROJECT_ROOT / "data" / "axes"
SUFFIX = "_hidden_states.npz"


def compute_model(model_key: str) -> bool:
    src = AXES_DIR / f"{model_key}{SUFFIX}"
    if not src.exists():
        print(f"SKIP {model_key}: {src.name} not found", flush=True)
        return False
    hs = np.load(src, allow_pickle=True)
    out = {}
    for key in hs.files:
        if not key.endswith("_train_pos_layers"):
            continue
        axis = key[: -len("_train_pos_layers")]
        states = hs[key]
        if len(states.shape) != 3:  # (n_samples, n_layers, hidden_dim)
            continue
        norms = np.linalg.norm(states.astype(np.float32), axis=2)
        out[axis] = {str(l): float(np.median(norms[:, l]))
                     for l in range(norms.shape[1])}
    dst = AXES_DIR / f"{model_key}_hnorm.json"
    json.dump(out, open(dst, "w"), indent=1)
    n_layers = len(next(iter(out.values()))) if out else 0
    print(f"OK {model_key}: {len(out)} axes x {n_layers} layers -> {dst.name}", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="model key, e.g. qwen3.5_9b")
    ap.add_argument("--all", action="store_true", help="all models with hidden_states npz")
    args = ap.parse_args()
    if args.all:
        models = sorted(p.name[: -len(SUFFIX)] for p in AXES_DIR.glob(f"*{SUFFIX}"))
    elif args.model:
        models = [args.model]
    else:
        ap.error("pass --model KEY or --all")
    done = sum(compute_model(m) for m in models)
    print(f"done: {done}/{len(models)}", flush=True)
    if done == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
