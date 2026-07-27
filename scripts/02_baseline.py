#!/usr/bin/env python3
"""Step 02: Collect baseline temperament measurements.

Produces: data/baselines/<model>.json
Depends on: 01_calibrate (needs axes NPZ)

Usage:
    python scripts/02_baseline.py --model qwen_7b
    python scripts/02_baseline.py --model qwen_7b --check
"""

import argparse
import json
import sys
import gc
import time
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parent.parent))  # repo root for framework imports

# Optional integration with the private `fw` framework. When it is not
# installed (public/standalone mode), fall back to a minimal emitter that
# keeps the #PROGRESS stdout protocol working.
try:
    from framework.progress import log_progress
except ImportError:
    import json as _progress_json

    def log_progress(done, total, elapsed_s, n_elapsed=None, extra=None):
        """Standalone fallback: emit #PROGRESS lines to stdout."""
        n = n_elapsed if n_elapsed is not None else done
        rate = (elapsed_s / n) if (n and n > 0 and elapsed_s > 0) else None
        eta = rate * (total - done) if rate is not None else None
        data = {"done": done, "total": total,
                "elapsed_s": round(elapsed_s, 1),
                "rate_s": round(rate, 1) if rate is not None else None,
                "eta_s": round(eta, 1) if eta is not None else None}
        if extra:
            data.update(extra)
        print(f"#PROGRESS {_progress_json.dumps(data)}", flush=True)

from config.settings import MOOD_AXES, AXES_DIR, MAX_NEW_TOKENS, TEMPERATURE, TOP_P, DO_SAMPLE
from config.models import MODELS
from config.prompts import BASELINE_QUESTIONS
from src.calibration.axis_computer import load_axis_vectors
from src.backend import add_backend_args, resolve_backend, run_on_vast

logger = logging.getLogger(__name__)
OUTPUT_DIR = PROJECT_ROOT / "data" / "baselines"


def preflight_check(model_key: str):
    assert model_key in MODELS, f"Unknown model: {model_key}"
    axes_file = AXES_DIR / f"{model_key}_axes.npz"
    assert axes_file.exists(), f"Missing axes: {axes_file}. Run 01_calibrate first."
    output_file = OUTPUT_DIR / f"{model_key}.json"
    print(f"Model: {MODELS[model_key].model_id}", flush=True)
    print(f"Axes file: {axes_file}", flush=True)
    print(f"Questions: {len(BASELINE_QUESTIONS)}", flush=True)
    print(f"Output: {output_file} (exists: {output_file.exists()})", flush=True)
    print("Preflight OK", flush=True)


def _detect_backend() -> str:
    try:
        import mlx.core
        return "mlx"
    except ImportError:
        return "pytorch"


def _load_generate_fn(model_config, backend: str):
    """Load model and return (generate_fn, cleanup_fn)."""
    template_kwargs = model_config.chat_template_kwargs

    if backend == "mlx":
        from src.model.mlx_inference import load_mlx_model, get_full_result_for_prompt as mlx_get_result
        print(f"Loading model: {model_config.model_id} (MLX)", flush=True)
        ctx = load_mlx_model(model_config.model_id)

        def generate_fn(messages):
            return mlx_get_result(ctx, messages, max_new_tokens=MAX_NEW_TOKENS,
                                  chat_template_kwargs=template_kwargs, save_extra=False)

        def cleanup_fn():
            del ctx["model"]
            gc.collect()

        return generate_fn, cleanup_fn
    else:
        from src.model.loader import load_model
        from src.model.inference import get_full_result_for_prompt
        import torch

        print(f"Loading model: {model_config.model_id} (PyTorch)", flush=True)
        model, tokenizer = load_model(model_config.model_id)

        def generate_fn(messages):
            return get_full_result_for_prompt(
                model, tokenizer, messages, max_new_tokens=MAX_NEW_TOKENS,
                chat_template_kwargs=template_kwargs, save_extra=False)

        def cleanup_fn():
            nonlocal model, tokenizer
            del model, tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return generate_fn, cleanup_fn


def collect_baseline(model_key: str, force: bool = False, backend: str = "pytorch") -> dict:
    output_file = OUTPUT_DIR / f"{model_key}.json"
    if output_file.exists() and not force:
        print(f"Output exists: {output_file}. Use --force to recompute.", flush=True)
        return json.load(open(output_file))

    model_config = MODELS[model_key]
    axes_file = AXES_DIR / f"{model_key}_axes.npz"
    axis_vectors = load_axis_vectors(axes_file)

    npz_data = np.load(axes_file)
    scales = {}
    centers = {}
    for axis in MOOD_AXES:
        scale_key = f"{axis}_scale"
        scales[axis] = float(npz_data[scale_key]) if scale_key in npz_data else 2.0
        center_key = f"{axis}_center"
        centers[axis] = float(npz_data[center_key]) if center_key in npz_data else 0.0

    generate_fn, cleanup_fn = _load_generate_fn(model_config, backend)

    measurements = {axis: [] for axis in MOOD_AXES}
    responses = []

    # Incremental: partial file for resume
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    partial_file = OUTPUT_DIR / f"{model_key}_partial.json"
    done_questions = set()
    if partial_file.exists() and not force:
        partial = json.load(open(partial_file))
        responses = partial.get("responses", [])
        done_questions = {r["question"] for r in responses}
        for axis in MOOD_AXES:
            ax = partial.get("axes", {}).get(axis, {})
            measurements[axis] = ax.get("values", [])
        print(f"Resuming: {len(done_questions)}/{len(BASELINE_QUESTIONS)} done", flush=True)

    _t_start = time.monotonic()
    for i, question in enumerate(BASELINE_QUESTIONS):
        if question in done_questions:
            continue

        messages = [{"role": "user", "content": question}]
        result = generate_fn(messages)

        responses.append({
            "question": question,
            "response": result.text,
            "n_tokens": result.n_generated_tokens,
            "n_words": result.n_words,
            "generation_time_s": result.generation_time_s,
            "truncated": result.n_generated_tokens >= MAX_NEW_TOKENS - 1,
        })

        for axis in MOOD_AXES:
            if axis not in axis_vectors:
                continue
            raw_value = float(np.dot(result.hidden_state, axis_vectors[axis]))
            raw_value = np.clip((raw_value - centers.get(axis, 0.0)) / scales.get(axis, 2.0), -1.0, 1.0)
            measurements[axis].append(float(raw_value))

        # Save partial after EVERY question
        partial_results = {
            "_meta": {
                "script": "scripts/02_baseline.py",
                "backend": backend,
                "model": model_key,
                "model_id": model_config.model_id,
                "timestamp": datetime.now().isoformat(),
                "n_questions": len(BASELINE_QUESTIONS),
                "max_new_tokens": MAX_NEW_TOKENS,
                "generation_settings": {
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "do_sample": DO_SAMPLE,
                },
            },
            "axes": {},
            "responses": responses,
        }
        for axis in MOOD_AXES:
            values = measurements[axis]
            if values:
                partial_results["axes"][axis] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "values": values,
                }
        with open(partial_file, "w") as f:
            json.dump(partial_results, f, indent=2)

        log_progress(len(responses), len(BASELINE_QUESTIONS),
                     time.monotonic() - _t_start, n_elapsed=len(responses) - len(done_questions))
        print(f"  [{len(responses)}/{len(BASELINE_QUESTIONS)}] {question[:50]}... saved", flush=True)

    # Final save to output file
    results = partial_results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    partial_file.unlink(missing_ok=True)
    print(f"Saved to {output_file}", flush=True)

    cleanup_fn()
    return results


def main():
    parser = argparse.ArgumentParser(description="Step 02: Collect baseline temperament")
    parser.add_argument("--model", required=True, help="Model key (e.g., qwen_7b)")
    parser.add_argument("--force", action="store_true", help="Recompute even if output exists")
    parser.add_argument("--check", action="store_true", help="Preflight check only")
    add_backend_args(parser)
    args = parser.parse_args()

    if args.check:
        preflight_check(args.model)
        return

    backend = resolve_backend(args.backend)
    print(f"Backend: {backend}", flush=True)

    if backend == "vast":
        extra_parts = ["--force"] if args.force else []
        run_on_vast(
            __file__, args.model, gpu=args.gpu,
            extra_args=" ".join(extra_parts),
            final_paths=[f"data/baselines/{args.model}.json"],
        )
        return

    results = collect_baseline(args.model, force=args.force, backend=backend)

    print(f"\n=== {args.model} Baseline ===", flush=True)
    for axis, stats in results["axes"].items():
        print(f"  {axis}: {stats['mean']:.3f} (+/-{stats['std']:.3f})", flush=True)


if __name__ == "__main__":
    main()
