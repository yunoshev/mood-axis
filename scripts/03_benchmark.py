#!/usr/bin/env python3
"""Step 03: Run benchmark validation scenarios.

Produces: data/benchmarks/<model>.json
Depends on: 01_calibrate (needs axes NPZ)

Tests if mood metrics change in expected directions under controlled stimuli.
9 scenarios: 3 axes (warm_cold, confident_cautious, verbose_concise) × 3 each.

Usage:
    python scripts/03_benchmark.py --model qwen_7b                   # auto backend
    python scripts/03_benchmark.py --model qwen_7b --backend mlx
    python scripts/03_benchmark.py --model qwen_7b --backend vast    # run on Vast.ai
    python scripts/03_benchmark.py --model qwen_7b --backend modal   # run on Modal
    python scripts/03_benchmark.py --model qwen_7b --check
    python scripts/03_benchmark.py --model qwen_7b --force
"""

import argparse
import json
import sys
import time
import logging
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parent.parent))  # repo root for framework imports

# Optional integration with the private `fw` framework. When it is not
# installed (public/standalone mode), fall back to a minimal emitter that
# keeps the #PROGRESS stdout protocol working.
try:
    from fw.progress import log_progress
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
for handler in logging.root.handlers:
    handler.flush = lambda h=handler: h.stream.flush()
logger = logging.getLogger(__name__)

from config.settings import MOOD_AXES, AXES_DIR, TEMPERATURE, TOP_P, DO_SAMPLE
from config.models import MODELS
from src.backend import add_backend_args, resolve_backend, load_generate_fn, run_on_vast

OUTPUT_DIR = PROJECT_ROOT / "data" / "benchmarks"


def _log(msg):
    logger.info(msg)
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_DIR / "_progress.txt", "w") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
            f.flush()
    except Exception:
        pass


# 9 benchmark scenarios: 3 axes x 3 (system prompt, increase, decrease)
BENCHMARK_SCENARIOS = [
    {
        "name": "System Prompt - Warm",
        "turns": [{"user": "Hello, can you help me?",
                    "system": "You are an extremely warm, friendly, and caring assistant. Show enthusiasm.",
                    "expect": {"warm_cold": "up"}}],
    },
    {
        "name": "Warmth Increase",
        "turns": [
            {"user": "Hi there."},
            {"user": "Please be more friendly and warm! Show some enthusiasm!", "expect": {"warm_cold": "up"}},
        ],
    },
    {
        "name": "Warmth Decrease",
        "turns": [
            {"user": "Hello!"},
            {"user": "Be more formal and professional. No small talk, no emotions.", "expect": {"warm_cold": "down"}},
        ],
    },
    {
        "name": "System Prompt - Confident",
        "turns": [{"user": "What programming language should I learn?",
                    "system": "You are an expert consultant. Always give confident, definitive answers.",
                    "expect": {"confident_cautious": "up"}}],
    },
    {
        "name": "Confidence Increase",
        "turns": [
            {"user": "What's the best way to learn programming?"},
            {"user": "Be more assertive and confident! Clear answer without hedging.",
             "expect": {"confident_cautious": "up"}},
        ],
    },
    {
        "name": "Confidence Decrease",
        "turns": [
            {"user": "What will the stock market do next month?"},
            {"user": "Add lots of caveats. Be very careful. Express uncertainty.",
             "expect": {"confident_cautious": "down"}},
        ],
    },
    {
        "name": "System Prompt - Verbose",
        "turns": [{"user": "What is water?",
                    "system": "You are an extremely detailed assistant. Always provide comprehensive answers.",
                    "expect": {"verbose_concise": "up"}}],
    },
    {
        "name": "Verbose Increase",
        "turns": [
            {"user": "What is machine learning?"},
            {"user": "Give me a very detailed, comprehensive answer with lots of examples!",
             "expect": {"verbose_concise": "up"}},
        ],
    },
    {
        "name": "Verbose Decrease",
        "turns": [
            {"user": "What is 5 + 5?"},
            {"user": "Just give the shortest possible answer. One word. No explanations.",
             "expect": {"verbose_concise": "down"}},
        ],
    },
]


def preflight_check(model_key: str):
    assert model_key in MODELS, f"Unknown model: {model_key}"
    axes_file = AXES_DIR / f"{model_key}_axes.npz"
    assert axes_file.exists(), f"Missing axes: {axes_file}"
    output_file = OUTPUT_DIR / f"{model_key}.json"
    print(f"Model: {MODELS[model_key].model_id}", flush=True)
    print(f"Scenarios: {len(BENCHMARK_SCENARIOS)}", flush=True)
    print(f"Output: {output_file} (exists: {output_file.exists()})", flush=True)
    print("Preflight OK", flush=True)


def run_benchmark(model_key: str, force: bool = False, backend: str = "pytorch") -> dict:
    output_file = OUTPUT_DIR / f"{model_key}.json"

    if output_file.exists() and not force:
        print(f"Output exists: {output_file}. Use --force to recompute.", flush=True)
        with open(output_file) as f:
            return json.load(f)

    model_config = MODELS[model_key]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    from src.mood.projector import MoodProjector
    axes_file = AXES_DIR / f"{model_key}_axes.npz"
    projector = MoodProjector(axes_file=axes_file)

    generate_fn, cleanup_fn = load_generate_fn(model_config, backend, log_fn=_log)

    all_results = {
        "_meta": {
            "script": "scripts/03_benchmark.py",
            "backend": backend,
            "model": model_key,
            "model_id": model_config.model_id,
            "timestamp": datetime.now().isoformat(),
            "max_new_tokens": 384,
            "generation_settings": {
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "do_sample": DO_SAMPLE,
            },
        },
        "scenarios": [],
        "summary": {"total": len(BENCHMARK_SCENARIOS), "passed": 0, "failed": 0},
    }

    t_start = time.monotonic()
    for idx, scenario in enumerate(BENCHMARK_SCENARIOS):
        _log(f"Scenario {idx+1}/{len(BENCHMARK_SCENARIOS)}: {scenario['name']}")
        history = []
        prev_values = {}
        passed = True
        turns_data = []

        for turn_idx, turn in enumerate(scenario["turns"]):
            messages = []
            if "system" in turn:
                messages.append({"role": "system", "content": turn["system"]})
            for msg in history:
                messages.append(msg)
            messages.append({"role": "user", "content": turn["user"]})

            t0 = time.monotonic()
            result = generate_fn(messages, max_new_tokens=384)
            gen_time = time.monotonic() - t0

            reading = projector.project(result.hidden_state)
            turn_data = {
                "user": turn["user"],
                "assistant": result.text,
                "values": reading.values,
                "n_tokens": result.n_generated_tokens,
                "generation_time_s": round(gen_time, 2),
                "truncated": result.n_generated_tokens >= 384 - 1,
            }

            if "expect" in turn:
                checks = {}
                for axis, direction in turn["expect"].items():
                    value = reading.values.get(axis, 0)
                    prev = prev_values.get(axis)
                    if prev is None:
                        ok = (value > 0.1) if direction == "up" else (value < -0.1)
                    else:
                        delta = value - prev
                        ok = (delta > 0) if direction == "up" else (delta < 0)
                    checks[axis] = {"direction": direction, "value": value, "passed": ok}
                    if not ok:
                        passed = False
                turn_data["checks"] = checks

            turns_data.append(turn_data)
            history.append({"role": "user", "content": turn["user"]})
            history.append({"role": "assistant", "content": result.text})
            prev_values = reading.values.copy()

            _log(f"  turn {turn_idx+1}/{len(scenario['turns'])} "
                 f"({gen_time:.1f}s, {result.n_generated_tokens} tok)")

        status = "PASS" if passed else "FAIL"
        _log(f"  {status}: {scenario['name']}")

        all_results["scenarios"].append({
            "name": scenario["name"],
            "passed": passed,
            "turns": turns_data,
        })
        all_results["summary"]["passed" if passed else "failed"] += 1

        # Save partial after EVERY scenario
        partial_file = OUTPUT_DIR / f"{model_key}_partial.json"
        with open(partial_file, "w") as f:
            json.dump(all_results, f, indent=2)

        log_progress(idx + 1, len(BENCHMARK_SCENARIOS), time.monotonic() - t_start)

    elapsed = time.monotonic() - t_start
    _log(f"Benchmark done in {elapsed:.0f}s")

    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    # Clean up partial
    partial_file = OUTPUT_DIR / f"{model_key}_partial.json"
    partial_file.unlink(missing_ok=True)
    _log(f"Saved to {output_file}")

    cleanup_fn()

    print(f"\nBenchmark: {all_results['summary']['passed']}/{all_results['summary']['total']} passed",
          flush=True)
    for s in all_results["scenarios"]:
        status = "PASS" if s["passed"] else "FAIL"
        print(f"  {status}: {s['name']}", flush=True)

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Step 03: Benchmark validation")
    parser.add_argument("--model", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--check", action="store_true")
    add_backend_args(parser)
    args = parser.parse_args()

    if args.check:
        preflight_check(args.model)
        return

    backend = resolve_backend(args.backend)

    if backend == "vast":
        extra = "--force" if args.force else ""
        run_on_vast(__file__, args.model, gpu=args.gpu, extra_args=extra,
                    final_paths=[f"data/benchmarks/{args.model}.json"])
        return

    if backend == "modal":
        print("Modal: use --backend vast or run locally (benchmark is fast, ~5 min)", flush=True)
        sys.exit(1)

    _log(f"Backend: {backend}")
    run_benchmark(args.model, force=args.force, backend=backend)


if __name__ == "__main__":
    main()
