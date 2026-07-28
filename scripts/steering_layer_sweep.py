#!/usr/bin/env python3
"""Sweep steering injection layer + find safe alpha range per model.

Phase A: 1 axis (warm_cold), 5 questions, alphas (-10, 0, +10).
         Tests 10 evenly spaced layers from 25% to 95% depth.
Phase B: At best layer, calibrate perturbation strength.
         Tests escalating perturbation %s [1,3,5,10,15,20,30,50] with coherence check.

Usage:
    python scripts/steering_layer_sweep.py --model qwen_7b --backend mlx
    python scripts/steering_layer_sweep.py --model all_featured --backend mlx
"""

import argparse
import json
import logging
import os
import time
import sys
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

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
from config.models import get_model_config, MODELS
from config.settings import MOOD_AXES, DATA_DIR, STEER_SCALE


class _AutoFlush:
    """Wraps a file object to flush after every write."""
    def __init__(self, f):
        self._f = f
    def write(self, s):
        self._f.write(s)
        self._f.flush()
    def flush(self):
        self._f.flush()
    def fileno(self):
        return self._f.fileno()


_log_file = os.environ.get("MOOD_LOG_FILE")

if _log_file:
    _log_f = open(_log_file, 'a')
    _writer = _AutoFlush(_log_f)
    sys.stdout = _writer
    sys.stderr = _writer
    _handler = logging.StreamHandler(_writer)
    _handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.basicConfig(level=logging.INFO, handlers=[_handler])
else:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)

AXIS = "warm_cold"
N_QUESTIONS = 5
ALPHAS = [-20.0, 0.0, 20.0]  # ±20% perturbation for layer comparison
N_LAYER_STEPS = 10  # layers from 25% to 95%
MAX_NEW_TOKENS = 384
OUTPUT_DIR = DATA_DIR / "steering_sweep"

# Phase B: perturbation scale calibration
CALIBRATION_PCTS = [1, 3, 5, 10, 15, 20, 30, 50]  # perturbation % of h_norm
REP_THRESHOLD = 0.3  # max 3-gram repetition rate for "safe" text
MIN_TOKENS = 50  # minimum tokens for text to be considered valid

FEATURED = [
    "qwen3.5_9b", "deepseek_r1_14b", "gemma3_12b",
    "phi4", "llama_8b", "gpt_oss_20b",
    "mistral_7b", "qwen_7b", "gemma_9b", "qwen3_8b",
]


def load_questions():
    """Load first N_QUESTIONS from calibration baseline."""
    baseline_file = DATA_DIR / "calibration" / "baseline.json"
    if baseline_file.exists():
        data = json.load(open(baseline_file))
        questions = data.get("questions", data.get("items", []))
        if isinstance(questions, list):
            if isinstance(questions[0], dict):
                return [q["question"] for q in questions[:N_QUESTIONS]]
            return questions[:N_QUESTIONS]
    # Fallback
    return [
        "How do time zones work?",
        "What is RAM in a computer?",
        "How do caves form?",
        "What is the Dewey Decimal System?",
        "How does a compass work?",
    ]


def _trigram_rep_rate(text: str) -> float:
    """Fraction of repeated 3-grams in text."""
    words = text.split()
    if len(words) < 4:
        return 1.0
    trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
    return 1.0 - len(set(trigrams)) / len(trigrams)


def _generate_text_mlx(ctx, model, tokenizer, messages, alpha, direction_mx,
                        target_layer, max_tokens, chat_kwargs):
    """Generate steered text and return (text, n_tokens)."""
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler
    from scripts.steering_basic import _SteeringLayerMLX

    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        from src.model.mlx_inference import build_prompt
        prompt = build_prompt(ctx, messages, chat_kwargs)
    else:
        # Base model fallback: raw question as prompt
        prompt = messages[0]["content"] + "\n"
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)

    inner = getattr(model, "model", getattr(model, "transformer", model))
    layers = getattr(inner, "layers", getattr(inner, "h", None))
    original_layer = layers[target_layer]
    layers[target_layer] = _SteeringLayerMLX(original_layer, direction_mx, alpha)

    prompt_mx = mx.array(prompt_tokens)
    sampler = make_sampler(temp=0.7, top_p=0.9)
    eos_ids = ctx.get("eos_token_ids", {tokenizer.eos_token_id})

    generated = []
    for token, _ in generate_step(prompt_mx, model, max_tokens=max_tokens, sampler=sampler):
        tok_id = token.item() if hasattr(token, 'item') else int(token)
        generated.append(tok_id)
        if tok_id in eos_ids or len(generated) >= max_tokens:
            break

    layers[target_layer] = original_layer
    from src.model.mlx_inference import check_think_leak
    raw = tokenizer.decode(generated, skip_special_tokens=False)
    leak = check_think_leak(raw, chat_kwargs)
    if leak:
        raise RuntimeError(
            f"THINK-LEAK ({leak}): no-think sweep generation produced reasoning. "
            f"First 200 chars: {raw[:200]!r}")
    text = tokenizer.decode(generated)
    mx.metal.clear_cache()
    return text, len(generated)


def calibrate_scale(ctx, model, tokenizer, questions, axis_vector,
                    h_norm, target_layer, chat_kwargs, scale, center, zero_mean,
                    progress_offset=0, progress_total=0, sweep_t0=0):
    """Phase B: find optimal perturbation strength (% of h_norm).

    Tests escalating perturbation percentages at the best layer.
    Each pct value = alpha for the direction_unit (1% of h_norm per unit).
    Returns (max_safe_pct, calibration_results).
    """
    import mlx.core as mx

    # Base direction: 1% of h_norm per unit
    direction_unit = axis_vector.astype(np.float32) * h_norm * 0.01
    direction_mx = mx.array(direction_unit)

    calibration_results = []
    max_safe_pct = 0
    consecutive_unsafe = 0

    logger.info(f"  Phase B: scale calibration at L{target_layer} (zero_mean={zero_mean:+.3f})")

    for pct in CALIBRATION_PCTS:
        texts = []
        projs = []
        for q in questions:
            messages = [{"role": "user", "content": q}]
            # pct as alpha: direction_unit * pct = axis * h_norm * pct/100
            text, n_tok = _generate_text_mlx(
                ctx, model, tokenizer, messages, pct, direction_mx,
                target_layer, MAX_NEW_TOKENS, chat_kwargs
            )
            texts.append(text)

            # Clean forward pass for projection
            from src.model.mlx_inference import _forward_extract_hidden, build_prompt
            if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
                prompt = build_prompt(ctx, messages, chat_kwargs)
            else:
                prompt = messages[0]["content"] + "\n"
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
            gen_tokens = tokenizer.encode(text, add_special_tokens=False)
            full_seq = prompt_tokens + gen_tokens
            per_layer_last, gen_last_n = _forward_extract_hidden(ctx, full_seq, len(prompt_tokens))

            # Same pipeline aggregation + unclipped projection as phase A
            # (see the layer-test loop for why last-token/last-layer + clip
            # zeroed Gemma's effects).
            from src.model.mlx_inference import _aggregate_decay
            if gen_last_n is not None and len(gen_last_n) > 0:
                hs_np = _aggregate_decay(gen_last_n)
            else:
                hs_np = np.zeros_like(axis_vector, dtype=np.float32)

            raw = float(np.dot(np.asarray(hs_np, dtype=np.float32), axis_vector))
            proj = float((raw - center) / scale)
            projs.append(proj)
            mx.metal.clear_cache()

        reps = [_trigram_rep_rate(t) for t in texts]
        mean_rep = float(np.mean(reps))
        mean_toks = float(np.mean([len(t.split()) for t in texts]))
        mean_proj = float(np.mean(projs))
        effect = mean_proj - zero_mean
        safe = mean_rep < REP_THRESHOLD and mean_toks > MIN_TOKENS

        result = {
            "pct": pct,
            "effect": round(effect, 4),
            "mean_proj": round(mean_proj, 4),
            "rep_rate": round(mean_rep, 4),
            "mean_tokens": round(mean_toks, 1),
            "safe": safe,
        }
        calibration_results.append(result)

        status = "SAFE" if safe else "UNSAFE"
        logger.info(f"    {pct:>3d}%: effect={effect:+.4f} rep={mean_rep:.3f} toks={mean_toks:.0f} → {status}")

        # Report progress for fw status
        if progress_total > 0 and sweep_t0 > 0:
            pct_done = progress_offset + len(calibration_results)
            total_elapsed = time.time() - sweep_t0
            log_progress(pct_done, progress_total, total_elapsed)

        if safe:
            max_safe_pct = pct
            consecutive_unsafe = 0
        else:
            consecutive_unsafe += 1
            if consecutive_unsafe >= 2:
                logger.info(f"    Stopping: 2 consecutive unsafe percentages")
                break

    logger.info(f"  Max safe perturbation: {max_safe_pct}%")
    return max_safe_pct, calibration_results


def sweep_model(model_key, backend="mlx"):
    """Run layer sweep + alpha boundary search for one model."""
    model_config = get_model_config(model_key)
    questions = load_questions()

    # Load axis vector
    axes_file = DATA_DIR / "axes" / f"{model_key}_axes.npz"
    if not axes_file.exists():
        logger.error(f"No axes file: {axes_file}")
        return None

    npz = np.load(axes_file, allow_pickle=True)
    if AXIS not in npz:
        logger.error(f"Axis {AXIS} not in {axes_file}")
        return None

    axis_vector = npz[AXIS]
    scale = float(npz.get(f"{AXIS}_scale", 1.0))
    center = float(npz.get(f"{AXIS}_center", 0.0))

    # Load hidden state norms per layer for alpha normalization.
    # Prefer the tiny precomputed <m>_hnorm.json (same medians, all layers) so
    # worker nodes don't need the multi-hundred-MB hidden_states npz.
    hs_file = DATA_DIR / "axes" / f"{model_key}_hidden_states.npz"
    layer_norms = {}
    hnorm_file = DATA_DIR / "axes" / f"{model_key}_hnorm.json"
    if hnorm_file.exists():
        with open(hnorm_file) as f:
            ax_map = json.load(f).get(AXIS) or {}
        layer_norms = {int(k): float(v) for k, v in ax_map.items()
                       if np.isfinite(float(v)) and float(v) > 0}
    if not layer_norms and hs_file.exists():
        try:
            hs = np.load(hs_file, allow_pickle=True)
            key = f"{AXIS}_train_pos_layers"
            if key in hs:
                states = hs[key]  # (n_samples, n_layers, hidden_dim)
                if len(states.shape) == 3:
                    for li in range(states.shape[1]):
                        norms = np.linalg.norm(states[:, li, :].astype(np.float32), axis=1)
                        med = float(np.median(norms))
                        # Filter inf/nan — fallback to 1.0
                        if np.isfinite(med) and med > 0:
                            layer_norms[li] = med
                        else:
                            layer_norms[li] = 1.0
            del hs
        except Exception as e:
            logger.warning(f"Could not load hidden state norms: {e}")

    # Load model
    if backend == "mlx":
        from src.model.mlx_inference import load_mlx_model
        import mlx.core as mx

        ctx = load_mlx_model(model_config.model_id, quantize=model_config.quantize,
                            mlx_model_id=model_config.mlx_model_id)
        model = ctx["model"]
        tokenizer = ctx["tokenizer"]
        n_layers = ctx["n_layers"]
    else:
        raise ValueError("Only mlx backend supported for sweep")

    logger.info(f"Model: {model_config.display_name}, {n_layers} layers, hidden_dim={model_config.hidden_dim}")
    if layer_norms:
        sample_norms = [layer_norms.get(i, 0) for i in sorted(layer_norms.keys())]
        logger.info(f"Hidden state norms: min={min(sample_norms):.0f}, max={max(sample_norms):.0f}")

    # Determine layers to test
    min_layer = max(1, int(n_layers * 0.25))
    max_layer = n_layers - 1
    step = max(1, (max_layer - min_layer) // (N_LAYER_STEPS - 1))
    test_layers = list(range(min_layer, max_layer + 1, step))
    if max_layer not in test_layers:
        test_layers.append(max_layer)

    logger.info(f"Testing {len(test_layers)} layers: {test_layers}")

    # Import steering wrapper
    from scripts.steering_basic import _SteeringLayerMLX
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler

    results = []
    chat_kwargs = model_config.chat_template_kwargs or {}
    sweep_t0 = time.time()

    for layer_idx in test_layers:
        t0 = time.time()
        projections_by_alpha = {}

        # Normalize direction by hidden state norm at this layer
        # perturbation = alpha * axis_vector * h_norm * STEER_SCALE
        # relative perturbation = alpha * STEER_SCALE (e.g., alpha=10 → 10%)
        h_norm = layer_norms.get(layer_idx, 1.0)
        direction_scaled = axis_vector.astype(np.float32) * h_norm * STEER_SCALE
        direction_mx = mx.array(direction_scaled)
        logger.info(f"  L{layer_idx}: h_norm={h_norm:.0f}, perturbation(a=10)={10*STEER_SCALE*100:.1f}%")

        for alpha in ALPHAS:
            projs = []

            for q_idx, question in enumerate(questions):
                # Build prompt
                messages = [{"role": "user", "content": question}]
                if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
                    from src.model.mlx_inference import build_prompt
                    prompt = build_prompt(ctx, messages, chat_kwargs)
                else:
                    prompt = question + "\n"
                prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)

                # Install steering hook
                inner = getattr(model, "model", getattr(model, "transformer", model))
                layers = getattr(inner, "layers", getattr(inner, "h", None))
                original_layer = layers[layer_idx]
                layers[layer_idx] = _SteeringLayerMLX(original_layer, direction_mx, alpha)

                # Generate
                prompt_mx = mx.array(prompt_tokens)
                sampler = make_sampler(temp=0.7, top_p=0.9)
                eos_ids = ctx.get("eos_token_ids", {tokenizer.eos_token_id})

                generated = []
                for token, _ in generate_step(prompt_mx, model, max_tokens=MAX_NEW_TOKENS, sampler=sampler):
                    tok_id = token.item() if hasattr(token, 'item') else int(token)
                    generated.append(tok_id)
                    if tok_id in eos_ids or len(generated) >= MAX_NEW_TOKENS:
                        break

                # Restore original layer
                layers[layer_idx] = original_layer

                # Get hidden state for projection (clean forward pass), using
                # the SAME aggregation as the whole pipeline (decay over tokens,
                # weighted mix of last 4 layers). Measuring last-token/last-layer
                # instead put Gemma's projections 100% into the +-1 clip (its
                # late-layer scale differs wildly from the aggregated one),
                # zeroing every effect.
                from src.model.mlx_inference import (_forward_extract_hidden,
                                                     _aggregate_decay)
                full_seq = prompt_tokens + generated
                per_layer_last, gen_last_n = _forward_extract_hidden(ctx, full_seq, len(prompt_tokens))

                if gen_last_n is not None and len(gen_last_n) > 0:
                    hs_np = _aggregate_decay(gen_last_n)
                else:
                    hs_np = np.zeros(model_config.hidden_dim, dtype=np.float32)

                # Project onto axis — unclipped: the sweep compares effect
                # magnitudes, and clipping silently saturates them.
                raw = float(np.dot(np.asarray(hs_np, dtype=np.float32), axis_vector))
                proj = float((raw - center) / scale)
                projs.append(proj)

                mx.metal.clear_cache()

            projections_by_alpha[alpha] = {
                "mean": float(np.mean(projs)),
                "std": float(np.std(projs)),
                "values": projs,
            }

        elapsed = time.time() - t0

        # Compute effect: mean(alpha=+max) - mean(alpha=-max)
        neg_alpha = min(ALPHAS)
        pos_alpha = max(ALPHAS)
        neg_mean = projections_by_alpha[neg_alpha]["mean"]
        pos_mean = projections_by_alpha[pos_alpha]["mean"]
        zero_mean = projections_by_alpha[0.0]["mean"]
        effect = pos_mean - neg_mean

        layer_result = {
            "layer": layer_idx,
            "layer_pct": round(layer_idx / n_layers * 100, 1),
            "effect": round(effect, 4),
            "neg_mean": round(neg_mean, 4),
            "zero_mean": round(zero_mean, 4),
            "pos_mean": round(pos_mean, 4),
            "time_s": round(elapsed, 1),
        }
        results.append(layer_result)

        layer_done = len(results)
        total_elapsed = time.time() - sweep_t0
        rate = total_elapsed / layer_done if layer_done else 0
        eta = rate * (len(test_layers) - layer_done)

        logger.info(
            f"  L{layer_idx} ({layer_result['layer_pct']:.0f}%): "
            f"effect={effect:+.4f} "
            f"[neg={neg_mean:+.3f}, zero={zero_mean:+.3f}, pos={pos_mean:+.3f}] "
            f"({elapsed:.1f}s)"
        )
        # Progress for fw status
        grand_total = len(test_layers) + len(CALIBRATION_PCTS)
        log_progress(layer_done, grand_total, total_elapsed)

    # Find best layer
    best = max(results, key=lambda r: abs(r["effect"]))
    logger.info(f"  BEST: L{best['layer']} ({best['layer_pct']:.0f}%) effect={best['effect']:+.4f}")

    # Phase B: calibrate perturbation scale at best layer
    best_layer = best["layer"]
    best_h_norm = layer_norms.get(best_layer, 1.0)
    zero_mean = best["zero_mean"]
    grand_total = len(test_layers) + len(CALIBRATION_PCTS)
    max_safe_pct, scale_results = calibrate_scale(
        ctx, model, tokenizer, questions, axis_vector,
        best_h_norm, best_layer, chat_kwargs, scale, center, zero_mean,
        progress_offset=len(test_layers), progress_total=grand_total,
        sweep_t0=sweep_t0,
    )

    # Current default layer
    ax_file = DATA_DIR / "steering" / f"{model_key}_basic.json"
    current_layer = None
    if ax_file.exists():
        sd = json.load(open(ax_file))
        current_layer = sd.get("axes", {}).get(AXIS, {}).get("target_layer")

    output = {
        "model": model_key,
        "model_id": model_config.model_id,
        "n_layers": n_layers,
        "hidden_dim": model_config.hidden_dim,
        "axis": AXIS,
        "n_questions": N_QUESTIONS,
        "alphas": ALPHAS,
        "steer_scale": 0.01,
        "current_target_layer": current_layer,
        "best_layer": best["layer"],
        "best_layer_pct": best["layer_pct"],
        "best_effect": best["effect"],
        "max_safe_pct": max_safe_pct,
        "recommended_range": [-max_safe_pct, max_safe_pct],
        "safe_alpha_range": [-max_safe_pct, max_safe_pct],
        "scale_results": scale_results,
        "timestamp": datetime.now().isoformat(),
        "layers": results,
    }

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "" if AXIS == "warm_cold" else f"_{AXIS}"
    out_file = OUTPUT_DIR / f"{model_key}_sweep{suffix}.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved to {out_file}")

    return output


def main():
    global AXIS
    parser = argparse.ArgumentParser(description="Steering layer sweep")
    parser.add_argument("--model", required=True, help="Model key or 'all_featured'")
    parser.add_argument("--backend", default="mlx", choices=["mlx"])
    parser.add_argument("--axis", default=AXIS,
                        help="Axis to sweep (default warm_cold). Non-default axes "
                             "write <model>_sweep_<axis>.json to keep the main sweep intact.")
    args = parser.parse_args()
    AXIS = args.axis

    if args.model == "all_featured":
        models = FEATURED
    elif args.model == "all":
        models = list(MODELS.keys())
    else:
        models = [args.model]

    all_results = []
    for mk in models:
        if mk not in MODELS:
            logger.warning(f"Unknown model: {mk}, skipping")
            continue
        logger.info(f"\n{'='*60}\n  Sweeping {mk}\n{'='*60}")
        try:
            result = sweep_model(mk, args.backend)
            if result:
                all_results.append(result)
        except Exception as e:
            logger.error(f"Failed {mk}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    if all_results:
        print(f"\n{'='*70}")
        print(f"{'Model':<20} {'Current':>8} {'Best':>8} {'Effect':>8} {'Improvement':>12}")
        print("-" * 70)
        for r in all_results:
            curr = r.get("current_target_layer")
            curr_str = f"L{curr}" if curr is not None else "?"
            curr_eff = 0
            for lr in r["layers"]:
                if lr["layer"] == curr:
                    curr_eff = lr["effect"]
            best_eff = r["best_effect"]
            improvement = abs(best_eff) - abs(curr_eff) if curr_eff else abs(best_eff)
            print(f"{r['model']:<20} {curr_str:<8} L{r['best_layer']:<7} {best_eff:>+8.4f} {improvement:>+11.4f}")


if __name__ == "__main__":
    main()
