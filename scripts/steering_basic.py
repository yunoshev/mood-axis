#!/usr/bin/env python3
"""Steering experiment: add axis vectors to hidden states during generation.

Produces: data/steering/<model>_basic.json
Depends on: 01_calibrate (needs axes NPZ)

Two-pass evaluation:
  Pass 1: Generate with steering hook (alpha * axis_vector added at target layer)
  Pass 2: Re-encode steered text, forward WITHOUT hook, project onto all axes

This measures "what the model's own clean representation says about the steered text."

Usage:
    python scripts/steering_basic.py --model qwen_7b
    python scripts/steering_basic.py --model qwen_7b --axes warm_cold,formal_casual
    python scripts/steering_basic.py --model qwen_7b --check
    python scripts/steering_basic.py --model qwen_7b --backend mlx
    python scripts/steering_basic.py --model qwen_7b --backend vast --gpu RTX_4090
"""

import argparse
import json
import os
import sys
import gc
import time
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

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

from config.settings import (MOOD_AXES, AXES_DIR, DATA_DIR, MAX_NEW_TOKENS, STEER_SCALE,
                             STEERING_MAX_NEW_TOKENS, STEERING_MAX_NEW_TOKENS_PER_AXIS)
from config.models import MODELS
from config.prompts import BASELINE_QUESTIONS
from src.calibration.axis_computer import load_axis_vectors
from src.time_budget import continuation_required


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

OUTPUT_DIR = PROJECT_ROOT / "data" / "steering"

DEFAULT_ALPHAS = [-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0]
TARGET_LAYER_RATIO_FALLBACK = 2 / 3
SEED = 42
BOOTSTRAP_N = 1000  # bootstrap resamples for slope CI

# Effective range thresholds
REP_THRESHOLD = 0.3  # 3-gram repetition rate above this = degenerate text


# --- Adaptive target layer ---

def find_peak_layer(model_key: str, axes: list, n_layers: int) -> tuple[int, dict]:
    """Find the layer with highest mean d' across axes using calibration hidden states.

    Returns (peak_layer_idx, per_axis_peak_info).
    Falls back to TARGET_LAYER_RATIO_FALLBACK * n_layers if hidden states unavailable.
    """
    hs_file = AXES_DIR / f"{model_key}_hidden_states.npz"
    if not hs_file.exists():
        fallback = int(n_layers * TARGET_LAYER_RATIO_FALLBACK)
        logger.info(f"No hidden_states.npz for {model_key}, using fallback layer {fallback}")
        return fallback, {}

    hs = np.load(hs_file)
    axis_vectors_file = AXES_DIR / f"{model_key}_axes.npz"
    npz = np.load(axis_vectors_file)

    per_axis_peaks = {}
    layer_scores = np.zeros(n_layers)

    for axis in axes:
        pos_key = f"{axis}_train_pos_layers"
        neg_key = f"{axis}_train_neg_layers"
        if pos_key not in hs or neg_key not in hs:
            continue

        pos_layers = hs[pos_key].astype(np.float32)  # (n_samples, n_layers, hidden_dim)
        neg_layers = hs[neg_key].astype(np.float32)

        axis_vec = npz[axis]  # (hidden_dim,)
        axis_vec_norm = axis_vec / (np.linalg.norm(axis_vec) + 1e-10)

        # Per-layer d': project pos/neg onto axis direction, compute d'
        n_available_layers = min(pos_layers.shape[1], n_layers)
        dprimes = np.zeros(n_available_layers)
        for li in range(n_available_layers):
            pos_proj = pos_layers[:, li, :] @ axis_vec_norm
            neg_proj = neg_layers[:, li, :] @ axis_vec_norm
            n1, n2 = len(pos_proj), len(neg_proj)
            v1 = np.var(pos_proj, ddof=1) if n1 > 1 else 1e-8
            v2 = np.var(neg_proj, ddof=1) if n2 > 1 else 1e-8
            pooled = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
            dprimes[li] = abs(float((np.mean(pos_proj) - np.mean(neg_proj)) / max(pooled, 1e-10)))

        peak_li = int(np.argmax(dprimes))
        per_axis_peaks[axis] = {
            "peak_layer": peak_li,
            "peak_dprime": round(float(dprimes[peak_li]), 2),
            "final_dprime": round(float(dprimes[-1]), 2),
        }
        layer_scores[:n_available_layers] += dprimes

    # Use the layer with highest mean d' across axes
    # Skip early layers (< 25% depth) — high d' there is usually an embedding artifact
    min_layer = max(1, n_layers // 4)
    layer_scores[:min_layer] = 0
    peak_layer = int(np.argmax(layer_scores))

    logger.info(f"Adaptive target layer: {peak_layer}/{n_layers} "
                f"(fallback would be {int(n_layers * TARGET_LAYER_RATIO_FALLBACK)}, "
                f"min_layer={min_layer})")
    for axis, info in per_axis_peaks.items():
        logger.info(f"  {axis}: peak@L{info['peak_layer']} d'={info['peak_dprime']}, "
                     f"final d'={info['final_dprime']}")

    return peak_layer, per_axis_peaks


def bootstrap_slope_ci(alphas: list, per_question_projections: list[list[float]],
                       n_boot: int = BOOTSTRAP_N, ci: float = 0.95) -> dict:
    """Bootstrap CI for dose-response slope.

    Args:
        alphas: list of alpha values
        per_question_projections: list of lists, per_question_projections[i] = projections for alphas[i]
    Returns:
        dict with slope, ci_low, ci_high, se
    """
    rng = np.random.RandomState(SEED)
    alpha_arr = np.array(alphas)
    n_questions = len(per_question_projections[0])

    slopes = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n_questions, size=n_questions)
        means = np.array([np.mean([per_question_projections[ai][j] for j in idx])
                          for ai in range(len(alphas))])
        slopes[b], _ = np.polyfit(alpha_arr, means, 1)

    lo = (1 - ci) / 2
    hi = 1 - lo
    return {
        "slope_mean": round(float(np.mean(slopes)), 6),
        "slope_se": round(float(np.std(slopes)), 6),
        "ci_low": round(float(np.percentile(slopes, lo * 100)), 6),
        "ci_high": round(float(np.percentile(slopes, hi * 100)), 6),
    }


# --- Effective alpha range ---

def _trigram_rep_rate(text: str) -> float:
    """Fraction of repeated 3-grams in text. 0=no repeats, 1=all repeats."""
    words = text.split()
    if len(words) < 4:
        return 1.0
    trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
    return 1.0 - len(set(trigrams)) / len(trigrams)


def compute_effective_range(axis_data: dict, axis_name: str,
                            rep_threshold: float = REP_THRESHOLD) -> dict:
    """Find effective alpha range where steering works without text degradation.

    Two criteria (both must hold):
      1. Text quality: mean 3-gram repetition rate < rep_threshold
      2. Monotonicity: target projection doesn't reverse vs previous alpha

    Returns dict with alpha_min, alpha_max, per-alpha quality metrics.
    """
    alphas = sorted(axis_data["alphas"])
    results = axis_data["results"]
    alpha_means = axis_data["alpha_means"]

    per_alpha = []
    for alpha in alphas:
        texts = [r["steered_text"] for r in results[str(float(alpha))]]
        reps = [_trigram_rep_rate(t) for t in texts]
        mean_rep = float(np.mean(reps))
        target = alpha_means[str(float(alpha))][axis_name]
        per_alpha.append({
            "alpha": alpha,
            "target": target,
            "rep_3gram": round(mean_rep, 4),
            "text_ok": mean_rep < rep_threshold,
        })

    # Find effective range: walk outward from alpha=0
    # Negative direction
    zero_idx = next((i for i, p in enumerate(per_alpha) if p["alpha"] == 0), None)
    if zero_idx is None:
        # No alpha=0, use middle
        zero_idx = len(per_alpha) // 2

    alpha_min = 0.0
    for i in range(zero_idx - 1, -1, -1):
        p = per_alpha[i]
        prev = per_alpha[i + 1]
        # Monotonicity: target should decrease (more negative) as alpha decreases
        monotonic = p["target"] <= prev["target"] + 0.05  # small tolerance
        if p["text_ok"] and monotonic:
            alpha_min = p["alpha"]
        else:
            break

    # Positive direction
    alpha_max = 0.0
    for i in range(zero_idx + 1, len(per_alpha)):
        p = per_alpha[i]
        prev = per_alpha[i - 1]
        # Monotonicity: target should increase as alpha increases
        monotonic = p["target"] >= prev["target"] - 0.05  # small tolerance
        if p["text_ok"] and monotonic:
            alpha_max = p["alpha"]
        else:
            break

    # Effective slope: recompute only within effective range
    effective = [p for p in per_alpha if alpha_min <= p["alpha"] <= alpha_max]
    if len(effective) >= 2:
        a_vals = np.array([p["alpha"] for p in effective])
        t_vals = np.array([p["target"] for p in effective])
        eff_slope, eff_intercept = np.polyfit(a_vals, t_vals, 1)
    else:
        eff_slope, eff_intercept = 0.0, 0.0

    # Total effect: target(alpha_max) - target(alpha_min)
    target_at_min = next((p["target"] for p in per_alpha if p["alpha"] == alpha_min), 0)
    target_at_max = next((p["target"] for p in per_alpha if p["alpha"] == alpha_max), 0)

    return {
        "alpha_min": alpha_min,
        "alpha_max": alpha_max,
        "effective_slope": round(float(eff_slope), 6),
        "total_effect": round(target_at_max - target_at_min, 4),
        "per_alpha_quality": per_alpha,
    }


# --- MLX Steering ---

class _SteeringLayerMLX:
    """Wraps an MLX transformer layer to inject alpha * direction into hidden states."""
    def __init__(self, layer, direction_mx, alpha: float):
        self._layer = layer
        self._dir = direction_mx  # (hidden_dim,) mx.array
        self._alpha = alpha

    def __call__(self, h, *args, **kwargs):
        result = self._layer(h, *args, **kwargs)
        if isinstance(result, tuple):
            return (result[0] + self._alpha * self._dir,) + result[1:]
        return result + self._alpha * self._dir

    def __getattr__(self, name):
        return getattr(self._layer, name)


def _generate_steered_mlx(ctx, messages, alpha, direction_mx, target_layer_idx,
                           max_new_tokens, chat_template_kwargs=None):
    """Pass 1 (MLX): generate with steering vector injected at target layer."""
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler
    from src.model.mlx_inference import build_prompt, check_think_leak

    tokenizer = ctx["tokenizer"]
    layers = ctx["layers"]
    original = layers[target_layer_idx]
    layers[target_layer_idx] = _SteeringLayerMLX(original, direction_mx, alpha)

    try:
        prompt_text = build_prompt(ctx, messages, chat_template_kwargs)
        prompt_tokens = tokenizer.encode(prompt_text)
        prompt_mx = mx.array(prompt_tokens)
        mx.random.seed(SEED)
        sampler = make_sampler(temp=0.7, top_p=0.9)
        eos_id = tokenizer.eos_token_id
        generated_tokens = []
        gen = generate_step(
            prompt_mx, ctx["model"], max_tokens=max_new_tokens, sampler=sampler,
        )
        for token, _logprobs in gen:
            tok_id = token.item() if hasattr(token, "item") else int(token)
            generated_tokens.append(tok_id)
            if tok_id == eos_id or len(generated_tokens) >= max_new_tokens:
                break
        gen.close()
        del gen, prompt_mx
        # Raw decode for the leak check: think tags are special tokens and
        # disappear from the default decode (how the deepseek leak hid).
        raw = tokenizer.decode(generated_tokens, skip_special_tokens=False)
        leak = check_think_leak(raw, chat_template_kwargs)
        if leak:
            raise RuntimeError(
                f"THINK-LEAK ({leak}): no-think generation produced reasoning. "
                f"First 200 chars: {raw[:200]!r}")
        text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        # Strong steering can override the no-think prefill in R1-style models,
        # producing a residual reasoning preamble that closes with </think>
        # BEFORE the real answer (past the 600-char head that check_think_leak
        # scans). The response of a reasoning model is the post-</think> answer,
        # so keep only that — standard R1 extraction, not a length truncation.
        if "</think>" in text:
            answer = text.rsplit("</think>", 1)[1].lstrip()
            if answer:
                text = answer
        return text
    finally:
        layers[target_layer_idx] = original


def _evaluate_text_mlx(ctx, question, response_text, chat_template_kwargs=None):
    """Pass 2 (MLX): encode [question, response], forward without hook → hidden state."""
    from src.model.mlx_inference import (
        build_prompt,
        _forward_extract_hidden,
        _aggregate_decay,
        _resolve_no_think_prefill,
    )

    tokenizer = ctx["tokenizer"]
    prompt_text = build_prompt(
        ctx, [{"role": "user", "content": question}], chat_template_kwargs)
    if _resolve_no_think_prefill(ctx, chat_template_kwargs):
        # Prefill models (gpt_oss/deepseek): mirror generation-time tokens
        # exactly — prompt(with prefill) + response. Re-rendering the
        # assistant turn through the template would drop the prefill from its
        # generation-time position (05b verify: cos 0.80 vs 1.00).
        prompt_ids = tokenizer.encode(prompt_text)
        full_ids = prompt_ids + tokenizer.encode(response_text, add_special_tokens=False)
        prompt_len = min(len(prompt_ids), len(full_ids) - 1)
    else:
        # Bit-identical to the construction used for the 9 clean models.
        full_text = build_prompt(
            ctx,
            [{"role": "user", "content": question},
             {"role": "assistant", "content": response_text}],
            chat_template_kwargs,
        )
        full_ids = tokenizer.encode(full_text)
        prompt_len = min(len(tokenizer.encode(prompt_text)), len(full_ids) - 1)
    _, gen_last_n = _forward_extract_hidden(ctx, full_ids, prompt_len)
    return _aggregate_decay(gen_last_n)


# --- PyTorch Steering hook ---

def _find_layers(model):
    """Find transformer layers module across architectures."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError(
        "Cannot find transformer layers. "
        f"Model type: {type(model).__name__}, attrs: {dir(model)}"
    )


def _make_steering_hook(alpha, direction_tensor):
    """Create a forward hook that adds alpha * direction to hidden states."""
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hs = output[0]
            hs = hs + alpha * direction_tensor
            return (hs,) + output[1:]
        else:
            # Some transformers versions return a single tensor
            return output + alpha * direction_tensor
    return hook_fn


@torch.no_grad()
def _generate_steered(model, tokenizer, messages, alpha, direction_tensor,
                      target_layer_idx, max_new_tokens, chat_template_kwargs=None):
    """Generate text with steering hook active. Returns generated text."""
    from src.model.inference import _apply_chat_template

    layers = _find_layers(model)
    hook = layers[target_layer_idx].register_forward_hook(
        _make_steering_hook(alpha, direction_tensor)
    )

    try:
        input_text = _apply_chat_template(tokenizer, messages, chat_template_kwargs)
        inputs = tokenizer(input_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)
        prompt_length = input_ids.shape[1]

        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(SEED)

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated_ids = outputs[0, prompt_length:]
        return tokenizer.decode(generated_ids, skip_special_tokens=True)
    finally:
        hook.remove()


@torch.no_grad()
def _evaluate_text(model, tokenizer, question, response_text, chat_template_kwargs=None):
    """Pass 2: encode [question, response] and forward WITHOUT hook to get clean projections."""
    from src.model.inference import _apply_chat_template, aggregate_hidden_states

    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": response_text},
    ]
    input_text = _apply_chat_template(tokenizer, messages, chat_template_kwargs)
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )

    # Use response tokens (approximate: last half of sequence)
    response_tokens = tokenizer.encode(response_text, add_special_tokens=False)
    n_response_tokens = len(response_tokens)
    n_response_tokens = min(n_response_tokens, input_ids.shape[1])

    hidden_state = aggregate_hidden_states(
        outputs.hidden_states, num_generated_tokens=n_response_tokens,
    )
    return hidden_state


# --- Main logic ---

def preflight_check(model_key: str, axes: list):
    assert model_key in MODELS, f"Unknown model: {model_key}"
    axes_file = AXES_DIR / f"{model_key}_axes.npz"
    assert axes_file.exists(), f"Missing axes: {axes_file}. Run 01_calibrate first."
    npz = np.load(axes_file)
    available = npz["_axes"].tolist()
    for a in axes:
        assert a in available, f"Axis {a} not in {axes_file} (available: {available})"
    output_file = OUTPUT_DIR / f"{model_key}_basic.json"
    print(f"Model: {MODELS[model_key].model_id}", flush=True)
    print(f"Axes: {axes}", flush=True)
    print(f"Alphas: {DEFAULT_ALPHAS}", flush=True)
    print(f"Questions: {len(BASELINE_QUESTIONS)}", flush=True)
    print(f"Output: {output_file} (exists: {output_file.exists()})", flush=True)
    print("Preflight OK", flush=True)


def run_steering(model_key: str, axes: list, alphas: list,
                 force: bool = False, backend: str = "pytorch",
                 time_budget_min: int = 0, time_budget_exit_code: int = 75,
                 target_layer_override: int = None, out_suffix: str = "",
                 alphas_explicit: bool = False, question_range: tuple = None,
                 max_new_tokens_override: int = None) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"{model_key}_basic{out_suffix}.json"
    partial_file = OUTPUT_DIR / f"{model_key}_basic{out_suffix}_partial.json"

    # Load existing partial results for resume
    existing = {}
    if partial_file.exists():
        raw = json.load(open(partial_file))
        # Merge completed axes + WIP alpha checkpoints
        if "axes" in raw:
            existing = dict(raw["axes"])
        # Preserve _wip_<axis> keys for per-alpha resume
        for k, v in raw.items():
            if k.startswith("_wip_"):
                existing[k] = v
        done = [k for k in existing if not k.startswith("_wip_")]
        wip = [k.replace("_wip_", "") for k in existing if k.startswith("_wip_")]
        logger.info(f"Resuming: done={done}, wip={wip}")

    if output_file.exists() and not force:
        print(f"Output exists: {output_file}. Use --force to recompute.", flush=True)
        return json.load(open(output_file))

    model_config = MODELS[model_key]
    axes_file = AXES_DIR / f"{model_key}_axes.npz"

    # Load axis vectors, scales, and centers
    npz_data = np.load(axes_file)
    axis_vectors = {a: npz_data[a] for a in axes}
    scales = {}
    for a in axes:
        sk = f"{a}_scale"
        scales[a] = float(npz_data[sk]) if sk in npz_data else 2.0

    # All axes for cross-axis projection
    all_axes = npz_data["_axes"].tolist()
    all_vectors = {a: npz_data[a] for a in all_axes}
    all_scales = {}
    all_centers = {}
    for a in all_axes:
        sk = f"{a}_scale"
        all_scales[a] = float(npz_data[sk]) if sk in npz_data else 2.0
        ck = f"{a}_center"
        all_centers[a] = float(npz_data[ck]) if ck in npz_data else 0.0

    # Load model
    if backend == "mlx":
        from src.model.mlx_inference import load_mlx_model
        ctx = load_mlx_model(model_config.model_id, quantize=model_config.quantize,
                             mlx_model_id=model_config.mlx_model_id)
        tokenizer = ctx["tokenizer"]
        n_layers = ctx["n_layers"]
    else:
        from src.model.loader import load_model
        model, tokenizer = load_model(model_config.model_id)
        n_layers = len(_find_layers(model))

    template_kwargs = model_config.chat_template_kwargs

    # Target layer: MUST come from sweep results (steering_layer_sweep.py)
    sweep_file = DATA_DIR / "steering_sweep" / f"{model_key}_sweep.json"
    if not sweep_file.exists():
        raise FileNotFoundError(
            f"Sweep file not found: {sweep_file}\n"
            f"Run: python scripts/steering_layer_sweep.py --model {model_key} --backend {backend}"
        )
    sweep_data = json.load(open(sweep_file))
    target_layer = sweep_data["best_layer"]
    peak_info = {"source": "sweep", "best_effect": sweep_data["best_effect"],
                 "best_layer_pct": sweep_data["best_layer_pct"]}
    if target_layer_override is not None:
        # Mid-band selection rule applied by the operator when the sweep's
        # argmax is a noise-level tie (phase A uses only 5 questions): prefer
        # the best 30-70%-depth layer when its effect is >=80% of the global max.
        sw = next((r for r in sweep_data.get("layers", [])
                   if r["layer"] == target_layer_override), None)
        target_layer = target_layer_override
        peak_info = {"source": "cli_override_midband_rule",
                     "sweep_best_layer": sweep_data["best_layer"],
                     "sweep_best_effect": sweep_data["best_effect"],
                     "override_effect": sw["effect"] if sw else None,
                     "best_layer_pct": sw["layer_pct"] if sw else None}

    # Adaptive alphas from sweep's recommended range (perturbation %).
    # Skipped when the operator passes --alphas explicitly (e.g. a per-axis
    # re-run at an overridden layer whose range comes from that axis's own
    # sweep, not the warm_cold sweep this file reads).
    rec_range = sweep_data.get("recommended_range") or sweep_data.get("safe_alpha_range")
    if alphas_explicit:
        logger.info(f"Using explicit alphas (adaptive override skipped): {alphas}")
        rec_range = None
    if rec_range:
        safe_min, safe_max = rec_range
        safe_max = min(safe_max, 50)
        safe_min = max(safe_min, -50)
        if safe_max > 0:
            alphas = sorted(set(np.linspace(safe_min, safe_max, 7).round(2).tolist() + [0.0]))
        else:
            # Fallback: no safe range found, use conservative alphas
            alphas = [-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0]
        logger.info(f"Adaptive alphas from sweep: {alphas} (range: [{safe_min}, {safe_max}])")
    else:
        logger.info(f"No recommended_range in sweep, using provided alphas: {alphas}")

    logger.info(f"Model: {n_layers} layers, steering at layer {target_layer} "
                f"(from sweep: effect={sweep_data['best_effect']:+.4f}, {sweep_data['best_layer_pct']:.0f}% depth)")

    # Compute hidden state norm at target layer for alpha normalization
    # With STEER_SCALE=0.01: alpha=1 → 1% of h_norm along axis direction
    # Prefer precomputed per-axis/per-layer medians (<m>_hnorm.json, tiny) so
    # worker nodes don't need the multi-hundred-MB hidden_states npz.
    h_norm = None
    hnorm_file = AXES_DIR / f"{model_key}_hnorm.json"
    if hnorm_file.exists():
        hn = json.load(open(hnorm_file))
        ax_map = hn.get(axes[0]) or (next(iter(hn.values())) if hn else None)
        if ax_map and str(target_layer) in ax_map:
            h_norm = float(ax_map[str(target_layer)])
            logger.info(f"h_norm from {hnorm_file.name} (axis={axes[0]}, layer={target_layer})")
    if h_norm is None:
        hs_file = AXES_DIR / f"{model_key}_hidden_states.npz"
        if not hs_file.exists():
            raise FileNotFoundError(
                f"Neither {hnorm_file.name} nor hidden states found: {hs_file}\n"
                f"Run: python scripts/01_calibrate.py --model {model_key}"
            )
        hs = np.load(hs_file, allow_pickle=True)
        norms = []
        for ax in axes[:1]:  # one axis is enough for norm estimate
            key = f"{ax}_train_pos_layers"
            if key in hs:
                states = hs[key]  # (n_samples, n_layers, hidden_dim)
                if len(states.shape) == 3 and target_layer < states.shape[1]:
                    layer_states = states[:, target_layer, :]
                    norms.extend(np.linalg.norm(layer_states.astype(np.float32), axis=1).tolist())
        del hs
        if not norms:
            raise ValueError(
                f"Could not compute h_norm from {hs_file} at layer {target_layer}. "
                f"Hidden states may be in wrong format or target layer out of range."
            )
        h_norm = float(np.median(norms))
    if not np.isfinite(h_norm) or h_norm <= 0:
        raise ValueError(f"Invalid h_norm={h_norm} at layer {target_layer}. Check hidden states.")
    logger.info(f"Alpha normalization: h_norm={h_norm:.1f}, STEER_SCALE={STEER_SCALE} "
                f"(alpha=1 → {STEER_SCALE*100:.1f}% perturbation)")

    # Question sharding: (start, end) subset of BASELINE_QUESTIONS, keeping the
    # ORIGINAL question_idx so shards from different ranges merge cleanly. Lets
    # one axis be split across nodes by question (e.g. 3x70 instead of 1x210).
    if question_range is not None:
        qs, qe = question_range
        q_items = [(i, BASELINE_QUESTIONS[i])
                   for i in range(qs, min(qe, len(BASELINE_QUESTIONS)))]
        logger.info(f"Question shard: idx [{qs}:{qe}) → {len(q_items)} questions")
    else:
        q_items = list(enumerate(BASELINE_QUESTIONS))
    n_questions = len(q_items)
    results_by_axis = {}
    _steer_total = len(axes) * len(alphas) * n_questions
    _steer_done = 0
    _steer_done_at_start = 0  # updated after counting resumed items
    _steer_t_start = time.monotonic()

    def _save_checkpoint(data, path):
        """Atomic save: write to tmp then rename."""
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        tmp.rename(path)

    for axis_idx, axis in enumerate(axes):
        if axis in existing and not force:
            logger.info(f"Axis {axis} already done (resume), skipping")
            # Keep results (per-question texts) across resumes — the site
            # playground and the judge both need them in the final file.
            results_by_axis[axis] = existing[axis]
            _steer_done += len(alphas) * n_questions
            _steer_done_at_start = _steer_done
            continue

        logger.info(f"=== Axis {axis_idx+1}/{len(axes)}: {axis} ===")

        # Per-axis token budget (0%-truncation goal). CLI override wins; else the
        # per-axis dict (verbose_concise runs longest); else the steering default.
        mnt = (max_new_tokens_override
               or STEERING_MAX_NEW_TOKENS_PER_AXIS.get(axis, STEERING_MAX_NEW_TOKENS))
        logger.info(f"  max_new_tokens={mnt} (axis={axis})")

        if backend == "mlx":
            import mlx.core as mx
            # Scale direction: alpha=1 → STEER_SCALE (1%) of hidden state norm
            direction_scaled = axis_vectors[axis] * h_norm * STEER_SCALE
            direction_mx = mx.array(direction_scaled)
        else:
            direction_scaled = axis_vectors[axis] * h_norm * STEER_SCALE
            direction = torch.tensor(
                direction_scaled, dtype=model.dtype, device=model.device
            )

        # Resume: check for per-alpha checkpoint within this axis
        alpha_checkpoint = existing.get(f"_wip_{axis}", {})
        alpha_results = {}
        total_steps = len(alphas) * n_questions
        step = 0

        for alpha in alphas:
            alpha_key = str(alpha)

            # Skip already-completed alphas (resume)
            if alpha_key in alpha_checkpoint:
                alpha_results[alpha_key] = alpha_checkpoint[alpha_key]
                step += n_questions
                _steer_done += n_questions
                _steer_done_at_start = _steer_done
                logger.info(f"  alpha={alpha:+.0f} — resumed from checkpoint [{step}/{total_steps}]")
                continue

            # Resume mid-alpha progress (per-question checkpoint for slow models:
            # gemma-class ~9min/item can't finish a 30-question alpha in one 25-min
            # chunk, so per-alpha-only saves never persist → restart at 0 forever).
            wipq_key = f"_wipq_{axis}_{alpha_key}"
            per_question = existing.get(wipq_key, [])
            _done_qidx = {r["question_idx"] for r in per_question}
            if per_question:
                step += len(per_question)
                _steer_done += len(per_question)
                _steer_done_at_start = _steer_done
                logger.info(f"  alpha={alpha:+.0f} — resumed mid-alpha "
                            f"[{len(per_question)}/{n_questions} q done]")
            for q_idx, question in q_items:
                if q_idx in _done_qidx:
                    continue
                step += 1
                t0 = time.time()
                messages = [{"role": "user", "content": question}]

                # Pass 1: generate with steering
                if backend == "mlx":
                    steered_text = _generate_steered_mlx(
                        ctx, messages, alpha, direction_mx,
                        target_layer, mnt, template_kwargs,
                    )
                else:
                    steered_text = _generate_steered(
                        model, tokenizer, messages, alpha, direction,
                        target_layer, mnt, template_kwargs,
                    )

                # Pass 2: evaluate clean projection
                if backend == "mlx":
                    hidden_state = _evaluate_text_mlx(
                        ctx, question, steered_text, template_kwargs,
                    )
                else:
                    hidden_state = _evaluate_text(
                        model, tokenizer, question, steered_text, template_kwargs,
                    )

                # Project onto ALL axes
                projections = {}
                for a in all_axes:
                    raw = float(np.dot(hidden_state, all_vectors[a]))
                    projections[a] = float(np.clip((raw - all_centers[a]) / all_scales[a], -1.0, 1.0))

                elapsed = time.time() - t0
                per_question.append({
                    "question_idx": q_idx,
                    "question": question,
                    "steered_text": steered_text,
                    "projections": projections,
                    "n_tokens": len(tokenizer.encode(steered_text, add_special_tokens=False)),
                    "time_s": round(elapsed, 1),
                })

                _steer_done += 1
                if step % 5 == 0 or step == total_steps:
                    logger.info(
                        f"  alpha={alpha:+.0f} q={q_idx+1}/{n_questions} "
                        f"({elapsed:.1f}s, {axis}={projections[axis]:+.3f}) "
                        f"[{step}/{total_steps}]"
                    )
                    log_progress(_steer_done, _steer_total,
                                 time.monotonic() - _steer_t_start,
                                 n_elapsed=_steer_done - _steer_done_at_start)

                # Per-question durable checkpoint: persist WIP after EVERY question
                # so slow models resume mid-alpha instead of restarting (the per-alpha
                # save alone never fires for gemma-class within a 25-min chunk).
                existing[wipq_key] = per_question
                _save_checkpoint({
                    "_meta": {"script": "scripts/steering_basic.py", "model": model_key,
                              "model_id": model_config.model_id,
                              "timestamp": datetime.now().isoformat(), "status": "partial"},
                    "axes": results_by_axis,
                    **{k: v for k, v in existing.items()},
                }, partial_file)
                # Clean mid-alpha exit if the time budget is hit (progress is saved).
                if continuation_required(
                    expired=bool(time_budget_min and
                                 (time.monotonic() - _steer_t_start) > time_budget_min * 60),
                    done=_steer_done, total=_steer_total,
                ):
                    logger.info(f"Time budget {time_budget_min} min reached mid-alpha at "
                                f"{_steer_done}/{_steer_total} (alpha={alpha:+.0f}, "
                                f"{len(per_question)}/{n_questions} q); clean exit "
                                f"{time_budget_exit_code}")
                    sys.exit(time_budget_exit_code)

            # Aggregate for this alpha
            mean_proj = {}
            for a in all_axes:
                vals = [r["projections"][a] for r in per_question]
                mean_proj[a] = float(np.mean(vals))

            alpha_results[alpha_key] = {
                "mean_projections": mean_proj,
                "results": per_question,
            }

            # Save per-alpha checkpoint (work-in-progress for this axis)
            existing[f"_wip_{axis}"] = alpha_results
            existing.pop(f"_wipq_{axis}_{alpha_key}", None)  # alpha done → clear mid-alpha WIP
            _save_checkpoint({
                "_meta": {
                    "script": "scripts/steering_basic.py",
                    "model": model_key,
                    "model_id": model_config.model_id,
                    "timestamp": datetime.now().isoformat(),
                    "status": "partial",
                },
                "axes": results_by_axis,
                **{k: v for k, v in existing.items()},
            }, partial_file)
            logger.info(f"  Checkpoint: {axis} alpha={alpha:+.0f} saved [{step}/{total_steps}]")

            # Free MLX metal cache + cyclic Python references after each alpha
            if backend == "mlx":
                import mlx.core as mx
                mx.metal.clear_cache()
            gc.collect()

            # fw2 shard contract: clean exit AFTER the durable per-alpha save,
            # BEFORE the hard timeout (exit 124 does not respool)
            if continuation_required(
                expired=bool(
                    time_budget_min
                    and (time.monotonic() - _steer_t_start) > time_budget_min * 60
                ),
                done=_steer_done,
                total=_steer_total,
            ):
                logger.info(f"Time budget {time_budget_min} min reached at "
                            f"{_steer_done}/{_steer_total}; clean exit "
                            f"{time_budget_exit_code}")
                sys.exit(time_budget_exit_code)

        # Compute dose-response metrics
        alpha_vals = np.array(alphas, dtype=float)
        target_means = np.array([
            alpha_results[str(a)]["mean_projections"][axis] for a in alphas
        ])

        # Linear regression: projection = slope * alpha + intercept
        if len(alpha_vals) > 1:
            slope, intercept = np.polyfit(alpha_vals, target_means, 1)
        else:
            slope, intercept = 0.0, float(target_means[0])

        # Bootstrap CI for slope
        per_q_proj = []
        for a in alphas:
            per_q_proj.append([r["projections"][axis] for r in alpha_results[str(a)]["results"]])
        slope_ci = bootstrap_slope_ci(alphas, per_q_proj)
        logger.info(f"  slope={slope:.4f}, 95% CI=[{slope_ci['ci_low']:.4f}, {slope_ci['ci_high']:.4f}], "
                     f"SE={slope_ci['slope_se']:.4f}")

        # Cross-axis leakage: ratio of other axes' slopes to target slope
        leakage = {}
        for a in all_axes:
            if a == axis:
                continue
            other_means = np.array([
                alpha_results[str(al)]["mean_projections"][a] for al in alphas
            ])
            if len(alpha_vals) > 1:
                other_slope, _ = np.polyfit(alpha_vals, other_means, 1)
            else:
                other_slope = 0.0
            leakage[a] = round(abs(other_slope / slope) if abs(slope) > 1e-6 else 0.0, 4)

        axis_result = {
            "target_layer": target_layer,
            "peak_layer_info": peak_info.get(axis, {}),
            "alphas": alphas,
            "slope": round(float(slope), 6),
            "slope_ci": slope_ci,
            "intercept": round(float(intercept), 4),
            "alpha_means": {
                str(a): alpha_results[str(a)]["mean_projections"] for a in alphas
            },
            "leakage": leakage,
            "results": {
                str(a): alpha_results[str(a)]["results"] for a in alphas
            },
        }

        # Compute effective alpha range (safe — don't crash on failure)
        try:
            eff = compute_effective_range(axis_result, axis)
        except Exception as e:
            logger.warning(f"  compute_effective_range failed: {e}. Use --analyze later.")
            eff = {"alpha_min": alphas[0], "alpha_max": alphas[-1], "total_effect": 0, "effective_slope": 0, "per_alpha_quality": []}
        axis_result["effective_range"] = eff

        results_by_axis[axis] = axis_result

        # Clean up WIP checkpoint for this axis
        existing.pop(f"_wip_{axis}", None)

        # Save partial after each axis (full results on disk)
        _save_checkpoint({
            "_meta": {
                "script": "scripts/steering_basic.py",
                "model": model_key,
                "model_id": model_config.model_id,
                "timestamp": datetime.now().isoformat(),
                "status": "partial",
            },
            "axes": results_by_axis,
            **{k: v for k, v in existing.items() if k.startswith("_wip_")},
        }, partial_file)

        # Note: we keep "results" (with texts) in results_by_axis so they
        # persist in the partial checkpoint. Texts are needed for the
        # steering playground. Memory is freed when model is unloaded.
        eff_str = f"[{eff['alpha_min']}, {eff['alpha_max']}]" if eff.get("total_effect") else "N/A"
        logger.info(f"  {axis}: slope={slope:.4f}, effective={eff_str}, "
                    f"total_effect={eff.get('total_effect', 0):.3f}, leakage_max={max(leakage.values()) if leakage else 0:.3f}")
        logger.info(f"  Saved partial to {partial_file}")

    # Final output — results_by_axis has full data including texts
    full_axes = results_by_axis

    final = {
        "_meta": {
            "script": "scripts/steering_basic.py",
            "model": model_key,
            "model_id": model_config.model_id,
            "timestamp": datetime.now().isoformat(),
            "n_questions": n_questions,
            "n_alphas": len(alphas),
            "alphas": alphas,
            "target_layer": target_layer,
            "target_layer_method": "adaptive_peak_dprime" if peak_info else "fallback_2_3",
            "peak_layer_info": peak_info,
            "n_layers": n_layers,
            "seed": SEED,
            "bootstrap_n": BOOTSTRAP_N,
        },
        "config": {
            "target_layer_ratio_fallback": TARGET_LAYER_RATIO_FALLBACK,
            "max_new_tokens": {a: (max_new_tokens_override
                                   or STEERING_MAX_NEW_TOKENS_PER_AXIS.get(a, STEERING_MAX_NEW_TOKENS))
                               for a in axes},
            "temperature": 0.7,
            "top_p": 0.9,
        },
        "axes": full_axes,
    }

    with open(output_file, "w") as f:
        json.dump(final, f, indent=2)
    logger.info(f"Saved to {output_file}")

    # Clean up partial
    if partial_file.exists():
        partial_file.unlink()

    # Clean up model
    if backend == "mlx":
        del ctx
    else:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return final


def main():
    parser = argparse.ArgumentParser(description="Steering: add axis vectors to hidden states")
    parser.add_argument("--model", required=True, help="Model key (e.g., qwen_7b)")
    parser.add_argument("--axes", default=None,
                        help="Comma-separated axes to steer (default: all MOOD_AXES)")
    parser.add_argument("--alphas", default=None,
                        help="Comma-separated alpha values (default: -5,-2,-1,0,1,2,5)")
    parser.add_argument("--force", action="store_true", help="Recompute even if output exists")
    parser.add_argument("--check", action="store_true", help="Preflight check only")
    parser.add_argument("--analyze", action="store_true",
                        help="Recompute effective_range from existing results (CPU only, no GPU)")
    parser.add_argument("--backend", choices=["pytorch", "mlx", "vast"], default="pytorch",
                        help="Backend: mlx (fast, Apple Silicon), pytorch (CUDA/MPS), vast (cloud GPU)")
    parser.add_argument("--gpu", default="RTX_4090", help="GPU type for vast backend")
    parser.add_argument("--time-budget-min", type=int, default=0,
                        help="Clean-exit budget for fw2 shards (0 = unlimited)")
    parser.add_argument("--time-budget-exit-code", type=int, default=75)
    parser.add_argument("--target-layer", type=int, default=None,
                        help="Override sweep best_layer (mid-band rule for "
                             "noise-level sweep ties); recorded in output")
    parser.add_argument("--out-suffix", default="",
                        help="Suffix for output/partial filenames (e.g. _ax_warm_cold); "
                             "lets per-axis shards of one model run on different nodes "
                             "without clobbering each other. Merge with "
                             "merge_steering_parts.py")
    parser.add_argument("--question-range", default=None,
                        help="Shard questions: 'START:END' (0-based, END exclusive) "
                             "over BASELINE_QUESTIONS. Original question_idx is kept "
                             "so per-range shards merge cleanly. Split one axis across "
                             "nodes, e.g. 0:10 / 10:20 / 20:30.")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="Override per-axis token budget (STEERING_MAX_NEW_TOKENS*). "
                             "Use to canary a higher cap before re-running the wave.")
    args = parser.parse_args()

    axes = args.axes.split(",") if args.axes else list(MOOD_AXES)
    alphas = [float(x) for x in args.alphas.split(",")] if args.alphas else DEFAULT_ALPHAS
    question_range = None
    if args.question_range:
        _qs, _qe = args.question_range.split(":")
        question_range = (int(_qs), int(_qe))

    if args.check:
        preflight_check(args.model, axes)
        return

    if args.analyze:
        # Recompute effective_range from existing results (no GPU needed)
        output_file = OUTPUT_DIR / f"{args.model}_basic.json"
        if not output_file.exists():
            print(f"No steering data found: {output_file}", flush=True)
            return
        with open(output_file) as f:
            data = json.load(f)
        print(f"\n=== {args.model} Effective Range Analysis ===", flush=True)
        print(f"  {'Axis':<25} {'Range':>12} {'Total':>8} {'Slope':>8}", flush=True)
        print(f"  {'-'*55}", flush=True)
        for axis_name, axis_data in data["axes"].items():
            eff = compute_effective_range(axis_data, axis_name)
            axis_data["effective_range"] = eff
            r = f"[{eff['alpha_min']}, {eff['alpha_max']}]"
            print(f"  {axis_name:<25} {r:>12} {eff['total_effect']:>+8.3f} {eff['effective_slope']:>+8.4f}", flush=True)
            # Per-alpha detail
            for p in eff["per_alpha_quality"]:
                marker = " " if p["text_ok"] else "X"
                in_range = "<<" if eff["alpha_min"] <= p["alpha"] <= eff["alpha_max"] else "  "
                print(f"    {marker} alpha={p['alpha']:+6.1f}: target={p['target']:+.3f} rep={p['rep_3gram']:.3f} {in_range}", flush=True)
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n  Updated {output_file}", flush=True)
        return

    if args.backend == "vast":
        from src.backend import run_on_vast
        extra = ""
        if args.axes:
            extra += f" --axes={args.axes}"
        if args.alphas:
            extra += f" --alphas={args.alphas}"
        if args.force:
            extra += " --force"
        # Upload partial checkpoint for resume
        partial = f"data/steering/{args.model}_basic_partial.json"
        upload = [partial] if (Path(__file__).resolve().parent.parent / partial).exists() else []
        run_on_vast(
            __file__, args.model, args.gpu,
            extra_args=extra,
            sync_paths=["data/steering/"],
            final_paths=["data/steering/"],
            upload_extra=upload,
        )
        return

    results = run_steering(args.model, axes, alphas, force=args.force, backend=args.backend,
                           time_budget_min=args.time_budget_min,
                           time_budget_exit_code=args.time_budget_exit_code,
                           target_layer_override=args.target_layer,
                           out_suffix=args.out_suffix,
                           alphas_explicit=args.alphas is not None,
                           question_range=question_range,
                           max_new_tokens_override=args.max_new_tokens)

    print(f"\n=== {args.model} Steering Results ===", flush=True)
    for axis, data in results["axes"].items():
        slope = data["slope"]
        leakage = data.get("leakage", {})
        if leakage:
            vals = [v if isinstance(v, (int, float)) else v.get("ratio", 0) for v in leakage.values()]
            max_leak = max(vals)
        else:
            max_leak = 0
        eff = data.get("effective_range", {})
        eff_range = f"[{eff.get('alpha_min', '?')}, {eff.get('alpha_max', '?')}]"
        total = eff.get("total_effect", 0)
        print(f"  {axis}: slope={slope:+.4f}, effective={eff_range}, "
              f"total_effect={total:+.3f}, max_leakage={max_leak:.3f}", flush=True)


if __name__ == "__main__":
    main()
