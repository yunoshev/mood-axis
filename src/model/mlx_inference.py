"""MLX inference backend for Mood Axis calibration.

Faster than PyTorch MPS on Apple Silicon. Loads models in native precision
(bf16/fp16) by default — same hidden states as PyTorch, just faster.

Two-pass approach:
1. Generate text autoregressively (mlx_lm.generate.generate_step)
2. Full forward pass on (prompt + generated) to extract hidden states

This is correct because causal attention guarantees identical hidden states
whether computed autoregressively or in a single pass.

Requirements: pip install mlx mlx-lm
"""

import os
import time
import numpy as np
from typing import List, Dict, Optional

from src.model.inference import GenerationResult
from config.settings import (
    HIDDEN_LAYERS_TO_USE,
    TOKEN_WEIGHT_DECAY,
    LAYER_WEIGHTS,
    CALIBRATION_MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_P,
)


# ---------------------------------------------------------------------------
# Model ID resolution
# ---------------------------------------------------------------------------

_MLX_NATIVE_OVERRIDES = {
    # Ministral 3: HF weights use mistral3 (multimodal) config but model is text-only.
    # mlx_lm loads it as multimodal → broken generation (garbage tokens).
    # mlx-community conversion has correct weight mapping.
    "mistralai/Ministral-3-8B-Instruct-2512": "mlx-community/Ministral-3-8B-Instruct-2512",
    # Base: no mlx-community conversion exists (2026-07-17); Aratako TextOnly is
    # the unwrapped text-only repack of the same weights — smoke-test gated.
    "mistralai/Ministral-3-8B-Base-2512": "Aratako/Ministral-3-8B-Base-2512-TextOnly",
    "mistralai/Ministral-3-8B-Reasoning-2512": "mlx-community/Ministral-3-8B-Reasoning-2512-bf16",
    "mistralai/Ministral-3-14B-Instruct-2512": "mlx-community/Ministral-3-14B-Instruct-2512-bf16",
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506": "mlx-community/Mistral-Small-3.2-24B-Instruct-2506-bf16",
    # Gemma 4 26B-A4B: HF weights are multimodal-wrapped; mlx-community bf16 is text-only.
    # mlx_lm gemma4 arch was incomplete in 0.31.3 (2026-04) — MUST pass a load test first.
    "google/gemma-4-26B-A4B-it": "mlx-community/gemma-4-26b-a4b-it-bf16",
    # Gemma 4 E4B: DEFERRED — no working MLX-compatible bf16 text-only conversion.
    # HF weights: language_model.model.layers.* (multimodal wrapper) — mlx_lm fails strict load.
    # mlx-community/gemma-4-e4b-it-bf16: still full multimodal (audio_tower + vision_tower + language_model).
    # principled-intelligence/...-text-only / AlexHung29629/...-text: weights named bare `layers.*`
    #   (no `model.` prefix) — mlx_lm expects `model.layers.*`. Non-standard conversion.
    # See memory/future_models.md for revisit plan.
    # "google/gemma-4-E4B-it": "???",  # uncomment when proper bf16 text-only appears
}


def _resolve_mlx_model_id(model_name: str, quantize: str = None) -> str:
    """Resolve model ID for mlx_lm.load().

    quantize=None: load original HF model in native precision (bf16/fp16).
    quantize="4bit"/"8bit": load mlx-community quantized version.
    """
    if quantize is None:
        # Native precision: use override if HF weights are buggy, else load directly
        return _MLX_NATIVE_OVERRIDES.get(model_name, model_name)

    suffix = {"4bit": "-4bit", "8bit": "-8bit"}.get(quantize)
    if suffix is None:
        raise ValueError(f"Unknown quantize={quantize!r}. Use '4bit' or '8bit'.")

    if model_name.startswith("mlx-community/"):
        return model_name  # already an mlx-community ID

    # org/Model-Name → mlx-community/Model-Name-Nbit
    parts = model_name.split("/")
    if len(parts) == 2:
        return f"mlx-community/{parts[1]}{suffix}"
    return model_name


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _detect_model_structure(model) -> dict:
    """Detect inner model, layers, norm, SWA, embed_scale.

    Handles: Llama, Qwen, Phi, Mistral (model.model.layers)
             Gemma-3 (model.language_model.model.layers)
             EXAONE (model.transformer.h)
             Nemotron-H (model.backbone.layers)
    """
    import mlx.core as mx

    inner = None
    lm_wrapper = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        inner = model.model
    elif (hasattr(model, "model") and hasattr(model.model, "language_model")
          and hasattr(model.model.language_model, "layers")):
        # Gemma 3n: Model.model = Gemma3n wrapper -> .language_model with .layers
        inner = model.model.language_model
    elif hasattr(model, "backbone") and hasattr(model.backbone, "layers"):
        # Nemotron-H: model.backbone -> NemotronHModel with .layers, .norm_f, .embeddings
        inner = model.backbone
    elif hasattr(model, "language_model"):
        lm_wrapper = model.language_model
        if hasattr(lm_wrapper, "model") and hasattr(lm_wrapper.model, "layers"):
            inner = lm_wrapper.model
        elif hasattr(lm_wrapper, "layers"):
            inner = lm_wrapper
    elif hasattr(model, "transformer"):
        # EXAONE: model.transformer with .h (layers), .wte (embed), .ln_f (norm)
        inner = model.transformer
    if inner is None:
        raise RuntimeError("Cannot find inner model structure (expected model.model.layers)")

    layers = getattr(inner, "layers", None) or getattr(inner, "h", None)
    if layers is None:
        raise RuntimeError("Cannot find layers on inner model")
    n_layers = len(layers)
    norm = (getattr(inner, "norm", None) or getattr(inner, "norm_f", None)
            or getattr(inner, "final_layernorm", None) or getattr(inner, "ln_f", None))

    # SWA (Gemma-3 interleaved global/sliding)
    has_swa = hasattr(inner, "swa_idx") and inner.swa_idx is not None
    sliding_window = getattr(inner, "sliding_window", None) or getattr(inner, "window_size", None)
    swa_pattern = getattr(inner, "sliding_window_pattern", 0)

    # Gemma-3 scales embeddings by sqrt(hidden_size)
    embed_scale = None
    hidden_size = getattr(getattr(inner, "args", None), "hidden_size", None)
    if hidden_size:
        import inspect
        try:
            src = inspect.getsource(type(inner).__call__)
            if "hidden_size**0.5" in src or "hidden_size ** 0.5" in src:
                embed_scale = mx.array(hidden_size ** 0.5, mx.bfloat16)
        except (TypeError, OSError):
            pass

    # Detect model_type from inner.args or model.args (Nemotron-H stores args on top-level model, not backbone)
    model_type = getattr(getattr(inner, "args", None), "model_type", "")
    if not model_type:
        model_type = getattr(getattr(model, "args", None), "model_type", "")
    if not model_type:
        model_type = getattr(model, "model_type", "")

    # Hybrid SSM: models with mixed SSM + attention layers.
    # Manual layer-by-layer forward breaks because layers need special masks/cache.
    # Qwen 3.5: interleaved linear_attn + full attention (has ssm_idx/fa_idx)
    # Falcon-H1 / Nemotron-H: parallel Mamba-2 + attention (has mamba layers)
    is_hybrid_ssm = hasattr(inner, "ssm_idx") and hasattr(inner, "fa_idx")
    if not is_hybrid_ssm:
        if model_type in ("falcon_h1", "nemotron_h"):
            is_hybrid_ssm = True
        elif hasattr(getattr(inner, "args", None) or object(), "mamba_n_heads"):
            is_hybrid_ssm = True
    hybrid_fa_idx = getattr(inner, "fa_idx", None)
    hybrid_ssm_idx = getattr(inner, "ssm_idx", None)

    # Gemma 4: shared-KV threading between layers + per-layer inputs + custom
    # masks — manual layer-by-layer forward can't reproduce this. Capture via
    # layer-class hook while running the model's own forward.
    # gemma3n: AltUp threads a (n_streams, B, T, D) stack + per-layer inputs
    # through layers — manual replication impossible, hook the layer class.
    use_layer_hook = (model_type == "gemma4_text" or "gemma4" in model_type
                      or "gemma3n" in model_type)

    # Ministral3 / Llama-4 style: layers need attn_scale argument
    needs_attn_scale = False
    attn_scale_params = None
    args_obj = getattr(inner, "args", None)
    if args_obj is not None:
        rope_params = getattr(args_obj, "rope_parameters", None) or {}
        if "llama_4_scaling_beta" in rope_params:
            needs_attn_scale = True
            attn_scale_params = {
                "beta": rope_params["llama_4_scaling_beta"],
                "max_position_embeddings": rope_params.get(
                    "original_max_position_embeddings",
                    getattr(args_obj, "max_position_embeddings", 32768),
                ),
            }

    return {
        "inner": inner,
        "layers": layers,
        "n_layers": n_layers,
        "norm": norm,
        "model_type": model_type,
        "has_swa": has_swa,
        "sliding_window": sliding_window,
        "sliding_window_pattern": swa_pattern,
        "embed_scale": embed_scale,
        "is_hybrid_ssm": is_hybrid_ssm,
        "use_layer_hook": use_layer_hook,
        "hybrid_fa_idx": hybrid_fa_idx,
        "hybrid_ssm_idx": hybrid_ssm_idx,
        "needs_attn_scale": needs_attn_scale,
        "attn_scale_params": attn_scale_params,
    }


def load_mlx_model(model_name: str, quantize: str = None,
                   mlx_model_id: str = None) -> dict:
    """Load model via MLX. Native precision by default, or quantized if specified.

    Args:
        model_name: HuggingFace model ID (e.g. "Qwen/Qwen2.5-7B-Instruct")
        quantize: None (fp16/bf16), "4bit", or "8bit" — resolves to the
            mlx-community quantized repo unless mlx_model_id is given
        mlx_model_id: explicit MLX repo override (takes precedence over
            automatic resolution; e.g. a specific mlx-community 8-bit repo)

    Returns ctx dict with model, tokenizer, layers, norm, etc.
    ctx['precision'] records the effective precision ("native"/"8bit"/"4bit").
    """
    try:
        from mlx_lm import load
        import mlx.core as mx
        import mlx.utils
    except ImportError:
        raise ImportError(
            "MLX backend requires mlx and mlx-lm. Install with:\n"
            "  pip install mlx mlx-lm"
        )

    if mlx_model_id is None:
        mlx_model_id = _resolve_mlx_model_id(model_name, quantize=quantize)
    precision = quantize or "native"
    q_label = f" ({quantize})" if quantize else ""

    print(f"Loading MLX model {mlx_model_id}{q_label}...", flush=True)
    print("  Loading weights...", flush=True)
    t0 = time.time()
    model, tokenizer = load(
        mlx_model_id,
        tokenizer_config={"trust_remote_code": True},
    )
    load_s = time.time() - t0

    ctx = _detect_model_structure(model)

    # F10 resolved 2026-07-18: gemma3n AltUp handled by the layerhook path
    # (stream-0 capture in _forward_extract_hidden_layerhook)

    ctx["model"] = model
    ctx["tokenizer"] = tokenizer
    ctx["mlx_model_id"] = mlx_model_id
    ctx["hf_model_id"] = model_name
    ctx["precision"] = precision

    # Build EOS token set — some models have multiple EOS tokens (e.g. Gemma 3: [1, 106])
    eos_ids = {tokenizer.eos_token_id} if isinstance(tokenizer.eos_token_id, int) else set(tokenizer.eos_token_id or [])
    try:
        import json as _json
        from huggingface_hub import hf_hub_download
        cfg_path = hf_hub_download(mlx_model_id, "config.json")
        with open(cfg_path) as _f:
            cfg_eos = _json.load(_f).get("eos_token_id")
        if isinstance(cfg_eos, list):
            eos_ids.update(cfg_eos)
        elif isinstance(cfg_eos, int):
            eos_ids.add(cfg_eos)
    except Exception:
        pass

    # Auto-detect <|im_end|> and other chat-template stop tokens not in eos_token_id
    for stop_tok in ["<|im_end|>", "<|eot_id|>", "<end_of_turn>"]:
        try:
            tid = tokenizer.convert_tokens_to_ids(stop_tok)
            if tid is not None and tid != tokenizer.unk_token_id:
                eos_ids.add(tid)
        except Exception:
            pass

    ctx["eos_token_ids"] = eos_ids

    n_params = sum(v.size for _, v in mlx.utils.tree_flatten(model.parameters()))
    ctx["n_params"] = n_params

    print(
        f"  Model loaded: {n_params / 1e9:.2f}B params, "
        f"{ctx['n_layers']}L, MLX ({load_s:.1f}s)",
        flush=True,
    )
    return ctx


# ---------------------------------------------------------------------------
# Chat template
# ---------------------------------------------------------------------------

def _finish_prompt_text(text, no_think_prefill=None):
    """Strip an auto-opened bare "<think>" tag, then splice on the no-think prefill.

    The bare-tag strip alone does NOT reliably force no-think (see
    _resolve_no_think_prefill docstring) -- several templates render it, but
    even when it's absent the model may open its own think block from
    training. `no_think_prefill` (config.models.ModelConfig.no_think_prefill)
    is a pre-"closed" empty think block (or, for Harmony/GPT-OSS, a forced
    channel switch) appended after the generation-prompt tail so the model's
    first generated tokens land past the reasoning segment.
    """
    if text.rstrip().endswith("<think>"):
        text = text.rstrip()[:-len("<think>")].rstrip() + "\n"
    if no_think_prefill:
        text = text + no_think_prefill
    return text


def _resolve_no_think_prefill(ctx, chat_template_kwargs):
    """Look up ModelConfig.no_think_prefill when EXPLICITLY opted in.

    Background (2026-07-23): chat_template_kwargs={"enable_thinking": False}
    is a no-op for several reasoning models -- their chat template has no
    enable_thinking conditional, so think_on/off generations come out
    byte-identical (data/analysis/gptoss_probe_check.json).

    Activation is OPT-IN via chat_template_kwargs={"no_think_prefill": True}
    (currently only scripts/07_thinking.py think_off), NOT via the long-standing
    enable_thinking=False in model configs: every v2 dataset (axes, stability,
    drift) for the no-op models was generated WITH thinking, and silently
    flipping the regime mid-corpus would make resumed runs (e.g. deepseek drift
    33/60) unmixable with their own earlier scenarios. Revisit after Part 1
    data collection is complete.
    ctx["hf_model_id"] (set by load_mlx_model) is the ORIGINAL model_id used
    to look the config up -- NOT ctx["mlx_model_id"], which may be an
    mlx-community override repo not present in the registry.
    """
    if not chat_template_kwargs or chat_template_kwargs.get("no_think_prefill") is not True:
        return None
    from config.models import get_model_by_id
    cfg = get_model_by_id(ctx.get("hf_model_id", ""))
    return cfg.no_think_prefill if cfg is not None else None


def _resolve_no_think_prefill_tok(tokenizer, chat_template_kwargs):
    """Fallback prefill resolution for callers without ctx: registry lookup via
    tokenizer.name_or_path, with reverse lookup through _MLX_NATIVE_OVERRIDES
    for models loaded from an mlx-community override repo."""
    if not chat_template_kwargs or chat_template_kwargs.get("no_think_prefill") is not True:
        return None
    from config.models import get_model_by_id
    name = getattr(tokenizer, "name_or_path", "") or ""
    cfg = get_model_by_id(name)
    if cfg is None:
        for hf_id, mlx_repo in _MLX_NATIVE_OVERRIDES.items():
            if mlx_repo == name:
                cfg = get_model_by_id(hf_id)
                break
    return cfg.no_think_prefill if cfg is not None else None


def _apply_chat_template(tokenizer, messages, chat_template_kwargs=None, no_think_prefill=None):
    """Apply chat template with fallbacks (same logic as PyTorch version).

    Default-safe since 2026-07-27: when the caller does not pass
    no_think_prefill explicitly, it is resolved from the model registry via
    the tokenizer — mirroring src/model/inference.py. Background: steering
    and sweep called this builder (or the raw tokenizer) without the prefill
    argument, so gpt_oss + deepseek steering silently ran in thinking mode.
    """
    if no_think_prefill is None:
        no_think_prefill = _resolve_no_think_prefill_tok(tokenizer, chat_template_kwargs)
    extra = {k: v for k, v in (chat_template_kwargs or {}).items()
                    if k != "no_think_prefill"}  # internal opt-in flag, not a template kwarg

    if not hasattr(tokenizer, "chat_template") or not tokenizer.chat_template:
        # Completion format for base models (standard in representation engineering literature)
        style = next((m["content"] for m in messages if m["role"] == "system"), "")
        question = next((m["content"] for m in messages if m["role"] == "user"), "")
        if style:
            return f"{style}\n\nQ: {question}\nA:"
        return f"Q: {question}\nA:"

    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **extra)
        # Strip thinking mode prompts (GLM-Z1, DeepSeek-R1 etc.), then splice
        # on the no-think prefill (no-op if not given).
        return _finish_prompt_text(text, no_think_prefill)
    except Exception as e:
        if "System role not supported" not in str(e):
            raise
        merged = []
        sys_content = ""
        for msg in messages:
            if msg["role"] == "system":
                sys_content = msg["content"]
            elif msg["role"] == "user" and sys_content:
                merged.append({"role": "user",
                               "content": f"{sys_content}\n\n{msg['content']}"})
                sys_content = ""
            else:
                merged.append(msg)
        text = tokenizer.apply_chat_template(
            merged, tokenize=False, add_generation_prompt=True, **extra)
        return _finish_prompt_text(text, no_think_prefill)


def build_prompt(ctx, messages, chat_template_kwargs=None):
    """The ONLY sanctioned way to build a generation prompt.

    Applies the chat template AND the model's no-think prefill, resolved from
    ModelConfig via ctx["hf_model_id"] (exact, works for mlx-community
    override repos). Never call tokenizer.apply_chat_template directly for
    generation prompts — audit 2026-07-27 found 9 такие call sites, and two
    models' steering ran in thinking mode for a week because of it.
    """
    return _apply_chat_template(
        ctx["tokenizer"], messages, chat_template_kwargs,
        no_think_prefill=_resolve_no_think_prefill(ctx, chat_template_kwargs))


_THINK_LEAK_MARKERS = ("<think>", "</think>", "<|channel|>analysis", "assistantanalysis")


def check_think_leak(raw_text, chat_template_kwargs=None):
    """Return the marker proving a no-think generation leaked reasoning, else None.

    Pass text decoded with skip_special_tokens=False: think tags are special
    tokens and vanish from the default decode — exactly how the deepseek
    steering leak stayed invisible. Fires only when the caller actually
    requested no-think via chat_template_kwargs.
    """
    kw = chat_template_kwargs or {}
    if not (kw.get("enable_thinking") is False or kw.get("no_think_prefill") is True):
        return None
    head = raw_text[:600]
    for m in _THINK_LEAK_MARKERS:
        if m in head:
            return m
    low = head.lstrip().lower()
    if low.startswith(("okay, so i need", "okay, so i'm trying", "alright, so i need")):
        return "r1-monologue-prefix"
    return None


# ---------------------------------------------------------------------------
# Hidden states extraction (second forward pass)
# ---------------------------------------------------------------------------

def _forward_extract_hidden(ctx, token_ids, prompt_len):
    """Extract hidden states via forward pass.

    For standard models: manual layer-by-layer pass (memory efficient).
    For hybrid SSM models (Qwen 3.5): hook-based approach using model's own forward.

    Returns:
        per_layer_last: list of numpy (hidden_dim,) — all layers, last generated token
        gen_last_n: list of numpy (n_gen, hidden_dim) — last N layers, all generated positions
    """
    if ctx.get("is_hybrid_ssm", False):
        return _forward_extract_hidden_hooked(ctx, token_ids, prompt_len)
    if ctx.get("use_layer_hook", False):
        return _forward_extract_hidden_layerhook(ctx, token_ids, prompt_len)
    return _forward_extract_hidden_manual(ctx, token_ids, prompt_len)


def _forward_extract_hidden_hooked(ctx, token_ids, prompt_len):
    """Monkey-patch inner model's __call__ to capture per-layer hidden states.

    For hybrid SSM models where manual layer-by-layer pass doesn't work
    because layers need special masks/cache that only the model knows how to create.
    Covers: Qwen 3.5 (interleaved SSM+attn), Falcon-H1 / Nemotron-H (parallel Mamba-2+attn).
    Same approach as multipass/trajectory_inference_mlx.py.
    """
    import mlx.core as mx

    inner = ctx["inner"]
    n_layers = ctx["n_layers"]
    n_keep = HIDDEN_LAYERS_TO_USE
    cls = inner.__class__

    tokens = mx.array([token_ids])
    captured = []  # list of hidden state tensors after each layer

    original_call = cls.__call__

    model_type = ctx.get("model_type", "")
    is_falcon_h1 = model_type == "falcon_h1"
    is_nemotron_h = model_type == "nemotron_h"
    needs_make_cache = is_falcon_h1  # Nemotron-H uses [None]*n_layers like original code

    def patched_call(self, inputs, cache=None, input_embeddings=None):
        captured.clear()
        if input_embeddings is not None:
            h = input_embeddings
        else:
            # embed_tokens (most models), embeddings (Nemotron-H)
            embed_fn = getattr(self, "embed_tokens", None) or getattr(self, "embeddings", None)
            h = embed_fn(inputs)
        # Row 0: embedding output (pre layer-0), matching HF hidden_states[0]
        captured.append(h)

        if cache is None:
            if needs_make_cache:
                # Falcon-H1 / Nemotron-H: layers need proper cache objects
                outer_model = ctx["model"]
                if hasattr(outer_model, "make_cache"):
                    cache = outer_model.make_cache()
                else:
                    cache = [None] * len(self.layers)
            else:
                cache = [None] * len(self.layers)

        # Create masks the way the model expects
        from mlx_lm.models.base import create_attention_mask
        try:
            from mlx_lm.models.base import create_ssm_mask
            has_ssm_mask = True
        except ImportError:
            has_ssm_mask = False

        if is_falcon_h1:
            # Falcon-H1: cache elements are (mamba_cache, attn_cache) tuples
            mamba_mask = create_ssm_mask(h, cache[0][0])
            attn_mask = create_attention_mask(h, cache[0][1])
            for layer, c in zip(self.layers, cache):
                h = layer(h, cache=c, attn_mask=attn_mask, mamba_mask=mamba_mask)
                captured.append(h)
            return self.final_layernorm(h)

        if is_nemotron_h:
            # Nemotron-H: block_type "M" (Mamba), "*" (attn), "-" (MLP, no cache)
            attn_mask = create_attention_mask(h, cache[self.fa_idx])
            ssm_mask = create_ssm_mask(h, cache[self.ssm_idx])
            cache_counter = 0
            for layer in self.layers:
                if layer.block_type in ("M", "*"):
                    c = cache[cache_counter]
                    cache_counter += 1
                else:
                    c = None
                mask = attn_mask if layer.block_type == "*" else ssm_mask
                h = layer(h, mask=mask, cache=c)
                captured.append(h)
            return self.norm_f(h)

        # Qwen 3.5 / generic hybrid SSM
        fa_mask = create_attention_mask(h, cache[self.fa_idx] if hasattr(self, 'fa_idx') else None)
        if has_ssm_mask and hasattr(self, 'ssm_idx'):
            ssm_mask = create_ssm_mask(h, cache[self.ssm_idx] if hasattr(self, 'ssm_idx') else None)
        else:
            ssm_mask = fa_mask

        for layer, c in zip(self.layers, cache):
            mask = ssm_mask if getattr(layer, "is_linear", False) else fa_mask
            h = layer(h, mask=mask, cache=c)
            captured.append(h)

        return self.norm(h)

    cls.__call__ = patched_call
    try:
        inner(tokens)
    finally:
        cls.__call__ = original_call

    # Extract what we need.
    # captured layout: [embedding, layer_0, ..., layer_{n-1}] → n_layers+1 rows.
    # F2: apply final norm to the TOP layer only (HF hidden_states[-1] is post-norm).
    norm_fn = ctx.get("norm")
    if norm_fn is not None:
        captured[-1] = norm_fn(captured[-1])

    # F1: token window = states that PRODUCED each generated token:
    # last prompt token state + g0..g(n-2). The final generated token's own
    # state (last position) produced nothing kept → excluded.
    # Last-token state = index -2 of the full sequence.
    per_layer_last = []
    gen_last_n = []
    n_rows = len(captured)  # n_layers + 1
    for i in range(n_rows):
        h = captured[i]
        per_layer_last.append(h[0, -2, :])
        if i >= n_rows - n_keep:
            gen_last_n.append(h[0, prompt_len - 1:-1, :])

    mx.eval(*per_layer_last, *gen_last_n)

    per_layer_np = [np.array(v.astype(mx.float32)) for v in per_layer_last]
    gen_np = [np.array(v.astype(mx.float32)) for v in gen_last_n]

    return per_layer_np, gen_np


def _forward_extract_hidden_layerhook(ctx, token_ids, prompt_len):
    """Capture per-layer hidden states by patching the LAYER class __call__
    and running the model's own forward pass unchanged.

    Needed for models whose forward threads extra state between layers
    (Gemma 4: shared KV + per-layer inputs + custom mask schedule) —
    replicating that manually would silently diverge on upstream changes.
    """
    import mlx.core as mx

    inner = ctx["inner"]
    layers = ctx["layers"]
    n_keep = HIDDEN_LAYERS_TO_USE

    tokens = mx.array([token_ids])
    captured = []
    layer_cls = layers[0].__class__
    original_call = layer_cls.__call__

    def _stream0(v):
        # gemma3n AltUp: (n_streams, B, T, D) stack — stream 0 is the residual
        # stream (same convention as the v1 torch path: hidden_states h[0])
        return v[0] if getattr(v, "ndim", 3) == 4 else v

    def patched_call(self, x, *a, **k):
        # Row 0: input to layer 0 = scaled embedding (HF hidden_states[0])
        if not captured:
            captured.append(_stream0(x))
        out = original_call(self, x, *a, **k)
        h = out[0] if isinstance(out, tuple) else out
        captured.append(_stream0(h))
        return out

    layer_cls.__call__ = patched_call
    try:
        inner(tokens)
    finally:
        layer_cls.__call__ = original_call

    if len(captured) != ctx["n_layers"] + 1:
        raise RuntimeError(
            f"layerhook captured {len(captured)} rows, expected {ctx['n_layers'] + 1}")

    # F2: final norm on top layer only (HF hidden_states[-1] is post-norm)
    norm_fn = ctx.get("norm")
    if norm_fn is not None:
        captured[-1] = norm_fn(captured[-1])

    # F1: same token window as other paths (producing positions)
    per_layer_last = []
    gen_last_n = []
    n_rows = len(captured)
    for i in range(n_rows):
        h = captured[i]
        per_layer_last.append(h[0, -2, :])
        if i >= n_rows - n_keep:
            gen_last_n.append(h[0, prompt_len - 1:-1, :])

    mx.eval(*per_layer_last, *gen_last_n)

    per_layer_np = [np.array(v.astype(mx.float32)) for v in per_layer_last]
    gen_np = [np.array(v.astype(mx.float32)) for v in gen_last_n]
    return per_layer_np, gen_np


def _forward_extract_hidden_manual(ctx, token_ids, prompt_len):
    """Manual layer-by-layer forward pass for standard transformer models.

    Memory-efficient: only captures what we need.
    """
    import mlx.core as mx
    from mlx_lm.models.llama import create_attention_mask

    inner = ctx["inner"]
    layers = ctx["layers"]
    n_layers = ctx["n_layers"]

    tokens = mx.array([token_ids])
    embed_fn = getattr(inner, "embed_tokens", None) or getattr(inner, "wte", None)
    h = embed_fn(tokens)

    if ctx.get("embed_scale") is not None:
        h = h * ctx["embed_scale"]

    # Causal mask
    fa_mask = create_attention_mask(h, cache=None, return_array=True)

    # SWA mask for Gemma-3 etc.
    swa_pattern = ctx.get("sliding_window_pattern", 0)
    swa_mask = None
    if swa_pattern > 1:
        swa_mask = create_attention_mask(
            h, cache=None, window_size=ctx.get("sliding_window"), return_array=True)
    elif ctx.get("has_swa", False):
        swa_mask = create_attention_mask(
            h, cache=None, window_size=ctx.get("sliding_window"), return_array=True)

    # Ministral3 / Llama-4: compute attn_scale if needed
    attn_scale = None
    if ctx.get("needs_attn_scale"):
        asp = ctx["attn_scale_params"]
        seq_len = h.shape[1]
        scaling = 1 + asp["beta"] * mx.log(
            1 + mx.floor(mx.arange(seq_len) / asp["max_position_embeddings"])
        )
        attn_scale = scaling[:, None].astype(h.dtype)  # (seq, 1)

    n_keep = HIDDEN_LAYERS_TO_USE  # last N rows for aggregation
    # captured layout mirrors HF hidden_states: [embedding, layer_0, ..., layer_{n-1}]
    # → n_layers+1 rows, last row post-final-norm (applied below).
    captured = [h]

    for i, layer in enumerate(layers):
        if swa_pattern > 1:
            is_global = (i % swa_pattern == swa_pattern - 1)
            mask = fa_mask if is_global else swa_mask
        elif ctx.get("has_swa") and hasattr(layer, "use_sliding") and layer.use_sliding:
            mask = swa_mask
        else:
            mask = fa_mask

        if attn_scale is not None:
            h = layer(h, attn_scale, mask, cache=None)
        else:
            h = layer(h, mask, cache=None)

        # MoE layers (e.g. gemma4_text) return (hidden, aux) tuples
        if isinstance(h, tuple):
            h = h[0]

        captured.append(h)

    # F2: apply final norm to the TOP layer only (HF hidden_states[-1] is post-norm)
    norm_fn = ctx.get("norm")
    if norm_fn is not None:
        captured[-1] = norm_fn(captured[-1])

    # F1: token window = states that PRODUCED each generated token:
    # last prompt token state + g0..g(n-2); last generated token's own state excluded.
    # Last-token state = index -2 of the full sequence.
    per_layer_last = []  # (hidden_dim,) at last-token position, n_layers+1 rows
    gen_last_n = []  # (n_gen, hidden_dim) at producing positions, last N rows
    n_rows = len(captured)
    for i in range(n_rows):
        hs = captured[i]
        per_layer_last.append(hs[0, -2, :])
        if i >= n_rows - n_keep:
            gen_last_n.append(hs[0, prompt_len - 1:-1, :])

    # Single eval for all captured tensors
    mx.eval(*per_layer_last, *gen_last_n)

    # Convert to numpy float32
    per_layer_np = [np.array(v.astype(mx.float32)) for v in per_layer_last]
    gen_np = [np.array(v.astype(mx.float32)) for v in gen_last_n]

    return per_layer_np, gen_np


# ---------------------------------------------------------------------------
# Aggregation (pure numpy — no torch dependency)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# D3: think/answer segmentation of generated tokens
# ---------------------------------------------------------------------------

# (open_marker, close_marker) per model family. Order matters: first match wins.
THINK_SEGMENT_MARKERS = [
    ("<think>", "</think>"),                                    # DeepSeek R1, Qwen3/3.5, GLM-Z1, VibeThinker
    ("<|begin_of_thought|>", "<|end_of_thought|>"),             # OpenThoughts-style
    ("[THINK]", "[/THINK]"),                                    # Ministral 3 Reasoning
    ("<|channel>thought\n", "<channel|>"),                      # Gemma 4
    ("<|channel|>analysis<|message|>", "<|channel|>final<|message|>"),  # GPT-OSS harmony
]


def _char_to_token_index(tokenizer, token_ids, char_pos):
    """Smallest k such that decode(token_ids[:k]) covers char_pos characters."""
    lo, hi = 0, len(token_ids)
    while lo < hi:
        mid = (lo + hi) // 2
        if len(tokenizer.decode(token_ids[:mid])) >= char_pos:
            hi = mid
        else:
            lo = mid + 1
    return lo


def segment_think_tokens(tokenizer, gen_token_ids, gen_text):
    """Split generated tokens into think and answer segments.

    Returns (think_range, answer_start): think_range is (start, end) token
    indices (marker tokens included) or None if no think block; answer_start
    is the token index where the answer segment begins. If the close marker
    is missing (truncated thought), the whole generation is think and
    answer_start == len(gen_token_ids).
    """
    for open_m, close_m in THINK_SEGMENT_MARKERS:
        o = gen_text.find(open_m)
        if o < 0:
            continue
        t_start = _char_to_token_index(tokenizer, gen_token_ids, o + 1) - 1
        t_start = max(t_start, 0)
        c = gen_text.find(close_m, o + len(open_m))
        if c < 0:
            return (t_start, len(gen_token_ids)), len(gen_token_ids)
        end_char = c + len(close_m)
        t_end = _char_to_token_index(tokenizer, gen_token_ids, end_char)
        return (t_start, t_end), t_end
    # R1-style: the chat template opens the think block itself, so the
    # generation contains only the CLOSE marker (e.g. "</think>").
    for open_m, close_m in THINK_SEGMENT_MARKERS:
        c = gen_text.find(close_m)
        if c < 0:
            continue
        end_char = c + len(close_m)
        t_end = _char_to_token_index(tokenizer, gen_token_ids, end_char)
        return (0, t_end), t_end
    return None, 0


def _aggregate_decay(gen_layer_states, weight_decay=TOKEN_WEIGHT_DECAY,
                     layer_weights=None):
    """Aggregate hidden states: decay across tokens, weighted mean across layers.

    Same math as inference.aggregate_hidden_states but in pure numpy.
    """
    if layer_weights is None:
        layer_weights = LAYER_WEIGHTS

    aggregated = []
    for layer_h in gen_layer_states:
        n_tok = layer_h.shape[0]
        if n_tok == 0:
            continue
        weights = np.array([weight_decay ** (n_tok - 1 - i) for i in range(n_tok)])
        weights /= weights.sum()
        weighted = (layer_h * weights[:, None]).sum(axis=0)
        aggregated.append(weighted)

    if not aggregated:
        dim = gen_layer_states[0].shape[-1] if gen_layer_states else 1
        return np.zeros(dim, dtype=np.float32)

    stacked = np.stack(aggregated)
    lw = np.array(layer_weights[:len(aggregated)], dtype=np.float32)
    lw /= lw.sum()
    return (stacked * lw[:, None]).sum(axis=0).astype(np.float32)


def _aggregate_last_token(gen_layer_states, layer_weights=None):
    """Aggregate last generated token across last N layers."""
    if layer_weights is None:
        layer_weights = LAYER_WEIGHTS

    last_vecs = [h[-1, :] for h in gen_layer_states if h.shape[0] > 0]
    if not last_vecs:
        dim = gen_layer_states[0].shape[-1] if gen_layer_states else 1
        return np.zeros(dim, dtype=np.float32)

    stacked = np.stack(last_vecs)
    lw = np.array(layer_weights[:len(last_vecs)], dtype=np.float32)
    lw /= lw.sum()
    return (stacked * lw[:, None]).sum(axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Shared-prefix prompt cache (opt-in: MOOD_PROMPT_CACHE=1)
#
# Within one calibration pole all prompts share the templated system prefix.
# The cache keeps the KV state of the longest common token prefix between
# consecutive prompts; before each generation it is trimmed back to that
# prefix, so only the per-sample suffix is prefilled. Pass 2 (hidden-state
# forward) never uses the cache — extracted states are unaffected.
# ---------------------------------------------------------------------------

def _get_prompt_cache(ctx: dict, model, prompt_tokens: List[int]):
    """Return (cache, n_reused_tokens) or (None, 0) if unsupported (e.g. SSM)."""
    from mlx_lm.models.cache import (
        make_prompt_cache, trim_prompt_cache, can_trim_prompt_cache)

    pc = ctx.get("_prompt_cache")
    if pc is None:
        cache = make_prompt_cache(model)
        if not can_trim_prompt_cache(cache):
            ctx["_prompt_cache_disabled"] = True
            return None, 0
        pc = ctx["_prompt_cache"] = {"cache": cache, "prefix_tokens": []}
    cache = pc["cache"]

    # Longest common prefix with what the cache currently holds; always leave
    # at least one token to feed so generate_step gets a non-empty prompt.
    common = 0
    max_common = len(prompt_tokens) - 1
    for a, b in zip(pc["prefix_tokens"], prompt_tokens):
        if a != b or common >= max_common:
            break
        common += 1

    offset = max(getattr(layer, "offset", 0) for layer in cache)
    if offset > common:
        trim_prompt_cache(cache, offset - common)
    pc["prefix_tokens"] = list(prompt_tokens)
    return cache, common


def warm_prompt_cache(ctx: dict, messages: List[Dict[str, str]],
                      chat_template_kwargs: Optional[dict] = None) -> int:
    """Prefill the prompt cache for `messages` (all tokens but the last).

    Call once per pole before the sample loop so the shared prefix is computed
    outside the first timed generation. No-op unless MOOD_PROMPT_CACHE=1.
    Returns the number of tokens prefilled.
    """
    if os.environ.get("MOOD_PROMPT_CACHE") != "1" or ctx.get("_prompt_cache_disabled"):
        return 0
    import mlx.core as mx
    tokenizer, model = ctx["tokenizer"], ctx["model"]
    prompt_text = build_prompt(ctx, messages, chat_template_kwargs)
    tokens = tokenizer.encode(prompt_text)
    cache, common = _get_prompt_cache(ctx, model, tokens)
    if cache is None:
        return 0
    n_new = len(tokens) - 1 - common
    if n_new > 0:
        model(mx.array(tokens[common:-1])[None], cache=cache)
        mx.eval([layer.state for layer in cache])
    return n_new


# ---------------------------------------------------------------------------
# Main: generate + extract hidden states
# ---------------------------------------------------------------------------

def generate_with_hidden_states(
    ctx: dict,
    messages: List[Dict[str, str]],
    max_new_tokens: int = CALIBRATION_MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    do_sample: bool = True,
    seed: Optional[int] = None,
    chat_template_kwargs: Optional[dict] = None,
    save_extra: bool = True,
    stop_strings: Optional[List[str]] = None,
    segment_think: bool = False,
) -> GenerationResult:
    """Generate text and extract hidden states using MLX.

    Pass 1: Autoregressive generation (produces tokens)
    Pass 2: Full forward on (prompt + generated) for hidden states
    """
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler

    tokenizer = ctx["tokenizer"]
    model = ctx["model"]

    if seed is not None:
        mx.random.seed(seed)

    # 1. Tokenize with chat template
    # no_think_prefill (if resolved) is appended to prompt_text BEFORE
    # tokenization, so it's counted in prompt_len like the rest of the
    # prompt -- excluded from generated_tokens and from the "generated
    # positions" window that hidden-state extraction slices out below
    # (_forward_extract_hidden* all slice from prompt_len - 1 onward).
    prompt_text = build_prompt(ctx, messages, chat_template_kwargs)
    prompt_tokens = tokenizer.encode(prompt_text)
    prompt_len = len(prompt_tokens)

    # 2. Generate tokens autoregressively
    t0 = time.time()
    prompt_mx = mx.array(prompt_tokens)
    generated_tokens = []

    temp = temperature if do_sample else 0.0
    tp = top_p if do_sample else 0.0
    sampler = make_sampler(temp=temp, top_p=tp)

    eos_ids = ctx.get("eos_token_ids", {tokenizer.eos_token_id})
    _stop_strings = stop_strings or []

    step_kwargs = {}
    if os.environ.get("MOOD_PROMPT_CACHE") == "1" and not ctx.get("_prompt_cache_disabled"):
        _cache, _n_reused = _get_prompt_cache(ctx, model, prompt_tokens)
        if _cache is not None:
            step_kwargs["prompt_cache"] = _cache
            prompt_mx = mx.array(prompt_tokens[_n_reused:])

    _repetition_stopped = False
    for token, _logprobs in generate_step(
        prompt_mx, model, max_tokens=max_new_tokens, sampler=sampler,
        **step_kwargs,
    ):
        tok_id = token.item() if hasattr(token, 'item') else int(token)
        generated_tokens.append(tok_id)
        if tok_id in eos_ids or len(generated_tokens) >= max_new_tokens:
            break
        # Detect degenerate repetition (e.g. Falcon-H1 SSM loops in multi-turn)
        n_gen = len(generated_tokens)
        if n_gen >= 128 and n_gen % 32 == 0:
            tail = generated_tokens[-64:]
            for period in range(4, 33):
                pattern = tail[-period:]
                reps = 0
                for off in range(period, len(tail), period):
                    start = len(tail) - off - period
                    if start < 0:
                        break
                    if tail[start:start + period] == pattern:
                        reps += 1
                    else:
                        break
                if reps >= 3:
                    _repetition_stopped = True
                    break
            if _repetition_stopped:
                # Trim to before the repetition started (keep at least 1 token)
                n_trim = period * (reps + 1)
                if n_trim >= len(generated_tokens):
                    import logging
                    logging.getLogger(__name__).warning(
                        f"Repetition trim ({n_trim}) would empty all "
                        f"{len(generated_tokens)} generated tokens — keeping 1 token"
                    )
                    generated_tokens = generated_tokens[:1]
                else:
                    generated_tokens = generated_tokens[:-n_trim]
                break
        if _stop_strings:
            # Check last ~20 chars for stop strings (cheap, avoids full decode each step)
            tail_str = tokenizer.decode(generated_tokens[-6:], skip_special_tokens=True)
            if any(s in tail_str for s in _stop_strings):
                # Trim tokens that are part of / after the stop string
                full = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                for s in _stop_strings:
                    if s in full:
                        cut = full.index(s)
                        generated_tokens = tokenizer.encode(
                            full[:cut], add_special_tokens=False)
                        break
                break

    gen_time = time.time() - t0
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    n_gen = len(generated_tokens)

    if _repetition_stopped:
        import logging
        logging.getLogger(__name__).warning(
            f"Repetition loop detected after {n_gen} tokens — generation stopped early"
        )

    if n_gen == 0:
        raise ValueError("No tokens were generated")

    # 3. Full forward pass for hidden states
    t1 = time.time()
    full_sequence = prompt_tokens + generated_tokens
    per_layer_last, gen_last_n = _forward_extract_hidden(ctx, full_sequence, prompt_len)
    fwd_time = time.time() - t1

    total_time = gen_time + fwd_time

    # 4. Aggregate
    aggregated = _aggregate_decay(gen_last_n)
    last_token = _aggregate_last_token(gen_last_n)

    # 5. Extra data (per_layer_states, token_states)
    per_layer_arr = None
    token_states_arr = None

    if save_extra and per_layer_last:
        # float32: Gemma-class residual streams overflow float16
        # (gemma3_12b: 24/49 layers non-finite in the v2 wave-1 fp16 dumps)
        per_layer_arr = np.stack(per_layer_last).astype(np.float32)

        if gen_last_n:
            stacked = np.stack(gen_last_n)  # (N_layers, n_gen, hidden)
            lw = np.array(LAYER_WEIGHTS[:len(gen_last_n)], dtype=np.float32)
            lw /= lw.sum()
            token_states_arr = (stacked * lw[:, None, None]).sum(axis=0).astype(np.float32)

    # 6. D3: think/answer segment aggregates (opt-in; frozen pipeline unaffected)
    think_state = answer_state = None
    think_state_uniform = answer_state_uniform = None
    think_range = None
    answer_start = 0
    if segment_think and gen_last_n:
        # decode WITH specials: think markers are special tokens in many tokenizers
        seg_text = tokenizer.decode(generated_tokens)
        think_range, answer_start = segment_think_tokens(
            tokenizer, generated_tokens, seg_text)
        # per-token layer-weighted aggregate — feeds trajectories + uniform means
        # (decay aggregates weight the segment tail; uniform is the robustness check)
        if token_states_arr is None:
            stacked = np.stack(gen_last_n)
            lw = np.array(LAYER_WEIGHTS[:len(gen_last_n)], dtype=np.float32)
            lw /= lw.sum()
            token_states_arr = (stacked * lw[:, None, None]).sum(axis=0).astype(np.float32)
        if think_range is not None:
            ts, te = think_range
            think_state = _aggregate_decay([h[ts:te] for h in gen_last_n])
            if te > ts:
                think_state_uniform = token_states_arr[ts:te].mean(axis=0)
        if answer_start < n_gen:
            answer_state = _aggregate_decay([h[answer_start:] for h in gen_last_n])
            answer_state_uniform = token_states_arr[answer_start:].mean(axis=0)

    return GenerationResult(
        text=generated_text,
        hidden_state=aggregated,
        hidden_state_last_token=last_token,
        n_generated_tokens=n_gen,
        n_words=len(generated_text.split()),
        per_layer_states=per_layer_arr,
        token_states=token_states_arr,
        top_k_ids=None,
        top_k_logprobs=None,
        generation_time_s=total_time,
        think_state=think_state,
        answer_state=answer_state,
        think_state_uniform=think_state_uniform,
        answer_state_uniform=answer_state_uniform,
        think_token_range=think_range,
        answer_token_start=answer_start,
    )


# ---------------------------------------------------------------------------
# Convenience wrapper (matching PyTorch API name)
# ---------------------------------------------------------------------------

def get_full_result_for_prompt(
    ctx: dict,
    messages: List[Dict[str, str]],
    max_new_tokens: int = CALIBRATION_MAX_NEW_TOKENS,
    seed: int = 42,
    chat_template_kwargs: Optional[dict] = None,
    save_extra: bool = True,
    stop_strings: Optional[List[str]] = None,
    segment_think: bool = False,
) -> GenerationResult:
    """Convenience function matching PyTorch API."""
    return generate_with_hidden_states(
        ctx=ctx,
        messages=messages,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        seed=seed,
        chat_template_kwargs=chat_template_kwargs,
        save_extra=save_extra,
        stop_strings=stop_strings,
        segment_think=segment_think,
    )
