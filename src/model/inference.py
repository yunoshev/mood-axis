"""Inference utilities with hidden states extraction for Mood Axis."""

import logging
import time

import torch
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer

from config.settings import (
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_P,
    DO_SAMPLE,
    CALIBRATION_MAX_NEW_TOKENS,
    HIDDEN_LAYERS_TO_USE,
    TOKEN_WEIGHT_DECAY,
    LAYER_WEIGHTS,
    TOP_K_LOGITS,
)

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result of a generation with hidden states."""
    text: str
    hidden_state: np.ndarray  # Aggregated hidden state (decay aggregation)
    raw_hidden_states: Optional[List[torch.Tensor]] = None
    hidden_state_last_token: Optional[np.ndarray] = None
    n_generated_tokens: int = 0
    n_words: int = 0
    per_layer_states: Optional[np.ndarray] = None  # (num_all_layers, hidden_dim) fp32
    token_states: Optional[np.ndarray] = None  # (n_generated_tokens, hidden_dim) fp16 (fp32 on overflow)
    top_k_ids: Optional[np.ndarray] = None  # (n_generated_tokens, TOP_K_LOGITS) int32
    top_k_logprobs: Optional[np.ndarray] = None  # (n_generated_tokens, TOP_K_LOGITS) float32
    generation_time_s: float = 0.0
    # D3 (thinking): populated only when segment_think=True is requested.
    # Token ranges are indices into the generated-token window (F1 layout).
    think_state: Optional[np.ndarray] = None    # decay-aggregate over think segment
    answer_state: Optional[np.ndarray] = None   # decay-aggregate over answer segment
    think_state_uniform: Optional[np.ndarray] = None   # uniform mean over think segment
    answer_state_uniform: Optional[np.ndarray] = None  # uniform mean over answer segment
    think_token_range: Optional[tuple] = None   # (start, end) or None if no think block
    answer_token_start: int = 0


def _resolve_no_think_prefill(tokenizer, chat_template_kwargs):
    """Look up ModelConfig.no_think_prefill when EXPLICITLY opted in
    (chat_template_kwargs={"no_think_prefill": True} — see mlx_inference.py
    rationale: v2 corpora were generated WITH thinking for no-op models;
    opt-in prevents silently flipping the regime mid-corpus).

    Background (2026-07-23): chat_template_kwargs={"enable_thinking": False}
    is a no-op for several reasoning models -- their chat template has no
    enable_thinking conditional at all, so think_on/off generations come out
    byte-identical (see data/analysis/gptoss_probe_check.json). Mirrors
    src/model/mlx_inference.py's ctx-based lookup, but this (PyTorch) path has
    no ctx dict -- tokenizer.name_or_path is set by
    AutoTokenizer.from_pretrained(model_config.model_id) in
    src/model/loader.py::load_model, so it matches the registry's model_id
    exactly and get_model_by_id() resolves the same ModelConfig transparently
    (no call-site changes needed in 01_calibrate.py / scripts/07_thinking.py).
    """
    if not chat_template_kwargs or chat_template_kwargs.get("no_think_prefill") is not True:
        return None
    from config.models import get_model_by_id
    model_id = getattr(tokenizer, "name_or_path", None)
    cfg = get_model_by_id(model_id) if model_id else None
    return cfg.no_think_prefill if cfg is not None else None


def _finish_prompt_text(text, no_think_prefill=None):
    """Strip an auto-opened bare "<think>" tag, then splice on the no-think prefill.

    See src/model/mlx_inference.py::_finish_prompt_text (same logic, kept
    duplicated since this module has no shared import point with the MLX
    backend).
    """
    if text.rstrip().endswith("<think>"):
        text = text.rstrip()[:-len("<think>")].rstrip() + "\n"
    if no_think_prefill:
        text = text + no_think_prefill
    return text


def _apply_chat_template(tokenizer, messages, chat_template_kwargs=None):
    """Apply chat template with fallback for models that don't support system role.

    Base models (no chat template) get plain text formatting.
    """
    extra_kwargs = {k: v for k, v in (chat_template_kwargs or {}).items()
                    if k != "no_think_prefill"}  # internal opt-in flag, not a template kwarg
    no_think_prefill = _resolve_no_think_prefill(tokenizer, extra_kwargs)

    if tokenizer.chat_template is None:
        # Completion format for base models (standard in representation engineering literature)
        style = next((m["content"] for m in messages if m["role"] == "system"), "")
        question = next((m["content"] for m in messages if m["role"] == "user"), "")
        if style:
            return f"{style}\n\nQ: {question}\nA:"
        return f"Q: {question}\nA:"

    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            **extra_kwargs,
        )
        # Strip thinking mode prompts (GLM-Z1, DeepSeek-R1 etc.)
        # These models add <think> at the end of generation prompt,
        # forcing chain-of-thought. We strip it for non-thinking calibration,
        # then splice on the no-think prefill (no-op if not given).
        return _finish_prompt_text(text, no_think_prefill)
    except Exception as e:
        if "System role not supported" not in str(e):
            raise
        merged = []
        system_content = ""
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            elif msg["role"] == "user" and system_content:
                merged.append({"role": "user", "content": f"{system_content}\n\n{msg['content']}"})
                system_content = ""
            else:
                merged.append(msg)
        text = tokenizer.apply_chat_template(
            merged, tokenize=False, add_generation_prompt=True,
            **extra_kwargs,
        )
        return _finish_prompt_text(text, no_think_prefill)


def format_chat_messages(
    user_message: str,
    system_message: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """Format messages for chat template."""
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    if history:
        for msg in history:
            messages.append(msg)
    messages.append({"role": "user", "content": user_message})
    return messages


def aggregate_hidden_states(
    hidden_states: Tuple[torch.Tensor, ...],
    num_generated_tokens: int,
    num_layers: int = HIDDEN_LAYERS_TO_USE,
    weight_decay: float = TOKEN_WEIGHT_DECAY,
    layer_weights: Optional[List[float]] = None,
) -> np.ndarray:
    """Aggregate hidden states from multiple layers and tokens.

    Uses weighted mean across tokens (more recent tokens get higher weight)
    and weighted mean across the last N layers.
    """
    if layer_weights is None:
        layer_weights = LAYER_WEIGHTS

    if num_generated_tokens == 0:
        num_generated_tokens = hidden_states[-1].shape[1]

    layers_to_use = hidden_states[-num_layers:]

    aggregated_layers = []
    for layer_states in layers_to_use:
        generated_states = layer_states[0, -num_generated_tokens:, :]
        num_tokens = generated_states.shape[0]
        weights = torch.tensor(
            [weight_decay ** (num_tokens - 1 - i) for i in range(num_tokens)],
            device=generated_states.device,
            dtype=generated_states.dtype,
        )
        weights = weights / weights.sum()
        weighted_state = (generated_states * weights.unsqueeze(1)).sum(dim=0)
        aggregated_layers.append(weighted_state)

    stacked = torch.stack(aggregated_layers, dim=0)
    layer_weights_tensor = torch.tensor(
        layer_weights[:len(aggregated_layers)],
        device=stacked.device,
        dtype=stacked.dtype,
    )
    layer_weights_tensor = layer_weights_tensor / layer_weights_tensor.sum()
    final_state = (stacked * layer_weights_tensor.view(-1, 1)).sum(dim=0)

    return final_state.cpu().float().numpy()


def aggregate_last_token(
    hidden_states: Tuple[torch.Tensor, ...],
    num_generated_tokens: int,
    num_layers: int = HIDDEN_LAYERS_TO_USE,
    layer_weights: Optional[List[float]] = None,
) -> np.ndarray:
    """Aggregate hidden states using only the last generated token."""
    if layer_weights is None:
        layer_weights = LAYER_WEIGHTS

    layers_to_use = hidden_states[-num_layers:]
    aggregated = [layer[0, -1, :] for layer in layers_to_use]

    stacked = torch.stack(aggregated, dim=0)
    lw = torch.tensor(
        layer_weights[:len(aggregated)],
        device=stacked.device,
        dtype=stacked.dtype,
    )
    lw = lw / lw.sum()
    return (stacked * lw.view(-1, 1)).sum(dim=0).cpu().float().numpy()


@torch.no_grad()
def generate_with_hidden_states(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: List[Dict[str, str]],
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    do_sample: bool = DO_SAMPLE,
    return_raw_states: bool = False,
    seed: Optional[int] = None,
    chat_template_kwargs: Optional[dict] = None,
    save_extra: bool = True,
    stop_strings: Optional[List[str]] = None,
) -> GenerationResult:
    """Generate text and extract hidden states."""
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    input_text = _apply_chat_template(tokenizer, messages, chat_template_kwargs)
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    prompt_length = input_ids.shape[1]

    from transformers import StoppingCriteria, StoppingCriteriaList

    class StopStringCriteria(StoppingCriteria):
        def __init__(self, tokenizer, stop_strings, prompt_length):
            self.tokenizer = tokenizer
            self.stop_strings = stop_strings
            self.prompt_length = prompt_length
        def __call__(self, input_ids, scores, **kwargs):
            generated = self.tokenizer.decode(
                input_ids[0, self.prompt_length:], skip_special_tokens=True)
            return any(s in generated for s in self.stop_strings)

    stopping_criteria = None
    if stop_strings:
        stopping_criteria = StoppingCriteriaList([
            StopStringCriteria(tokenizer, stop_strings, prompt_length)
        ])

    t0 = time.time()
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=temperature if do_sample else 1.0,
        top_p=top_p if do_sample else 1.0,
        do_sample=do_sample,
        output_hidden_states=True,
        output_scores=save_extra,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=model.config.eos_token_id,
        stopping_criteria=stopping_criteria,
    )
    generation_time_s = time.time() - t0

    generated_ids = outputs.sequences[0, prompt_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    num_generated = len(outputs.hidden_states)
    if num_generated == 0:
        raise ValueError("No tokens were generated")

    all_layer_states = []
    num_layers = len(outputs.hidden_states[0])

    for layer_idx in range(num_layers):
        layer_token_states = []
        for token_idx in range(num_generated):
            token_state = outputs.hidden_states[token_idx][layer_idx]
            layer_token_states.append(token_state)
        layer_states = torch.cat(layer_token_states, dim=1)
        all_layer_states.append(layer_states)

    hidden_states_tuple = tuple(all_layer_states)

    aggregated_state = aggregate_hidden_states(
        hidden_states_tuple, num_generated_tokens=num_generated,
    )
    last_token_state = aggregate_last_token(
        hidden_states_tuple, num_generated_tokens=num_generated,
    )

    per_layer = None
    token_states_arr = None
    top_k_ids_arr = None
    top_k_logprobs_arr = None

    if save_extra:
        # float32: Gemma-class hidden norms reach ~80k, above fp16 max (65504)
        per_layer = torch.stack(
            [layer[0, -1, :] for layer in all_layer_states], dim=0
        ).cpu().float().numpy()

        layers_for_tokens = all_layer_states[-HIDDEN_LAYERS_TO_USE:]
        stacked_for_tokens = torch.stack(
            [l[0] for l in layers_for_tokens], dim=0
        )
        lw = torch.tensor(
            LAYER_WEIGHTS[:len(layers_for_tokens)],
            device=stacked_for_tokens.device,
            dtype=stacked_for_tokens.dtype,
        )
        lw = lw / lw.sum()
        token_states_arr = (
            stacked_for_tokens * lw.view(-1, 1, 1)
        ).sum(dim=0).cpu().half().numpy()
        # fp16 keeps the (large) token array small, but can overflow to inf
        # for models with huge hidden norms — upcast to fp32 if that happens.
        if not np.isfinite(token_states_arr).all():
            logger.warning(
                "token_states overflowed fp16 (non-finite values); upcasting to float32"
            )
            token_states_arr = (
                stacked_for_tokens * lw.view(-1, 1, 1)
            ).sum(dim=0).cpu().float().numpy()

        if hasattr(outputs, "scores") and outputs.scores:
            tk_ids_list = []
            tk_lp_list = []
            for score in outputs.scores:
                logprobs = torch.log_softmax(score[0], dim=-1)
                topk_vals, topk_ids = torch.topk(logprobs, TOP_K_LOGITS)
                tk_ids_list.append(topk_ids.cpu().numpy())
                tk_lp_list.append(topk_vals.cpu().float().numpy())
            top_k_ids_arr = np.stack(tk_ids_list)
            top_k_logprobs_arr = np.stack(tk_lp_list)

    raw_states = hidden_states_tuple if return_raw_states else None

    return GenerationResult(
        text=generated_text,
        hidden_state=aggregated_state,
        raw_hidden_states=raw_states,
        hidden_state_last_token=last_token_state,
        n_generated_tokens=num_generated,
        n_words=len(generated_text.split()),
        per_layer_states=per_layer,
        token_states=token_states_arr,
        top_k_ids=top_k_ids_arr,
        top_k_logprobs=top_k_logprobs_arr,
        generation_time_s=generation_time_s,
    )


@torch.no_grad()
def get_full_result_for_prompt(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: List[Dict[str, str]],
    max_new_tokens: int = CALIBRATION_MAX_NEW_TOKENS,
    seed: int = 42,
    chat_template_kwargs: Optional[dict] = None,
    save_extra: bool = True,
    stop_strings: Optional[List[str]] = None,
) -> GenerationResult:
    """Convenience function to get full GenerationResult for a prompt."""
    return generate_with_hidden_states(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        seed=seed,
        chat_template_kwargs=chat_template_kwargs,
        save_extra=save_extra,
        stop_strings=stop_strings,
    )


@torch.no_grad()
def get_hidden_state_for_prompt(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: List[Dict[str, str]],
    max_new_tokens: int = CALIBRATION_MAX_NEW_TOKENS,
    seed: int = 42,
    chat_template_kwargs: Optional[dict] = None,
) -> Tuple[str, np.ndarray]:
    """Convenience function to get hidden state for a calibration prompt."""
    result = generate_with_hidden_states(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        seed=seed,
        chat_template_kwargs=chat_template_kwargs,
    )
    return result.text, result.hidden_state


@torch.no_grad()
def generate_response(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: List[Dict[str, str]],
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    do_sample: bool = DO_SAMPLE,
    chat_template_kwargs: Optional[dict] = None,
) -> str:
    """Generate a text response without hidden states (faster)."""
    input_text = _apply_chat_template(tokenizer, messages, chat_template_kwargs)
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    prompt_length = input_ids.shape[1]

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=temperature if do_sample else 1.0,
        top_p=top_p if do_sample else 1.0,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=model.config.eos_token_id,
    )

    generated_ids = outputs[0, prompt_length:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)
