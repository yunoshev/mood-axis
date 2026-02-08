"""Inference utilities with hidden states extraction for Mood Axis."""

import time

import torch
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, str(__file__).rsplit("/src/", 1)[0])
from config.settings import (
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_P,
    DO_SAMPLE,
    HIDDEN_LAYERS_TO_USE,
    TOKEN_WEIGHT_DECAY,
    LAYER_WEIGHTS,
    TOP_K_LOGITS,
    DEVICE,
)


@dataclass
class GenerationResult:
    """Result of a generation with hidden states."""
    text: str
    hidden_state: np.ndarray  # Aggregated hidden state (decay aggregation)
    raw_hidden_states: Optional[List[torch.Tensor]] = None  # Per-token states if needed
    hidden_state_last_token: Optional[np.ndarray] = None  # Last-token aggregation
    n_generated_tokens: int = 0  # Number of tokens generated
    n_words: int = 0  # Word count of generated text
    per_layer_states: Optional[np.ndarray] = None  # (num_all_layers, hidden_dim) fp16, last token all layers
    token_states: Optional[np.ndarray] = None  # (n_generated_tokens, hidden_dim) fp16, per-token layer-aggregated
    top_k_ids: Optional[np.ndarray] = None  # (n_generated_tokens, TOP_K_LOGITS) int32
    top_k_logprobs: Optional[np.ndarray] = None  # (n_generated_tokens, TOP_K_LOGITS) float32
    generation_time_s: float = 0.0  # Wall time in seconds


def _apply_chat_template(tokenizer, messages):
    """Apply chat template with fallback for models that don't support system role.

    Some models (e.g., Gemma) don't support system messages in their chat template.
    For these, we merge the system message into the first user message.

    Base models (no chat template) get plain text formatting.
    """
    # Base models: no chat template — format as plain text
    if tokenizer.chat_template is None:
        parts = []
        for msg in messages:
            if msg["role"] == "system":
                parts.append(msg["content"])
            elif msg["role"] == "user":
                parts.append(msg["content"])
            elif msg["role"] == "assistant":
                parts.append(msg["content"])
        return "\n\n".join(parts) + "\n\n"

    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    except Exception as e:
        if "System role not supported" not in str(e):
            raise
        # Merge system message into first user message
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
        return tokenizer.apply_chat_template(
            merged, tokenize=False, add_generation_prompt=True,
        )


def format_chat_messages(
    user_message: str,
    system_message: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """Format messages for chat template.

    Args:
        user_message: Current user message
        system_message: Optional system prompt
        history: Optional conversation history

    Returns:
        List of message dicts
    """
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
    and weighted mean across the last N layers (deeper layers get higher weight).

    Args:
        hidden_states: Tuple of hidden states from all layers
            Each tensor is (batch_size, seq_len, hidden_dim)
        num_generated_tokens: Number of tokens that were generated (not prompt)
        num_layers: Number of last layers to use
        weight_decay: Weight decay factor for token weighting
        layer_weights: Weights for layer aggregation (default: LAYER_WEIGHTS from config)

    Returns:
        Aggregated hidden state as numpy array (hidden_dim,)
    """
    if layer_weights is None:
        layer_weights = LAYER_WEIGHTS

    if num_generated_tokens == 0:
        # Fallback: use all tokens if none were generated
        num_generated_tokens = hidden_states[-1].shape[1]

    # Take last N layers
    layers_to_use = hidden_states[-num_layers:]

    aggregated_layers = []
    for layer_states in layers_to_use:
        # Get only generated tokens (last num_generated_tokens)
        generated_states = layer_states[0, -num_generated_tokens:, :]  # (num_tokens, hidden_dim)

        # Create weights for tokens (exponential decay, more recent = higher)
        num_tokens = generated_states.shape[0]
        weights = torch.tensor(
            [weight_decay ** (num_tokens - 1 - i) for i in range(num_tokens)],
            device=generated_states.device,
            dtype=generated_states.dtype,
        )
        weights = weights / weights.sum()

        # Weighted mean across tokens
        weighted_state = (generated_states * weights.unsqueeze(1)).sum(dim=0)
        aggregated_layers.append(weighted_state)

    # Weighted mean across layers (deeper layers get higher weight)
    stacked = torch.stack(aggregated_layers, dim=0)
    layer_weights_tensor = torch.tensor(
        layer_weights[:len(aggregated_layers)],
        device=stacked.device,
        dtype=stacked.dtype,
    )
    layer_weights_tensor = layer_weights_tensor / layer_weights_tensor.sum()  # Normalize
    final_state = (stacked * layer_weights_tensor.view(-1, 1)).sum(dim=0)

    return final_state.cpu().float().numpy()


def aggregate_last_token(
    hidden_states: Tuple[torch.Tensor, ...],
    num_generated_tokens: int,
    num_layers: int = HIDDEN_LAYERS_TO_USE,
    layer_weights: Optional[List[float]] = None,
) -> np.ndarray:
    """Aggregate hidden states using only the last generated token.

    Takes the final generated token from each of the last N layers
    and computes a weighted average across layers.

    Args:
        hidden_states: Tuple of hidden states from all layers
        num_generated_tokens: Number of tokens that were generated
        num_layers: Number of last layers to use
        layer_weights: Weights for layer aggregation

    Returns:
        Aggregated hidden state as numpy array (hidden_dim,)
    """
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
) -> GenerationResult:
    """Generate text and extract hidden states.

    Args:
        model: The language model
        tokenizer: The tokenizer
        messages: Chat messages in standard format
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        do_sample: Whether to sample or use greedy decoding
        return_raw_states: Whether to return raw per-token hidden states
        seed: Random seed for reproducible sampling (only relevant when do_sample=True)

    Returns:
        GenerationResult with generated text and aggregated hidden state
    """
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    # Apply chat template
    input_text = _apply_chat_template(tokenizer, messages)

    # Tokenize
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    prompt_length = input_ids.shape[1]

    # Generate with hidden states and scores
    t0 = time.time()
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=temperature if do_sample else 1.0,
        top_p=top_p if do_sample else 1.0,
        do_sample=do_sample,
        output_hidden_states=True,
        output_scores=True,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    generation_time_s = time.time() - t0

    # Decode generated text
    generated_ids = outputs.sequences[0, prompt_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Process hidden states
    # outputs.hidden_states is a tuple of tuples:
    # outer tuple: one element per generated token
    # inner tuple: hidden states from each layer for that token
    num_generated = len(outputs.hidden_states)

    if num_generated == 0:
        # No tokens generated, use prompt hidden states
        raise ValueError("No tokens were generated")

    # Collect hidden states for all generated tokens
    # Stack along sequence dimension
    all_layer_states = []
    num_layers = len(outputs.hidden_states[0])

    for layer_idx in range(num_layers):
        layer_token_states = []
        for token_idx in range(num_generated):
            # Each hidden state is (batch, 1, hidden_dim) for generated tokens
            token_state = outputs.hidden_states[token_idx][layer_idx]
            layer_token_states.append(token_state)

        # Concat along seq dimension: (batch, num_generated, hidden_dim)
        layer_states = torch.cat(layer_token_states, dim=1)
        all_layer_states.append(layer_states)

    # all_layer_states is now a list of (batch, num_generated, hidden_dim) tensors
    # Convert to tuple for aggregate_hidden_states
    hidden_states_tuple = tuple(all_layer_states)

    # Aggregate: decay (production method)
    aggregated_state = aggregate_hidden_states(
        hidden_states_tuple,
        num_generated_tokens=num_generated,
    )

    # Aggregate: last token only (negligible overhead)
    last_token_state = aggregate_last_token(
        hidden_states_tuple,
        num_generated_tokens=num_generated,
    )

    # Per-layer: last generated token from ALL layers → (num_layers, hidden_dim) fp16
    per_layer = torch.stack(
        [layer[0, -1, :] for layer in all_layer_states], dim=0
    ).cpu().half().numpy()

    # Token-level: per-token weighted across last N layers → (n_tokens, hidden_dim) fp16
    layers_for_tokens = all_layer_states[-HIDDEN_LAYERS_TO_USE:]
    stacked_for_tokens = torch.stack(
        [l[0] for l in layers_for_tokens], dim=0
    )  # (N_layers, n_tokens, hidden_dim)
    lw = torch.tensor(
        LAYER_WEIGHTS[:len(layers_for_tokens)],
        device=stacked_for_tokens.device,
        dtype=stacked_for_tokens.dtype,
    )
    lw = lw / lw.sum()
    token_states_arr = (
        stacked_for_tokens * lw.view(-1, 1, 1)
    ).sum(dim=0).cpu().half().numpy()  # (n_tokens, hidden_dim)

    # Top-k logits from scores
    top_k_ids_arr = None
    top_k_logprobs_arr = None
    if hasattr(outputs, "scores") and outputs.scores:
        tk_ids_list = []
        tk_lp_list = []
        for score in outputs.scores:  # score: (batch=1, vocab_size)
            logprobs = torch.log_softmax(score[0], dim=-1)
            topk_vals, topk_ids = torch.topk(logprobs, TOP_K_LOGITS)
            tk_ids_list.append(topk_ids.cpu().numpy())
            tk_lp_list.append(topk_vals.cpu().float().numpy())
        top_k_ids_arr = np.stack(tk_ids_list)  # (n_tokens, TOP_K)
        top_k_logprobs_arr = np.stack(tk_lp_list)  # (n_tokens, TOP_K)

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
    max_new_tokens: int = 100,
) -> GenerationResult:
    """Convenience function to get full GenerationResult for a prompt.

    Returns the complete result with both hidden state aggregations,
    token count, and word count.

    Args:
        model: The language model
        tokenizer: The tokenizer
        messages: Chat messages
        max_new_tokens: Max tokens for generation

    Returns:
        Full GenerationResult
    """
    return generate_with_hidden_states(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # Deterministic for calibration
    )


@torch.no_grad()
def get_hidden_state_for_prompt(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: List[Dict[str, str]],
    max_new_tokens: int = 100,
) -> Tuple[str, np.ndarray]:
    """Convenience function to get hidden state for a calibration prompt.

    Args:
        model: The language model
        tokenizer: The tokenizer
        messages: Chat messages
        max_new_tokens: Max tokens for generation

    Returns:
        Tuple of (generated_text, hidden_state)
    """
    result = generate_with_hidden_states(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # Deterministic for calibration
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
) -> str:
    """Generate a text response without hidden states (faster).

    Args:
        model: The language model
        tokenizer: The tokenizer
        messages: Chat messages
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        do_sample: Whether to sample or use greedy decoding

    Returns:
        Generated text response
    """
    # Apply chat template
    input_text = _apply_chat_template(tokenizer, messages)

    # Tokenize
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    prompt_length = input_ids.shape[1]

    # Generate
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=temperature if do_sample else 1.0,
        top_p=top_p if do_sample else 1.0,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Decode
    generated_ids = outputs[0, prompt_length:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)
