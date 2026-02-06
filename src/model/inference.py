"""Inference utilities with hidden states extraction for Mood Axis."""

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
    DEVICE,
)


@dataclass
class GenerationResult:
    """Result of a generation with hidden states."""
    text: str
    hidden_state: np.ndarray  # Aggregated hidden state
    raw_hidden_states: Optional[List[torch.Tensor]] = None  # Per-token states if needed


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

    Returns:
        GenerationResult with generated text and aggregated hidden state
    """
    # Apply chat template
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Tokenize
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    prompt_length = input_ids.shape[1]

    # Generate with hidden states
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=temperature if do_sample else 1.0,
        top_p=top_p if do_sample else 1.0,
        do_sample=do_sample,
        output_hidden_states=True,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.pad_token_id,
    )

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

    # Aggregate
    aggregated_state = aggregate_hidden_states(
        hidden_states_tuple,
        num_generated_tokens=num_generated,
    )

    raw_states = hidden_states_tuple if return_raw_states else None

    return GenerationResult(
        text=generated_text,
        hidden_state=aggregated_state,
        raw_hidden_states=raw_states,
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
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

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
