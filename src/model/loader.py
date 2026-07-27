"""Model loading utilities for Mood Axis."""

import os
import logging
import torch
from typing import Tuple

# Suppress verbose output before importing HF libs
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.getLogger("httpx").setLevel(logging.WARNING)

from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")
warnings.filterwarnings("ignore", message=".*generation flags are not valid.*")
import transformers.utils.logging as hf_logging
hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()

from config.settings import DEVICE, DTYPE, DEFAULT_MODEL


def load_model(
    model_name: str = DEFAULT_MODEL,
    device: torch.device = DEVICE,
    dtype: torch.dtype = DTYPE,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a model and tokenizer with hidden states output enabled.

    Args:
        model_name: HuggingFace model name or path
        device: Device to load the model on
        dtype: Data type for model weights

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model {model_name} on {device} with {dtype}...", flush=True)

    print("  Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Yi-1.5-9B-Chat fast tokenizer has a decode bug: it strips SentencePiece
    # space prefixes, producing "HellohowareyouI'mfine" instead of "Hello, how are you".
    # We explicitly load LlamaTokenizer which decodes correctly.
    test_ids = tokenizer.encode("Hello, how are you?", add_special_tokens=False)
    if " " not in tokenizer.decode(test_ids):
        from transformers import LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("  Loading weights...", flush=True)
    if device.type == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            output_hidden_states=True,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=None,
            output_hidden_states=True,
            trust_remote_code=True,
        )
        model = model.to(device)

    model.eval()

    print(f"  Model loaded: {model.num_parameters() / 1e9:.2f}B params on {device}", flush=True)
    return model, tokenizer


class ModelManager:
    """Singleton-like manager for model and tokenizer."""

    _instance = None
    _model = None
    _tokenizer = None
    _model_name = None
    _device = None

    @classmethod
    def get_model(
        cls,
        model_name: str = DEFAULT_MODEL,
        device: torch.device = DEVICE,
        dtype: torch.dtype = DTYPE,
        force_reload: bool = False,
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        if (
            force_reload
            or cls._model is None
            or cls._model_name != model_name
            or cls._device != device
        ):
            if cls._model is not None:
                del cls._model
                cls._model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            cls._model, cls._tokenizer = load_model(model_name, device, dtype)
            cls._model_name = model_name
            cls._device = device

        return cls._model, cls._tokenizer

    @classmethod
    def clear(cls):
        """Clear the loaded model from memory."""
        if cls._model is not None:
            del cls._model
            cls._model = None
            cls._tokenizer = None
            cls._model_name = None
            cls._device = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
