"""Model loading utilities for Mood Axis."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple
import sys

sys.path.insert(0, str(__file__).rsplit("/src/", 1)[0])
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
    print(f"Loading model {model_name} on {device} with {dtype}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device.type == "cuda":
        # Use device_map="auto" and native dtype on CUDA.
        # This lets MoE/MXFP4 models (e.g. gpt-oss-20b) load in their
        # native precision instead of upcasting to fp16 and OOM-ing.
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            output_hidden_states=True,
            trust_remote_code=True,
        )
    else:
        # MPS/CPU: keep manual placement
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=None,
            output_hidden_states=True,
            trust_remote_code=True,
        )
        model = model.to(device)

    model.eval()

    print(f"Model loaded successfully. Parameters: {model.num_parameters() / 1e9:.2f}B")
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
        """Get or load the model and tokenizer.

        Args:
            model_name: HuggingFace model name or path
            device: Device to load the model on
            dtype: Data type for model weights
            force_reload: Force reloading even if already loaded

        Returns:
            Tuple of (model, tokenizer)
        """
        if (
            force_reload
            or cls._model is None
            or cls._model_name != model_name
            or cls._device != device
        ):
            # Clear existing model
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
