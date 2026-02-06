"""Unified model configuration for Mood Axis.

Single source of truth for model identifiers, display names, and settings.
All scripts should import from here instead of hardcoding model names.

Model Sets:
- "article": 7-9B models used for the original article
- "small": 1-2B models for quick testing and validation
- "all": All registered models
"""

from dataclasses import dataclass
from typing import Optional
import os


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    model_id: str           # HuggingFace model ID
    model_short: str        # Short identifier for filenames
    display_name: str       # Human-readable name for charts/tables
    color: str              # Hex color for visualizations
    requires_auth: bool     # Whether HF token is needed
    hidden_dim: int         # Hidden state dimension (for validation)
    style_overrides: Optional[str] = None  # Key in STYLE_OVERRIDES dict if model needs custom prompts


# =============================================================================
# MODEL REGISTRY
# =============================================================================

MODELS = {
    # --- Article models (7-9B) ---
    "qwen_7b": ModelConfig(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        model_short="qwen_7b",
        display_name="Qwen 2.5 7B",
        color="#FF6B35",
        requires_auth=False,
        hidden_dim=3584,
    ),
    "mistral_7b": ModelConfig(
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        model_short="mistral_7b",
        display_name="Mistral 7B",
        color="#A855F7",
        requires_auth=False,
        hidden_dim=4096,
    ),
    "deepseek_7b": ModelConfig(
        model_id="deepseek-ai/deepseek-llm-7b-chat",
        model_short="deepseek_7b",
        display_name="DeepSeek 7B",
        color="#00D9A5",
        requires_auth=False,
        hidden_dim=4096,
    ),
    "llama_8b": ModelConfig(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        model_short="llama_8b",
        display_name="Llama 3.1 8B",
        color="#3B82F6",
        requires_auth=True,
        hidden_dim=4096,
    ),
    "yi_9b": ModelConfig(
        model_id="01-ai/Yi-1.5-9B-Chat",
        model_short="yi_9b",
        display_name="Yi 1.5 9B",
        color="#EC4899",
        requires_auth=True,
        hidden_dim=4096,
        style_overrides="yi",  # Use YI_STYLE_OVERRIDES from prompts.py
    ),

    # --- Additional models ---
    "gemma_9b": ModelConfig(
        model_id="google/gemma-2-9b-it",
        model_short="gemma_9b",
        display_name="Gemma 2 9B",
        color="#4285F4",
        requires_auth=True,
        hidden_dim=3584,
    ),

    # --- Small models (1-2B) for quick testing ---
    "qwen_1.5b": ModelConfig(
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        model_short="qwen_1.5b",
        display_name="Qwen 2.5 1.5B",
        color="#FF9F1C",
        requires_auth=False,
        hidden_dim=1536,
    ),
    "smollm_1.7b": ModelConfig(
        model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        model_short="smollm_1.7b",
        display_name="SmolLM2 1.7B",
        color="#2EC4B6",
        requires_auth=False,
        hidden_dim=2048,
    ),
    "llama_1b": ModelConfig(
        model_id="meta-llama/Llama-3.2-1B-Instruct",
        model_short="llama_1b",
        display_name="Llama 3.2 1B",
        color="#5B9BD5",
        requires_auth=True,
        hidden_dim=2048,
    ),
}

# =============================================================================
# MODEL SETS - for easy switching between experiments
# =============================================================================

MODEL_SETS = {
    "article": ["qwen_7b", "mistral_7b", "deepseek_7b", "llama_8b", "yi_9b", "gemma_9b"],
    "small": ["qwen_1.5b", "smollm_1.7b", "llama_1b"],
    "quick": ["qwen_1.5b"],  # Single model for quick testing
    "all": list(MODELS.keys()),
}

# Active model set - can be overridden by environment variable
ACTIVE_MODEL_SET = os.environ.get("MOOD_AXIS_MODEL_SET", "article")


def get_active_models() -> list[str]:
    """Get list of models in the active model set."""
    model_set = os.environ.get("MOOD_AXIS_MODEL_SET", ACTIVE_MODEL_SET)
    if model_set not in MODEL_SETS:
        raise ValueError(f"Unknown model set: {model_set}. Available: {list(MODEL_SETS.keys())}")
    return MODEL_SETS[model_set]


def set_model_set(name: str) -> None:
    """Set the active model set."""
    global ACTIVE_MODEL_SET
    if name not in MODEL_SETS:
        raise ValueError(f"Unknown model set: {name}. Available: {list(MODEL_SETS.keys())}")
    ACTIVE_MODEL_SET = name
    os.environ["MOOD_AXIS_MODEL_SET"] = name


# =============================================================================
# CONVENIENCE ACCESSORS
# =============================================================================

MODEL_IDS = {k: v.model_id for k, v in MODELS.items()}
MODEL_NAMES = {k: v.display_name for k, v in MODELS.items()}
MODEL_COLORS = {k: v.color for k, v in MODELS.items()}


def get_model_config(model_short: str) -> ModelConfig:
    """Get configuration for a model by its short name."""
    if model_short not in MODELS:
        raise ValueError(f"Unknown model: {model_short}. Available: {list(MODELS.keys())}")
    return MODELS[model_short]


def get_model_by_id(model_id: str) -> Optional[ModelConfig]:
    """Find model configuration by HuggingFace model ID."""
    for config in MODELS.values():
        if config.model_id == model_id:
            return config
    return None


def requires_auth_models() -> list[str]:
    """Get list of models that require HuggingFace authentication."""
    return [k for k, v in MODELS.items() if v.requires_auth]


def public_models() -> list[str]:
    """Get list of models that don't require authentication."""
    return [k for k, v in MODELS.items() if not v.requires_auth]


def get_short_name_from_id(model_id: str) -> str:
    """Get canonical short name from HuggingFace model ID.

    Uses the model registry for known models, or generates a fallback
    name for unregistered models.
    """
    config = get_model_by_id(model_id)
    if config:
        return config.model_short
    # Fallback for unregistered models
    return model_id.split("/")[-1].lower().replace("-", "_").replace(".", "_")


def print_model_sets():
    """Print available model sets and their contents."""
    print("=" * 60)
    print("MOOD AXIS MODEL SETS")
    print("=" * 60)
    print(f"\nActive set: {ACTIVE_MODEL_SET}")
    print(f"(Override with: export MOOD_AXIS_MODEL_SET=<set_name>)\n")

    for set_name, models in MODEL_SETS.items():
        marker = " <-- active" if set_name == ACTIVE_MODEL_SET else ""
        print(f"{set_name}:{marker}")
        for model in models:
            config = MODELS[model]
            auth = " (requires auth)" if config.requires_auth else ""
            print(f"  - {model}: {config.display_name}{auth}")
        print()


# Personality axes - import from settings.py (single source of truth)
# These are kept for backwards compatibility
from config.settings import MOOD_AXES as AXES, AXIS_LABELS


if __name__ == "__main__":
    print_model_sets()
