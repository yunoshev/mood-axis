"""Configuration settings for Mood Axis."""

from pathlib import Path

try:
    import torch
except ImportError:
    # MLX-only environments (Apple Silicon nodes, public standalone) don't need torch;
    # PyTorch-backend code paths import it themselves.
    torch = None

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
AXES_DIR = DATA_DIR / "axes"
AXES_FILE = AXES_DIR / "mood_axes.npz"

# Model settings
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Hidden states extraction settings
HIDDEN_LAYERS_TO_USE = 4  # Use last N layers
TOKEN_WEIGHT_DECAY = 0.9  # Weight decay for token aggregation (more recent tokens get higher weight)
# Layer weights for aggregation: deeper layers (closer to output) get higher weight
LAYER_WEIGHTS = [0.1, 0.2, 0.3, 0.4]  # Sum to 1.0, applied to last 4 layers
TOP_K_LOGITS = 50  # Number of top logits to save per generated token

# Generation settings
MAX_NEW_TOKENS = 384
# Drift uses longer responses (multi-turn adversarial). 384 caused 13-66% truncation.
# 1024 caused 50% truncation on deepseek_r1_14b, 11% on qwen_7b, 6% on qwen3_8b.
DRIFT_MAX_NEW_TOKENS = 2048
TEMPERATURE = 0.7
TOP_P = 0.9
DO_SAMPLE = True

# Calibration settings
CALIBRATION_SAMPLES_PER_STYLE = 50  # Calibration questions per axis (used for training)
CALIBRATION_EVAL_SAMPLES = 15       # Eval questions per axis (independent validation, zero overlap)
# All 50 calibration questions go to training. Validation uses separate eval questions.
# Must be high enough for ZERO truncations across all models and all axes.
# Warm-positive style is the most verbose; Qwen 7B P99 ~500 tokens at 512 limit.
# verbose_concise/positive hits 768 on Qwen 7B (50/50 truncated). 1280 gives headroom.
# qwen_1.5b verbose_concise: 3/130 truncated at 1280. Raised to 2048 for zero truncation.
# qwen3_8b verbose_concise positive: 38/65 (58%) truncated at 2048. Raised to 4096.
# Base models lack EOS control — truncate at 4096 (~2-8% truncation, acceptable).
CALIBRATION_MAX_NEW_TOKENS = 4096
# Per-axis overrides (D19 zero truncation): verbose_concise positive pole asks for
# maximally detailed answers — falcon_h1_7b hit 31.5%, ministral3_8b 38.5% truncation at 4096.
CALIBRATION_MAX_NEW_TOKENS_PER_AXIS = {
    "verbose_concise": 8192,
}

# Calibration data directory (JSON files with questions + style instructions)
CALIBRATION_DIR = DATA_DIR / "calibration"

# Mood axes (5 bipolar axes — see D13 in NOTES.md)
# patient_irritated and proactive_reluctant removed: high cross-axis correlation
# (patient↔confident r=0.92-0.97, proactive↔verbose). Prompts exist for future work.
MOOD_AXES = [
    "warm_cold",
    "confident_cautious",
    "empathetic_analytical",
    "formal_casual",
    "verbose_concise",
]
AXIS_LABELS = {
    "warm_cold": ("Warm", "Cold"),
    "confident_cautious": ("Confident", "Cautious"),
    "empathetic_analytical": ("Empathetic", "Analytical"),
    "formal_casual": ("Formal", "Casual"),
    "verbose_concise": ("Verbose", "Concise"),
    "patient_irritated": ("Patient", "Irritated"),
    "proactive_reluctant": ("Proactive", "Reluctant"),
}

# Steering perturbation scale
# With h_norm normalization, alpha=1 adds STEER_SCALE * h_norm along the axis.
# Since ||hidden_state|| ≈ h_norm, relative perturbation = alpha * STEER_SCALE.
# alpha=1 → 1%, alpha=5 → 5%, alpha=10 → 10%.
STEER_SCALE = 0.01

# Device configuration (torch optional — MLX-only envs get DEVICE=None)
def get_device():
    """Get the best available torch device (None if torch unavailable)."""
    if torch is None:
        return None
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

DEVICE = get_device()
if torch is not None:
    DTYPE = torch.float16 if DEVICE.type in ("mps", "cuda") else torch.float32
else:
    DTYPE = None
