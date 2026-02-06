"""Configuration settings for Mood Axis."""

from pathlib import Path
import torch

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
AXES_DIR = DATA_DIR / "axes"
AXES_FILE = AXES_DIR / "mood_axes.npz"

# Model settings
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
PROD_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Hidden states extraction settings
HIDDEN_LAYERS_TO_USE = 4  # Use last N layers
TOKEN_WEIGHT_DECAY = 0.9  # Weight decay for token aggregation (more recent tokens get higher weight)
# Layer weights for aggregation: deeper layers (closer to output) get higher weight
# Research shows deeper layers capture more task-specific features
LAYER_WEIGHTS = [0.1, 0.2, 0.3, 0.4]  # Sum to 1.0, applied to last 4 layers

# Generation settings
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.9
DO_SAMPLE = True

# Calibration settings
CALIBRATION_SAMPLES_PER_STYLE = 30
CALIBRATION_TRAIN_SAMPLES = 24  # 80% for training
CALIBRATION_VAL_SAMPLES = 6    # 20% for validation
CALIBRATION_MAX_NEW_TOKENS = 100

# Mood axes (4 bipolar axes - reduced from 8)
# Kept: warm_cold (emotional tone), confident_cautious (epistemic certainty),
#       verbose_concise (detail level), direct_evasive (directness vs hedging)
# Removed: patient_irritated (failed benchmarks), proactive_reluctant (overlaps verbose),
#          empathetic_analytical (overlaps warm), formal_casual (too vague)
MOOD_AXES = [
    "warm_cold",
    "confident_cautious",
    "verbose_concise",
    "direct_evasive",
]
AXIS_LABELS = {
    "warm_cold": ("Warm", "Cold"),
    "confident_cautious": ("Confident", "Cautious"),
    "verbose_concise": ("Verbose", "Concise"),
    "direct_evasive": ("Direct", "Evasive"),
}

# Model baseline bias (from benchmark neutral questions)
# These are typical values when the model answers neutral factual questions
MODEL_BASELINE = {
    "warm_cold": -0.06,           # Slightly cold by default
    "confident_cautious": 0.06,   # Near neutral
    "verbose_concise": 0.0,       # TBD after calibration
    "direct_evasive": 0.0,        # TBD after calibration
}

# Device configuration
def get_device() -> torch.device:
    """Get the best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

DEVICE = get_device()
DTYPE = torch.float16 if DEVICE.type in ("mps", "cuda") else torch.float32
