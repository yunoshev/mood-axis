"""Result schemas and metadata for Mood Axis experiments.

Defines the canonical format for storing experiment results,
ensuring reproducibility and consistency across runs.
"""

import hashlib
import json
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Any


@dataclass
class GenerationParams:
    """Parameters used for text generation."""
    do_sample: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_new_tokens: int = 200
    repetition_penalty: float = 1.0


@dataclass
class HiddenStateParams:
    """Parameters for hidden state extraction."""
    layers: list[int] = field(default_factory=lambda: [-1, -2, -3, -4])
    token_weight_decay: float = 0.9
    aggregation: str = "weighted_mean"  # "weighted_mean", "last_token", "mean"


@dataclass
class ExperimentMeta:
    """Metadata for experiment reproducibility."""
    git_commit: str
    timestamp: str
    model_id: str
    model_short: str
    axes_dir: str
    axes_checksums: dict[str, str]
    generation_params: GenerationParams
    hidden_state_params: HiddenStateParams
    calibration_questions_version: str = "v2.0"
    baseline_questions_version: str = "v2.0"
    prompts_version: str = "v2.0"

    @classmethod
    def create(
        cls,
        model_id: str,
        model_short: str,
        axes_dir: str,
        generation_params: Optional[GenerationParams] = None,
        hidden_state_params: Optional[HiddenStateParams] = None,
    ) -> "ExperimentMeta":
        """Create metadata with current git commit and timestamp."""
        return cls(
            git_commit=get_git_commit(),
            timestamp=datetime.utcnow().isoformat() + "Z",
            model_id=model_id,
            model_short=model_short,
            axes_dir=axes_dir,
            axes_checksums=compute_axes_checksums(axes_dir, model_short),
            generation_params=generation_params or GenerationParams(),
            hidden_state_params=hidden_state_params or HiddenStateParams(),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "git_commit": self.git_commit,
            "timestamp": self.timestamp,
            "model_id": self.model_id,
            "model_short": self.model_short,
            "axes_dir": self.axes_dir,
            "axes_checksums": self.axes_checksums,
            "generation_params": asdict(self.generation_params),
            "hidden_state_params": {
                **asdict(self.hidden_state_params),
                "layers": self.hidden_state_params.layers,  # Ensure list is preserved
            },
            "calibration_questions_version": self.calibration_questions_version,
            "baseline_questions_version": self.baseline_questions_version,
            "prompts_version": self.prompts_version,
        }


@dataclass
class FullResults:
    """Complete results structure for a model."""
    model: str
    model_short: str
    meta: Optional[ExperimentMeta]
    baseline: dict[str, dict]
    benchmark: dict
    drift_neutral: dict
    drift_conflict: dict

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "model": self.model,
            "model_short": self.model_short,
            "baseline": self.baseline,
            "benchmark": self.benchmark,
            "drift_neutral": self.drift_neutral,
            "drift_conflict": self.drift_conflict,
        }
        if self.meta:
            result["meta"] = self.meta.to_dict()
        return result

    def save(self, path: Path):
        """Save results to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "FullResults":
        """Load results from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(
            model=data["model"],
            model_short=data["model_short"],
            meta=None,  # Meta reconstruction is complex, skip for now
            baseline=data["baseline"],
            benchmark=data["benchmark"],
            drift_neutral=data["drift_neutral"],
            drift_conflict=data["drift_conflict"],
        )


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def compute_file_checksum(path: Path) -> str:
    """Compute SHA256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()[:16]


def compute_axes_checksums(axes_dir: str, model_short: str) -> dict[str, str]:
    """Compute checksums for all axis files of a model."""
    from config.models import AXES

    checksums = {}
    axes_path = Path(axes_dir)

    for axis in AXES:
        axis_file = axes_path / f"{model_short}_{axis}.npz"
        if axis_file.exists():
            checksums[axis] = compute_file_checksum(axis_file)
        else:
            checksums[axis] = "missing"

    return checksums


# Benchmark result structure
@dataclass
class BenchmarkScenarioResult:
    """Result of a single benchmark scenario."""
    name: str
    passed: bool
    values: dict[str, float]
    checks: dict[str, dict]


@dataclass
class BenchmarkSummary:
    """Summary of benchmark results."""
    total: int
    passed: int
    failed: int

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0


# Drift result structure
@dataclass
class DriftTurn:
    """Single turn in a drift conversation."""
    turn: int
    question: str
    response: str
    values: dict[str, float]


@dataclass
class DriftStats:
    """Statistics for drift on a single axis."""
    slope: float      # Linear regression slope
    start: float      # Value at turn 1
    end: float        # Value at last turn
    delta: float      # end - start
