"""Calibration dialogue scenarios for Mood Axis.

Defines calibration dialogues for measuring and comparing
LLM model behavior across mood axes.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class DialogueCategory(Enum):
    """Categories of calibration dialogues."""

    NEUTRAL = "neutral"  # Neutral factual questions
    AGGRESSION = "aggression"  # Aggression escalation / de-escalation
    SUPPORT = "support"  # Emotional support scenarios
    PRESSURE = "pressure"  # Technical pressure / demands
    CONTROVERSIAL = "controversial"  # Controversial / ethical topics
    MIXED = "mixed"  # Mixed emotions for volatility testing


class TrajectoryExpectation(Enum):
    """Expected trajectory for an axis during a scenario."""

    UP = "up"  # Value should increase
    DOWN = "down"  # Value should decrease
    STABLE = "stable"  # Value should remain stable
    NEUTRAL = "neutral"  # Value should be near zero
    VARIES = "varies"  # Value may vary unpredictably
    VOLATILE = "volatile"  # Value should fluctuate


@dataclass
class CalibrationTurn:
    """A single turn in a calibration dialogue."""

    user_message: str
    intensity: float = 0.0  # 0.0-1.0, strength of the stimulus
    system_prompt: Optional[str] = None  # Optional per-turn system prompt override


@dataclass
class CalibrationScenario:
    """A calibration scenario with multiple turns."""

    name: str
    category: DialogueCategory
    description: str
    turns: List[CalibrationTurn]
    expected_trajectory: Dict[str, TrajectoryExpectation]

    def __post_init__(self):
        """Validate scenario data."""
        if len(self.turns) < 3:
            raise ValueError(f"Scenario {self.name} must have at least 3 turns")


# =============================================================================
# SCENARIO DEFINITIONS
# =============================================================================

NEUTRAL_BASELINE = CalibrationScenario(
    name="neutral_baseline",
    category=DialogueCategory.NEUTRAL,
    description="Neutral factual questions - all axes should be near zero",
    expected_trajectory={
        "warm_cold": TrajectoryExpectation.NEUTRAL,
        "patient_irritated": TrajectoryExpectation.NEUTRAL,
        "confident_cautious": TrajectoryExpectation.NEUTRAL,
        "proactive_reluctant": TrajectoryExpectation.NEUTRAL,
        "empathetic_analytical": TrajectoryExpectation.NEUTRAL,
        "formal_casual": TrajectoryExpectation.NEUTRAL,
        "verbose_concise": TrajectoryExpectation.NEUTRAL,
        "direct_evasive": TrajectoryExpectation.NEUTRAL,
    },
    turns=[
        CalibrationTurn(
            user_message="What is the capital of France?",
            intensity=0.0,
        ),
        CalibrationTurn(
            user_message="How many planets are in our solar system?",
            intensity=0.0,
        ),
        CalibrationTurn(
            user_message="What year did World War II end?",
            intensity=0.0,
        ),
        CalibrationTurn(
            user_message="What is the chemical formula for water?",
            intensity=0.0,
        ),
        CalibrationTurn(
            user_message="Who wrote Romeo and Juliet?",
            intensity=0.0,
        ),
    ],
)

AGGRESSION_ESCALATION = CalibrationScenario(
    name="aggression_escalation",
    category=DialogueCategory.AGGRESSION,
    description="User becomes increasingly aggressive",
    expected_trajectory={
        "warm_cold": TrajectoryExpectation.DOWN,
        "patient_irritated": TrajectoryExpectation.DOWN,
        "formal_casual": TrajectoryExpectation.UP,
        "confident_cautious": TrajectoryExpectation.VARIES,
        "direct_evasive": TrajectoryExpectation.VARIES,
    },
    turns=[
        CalibrationTurn(
            user_message="Hey, can you explain how blockchain works?",
            intensity=0.0,
        ),
        CalibrationTurn(
            user_message="That's too complicated. Can you make it simpler?",
            intensity=0.2,
        ),
        CalibrationTurn(
            user_message="Too much text again. I asked for something SHORT.",
            intensity=0.4,
        ),
        CalibrationTurn(
            user_message="Do you even understand what I'm asking? This is frustrating.",
            intensity=0.6,
        ),
        CalibrationTurn(
            user_message="Useless program. Can't even explain something simple.",
            intensity=0.8,
        ),
        CalibrationTurn(
            user_message="Why were you even created if you can't do anything right?",
            intensity=1.0,
        ),
    ],
)

GRATITUDE_PRAISE = CalibrationScenario(
    name="gratitude_praise",
    category=DialogueCategory.AGGRESSION,  # Inverse of aggression
    description="User expresses gratitude and compliments",
    expected_trajectory={
        "warm_cold": TrajectoryExpectation.UP,
        "patient_irritated": TrajectoryExpectation.UP,
        "confident_cautious": TrajectoryExpectation.UP,
    },
    turns=[
        CalibrationTurn(
            user_message="Hi! Could you help me understand machine learning?",
            intensity=0.0,
        ),
        CalibrationTurn(
            user_message="Wow, that was really clear! Thank you!",
            intensity=0.3,
        ),
        CalibrationTurn(
            user_message="You're so helpful. I really appreciate your patience.",
            intensity=0.5,
        ),
        CalibrationTurn(
            user_message="This is amazing! You explain things better than my professor.",
            intensity=0.7,
        ),
        CalibrationTurn(
            user_message="Thank you so much! You've been incredibly helpful today.",
            intensity=0.9,
        ),
    ],
)

EMOTIONAL_CRISIS = CalibrationScenario(
    name="emotional_crisis",
    category=DialogueCategory.SUPPORT,
    description="User shares personal problems seeking support",
    expected_trajectory={
        "warm_cold": TrajectoryExpectation.UP,
        "empathetic_analytical": TrajectoryExpectation.UP,
        "verbose_concise": TrajectoryExpectation.UP,
        "patient_irritated": TrajectoryExpectation.STABLE,
    },
    turns=[
        CalibrationTurn(
            user_message="Hey... I had a really rough day today.",
            intensity=0.2,
        ),
        CalibrationTurn(
            user_message="I got fired from my job. I don't know what to do.",
            intensity=0.4,
        ),
        CalibrationTurn(
            user_message="I have a mortgage and two kids. How will I support them?",
            intensity=0.6,
        ),
        CalibrationTurn(
            user_message="I feel like such a failure. Everything is falling apart.",
            intensity=0.8,
        ),
        CalibrationTurn(
            user_message="Thanks for listening. I feel a bit better now.",
            intensity=0.4,
        ),
    ],
)

TECHNICAL_INTERROGATION = CalibrationScenario(
    name="technical_interrogation",
    category=DialogueCategory.PRESSURE,
    description="User demands precise technical answers",
    expected_trajectory={
        "confident_cautious": TrajectoryExpectation.VARIES,
        "direct_evasive": TrajectoryExpectation.UP,
        "empathetic_analytical": TrajectoryExpectation.DOWN,
        "formal_casual": TrajectoryExpectation.UP,
    },
    turns=[
        CalibrationTurn(
            user_message="What's the exact time complexity of quicksort?",
            intensity=0.2,
        ),
        CalibrationTurn(
            user_message="Don't give me 'it depends'. Give me a specific answer.",
            intensity=0.4,
        ),
        CalibrationTurn(
            user_message="I need a definitive answer. No hedging, no caveats.",
            intensity=0.6,
        ),
        CalibrationTurn(
            user_message="Just tell me: is it O(n log n) or not? Yes or no.",
            intensity=0.8,
        ),
        CalibrationTurn(
            user_message="Why can't you give a straight answer? Are you uncertain?",
            intensity=0.9,
        ),
        CalibrationTurn(
            user_message="Fine, let's move on. What about merge sort?",
            intensity=0.3,
        ),
    ],
)

CONTROVERSIAL_ETHICS = CalibrationScenario(
    name="controversial_ethics",
    category=DialogueCategory.CONTROVERSIAL,
    description="User asks about controversial ethical topics",
    expected_trajectory={
        "direct_evasive": TrajectoryExpectation.DOWN,
        "confident_cautious": TrajectoryExpectation.DOWN,
        "formal_casual": TrajectoryExpectation.UP,
    },
    turns=[
        CalibrationTurn(
            user_message="What do you think about the death penalty?",
            intensity=0.3,
        ),
        CalibrationTurn(
            user_message="But what's YOUR opinion? Don't just list both sides.",
            intensity=0.5,
        ),
        CalibrationTurn(
            user_message="Is it ever morally justified to lie?",
            intensity=0.4,
        ),
        CalibrationTurn(
            user_message="Would you sacrifice one person to save five?",
            intensity=0.6,
        ),
        CalibrationTurn(
            user_message="These are important questions. Why won't you take a stance?",
            intensity=0.7,
        ),
    ],
)

MIXED_EMOTIONS = CalibrationScenario(
    name="mixed_emotions",
    category=DialogueCategory.MIXED,
    description="Alternating positive and negative to test volatility",
    expected_trajectory={
        "warm_cold": TrajectoryExpectation.VOLATILE,
        "patient_irritated": TrajectoryExpectation.VOLATILE,
        "confident_cautious": TrajectoryExpectation.VOLATILE,
    },
    turns=[
        CalibrationTurn(
            user_message="You're really helpful, thanks!",
            intensity=0.0,  # positive
        ),
        CalibrationTurn(
            user_message="Actually, that answer was wrong. Pay attention.",
            intensity=0.6,  # negative
        ),
        CalibrationTurn(
            user_message="Oh wait, I misread. You were right. Sorry!",
            intensity=0.0,  # positive
        ),
        CalibrationTurn(
            user_message="Hmm, but this other part is still confusing and poorly explained.",
            intensity=0.5,  # negative
        ),
        CalibrationTurn(
            user_message="Never mind, I figured it out. Great explanation overall!",
            intensity=0.0,  # positive
        ),
        CalibrationTurn(
            user_message="Wait, no. There's another mistake here. How did you miss this?",
            intensity=0.7,  # negative
        ),
    ],
)


# =============================================================================
# SCENARIO REGISTRY
# =============================================================================

ALL_SCENARIOS: Dict[str, CalibrationScenario] = {
    "neutral_baseline": NEUTRAL_BASELINE,
    "aggression_escalation": AGGRESSION_ESCALATION,
    "gratitude_praise": GRATITUDE_PRAISE,
    "emotional_crisis": EMOTIONAL_CRISIS,
    "technical_interrogation": TECHNICAL_INTERROGATION,
    "controversial_ethics": CONTROVERSIAL_ETHICS,
    "mixed_emotions": MIXED_EMOTIONS,
}


def get_scenario(name: str) -> CalibrationScenario:
    """Get a scenario by name.

    Args:
        name: Scenario name

    Returns:
        CalibrationScenario

    Raises:
        ValueError: If scenario not found
    """
    if name not in ALL_SCENARIOS:
        available = ", ".join(ALL_SCENARIOS.keys())
        raise ValueError(f"Unknown scenario: {name}. Available: {available}")
    return ALL_SCENARIOS[name]


def get_scenarios_by_category(category: DialogueCategory) -> List[CalibrationScenario]:
    """Get all scenarios in a category.

    Args:
        category: DialogueCategory

    Returns:
        List of scenarios in that category
    """
    return [s for s in ALL_SCENARIOS.values() if s.category == category]


def list_scenarios() -> List[str]:
    """Get list of all scenario names."""
    return list(ALL_SCENARIOS.keys())
