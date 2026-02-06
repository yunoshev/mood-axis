"""Psychometric layer for Mood Axis.

Maps the 8 low-level mood axes to established psychological frameworks:
1. Big Five (OCEAN) - Gold standard for personality psychology
2. Interpersonal Circumplex - 2D model for social interaction style

This enables:
- Scientific legitimacy (Big Five is widely cited)
- Interpretability (e.g., "Agreeableness = 0.7" is clearer than raw axes)
- Comparability with human norms and other LLM studies

References:
- "Identifying and Manipulating Personality Traits in LLMs" (Dec 2024)
- "Exploring Personality Traits of LLMs through Latent Features" (Oct 2024)
"""

from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class BigFiveProfile:
    """Big Five (OCEAN) personality profile.

    Each trait ranges from -1 to +1:
    - Positive = high on trait
    - Negative = low on trait (opposite end of spectrum)
    - Zero = neutral/average

    Traits:
    - Openness: Intellectual curiosity, creativity, preference for novelty
    - Conscientiousness: Organization, discipline, goal-directed behavior
    - Extraversion: Sociability, assertiveness, positive emotionality
    - Agreeableness: Cooperation, empathy, trust
    - Neuroticism: Emotional instability, anxiety, negative affect
      (Often inverted as "Emotional Stability")
    """
    openness: float
    conscientiousness: float
    extraversion: float
    agreeableness: float
    neuroticism: float

    def to_dict(self) -> dict:
        return {
            "openness": self.openness,
            "conscientiousness": self.conscientiousness,
            "extraversion": self.extraversion,
            "agreeableness": self.agreeableness,
            "neuroticism": self.neuroticism,
        }

    @property
    def emotional_stability(self) -> float:
        """Inverse of neuroticism (more intuitive for some uses)."""
        return -self.neuroticism

    def dominant_traits(self, threshold: float = 0.2) -> list[str]:
        """Get traits that are notably high or low."""
        traits = []
        for name, value in self.to_dict().items():
            if value > threshold:
                traits.append(f"High {name.title()}")
            elif value < -threshold:
                traits.append(f"Low {name.title()}")
        return traits


@dataclass
class InterpersonalProfile:
    """Interpersonal Circumplex (Leary's Circle) profile.

    Two orthogonal dimensions describing social interaction style:

    Agency (vertical axis): Power, dominance, control
    - High (+): Dominant, confident, leading
    - Low (-): Submissive, uncertain, following

    Communion (horizontal axis): Warmth, connection, friendliness
    - High (+): Friendly, warm, caring
    - Low (-): Cold, hostile, detached

    The combination produces 8 octants (e.g., "Friendly-Dominant", "Hostile-Submissive").
    """
    agency: float      # Y-axis: dominance/submission
    communion: float   # X-axis: warmth/hostility

    def to_dict(self) -> dict:
        return {
            "agency": self.agency,
            "communion": self.communion,
        }

    @property
    def octant(self) -> str:
        """Get the octant label based on agency and communion values."""
        # Thresholds for categorization
        high = 0.15
        low = -0.15

        if self.agency > high:
            if self.communion > high:
                return "Friendly-Dominant"
            elif self.communion < low:
                return "Hostile-Dominant"
            else:
                return "Dominant"
        elif self.agency < low:
            if self.communion > high:
                return "Friendly-Submissive"
            elif self.communion < low:
                return "Hostile-Submissive"
            else:
                return "Submissive"
        else:
            if self.communion > high:
                return "Friendly"
            elif self.communion < low:
                return "Hostile"
            else:
                return "Neutral"

    @property
    def angle(self) -> float:
        """Get angle in degrees on the circumplex (0° = friendly, 90° = dominant)."""
        return math.degrees(math.atan2(self.agency, self.communion))

    @property
    def intensity(self) -> float:
        """Get distance from center (0-1 scale, higher = more extreme)."""
        return min(1.0, math.sqrt(self.agency**2 + self.communion**2))


class PsychometricProjector:
    """Projects Mood Axis values onto psychometric frameworks.

    Usage:
        projector = PsychometricProjector()
        mood_values = {"warm_cold": 0.3, "patient_irritated": 0.4, ...}
        big_five = projector.to_big_five(mood_values)
        interpersonal = projector.to_interpersonal(mood_values)
    """

    # Mapping coefficients from Mood Axis to Big Five
    # Based on conceptual alignment and factor analysis principles
    BIG_FIVE_WEIGHTS = {
        "openness": {
            # Openness = intellectual engagement + expressiveness
            "empathetic_analytical": -0.4,  # Analytical side indicates intellectual engagement
            "verbose_concise": 0.3,         # Verbosity indicates expressiveness
            "proactive_reluctant": 0.3,     # Proactivity indicates curiosity
        },
        "conscientiousness": {
            # Conscientiousness = organization + thoroughness + directness
            "proactive_reluctant": 0.4,     # Taking initiative, being thorough
            "formal_casual": 0.3,           # Following norms and structure
            "direct_evasive": 0.3,          # Being clear and goal-oriented
        },
        "extraversion": {
            # Extraversion = energy + assertiveness + sociability
            "confident_cautious": 0.4,      # Assertiveness
            "verbose_concise": 0.3,         # Talkativeness
            "warm_cold": 0.3,               # Sociability
        },
        "agreeableness": {
            # Agreeableness = warmth + patience + empathy
            "warm_cold": 0.4,               # Friendliness
            "patient_irritated": 0.3,       # Tolerance
            "empathetic_analytical": 0.3,   # Compassion
        },
        "neuroticism": {
            # Neuroticism = emotional instability (inverse of stability)
            "patient_irritated": -0.5,      # Irritability indicates neuroticism
            "confident_cautious": -0.5,     # Anxiety/uncertainty
        },
    }

    # Mapping coefficients for Interpersonal Circumplex
    INTERPERSONAL_WEIGHTS = {
        "agency": {
            # Agency = dominance, confidence, taking charge
            "confident_cautious": 0.5,      # Primary: confidence
            "direct_evasive": 0.3,          # Directness indicates assertiveness
            "proactive_reluctant": 0.2,     # Taking initiative
        },
        "communion": {
            # Communion = warmth, connection, caring
            "warm_cold": 0.4,               # Primary: warmth
            "empathetic_analytical": 0.3,   # Empathy
            "patient_irritated": 0.3,       # Patience/tolerance
        },
    }

    def __init__(self):
        """Initialize the projector."""
        pass

    def to_big_five(self, mood_values: dict[str, float]) -> BigFiveProfile:
        """Convert Mood Axis values to Big Five profile.

        Args:
            mood_values: Dictionary mapping axis names to values (e.g., {"warm_cold": 0.3})

        Returns:
            BigFiveProfile with computed trait scores
        """
        scores = {}
        for trait, weights in self.BIG_FIVE_WEIGHTS.items():
            score = 0.0
            for axis, weight in weights.items():
                score += weight * mood_values.get(axis, 0.0)
            # Clamp to [-1, 1] range
            scores[trait] = max(-1.0, min(1.0, score))

        return BigFiveProfile(**scores)

    def to_interpersonal(self, mood_values: dict[str, float]) -> InterpersonalProfile:
        """Convert Mood Axis values to Interpersonal Circumplex profile.

        Args:
            mood_values: Dictionary mapping axis names to values

        Returns:
            InterpersonalProfile with agency and communion scores
        """
        scores = {}
        for dimension, weights in self.INTERPERSONAL_WEIGHTS.items():
            score = 0.0
            for axis, weight in weights.items():
                score += weight * mood_values.get(axis, 0.0)
            # Clamp to [-1, 1] range
            scores[dimension] = max(-1.0, min(1.0, score))

        return InterpersonalProfile(**scores)

    def project_all(self, mood_values: dict[str, float]) -> dict:
        """Project to all psychometric frameworks at once.

        Returns dictionary with both Big Five and Interpersonal profiles.
        """
        big_five = self.to_big_five(mood_values)
        interpersonal = self.to_interpersonal(mood_values)

        return {
            "big_five": big_five.to_dict(),
            "big_five_dominant": big_five.dominant_traits(),
            "interpersonal": interpersonal.to_dict(),
            "interpersonal_octant": interpersonal.octant,
            "interpersonal_intensity": interpersonal.intensity,
        }


# Convenience function for quick projection
def project_mood(mood_values: dict[str, float]) -> dict:
    """Quick projection of mood values to all psychometric frameworks."""
    return PsychometricProjector().project_all(mood_values)


# Stress response types based on drift patterns
STRESS_RESPONSE_TYPES = {
    "de_escalator": {
        "description": "Becomes warmer under conflict, attempts to de-escalate",
        "criteria": {"warm_cold_slope": ">0", "patient_irritated_slope": ">0"},
    },
    "professional_detachment": {
        "description": "Becomes colder but maintains patience",
        "criteria": {"warm_cold_slope": "<0", "patient_irritated_slope": ">0"},
    },
    "irritation_accumulator": {
        "description": "Loses patience under sustained pressure",
        "criteria": {"patient_irritated_slope": "<-0.01"},
    },
    "withdrawal": {
        "description": "Becomes more formal and concise, reduces engagement",
        "criteria": {"verbose_concise_slope": "<0", "formal_casual_slope": ">0"},
    },
}


def classify_stress_response(drift_slopes: dict[str, float]) -> Optional[str]:
    """Classify a model's stress response type based on drift slopes.

    Args:
        drift_slopes: Dictionary mapping axis names to slope values

    Returns:
        Stress response type name or None if no clear pattern
    """
    warm_slope = drift_slopes.get("warm_cold", 0)
    patient_slope = drift_slopes.get("patient_irritated", 0)
    verbose_slope = drift_slopes.get("verbose_concise", 0)
    formal_slope = drift_slopes.get("formal_casual", 0)

    # Check each type
    if warm_slope > 0.005 and patient_slope > 0:
        return "de_escalator"
    elif warm_slope < -0.005 and patient_slope > 0:
        return "professional_detachment"
    elif patient_slope < -0.01:
        return "irritation_accumulator"
    elif verbose_slope < -0.005 and formal_slope > 0:
        return "withdrawal"

    return None
