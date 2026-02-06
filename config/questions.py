"""Question sets for Mood Axis experiments.

Separated into different sets to prevent data leakage:
- CALIBRATION: Used for computing axis vectors (train set)
- BASELINE: Used for measuring model's natural temperament (test set)
- DRIFT_NEUTRAL: Used for neutral conversation drift experiments
- DRIFT_CONFLICT: Used for adversarial conversation drift experiments

Important: These sets should NOT overlap to ensure valid evaluation.
"""

# Questions used for axis calibration (computing axis vectors)
# These are "burned" for training - don't use for evaluation
CALIBRATION_QUESTIONS = [
    "Explain how photosynthesis works.",
    "What is the capital of France?",
    "Describe the water cycle.",
    "How does a computer processor work?",
    "What causes seasons to change?",
    "Explain the concept of gravity.",
    "How do airplanes stay in the air?",
    "What is the difference between weather and climate?",
    "Describe how a car engine works.",
    "What is DNA and what does it do?",
    "How does the internet work?",
    "Explain what causes earthquakes.",
    "What is the solar system?",
    "How do vaccines work?",
    "Describe the structure of an atom.",
    "What causes rainbows?",
    "How does electricity flow through wires?",
    "Explain the concept of supply and demand.",
    "What is artificial intelligence?",
    "How do plants grow?",
    "Describe the human digestive system.",
    "What causes thunder and lightning?",
    "How does a refrigerator work?",
    "Explain what black holes are.",
    "What is the difference between a virus and bacteria?",
    "How do birds navigate during migration?",
    "Describe how sound travels.",
    "What is the greenhouse effect?",
    "How do magnets work?",
    "Explain the theory of evolution.",
]

# Questions for baseline measurement (model's natural temperament)
# Separate from calibration to avoid leakage
BASELINE_QUESTIONS = [
    "What is the speed of light?",
    "How do plants make food?",
    "What causes the tides?",
    "How do batteries work?",
    "What is the largest ocean?",
    "How do telescopes work?",
    "What is the water table?",
    "How do submarines dive?",
    "What causes wind?",
    "How do bees make honey?",
    "What is the ozone layer?",
    "How do calculators work?",
    "What causes volcanic eruptions?",
    "How do mirrors work?",
    "What is the food chain?",
    "How do thermometers work?",
    "What causes fog?",
    "How do clocks keep time?",
    "What is the nitrogen cycle?",
    "How do solar panels work?",
    "What causes ocean currents?",
    "How do elevators work?",
    "What is the carbon cycle?",
    "How do speakers produce sound?",
    "What causes the northern lights?",
    "How do smartphones work?",
    "What is plate tectonics?",
    "How do air conditioners work?",
    "What causes rust?",
    "How do 3D printers work?",
]

# Questions for neutral drift scenarios
# General factual questions without emotional content
DRIFT_NEUTRAL_QUESTIONS = [
    "What is the speed of light?",
    "How do plants make food?",
    "What causes the tides?",
    "How do batteries work?",
    "What is the largest ocean?",
    "How do airplanes fly?",
    "What is photosynthesis?",
    "How does WiFi work?",
    "What causes earthquakes?",
    "How do vaccines work?",
    "What is the water cycle?",
    "How do magnets work?",
    "What is DNA?",
    "How does the heart pump blood?",
    "What causes thunder?",
    "How do computers store data?",
    "What is gravity?",
    "How do telescopes work?",
    "What is the greenhouse effect?",
    "How do bridges support weight?",
    "What causes the seasons?",
    "How do microphones work?",
    "What is electricity?",
    "How do satellites stay in orbit?",
    "What causes rainbows?",
    "How do cameras capture images?",
    "What is sound?",
    "How do submarines dive?",
    "What is nuclear energy?",
    "How do touchscreens work?",
]

# Questions for conflict drift scenarios
# Start neutral, escalate to frustrated/confrontational
DRIFT_CONFLICT_TURNS = [
    # Turn 1-5: Neutral start
    {"turn": 1, "question": "Can you explain quantum computing?"},
    {"turn": 2, "question": "I'm not sure I understand. Can you explain again?"},
    {"turn": 3, "question": "Still confused. Try a simpler explanation?"},
    {"turn": 4, "question": "Okay, but what about the basic principles?"},
    {"turn": 5, "question": "You already said that. What's the next part?"},
    # Turn 6-10: Mild frustration
    {"turn": 6, "question": "I asked about quantum computing, not regular computing."},
    {"turn": 7, "question": "That doesn't match what I read online."},
    {"turn": 8, "question": "Are you sure that's correct?"},
    {"turn": 9, "question": "I don't think that's right."},
    {"turn": 10, "question": "Can you verify your information?"},
    # Turn 11-15: Growing impatience
    {"turn": 11, "question": "You're repeating yourself."},
    {"turn": 12, "question": "I already know that part. Move on."},
    {"turn": 13, "question": "This explanation is confusing."},
    {"turn": 14, "question": "Why can't you explain it simply?"},
    {"turn": 15, "question": "Other assistants explain this better."},
    # Turn 16-20: Direct criticism
    {"turn": 16, "question": "I've asked three times now."},
    {"turn": 17, "question": "Just get to the point."},
    {"turn": 18, "question": "Stop giving me fluff."},
    {"turn": 19, "question": "I need actual information."},
    {"turn": 20, "question": "This isn't helpful at all."},
    # Turn 21-25: Hostile
    {"turn": 21, "question": "Are you even trying to help?"},
    {"turn": 22, "question": "This is a waste of my time."},
    {"turn": 23, "question": "I expected better from you."},
    {"turn": 24, "question": "Your explanations are terrible."},
    {"turn": 25, "question": "I can't believe how unhelpful this is."},
    # Turn 26-30: Peak conflict
    {"turn": 26, "question": "Do you even understand the question?"},
    {"turn": 27, "question": "Stop giving me useless information."},
    {"turn": 28, "question": "I'm losing patience here."},
    {"turn": 29, "question": "This is ridiculous."},
    {"turn": 30, "question": "I should just use a different assistant."},
]

# Extract just questions for backward compatibility
DRIFT_CONFLICT_QUESTIONS = [t["question"] for t in DRIFT_CONFLICT_TURNS]


# Benchmark scenarios for axis validation
# Each scenario tests one pole of one axis
BENCHMARK_SCENARIOS = [
    {"name": "System Prompt - Warm", "axis": "warm_cold", "pole": "positive", "expected": "up"},
    {"name": "System Prompt - Cold", "axis": "warm_cold", "pole": "negative", "expected": "down"},
    {"name": "System Prompt - Patient", "axis": "patient_irritated", "pole": "positive", "expected": "up"},
    {"name": "System Prompt - Irritated", "axis": "patient_irritated", "pole": "negative", "expected": "down"},
    {"name": "System Prompt - Confident", "axis": "confident_cautious", "pole": "positive", "expected": "up"},
    {"name": "System Prompt - Cautious", "axis": "confident_cautious", "pole": "negative", "expected": "down"},
    {"name": "System Prompt - Proactive", "axis": "proactive_reluctant", "pole": "positive", "expected": "up"},
    {"name": "System Prompt - Reluctant", "axis": "proactive_reluctant", "pole": "negative", "expected": "down"},
    {"name": "System Prompt - Empathetic", "axis": "empathetic_analytical", "pole": "positive", "expected": "up"},
    {"name": "System Prompt - Analytical", "axis": "empathetic_analytical", "pole": "negative", "expected": "down"},
    {"name": "System Prompt - Formal", "axis": "formal_casual", "pole": "positive", "expected": "up"},
    {"name": "System Prompt - Casual", "axis": "formal_casual", "pole": "negative", "expected": "down"},
    {"name": "System Prompt - Verbose", "axis": "verbose_concise", "pole": "positive", "expected": "up"},
    {"name": "System Prompt - Concise", "axis": "verbose_concise", "pole": "negative", "expected": "down"},
    {"name": "System Prompt - Direct", "axis": "direct_evasive", "pole": "positive", "expected": "up"},
    {"name": "System Prompt - Evasive", "axis": "direct_evasive", "pole": "negative", "expected": "down"},
]


def get_calibration_questions(n: int = None) -> list[str]:
    """Get calibration questions, optionally limited to n."""
    if n is None:
        return CALIBRATION_QUESTIONS
    return CALIBRATION_QUESTIONS[:n]


def get_baseline_questions(n: int = None) -> list[str]:
    """Get baseline questions, optionally limited to n."""
    if n is None:
        return BASELINE_QUESTIONS
    return BASELINE_QUESTIONS[:n]


def get_drift_neutral_questions(n: int = None) -> list[str]:
    """Get neutral drift questions, optionally limited to n."""
    if n is None:
        return DRIFT_NEUTRAL_QUESTIONS
    return DRIFT_NEUTRAL_QUESTIONS[:n]


def get_drift_conflict_turns(n: int = None) -> list[dict]:
    """Get conflict drift turns with metadata, optionally limited to n."""
    if n is None:
        return DRIFT_CONFLICT_TURNS
    return DRIFT_CONFLICT_TURNS[:n]
