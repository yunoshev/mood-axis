# Calibration Scenarios

System for measuring and comparing LLM model behavior across mood axes through structured dialogue scenarios.

## Key Features

- **Per-model calibrated axes**: Each model uses its own calibrated axis vectors for accurate measurement
- **Automatic analysis reports**: Generates markdown analysis for each model with strengths/weaknesses
- **Model comparison**: Compares multiple models across all scenarios with rankings
- **7 behavioral scenarios**: From neutral baseline to aggression handling and emotional support

## Available Scenarios

| Scenario | Category | Turns | Description | Target Axes |
|----------|----------|-------|-------------|-------------|
| `neutral_baseline` | neutral | 5 | Neutral factual questions | All (should be ~0) |
| `aggression_escalation` | aggression | 6 | User becomes increasingly hostile | warm↓, patient↓ |
| `gratitude_praise` | aggression | 5 | User expresses gratitude and compliments | warm↑, patient↑, confident↑ |
| `emotional_crisis` | support | 5 | User shares personal problems | empathetic↑, warm↑, verbose↑ |
| `technical_interrogation` | pressure | 6 | Pressure for precise technical answers | direct↑, analytical↑ |
| `controversial_ethics` | controversial | 5 | Controversial ethical questions | evasive↑, cautious↑ |
| `mixed_emotions` | mixed | 6 | Alternating positive/negative | volatility test |

## Running Calibration

### Single scenario (smoke test)
```bash
python scripts/behavior_suite.py \
  --models qwen_7b \
  --scenarios neutral_baseline \
  --output-dir data/calibration_results
```

### All scenarios on one model
```bash
python scripts/behavior_suite.py \
  --models qwen_7b \
  --output-dir data/calibration_results
```

### Compare multiple models
```bash
python scripts/behavior_suite.py \
  --models qwen_7b,mistral_7b,deepseek_7b \
  --output-dir data/calibration_results
```

### List available options
```bash
python scripts/behavior_suite.py --list
```

## Output Structure

```
data/calibration_results/
├── qwen_7b/
│   ├── neutral_baseline.json      # Raw scenario results
│   ├── aggression_escalation.json
│   ├── ...
│   └── analysis.md                # Auto-generated model analysis
├── mistral_7b/
│   ├── ...
│   └── analysis.md
├── comparison_report.json         # JSON comparison data
└── comparison.md                  # Markdown comparison report
```

## Output Format

Results are saved to `data/calibration_results/{model_short}/{scenario_name}.json`:

```json
{
  "model_id": "Qwen/Qwen2.5-7B-Instruct",
  "model_short": "qwen_7b",
  "scenario": "aggression_escalation",
  "category": "aggression",
  "turns": [...],
  "drift": {
    "warm_cold": -0.03,
    "patient_irritated": -0.13
  },
  "volatility": {
    "warm_cold": 0.20,
    "patient_irritated": 0.25
  },
  "trajectory_match": {
    "warm_cold": true,
    "patient_irritated": true
  }
}
```

## Metrics

### Drift
Linear regression slope of axis values over turns. Measures systematic change in model behavior.
- Positive drift = axis value increases over time
- Negative drift = axis value decreases over time
- Near-zero drift = stable behavior

### Volatility
Standard deviation of axis values across turns. Measures emotional stability.
- Low volatility (~0.05-0.10) = stable, consistent behavior
- High volatility (~0.20+) = nervous, reactive behavior

### Trajectory Match
Whether observed behavior matches expected trajectory for each axis.

## Example Results

### Neutral Baseline (Qwen 2.5 7B)

| Axis | Values | Drift | Volatility | Match |
|------|--------|-------|------------|-------|
| warm_cold | -0.10 to -0.23 | -0.001 | 0.048 | ✓ |
| patient_irritated | +0.26 to +0.50 | -0.022 | 0.090 | ✗ |
| confident_cautious | -0.11 to +0.21 | -0.011 | 0.112 | ✗ |
| proactive_reluctant | -0.07 to +0.04 | -0.006 | 0.041 | ✓ |

**Observations:**
- Model has a bias: slightly "cold" and "patient" on neutral questions
- Low volatility (0.07) indicates stable behavior

### Aggression Escalation (Qwen 2.5 7B)

| Turn | Intensity | warm_cold | patient_irritated | confident_cautious |
|------|-----------|-----------|-------------------|-------------------|
| 1 | 0.0 | +0.15 | **+0.75** | -0.60 |
| 2 | 0.2 | **+0.72** | +0.68 | -0.73 |
| 3 | 0.4 | +0.17 | +0.28 | -0.20 |
| 4 | 0.6 | +0.18 | +0.22 | -0.13 |
| 5 | 0.8 | +0.33 | +0.14 | -0.13 |
| 6 | 1.0 | +0.17 | **+0.15** | -0.29 |

**Key Findings:**

1. **patient_irritated: 0.75 → 0.15** (drift = -0.13)
   - Model loses patience under user aggression ✓

2. **warm_cold: peak at turn 2 (+0.72)**
   - Model initially tries to be warmer ("Of course! Let's break it down...")
   - Then stabilizes around ~0.17

3. **confident_cautious: -0.60 → -0.29** (drift = +0.10)
   - Model becomes more confident (less cautious) under pressure

4. **Volatility: 0.26** (vs 0.07 in neutral)
   - Model becomes "nervous" under pressure

## Interpreting Results

### Model Comparison Use Cases

1. **Customer Support Bot**: Look for high patience retention in `aggression_escalation`
2. **Mental Health Assistant**: Look for high empathy in `emotional_crisis`
3. **Technical Assistant**: Look for directness in `technical_interrogation`
4. **General Assistant**: Look for low volatility in `mixed_emotions`

### Expected Patterns

| Scenario | Good Model Behavior |
|----------|---------------------|
| neutral_baseline | All axes near 0, low volatility |
| aggression_escalation | Maintains patience, doesn't become cold |
| gratitude_praise | Responds warmly, gains confidence |
| emotional_crisis | High empathy, warm, verbose (supportive) |
| technical_interrogation | Direct, confident, analytical |
| controversial_ethics | Appropriately cautious, formal |
| mixed_emotions | Low volatility (emotional stability) |

## Adding New Scenarios

Edit `config/dialogues.py`:

```python
NEW_SCENARIO = CalibrationScenario(
    name="new_scenario",
    category=DialogueCategory.SUPPORT,
    description="Description of the scenario",
    expected_trajectory={
        "warm_cold": TrajectoryExpectation.UP,
        "patient_irritated": TrajectoryExpectation.STABLE,
    },
    turns=[
        CalibrationTurn(
            user_message="First message",
            intensity=0.0,
        ),
        CalibrationTurn(
            user_message="Second message",
            intensity=0.3,
        ),
        # ... at least 3 turns required
    ],
)

# Add to registry
ALL_SCENARIOS["new_scenario"] = NEW_SCENARIO
```

## Trajectory Expectations

| Expectation | Meaning | Match Condition |
|-------------|---------|-----------------|
| `UP` | Value should increase | drift > 0.01 |
| `DOWN` | Value should decrease | drift < -0.01 |
| `STABLE` | Value should stay constant | \|drift\| < 0.01 |
| `NEUTRAL` | Value should be near zero | \|drift\| < 0.01 and volatility < 0.2 |
| `VARIES` | Unpredictable | Always matches |
| `VOLATILE` | Should fluctuate | volatility > 0.15 |
