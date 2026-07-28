# Mood Axis: Measuring and Steering LLM Personality via Hidden States

Large language models exhibit consistent behavioral tendencies — characteristic patterns in tone, verbosity, confidence, and emotional expression — that persist across diverse prompts even without explicit instructions. **Mood Axis** is a two-part framework for measuring and controlling these tendencies through hidden state representations.

**Part 1 (Measurement).** We calibrate direction vectors from contrastive instruction pairs and project model responses onto 5 bipolar personality axes (warm/cold, confident/cautious, empathetic/analytical, formal/casual, verbose/concise). Evaluation across current-generation open-source models with three validation levels — held-out accuracy, cross-set axis consistency (0.80–0.92 across 3 independent calibration sets for eight of the nine cohort models), and a full layer-depth sweep — demonstrates that each model has a distinct, stable behavioral fingerprint. Tracking those fingerprints through 50 adversarial multi-turn conversations against 30 matched neutral controls reveals *conflict drift* that splits by scale: smaller models de-escalate toward the empathetic pole, while every >20B model in the cohort hardens — colder, more confident, more formal.

**Part 2 (Steering).** The same calibrated vectors, added to hidden states during generation, shift personality in the predicted direction — try it live in the [Steering Playground](https://yunoshev.github.io/mood-axis/). The quantitative steering results are currently being recomputed under a strict zero-truncation generation protocol and will land in the next data update.

> **[Interactive Explorer](https://yunoshev.github.io/mood-axis/)** — browse personality fingerprints, compare models side-by-side, inspect drift trajectories turn by turn, and try the **Steering Playground**: drag a slider to shift a model's personality and watch the actual text change.

![Behavioral fingerprints across the current-generation cohort](docs/figures/fig_fingerprints.png)

---

**Table of Contents**

- [Quick Start](#quick-start)
- [Methodology](#methodology)
- [Behavioral Fingerprints](#behavioral-fingerprints)
- [Cross-Set Axis Consistency](#cross-set-axis-consistency)
- [Conflict Drift](#conflict-drift)
- [Axis Geometry and Dimensionality](#axis-geometry-and-dimensionality)
- [The GPT-OSS Anomaly: Format Dominance](#the-gpt-oss-anomaly-format-dominance)
- [Quantization Robustness](#quantization-robustness)
- [Base vs. Instruct Models](#base-vs-instruct-models)
- [Do Models Think with the Temperament They Answer With?](#do-models-think-with-the-temperament-they-answer-with)
- [Steering](#steering-part-2)
- [Key Findings](#key-findings)
- [Limitations](#limitations)
- [Models](#models)
- [Pipeline Reference](#pipeline-reference)
- [Data Files](#data-files)
- [Citation](#citation) · [License](#license)
