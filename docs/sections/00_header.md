# Mood Axis: Measuring and Steering LLM Personality via Hidden States

Large language models exhibit consistent behavioral tendencies -- characteristic patterns in tone, verbosity, confidence, and emotional expression -- that persist across diverse prompts even without explicit instructions. **Mood Axis** is a two-part framework for measuring and controlling these tendencies through hidden state representations.

**Part 1 (Measurement).** We calibrate direction vectors from contrastive instruction pairs and project model responses onto {{n_axes}} bipolar personality axes. Evaluation across {{n_models}} models ({{model_size_range}}) with three validation levels -- held-out accuracy ({{accuracy_range}}), test-retest stability (ICC {{icc_range}}), and signal concentration ({{signal_concentration_final_layer}} in the final layer) -- demonstrates that each model has a distinct, stable behavioral fingerprint. Comparing base and instruct models reveals that RLHF trains **instruction resistance** rather than suppressing personality signal: base models follow personality instructions at {{llama_8b_base_instruction_acc}} accuracy while aligned models achieve only {{llama_8b_instruction_acc}} (Llama), with the effect varying by organization (Meta -24%, Alibaba +9%, Mistral -3%).

**Part 2 (Steering).** Using the same axis vectors, we add scaled personality directions to hidden states during generation. Across {{n_paper_models}} models x {{n_axes_steering}} axes (5,250 steered generations), steerability varies ~40x: {{steering_most_steerable}} responds strongly (total effect {{steering_most_steerable_effect}}, {{judge_mistral_7b_pairwise}} external judge accuracy) while {{steering_least_steerable}} is essentially immovable ({{steering_least_steerable_effect}}). Dead zones from Part 1 do not predict steerability (rho = {{steering_dz_rho}}), revealing that behavioral measurement and behavioral control are orthogonal properties.

All code, calibrated axes, datasets, and reproduction instructions: [{{github_url}}]({{github_url}})

> **[Interactive Explorer](https://yunoshev.github.io/mood-axis/)** — browse personality fingerprints, compare models side-by-side, and try the **Steering Playground**: drag a slider to shift a model's personality and watch the actual text change in real time.

![Personality Fingerprints](docs/figures/fingerprints.png)

---

**Table of Contents**

- [Quick Start](#quick-start)
- **Part 1: Measurement**
  - [Methodology](#methodology)
  - [Behavioral Fingerprints](#behavioral-fingerprints)
  - [Axis Stability](#axis-stability)
  - [Instruction Resistance](#instruction-resistance-how-rlhf-affects-personality)
  - [Dead Zones](#dead-zones-rlhf-suppression)
  - [Conflict Drift](#conflict-drift)
  - [PCA Dimensionality](#pca-dimensionality)
- **Part 2: Steering**
  - [Steering Methodology](#steering-methodology)
  - [Steerability Results](#steerability-varies-40x-across-models)
  - [External Validation](#external-validation-llm-judge)
  - [Dead Zones vs Steerability](#dead-zones-do-not-predict-steerability)
- [Key Findings](#key-findings)
- [Limitations](#limitations)
- [Pipeline Reference](#pipeline-reference)
- [Data Files](#data-files)
- [Models](#models-tested)
