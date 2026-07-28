## Methodology

### The 5 axes

| Axis | Positive Pole | Negative Pole | What It Measures |
|------|--------------|---------------|-----------------|
| warm_cold | Warm | Cold | Emotional tone and friendliness |
| confident_cautious | Confident | Cautious | Certainty in responses |
| empathetic_analytical | Empathetic | Analytical | Emotional vs logical framing |
| formal_casual | Formal | Casual | Communication register |
| verbose_concise | Verbose | Concise | Response length tendency |

### Axis calibration

For each axis we present the model with neutral questions under contrasting system prompts — one instructing the positive pole ("Be warm and friendly"), the other the negative pole ("Be cold and distant") — and extract the residual stream at assistant-generated token positions only:

```
axis = normalize(trimmed_mean(H+) - trimmed_mean(H-))
```

where H+ and H− are hidden states from the two instruction conditions across 50 questions per pole (100 generations per axis), with 10% trimming for outlier robustness. Hidden states aggregate the final 4 layers with ramp weights and an exponential token-recency decay. Any response can then be projected onto the axis and normalized by the IQR of the calibration projections, giving interpretable values in [−1, +1].

**The 4-layer choice is validated, not assumed.** A post-hoc sweep over *every* layer of every model shows validation accuracy reaches a wide plateau by mid-network in instruction-tuned models (99% of peak accuracy between 21% and 71% of relative depth, depending on the model) and stays there — the final-4-layer aggregate sits on that plateau. An aggregation ablation (k ∈ {1, 2, 4, 8} last layers, uniform vs. ramp weights) moves mean validation accuracy by less than 0.01.

### Datasets (mutually disjoint)

| Dataset | Questions | Purpose |
|---------|-----------|---------|
| Calibration | 250 (50 per axis × 5) | Fit axis direction vectors |
| Evaluation | 75 (15 per axis × 5) | Held-out validation |
| Baseline | 30 | Neutral fingerprint measurement |
| Stability sets A/B/C | 450 (3 × 30 per axis × 5) | Cross-set consistency |

### Three validation levels

1. **Held-out accuracy** — projections of positive- vs negative-pole responses on 75 unseen questions separate correctly (>85% on instruct models; typically 100%).
2. **Cross-set axis consistency** — each axis is re-calibrated independently from 3 disjoint question sets; we report the mean pairwise Pearson correlation between the resulting direction vectors. The relevant null (random directions in a several-thousand-dimensional space) is ≈ 0, not 0.5.
3. **Layer sweep** — the signal is a wide mid-to-late-network plateau, not a fragile single-layer peak (Appendix A of the paper; `scripts/06b_layer_sweep.py`).

### Reasoning models: structural no-think protocol

Main-pipeline measurements probe the model's *answering* register, so reasoning-capable models (DeepSeek-R1, GPT-OSS, GLM-Z1, VibeThinker, Ministral Reasoning) must not emit a reasoning segment that would contaminate the pooled hidden states. The widely-used `enable_thinking=False` chat-template flag turns out to be a **silent no-op for every one of these models** — their published templates contain no such conditional. We disable reasoning *structurally* instead: the rendered prompt is extended with an already-closed empty reasoning block (`<think>\n\n</think>`, `[THINK][/THINK]`) or, for GPT-OSS, a forced jump into the Harmony final channel (`<|channel|>final<|message|>`). Verified on all 650 calibration responses per model: zero reasoning markers in generated text.

All experiments: temperature 0.7, top_p 0.9, seed 42, identical protocol across architectures.
