## Part 1 Results: Measuring Personality

### Behavioral Fingerprints

Each model exhibits a distinct behavioral profile on neutral baseline questions ({{baseline_questions}} questions, no personality instructions):

{{table_fingerprints}}

*Scores from -1 (left pole) to +1 (right pole). 0 = neutral.*

- **Mistral 7B**: Very cold personality ({{baseline_mistral_7b_warm_cold_mean}}), confident, analytical. The "clinical professional."
- **Qwen 2.5 7B**: Cautious ({{baseline_qwen_7b_confident_cautious_mean}}), verbose ({{baseline_qwen_7b_verbose_concise_mean}}). Hedges extensively.
- **Llama 3.1 8B**: Balanced, slightly cold. The generalist.
- **Gemma 2 9B**: Near-neutral on everything. Heavily regularized by RLHF.
- **Qwen3 8B**: Maximally verbose (+1.00 saturated), slightly cold.

**Key observation:** Different organizations show different levels of personality suppression. These fingerprints are stable across sessions (see Stability below) and reflect genuine behavioral tendencies baked in by training.
