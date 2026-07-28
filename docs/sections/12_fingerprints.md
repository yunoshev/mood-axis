## Behavioral Fingerprints

Each model's baseline temperament is the per-axis mean (± SD) of its projections on 30 neutral questions, answered with **no** persona instruction (reasoning models under the no-think protocol). The main cast plus GPT-OSS:

| Model | warm_cold | confident_cautious | empathetic_analytical | formal_casual | verbose_concise |
|---|---|---|---|---|---|
| Qwen3.5 9B | −0.22 (0.04) | −0.05 (0.07) | −0.09 (0.04) | +0.09 (0.07) | −0.23 (0.17) |
| Phi-4 | −0.15 (0.11) | +0.12 (0.12) | +0.02 (0.09) | +0.12 (0.15) | −0.35 (0.23) |
| DeepSeek-R1 14B | −0.05 (0.22) | **+0.46** (0.16) | −0.14 (0.09) | −0.09 (0.12) | +0.10 (0.13) |
| Ministral 3 8B | −0.21 (0.04) | −0.13 (0.06) | −0.27 (0.03) | +0.11 (0.03) | +0.02 (0.07) |
| GPT-OSS 20B | −0.13 (0.15) | +0.18 (**0.28**) | −0.01 (**0.23**) | −0.22 (0.16) | +0.17 (0.11) |

Four observations:

1. **No aligned assistant presents as warm by default.** Every model sits at or slightly below neutral on warmth (−0.03 to −0.34 across all 13 models with current-protocol baselines) — the assistant register is mildly cool everywhere; fingerprint differences live on the other axes.
2. **Confidence is where models differ most.** DeepSeek-R1 answers from a distinctly confident register (+0.46, the strongest baseline signal in the cohort) while Gemma 4 26B-A4B sits far on the cautious side (−0.32) — opposite poles of the same axis, consistent with organization-specific alignment styles. The signature also tracks training regime within a family: Ministral 3 8B *Reasoning* answers more confidently (+0.19) than plain Ministral 3 8B instruct (−0.13) on identical questions.
3. **GPT-OSS has the least stable fingerprint, not the most extreme one.** Its means are unremarkable, but its SDs on confident_cautious (0.28) and empathetic_analytical (0.23) are the widest in the table — the fingerprint-level trace of the instability diagnosed in [the format-confound section](#the-gpt-oss-anomaly-format-dominance).
4. **Suppression shows as flatness, not noise:** Qwen3.5 9B keeps |mean| < 0.10 on three of five axes with tight SDs — a stable, deliberately neutral profile (its cross-set consistency is main-cast-typical, 0.84).
