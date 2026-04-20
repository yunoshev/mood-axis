### External Validation: LLM Judge

To verify probe-measured shifts correspond to **perceptually distinguishable** behavior changes, we use Claude Sonnet as an external judge:

1. **Pairwise forced-choice**: Given responses from alpha=-5 and alpha=+5 (randomized order), identify the warmer one
2. **Likert dose-response**: Rate warmth 1-5 across 5 alpha values, testing monotonic dose-response

{{table_judge}}

For steerable models (Mistral), the judge correctly identifies the steered direction {{judge_mistral_7b_pairwise}} of the time with strong dose-response (rho={{judge_mistral_7b_rho}}). For resistant models (Gemma), performance is at chance ({{judge_gemma_9b_pairwise}}, rho={{judge_gemma_9b_rho}}), confirming that resistance is genuine.
