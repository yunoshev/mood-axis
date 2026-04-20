### Dead Zones Do Not Predict Steerability

Mistral 7B: {{dead_mistral_7b_n}} dead zones, most steerable ({{steering_mistral_7b_total}}). Qwen3 8B: {{dead_qwen3_8b_n}} dead zones, least steerable ({{steering_qwen3_8b_total}}). Gemma 2 9B: {{dead_gemma_9b_n}} dead zones, moderate steerability.

Correlation between dead zone count and total steering effect: **rho = {{steering_dz_rho}}, p = {{steering_dz_p}}** -- effectively zero. Surface-level behavioral suppression (Part 1) and deep representational steerability (Part 2) are orthogonal properties.

With normalized alpha and optimal layer selection, all tested models show measurable steering effects. Gemma 3 12B shows the weakest effect (+0.58) -- consistent with its unique mid-layer peak / final-layer suppression alignment pattern. DeepSeek-R1 14B shows the strongest effect (+1.86) despite being a reasoning-distilled model.
