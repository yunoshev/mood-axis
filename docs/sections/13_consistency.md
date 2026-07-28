## Cross-Set Axis Consistency

Is a calibrated axis a property of the *model*, or an artifact of the particular calibration questions? We re-calibrate every axis three times from independent, disjoint question sets (A/B/C, 30 questions per axis each) and measure the mean pairwise correlation between the three resulting direction vectors.

![Cross-set axis consistency by model](docs/figures/fig_consistency.png)

**Eight of the nine cohort models land at 0.80–0.92** — far above the ≈ 0 null expected for random directions at this dimensionality. Axis vectors are a reproducible property of each model, not question noise.

The one exception is **GPT-OSS 20B (0.39)** — and its deficit is not sampling noise but a diagnosable property of its axis geometry: a dominant *format* response to persona prompts that its axes partly absorb. Removing a low-rank format subspace recovers most of the deficit (see [the format-confound section](#the-gpt-oss-anomaly-format-dominance)). Notably, MoE architecture does not explain it: Qwen3.6-35B-A3B and Gemma 4 26B-A4B — both sparse MoE models of comparable or larger scale — sit at 0.92 and 0.86.

Consistency is also decoupled from baseline magnitude: Qwen 3.5 9B pairs a deliberately neutral fingerprint with main-cast-typical consistency (0.84) — it stably reproduces suppressed responses rather than noise.
