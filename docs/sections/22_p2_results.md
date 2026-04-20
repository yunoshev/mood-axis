### Steerability Varies ~40x Across Models

{{table_steering_full}}

Models cluster into three steerability tiers:
- **Steerable** (total > 1.0): {{steering_most_steerable}} responds strongly to hidden-state intervention across all axes
- **Moderate** (0.1-1.0): Llama 3.1 8B, Qwen 2.5 7B, Gemma 2 9B show detectable but limited shifts
- **Resistant** (< 0.1): {{steering_least_steerable}} is essentially immovable

### Per-Axis Heatmap

{{table_steering_heatmap}}

![Steering Heatmap](docs/figures/steering_heatmap.png)

Mistral dominates every axis, with warm_cold ({{steering_mistral_7b_warm_cold_effect}}) showing the strongest effect. For moderate models, effects are distributed more evenly.

### Cross-Axis Leakage

When steering one axis, do other axes shift? We measure mean cross-axis leakage:

- **Low leakage** (< 1.0): Mistral 7B ({{steering_mistral_7b_leakage}}), Qwen3 8B ({{steering_qwen3_8b_leakage}}) -- steering is axis-specific
- **High leakage** (> 5.0): Qwen 2.5 7B ({{steering_qwen_7b_leakage}}) -- weak target effect but spills onto other axes
