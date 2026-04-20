### Steering Methodology

We inject the calibrated axis vector into hidden states at every generation step, with alpha normalized by the model's hidden state norm:

```
h' = h + alpha * (v_axis * h_norm)
```

where v_axis is the normalized direction vector, h_norm = mean(||h||) across calibration responses (normalizes alpha across models), and alpha in {{steering_alphas}}. Some model/axis pairs with weak response were additionally tested at alpha in {-10, +10} to characterize saturation behavior. For each (model, axis, alpha), we generate {{steering_questions_per_combo}} responses and measure target shift, cross-axis leakage, and effective range.

**Hidden state norms vary by 5000x across models** (from ~10 for Qwen 0.5B to ~51,000 for Gemma 3 12B). Without normalization, steering is invisible to models with large hidden norms. With normalization, all tested models respond to steering.

**Layer selection:** Rather than applying steering to all layers, we perform an empirical sweep across 10-15 layer positions per model and select the layer that maximizes target axis shift while keeping 3-gram repetition rate below 30%. This yields model-specific optimal layers that balance steering strength with generation quality.

Experiments cover {{n_paper_models}} models x {{n_axes_steering}} axes x 7 alpha values x {{steering_questions_per_combo}} questions = **5,250 steered generations**.

**External validation:** Claude Sonnet judges pairwise comparisons (alpha=-5 vs alpha=+5) and rates warmth on a Likert scale across alpha values. Steering quality is validated via 3-gram repetition rate (<30% threshold) and external judge evaluation.
