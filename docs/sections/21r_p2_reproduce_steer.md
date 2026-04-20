<details>
<summary>Reproduce: Steering</summary>

**Prerequisites:** Calibrated axes for the target model (`data/axes/{model}_axes.npz`). GPU required.

**Input:**
- `data/axes/{model}_axes.npz` -- axis direction vectors from calibration
- `config/settings.py` -- steering parameters (alpha values, questions per combo)

**Run:**
```bash
# Generate steered responses (~60-90 min per model)
python scripts/steering_basic.py --model qwen_7b

# Analyze steering results (CPU, ~10 seconds)
python scripts/steering_analyze.py
```

**Output:**
- `data/steering/qwen_7b_basic.json` -- steered responses and projections for all (axis, alpha) combinations
  - 5 axes x 7 alphas x 30 questions = 1,050 generations per model
  - Per-generation: text, target projection, all 7 axis projections (for leakage analysis)
- `data/analysis/steering_summary.json` -- aggregated effects, slopes, leakage metrics

**Validation:**
- Target axis projection should increase monotonically with alpha (positive slope)
- Cross-axis leakage < 1.0 for axis-specific models (Mistral)
- Total effect (sum of absolute slopes across axes) varies by model: Mistral ~3.3, Llama ~0.5, Qwen3 ~0.08

**Pre-computed:** Steering data for all 5 main models is included.

</details>
