<details>
<summary>Reproduce: Instruction Resistance (Base vs Instruct)</summary>

**Prerequisites:** Calibrated axes for both base and instruct model variants. GPU required. HuggingFace access tokens for gated models (Llama, Gemma).

**Input:**
- `data/axes/{model}_axes.npz` -- calibrated axis vectors (instruct model)
- `config/models.py` -- model registry with `is_base_model` and `instruct_counterpart` fields

**Run:**
```bash
# Run base vs instruct comparison (~10 min per model pair)
python scripts/07_experiments.py --model llama_8b

# This automatically:
# 1. Calibrates base model axes (if not cached)
# 2. Measures instruction-following accuracy for both variants
# 3. Computes McNemar's test for significance
```

**Output:**
- `data/experiments/llama_8b_exp_base_instruct.json` -- per-axis accuracy, McNemar chi2, p-values
- Includes: refusal direction analysis (Qwen only), decoding probe accuracy, amplification override

**Validation:**
- Base models: instruction accuracy >= 80% (signal exists)
- Instruct models: accuracy varies by organization (Meta ~63%, Alibaba ~97%, Mistral ~81%)
- McNemar p < 0.01 for Llama (significant after Bonferroni)

**Pre-computed:** Experiment data for Llama, Qwen, Mistral (base+instruct pairs) is included.

</details>
