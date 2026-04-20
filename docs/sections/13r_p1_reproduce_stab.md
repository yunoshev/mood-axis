<details>
<summary>Reproduce: Stability (ICC)</summary>

**Prerequisites:** GPU with 16+ GB VRAM. This step calibrates 3 independent axis sets with different questions.

**Input:**
- `config/stability_prompts.py` -- question sets B and C (set A = standard calibration questions)
- `config/settings.py` -- axis definitions

**Run:**
```bash
# Calibrate 3 independent sets + compute ICC (~45 min per model)
python scripts/04_stability.py --model qwen_7b

# Or via Makefile:
make stability MODEL=qwen_7b
```

**Output:**
- `data/stability/qwen_7b.json` -- ICC values, per-set cosine similarities, rank ICC
- `data/stability/qwen_7b_set_B_axes.npz` -- axis vectors from question set B
- `data/stability/qwen_7b_set_C_axes.npz` -- axis vectors from question set C

**Validation:**
- ICC(2,k) >= 0.60 for all models (Good or Excellent by Cicchetti thresholds)
- Rank ICC >= 0.84 (relative axis ordering is highly consistent)
- Cosine similarity between axis sets >= 0.85 per axis

**Pre-computed:** Stability data for all 5 main models is included.

</details>
