<details>
<summary><b>Reproduce: stability / consistency</b></summary>

**Input:** `config/stability_prompts.py` (3 × 150 questions) + `config/models.py`.

**Run:**
```bash
# GPU: calibrate the three independent sets (resumable; --set a|b|c to run one set)
python3 scripts/04_stability.py --model qwen3.5_9b --backend mlx calibrate

# CPU: cross-set consistency, rank-ICC, Procrustes congruence
python3 scripts/04_stability.py --model qwen3.5_9b analyze

# CPU: permutation null for the consistency statistic (all models)
python3 scripts/06h_consistency_null.py
```

**Output:**
- `data/stability/qwen3.5_9b.json` — per-axis cross-set correlations, ICC, Procrustes
- `data/analysis/consistency_null.json` — permutation-null z-scores per model

**Validation:** all cohort models except GPT-OSS: mean consistency ≥ 0.80; every model's consistency is many SDs above its permutation null.

**Pre-computed:** `v2/data/stability/<model>.json` + `v2/data/analysis/consistency_null.json`.

</details>
