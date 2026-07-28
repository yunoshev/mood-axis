<details>
<summary><b>Reproduce: fingerprints</b></summary>

**Input:** calibrated axes (`data/axes/<model>_axes.npz`, previous step) + `config/prompts.py` baseline questions.

**Run:**
```bash
python3 scripts/02_baseline.py --model qwen3.5_9b --backend mlx
```

**Output:** `data/baselines/qwen3.5_9b.json` — per-axis mean/std over 30 questions + full per-question projections and response texts.

**Validation:** projections lie in [−1, +1]; `_meta.n_questions == 30`.

**Pre-computed:** `v2/data/baselines/<model>.json` for the full cohort — these exact files back the fingerprint table above and the [Interactive Explorer](https://yunoshev.github.io/mood-axis/).

</details>
