## Quantization Robustness

Models up to ~14B are measured in bf16; models above 20B in 4-bit quantization (the precision most deployments actually run). To verify this mixed-precision design doesn't confound comparisons, every main-cast model was calibrated **twice** — once in bf16 and once in 4-bit — and the pairing repeated for all four >20B models.

Across all eight paired models (40 axis pairs): **axis directions are essentially precision-invariant** — mean cosine between bf16 and 4-bit axis vectors is 0.95 (per-model means 0.92–0.96, pooled minimum 0.85), and held-out validation accuracy is unchanged (0.996 in both precisions on average). The neutral point of an axis can shift slightly between precisions (mean 0.10 calibration-scale units, occasionally up to ~0.6), which is why every reported projection is normalized by the center and scale of its *own* precision's calibration.

<details>
<summary><b>Reproduce: quantization robustness</b></summary>

**Input:** paired axis files — `data/axes/<model>_axes.npz` and `data/axes/<model>_4bit_axes.npz` (calibrate the 4-bit twin with `python3 scripts/01_calibrate.py --model qwen3.5_9b_4bit --backend mlx`).

**Run:**
```bash
python3 scripts/06j_quant_robustness.py
```

**Output:** `data/analysis/quant_robustness.json` — per-axis cosines, center/scale shifts, validation accuracy in both precisions.

**Pre-computed:** `v2/data/analysis/quant_robustness.json`; 4-bit twin calibration metadata in `v2/data/axes_meta/`.

</details>
