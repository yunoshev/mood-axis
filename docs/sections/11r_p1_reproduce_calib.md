<details>
<summary>Reproduce: Calibration and Baseline</summary>

**Prerequisites:** GPU with 16+ GB VRAM (Apple Silicon MPS or CUDA). Model weights auto-downloaded from HuggingFace.

**Input:**
- `config/prompts.py` -- 210 calibration + 70 evaluation + 30 baseline questions
- `config/settings.py` -- axis definitions, generation parameters
- `config/models.py` -- model registry (HuggingFace IDs, hidden dimensions)

**Run:**
```bash
# Step 1: Calibrate axis vectors (~15 min per model on M1 Pro)
python scripts/01_calibrate.py --model qwen_7b

# Step 2: Measure default personality profile (~5 min)
python scripts/02_baseline.py --model qwen_7b

# Step 3: Validate with systematic manipulation (~10 min)
python scripts/03_benchmark.py --model qwen_7b

# Or all at once:
make pipeline MODEL=qwen_7b
```

**Output:**
- `data/axes/qwen_7b_axes.npz` -- 7 axis direction vectors (shape: 7 x 3584)
- `data/axes/qwen_7b_axes_meta.json` -- validation accuracy, d-prime, cosine similarity per axis
- `data/baselines/qwen_7b.json` -- 7-axis personality profile (mean, std, per-question projections)
- `data/benchmarks/qwen_7b.json` -- directional validation (does "be warm" shift warm_cold positive?)

**Validation:**
- `qwen_7b_axes_meta.json`: val_accuracy >= 85% on all 7 axes (typically 95-100%)
- `qwen_7b_axes_meta.json`: d_prime >= 2.0 on all axes (good separation)
- `qwen_7b.json`: baseline scores between -1 and +1, non-trivial std (> 0.05)

**Pre-computed:** All calibrated axes and baselines for 5 main models + variants are included in `data/`. Skip to `make analysis` for CPU-only exploration.

</details>
