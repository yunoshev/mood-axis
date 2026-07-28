## Quick Start

Everything runs standalone — plain `python3`, no job framework, no API keys (except the optional LLM-judge step).

```bash
git clone https://github.com/yunoshev/mood-axis.git
cd mood-axis
pip install -r requirements.txt

# Measure a model's personality (GPU or Apple Silicon; ~30-60 min per model)
python3 scripts/01_calibrate.py --model qwen3.5_9b --backend mlx   # calibrate 5 axis vectors
python3 scripts/02_baseline.py  --model qwen3.5_9b --backend mlx   # 30-question fingerprint

# Or via make:
make pipeline MODEL=qwen3.5_9b

# CPU-only: recompute the cross-model statistics from the shipped data
python3 scripts/06e_drift_stats.py        # drift slopes + significance
python3 scripts/06d_axis_geometry.py      # axis cosines, PCA, concept-cone rank
python3 scripts/07_figures.py             # all figures
```

- `--backend mlx` uses Apple MLX (2–5× faster on Apple Silicon; requires `pip install mlx mlx-lm`). Default backend is PyTorch (CUDA/CPU).
- Every GPU script supports `--check` for a preflight validation without loading the model.
- Model keys come from `config/models.py` (e.g. `qwen3.5_9b`, `phi4`, `deepseek_r1_14b`, `ministral3_8b`, `gpt_oss_20b`).
- Pre-computed outputs for the full cohort ship in this repo (see [Data Files](#data-files)), so every analysis step runs without any GPU time.
