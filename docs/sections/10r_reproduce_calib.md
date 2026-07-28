<details>
<summary><b>Reproduce: calibration</b></summary>

**Prerequisites:** 16+ GB VRAM/unified memory for ≤14B models (Apple Silicon via `--backend mlx`, or CUDA/CPU via the default PyTorch backend). Weights auto-download from HuggingFace.

**Input:**
- `config/prompts.py` — calibration / evaluation / baseline question sets
- `config/settings.py` — axis definitions, generation parameters
- `config/models.py` — model registry (HF IDs, per-model quirks)

**Run:**
```bash
python3 scripts/01_calibrate.py --model qwen3.5_9b --backend mlx
# Useful flags:
#   --axes warm_cold,verbose_concise   subset of axes
#   --check                            preflight only, no GPU
#   --time-budget-min 25               clean checkpoint-exit after N minutes (rerun to resume)
```

Calibration is resumable by default: it checkpoints after every sample and skips already-computed axes. `--force` recomputes from scratch.

**Output:**
- `data/axes/qwen3.5_9b_axes.npz` — 5 axis vectors + per-axis normalization center/scale
- `data/axes/qwen3.5_9b_axes_meta.json` — val_accuracy, val_dprime per axis
- `data/axes/qwen3.5_9b_hidden_states.npz` — archived per-layer states (reused by the CPU analyses)

**Validation:** `val_accuracy ≥ 0.85` on all 5 axes for instruct models (typically 1.00).

**Pre-computed:** axis metadata for the full cohort ships in `v2/data/axes_meta/`.

</details>
