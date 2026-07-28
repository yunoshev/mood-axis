## Steering (Part 2)

Part 1 established that personality is linearly encoded in hidden states. Part 2 asks whether those representations are *causal*: add the calibrated axis vector, scaled by a coefficient α, to the hidden states at a mid-network layer during generation —

```
h'_l = h_l + α · (v_axis × h_norm)
```

— and the generated text shifts along the targeted axis. The `h_norm` factor (median hidden-state L2 norm at the target layer) makes α comparable across architectures whose hidden-state norms differ by up to 5000×. Injection layers are selected per model by an empirical sweep (optimal layers cluster in the 25–65% depth range, consistent with middle layers encoding stylistic properties).

The effect is deliberately subtle: steered text shifts in tone and style while staying factually and topically intact — warm-steered text picks up exclamation marks, second-person address and positive framing; cold-steered text stays factually identical but tonally neutral. That subtlety is the point: it confirms the axes capture genuine internal tendencies rather than surface templates.

**Try it live:** the [Steering Playground](https://yunoshev.github.io/mood-axis/) shows real steered generations — drag the α slider and watch the same answer change temperament.

> **Data status:** the quantitative steering results (dose-response slopes, per-model steerability, external judge validation) are being **recomputed under a strict zero-truncation protocol** — earlier runs capped generation at 384 tokens, which truncated long steered responses and could bias per-axis effects. Updated numbers, figures, and playground texts will land in the next data update; the qualitative picture above is unaffected.

<details>
<summary><b>Reproduce: steering</b></summary>

**Input:** calibrated axes (`data/axes/<model>_axes.npz`) + layer-sweep result (`data/steering_sweep/<model>_sweep.json`).

**Run:**
```bash
# GPU: layer sweep (picks the injection layer)
python3 scripts/steering_layer_sweep.py --model qwen3.5_9b --backend mlx

# GPU: dose-response over alphas x questions (checkpointed per question, resumable)
python3 scripts/steering_basic.py --model qwen3.5_9b --backend mlx
#   --axes warm_cold          one axis
#   --analyze                 recompute stats from saved generations (CPU)

# Optional: external LLM judge (needs ANTHROPIC_API_KEY)
python3 scripts/steering_judge_prep.py && python3 scripts/steering_judge.py
```

**Output:** `data/steering/<model>_basic.json` — per-alpha generations, projections, slopes, leakage.

**Pre-computed:** `v2/data/steering/<model>_basic.json` (LFS) — these back the current playground; they predate the zero-truncation rerun noted above.

</details>
