## Steering (Part 2)

Part 1 established that personality is linearly encoded in hidden states. Part 2 asks whether those representations are *causal*: add the calibrated axis vector, scaled by a coefficient α, to the hidden states at a per-model mid-network layer during generation —

```
h'_l = h_l + α · STEER_SCALE · h_norm · v_axis     (STEER_SCALE = 0.01)
```

— and the generated text shifts along the targeted axis. The `h_norm` factor (median hidden-state L2 norm at the target layer) makes α comparable across architectures whose hidden-state norms differ by up to 5000×; with `STEER_SCALE = 0.01`, α = 1 is a 1% perturbation of the hidden-state norm. Injection layers are chosen per model by an empirical sweep, which also fixes each model's *safe* α range. We sweep α over that range (up to ±50% of the norm) at 7 values × 5 axes × 30 questions = **5,250 steered generations** across 5 models.

**Steerability varies ~6× across models — with a dense-model coherence limit.** Four of the five models are reliably steerable: DeepSeek-R1 14B (total effect **3.58**, the strongest), Ministral 3 8B (2.07), Qwen3.5 9B (1.62), and the mixture-of-experts GPT-OSS 20B (1.61) — all staying coherent across the full ±50% range. The dense **Phi-4 14B** is the exception (0.60): beyond ±30% its generations degenerate into repetition and emoji loops rather than cleanly shifting the trait, so it can only be steered over a narrower coherent range. Its lower steerability is inseparable from that lower robustness — a coherence limit, not simple immovability.

**External validation.** A blind LLM judge — shown steered responses at the extremes of each model's range and asked which better expresses the trait, plus a 1–5 dose-response rating — confirms the probe measurements: **94% mean pairwise accuracy** across all 25 model × axis batches (mean Likert ρ = 0.68). The robust models are near-perfect (Qwen3.5 9B and Ministral 3 8B at 100% pairwise), while Phi-4 sits at 83% — on some axes its steered outputs are not reliably distinguishable, precisely because near its coherence limit it degenerates rather than shifting cleanly. Cross-axis leakage is modest for every model (mean ratios 0.40–0.71, all < 1.0): steering predominantly moves the targeted axis rather than spilling onto others.

The effect is deliberately subtle: steered text shifts in tone and style while staying factually and topically intact — warm-steered text picks up exclamation marks, second-person address and positive framing; cold-steered text stays factually identical but tonally neutral. That subtlety is the point: it confirms the axes capture genuine internal tendencies rather than surface templates.

**Try it live:** the [Steering Playground](https://yunoshev.github.io/mood-axis/) shows real steered generations — drag the α slider and watch the same answer change temperament.

> **Data status:** the steering results, figures, and playground texts above were recomputed under a strict **zero-truncation protocol** — per-axis token budgets of 4096–8192, replacing an earlier 384-token cap that truncated long steered responses and biased per-axis effects. The five core models are on the rerun data; the 4-bit quantized twins in the playground retain the earlier runs.

<details>
<summary><b>Reproduce: steering</b></summary>

**Input:** calibrated axes (`data/axes/<model>_axes.npz`) + layer-sweep result (`data/steering_sweep/<model>_sweep.json`).

**Run:**
```bash
# GPU: layer sweep (picks the injection layer + safe alpha range)
python3 scripts/steering_layer_sweep.py --model qwen3.5_9b --backend mlx

# GPU: dose-response over alphas x questions (checkpointed per question, resumable)
python3 scripts/steering_basic.py --model qwen3.5_9b --backend mlx
#   --axes warm_cold          one axis
#   --analyze                 recompute stats from saved generations (CPU)

# CPU: cross-model summary (steerability, leakage, clusters)
python3 scripts/steering_analyze.py

# Optional: blind external LLM judge (zero-truncation, keys hidden from judge)
python3 scripts/steering_judge_prep.py       # build blind batches + answer keys
#   -> judge each batch with an LLM, writing data/judge_steering/verdicts/<m>_<axis>.json
python3 scripts/steering_judge_analyze.py    # score verdicts -> judge_results.json
```

**Output:** `data/steering/<model>_basic.json` (per-alpha generations, projections, slopes, leakage), `data/analysis/steering_summary.json`, `data/judge_steering/judge_results.json`.

**Pre-computed:** `v2/data/steering/<model>_basic.json`, `v2/data/analysis/steering_summary.json`, `v2/data/judge_steering/judge_results.json` (LFS) — these back the playground and the numbers above.

</details>
