## Base vs. Instruct Models

Do aligned models lose their personality, or learn to hide it? The evidence points to a third option: **instruction tuning sharpens personality representations rather than erasing or suppressing them.**

**Representational separability survives alignment.** For 75 held-out items (5 axes × 15 questions), we project hidden states from positive-pole vs negative-pole instructions onto each model's own calibrated axis. Separability is near-ceiling in *both* versions of all three families tested — 93–97% base vs 100% instruct (Llama-8B, Qwen-7B, Mistral-7B), with no McNemar test reaching significance. Simple suppression (which predicts collapse toward chance in instruct models) is ruled out.

**The axis direction pre-dates alignment.** Cross-calibration is the sharper test: an axis fit purely on a *base* model's hidden states still separates the corresponding *instruct* model's held-out states almost perfectly (**97.3%**), while the reverse transfer is noticeably weaker (80.4%, below even base models' own 90.3% within-model accuracy). Mean cosine between corresponding base and instruct axis vectors is 0.44. The asymmetry is the informative part: instruction tuning *sharpens and cleans* a direction that already exists in the base model rather than creating it — an instruct-only axis is a noisier, more idiosyncratic version of a direction the base model already encodes.

**Where a real alignment effect does show up:**
- Base models are least reliable specifically on *stylistic-register* axes: every base model's single weakest validation axis is warm_cold or formal_casual (Gemma-9B formal_casual 61.3%, Llama-8B warm_cold 76.0%, Mistral-7B formal_casual 76.9%, Qwen-7B warm_cold 79.1%) — while the corresponding instruct model is at or near 100% on that same axis.
- An independent linear probe (trained directly on hidden states, no per-model axis) decodes *which personality instruction produced a given state* at ceiling from instruct representations (100% across five axes) and slightly lower from base ones (87% Llama-8B, 96% Qwen-7B) — instruction tuning *sharpens* this encoding rather than reducing it, the same direction as the stylistic-axis effect above.
- Refusal-direction ablation on Qwen 2.5 partially restores base-like axis geometry (33–50% recovery) — where alignment does shift axis geometry, the change is tied to a specific RLHF component and is partly reversible.

**What this does not show:** none of the above is *behavioral* evidence that instruct models comply with personality-directed prompts less often — that requires judged behavioral labels we have not collected (see Limitations).

<details>
<summary><b>Reproduce: base vs instruct</b></summary>

**Input:** hidden states + axes for the base/instruct pairs (`llama_8b`/`llama_8b_base` etc.), produced by `01_calibrate.py`.

**Run:**
```bash
python3 scripts/06f_probe_stats.py          # item-level separability + McNemar (75 items)
python3 scripts/06g_cross_calibration.py    # base<->instruct axis transfer
```

**Output:** `data/analysis/probe_stats_75q.json`, `data/analysis/cross_calibration.json`.

**Pre-computed:** `v2/data/analysis/probe_stats_75q.json`, `v2/data/analysis/cross_calibration.json`.

</details>
