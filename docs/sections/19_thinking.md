## Do Models Think with the Temperament They Answer With?

Reasoning models expose a second register: the reasoning segment generated before the visible answer. Projecting think and answer segments separately (30 questions, paired by question, bootstrap 95% CIs), we pre-registered that thinking would be *colder and more analytical* than answering — "performative warmth". **The hypothesis fails in its universal form**, and the cohort splits three ways:

| Model | think − answer: warm_cold | confident_cautious | empathetic_analytical |
|---|---|---|---|
| DeepSeek-R1 14B | **+0.28** [+0.24, +0.33] | **−0.25** [−0.31, −0.18] | **+0.24** [+0.21, +0.26] |
| GPT-OSS 20B | **−0.41** [−0.54, −0.29] | **+0.39** [+0.29, +0.49] | **+0.24** [+0.14, +0.33] |
| GLM-Z1 9B | +0.08 [+0.06, +0.10] | **−0.15** [−0.18, −0.12] | +0.08 [+0.07, +0.10] |
| VibeThinker 3B | −0.10 [−0.13, −0.07] | **+0.65** [+0.61, +0.69] | **−0.25** [−0.27, −0.23] |
| Ministral 3 8B Reasoning | +0.01 [−0.00, +0.03] | −0.09 [−0.13, −0.06] | +0.02 [+0.01, +0.04] |

DeepSeek thinks *warmer* and more empathetically than it answers — the mirror image of the hypothesis. GPT-OSS thinks colder yet *more* empathetically. VibeThinker is the closest match but couples it with a huge confidence surge while thinking (+0.65, the largest effect in the study). The one near-regularity runs through confidence: **three of five models deliberate more cautiously than they deliver** — hedged reasoning behind a confident answer. Thinking temperament, like conflict drift, is a per-model property, not a law of the reasoning regime.

<details>
<summary><b>Reproduce: think vs answer</b></summary>

**Run:**
```bash
python3 scripts/07_thinking.py --model deepseek_r1_14b --backend mlx   # GPU: paired generations
python3 scripts/06l_think_stats.py                                     # CPU: paired contrasts + CIs
```

**Output:** `data/analysis/think_stats.json`. **Pre-computed:** `v2/data/analysis/think_stats.json`.

</details>
