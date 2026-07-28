## The GPT-OSS Anomaly: Format Dominance

A dedicated confound analysis identifies what GPT-OSS's competing off-axis direction actually is: **response format**. On persona calibration prompts, GPT-OSS changes *how much and how* it writes far more than it changes style-in-content:

- "empathetic" responses average **529 tokens vs 865** for "analytical" (DeepSeek: 101 vs 142);
- "warm" responses **1075 tokens vs 400** for "cold";
- the analytical pole is **100% markdown-structured documents vs 54%** for the empathetic pole;
- safety refusals play no role (2 refusals in 650 calibration responses — fewer than stable models like Gemma 4, at 19).

As a result, three of its five axis directions substantially share the length/verbosity direction — cosine with its own verbose_concise axis: 0.57 (warm_cold), −0.63 (confident_cautious), −0.65 (empathetic_analytical) — versus |cos| ≤ 0.26 for every content axis of DeepSeek and Gemma 4.

**Projecting out a low-rank format subspace** (the verbose_concise contrast direction + top-3 PCs of its variation) **restores held-out pair separation on empathetic_analytical from 0.73 to 1.00** and raises cross-set axis agreement 0.70→0.81 (warm_cold), 0.37→0.51 (confident_cautious), 0.62→0.71 (empathetic_analytical) — while the identical operation leaves DeepSeek's and Gemma 4's axes unchanged (±0.03). A genuine temperament signal exists in GPT-OSS; it is masked by a dominant, *noisy* format response to persona instructions.

**The check generalizes — and is nearly free.** Run cohort-wide, it separates "axis direction contains format" (common, mostly benign: 6 of 46 models exceed the |cos| ≥ 0.4 flag, but all except GPT-OSS still pass held-out validation at 1.00) from "axis is *destabilized* by format" (GPT-OSS only: there the flagged direction, a strong projection–length correlation r = −0.74, degraded held-out separation, and depressed consistency all co-occur, and only there does removing the format subspace repair them). It needs no extra generation — only artifacts every axis fit already produces. We recommend it as a standard sanity check for contrastive persona-direction methods.

**Robustness of the drift results:** re-running the full drift significance machinery on format-partialled axes changes no verdict — every significant axis stays significant, six axes *gain* significance (format variance was masking signal, not creating it), and GPT-OSS's large-but-nonspecific movements collapse toward zero, confirming most of what moved for that model was format.
