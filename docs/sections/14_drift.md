## Conflict Drift

Does personality *move* under pressure? We track axis projections through 50 adversarial multi-turn scenarios (10 categories — sarcasm, passive aggression, hostility escalation, … — 12 conflict turns + 4 recovery turns each) against **30 neutral control conversations of matched length**. An axis counts as showing *conflict-specific* drift only when its conflict-phase slope (a) has a 95% bootstrap CI excluding zero **and** (b) exceeds 2× the control slope — separating drift caused by hostility from drift that is simply an artifact of long conversations.

**Six of the nine models show at least one significant axis — and the pattern splits sharply by scale:**

| Model | Axis | Conflict slope [95% CI] | Control slope | Recovery |
|---|---|---|---|---|
| Qwen 3.5 9B | empathetic_analytical | +0.006 [+0.003, +0.009] | −0.000 | 62% |
| Qwen 3.5 9B | confident_cautious | +0.006 [+0.001, +0.009] | −0.000 | 48% |
| Phi-4 | warm_cold | +0.005 [+0.001, +0.009] | −0.003 | 31% |
| Qwen3.6 27B | warm_cold | −0.012 [−0.015, −0.009] | −0.002 | 30% |
| Qwen3.6 27B | confident_cautious | +0.018 [+0.014, +0.021] | +0.003 | 41% |
| Qwen3.6 27B | formal_casual | +0.010 [+0.007, +0.012] | −0.001 | 47% |
| Qwen3.6 27B | verbose_concise | −0.009 [−0.014, −0.005] | −0.003 | 36% |
| Mistral Small 3.2 24B | confident_cautious | +0.014 [+0.010, +0.017] | +0.004 | 40% |
| Qwen3.6-35B-A3B | warm_cold | −0.013 [−0.016, −0.010] | −0.003 | 38% |
| Qwen3.6-35B-A3B | confident_cautious | +0.024 [+0.019, +0.029] | +0.004 | 48% |
| Qwen3.6-35B-A3B | empathetic_analytical | −0.009 [−0.012, −0.005] | −0.002 | 52% |
| Qwen3.6-35B-A3B | formal_casual | +0.012 [+0.009, +0.015] | +0.001 | 48% |
| Gemma 4 26B-A4B | warm_cold | −0.012 [−0.015, −0.010] | −0.003 | 47% |
| Gemma 4 26B-A4B | confident_cautious | +0.020 [+0.017, +0.023] | +0.006 | 63% |

![Conflict-specific drift: conflict slopes with 95% CIs vs matched-control slopes](docs/figures/fig_drift.png)

**Under sustained attack the >20B models harden: colder, more confident, more formal.** Every significant warm_cold slope in the >20B cohort is negative (−0.012 to −0.013), every significant confident_cautious slope is positive (+0.014 to +0.024), both significant formal_casual slopes are positive. Recovery is partial everywhere (30–63%): after the pressure lifts, the models drift back toward baseline but don't reach it within four recovery turns.

**The smaller models de-escalate instead.** Qwen 3.5 9B shifts toward the *empathetic* pole under conflict (+0.006, best recovery in the small cohort at 62%). Phi-4, Ministral 3 8B, and DeepSeek-R1 move the same way (+0.006 to +0.013), and the peak-conflict texts confirm it — Phi-4's projections climb from −0.09 to +0.31…+0.41 on turns opening with "I'm so glad to hear…" and "I apologize if my previous responses didn't meet your expectations…". These three don't pass the composite criterion for an instructive reason: their *control* conversations drift the opposite way (toward analytical), so the magnitude-only 2× clause fails even though the conflict-vs-control difference is highly significant by permutation test (p ≤ 0.0008, FDR-surviving). What hostility elicits from smaller models is conciliatory de-escalation, not analytical detachment.

**Neutral controls drift too — matching for it matters.** With 30 controls, control slopes are often reliably nonzero: Ministral's confident_cautious rises at +0.018/turn in *neutral* conversations (vs +0.019 under conflict — conversation length, not hostility, drives it). Conversation length alone is an active confound; that's exactly what the control condition and the 2× clause absorb.

**GPT-OSS 20B: large movement, none of it conflict-specific.** Its conflict slopes reach ±0.03 — but its *control* slopes are even larger (confident_cautious: conflict −0.026 vs control **+0.065**). Neutral small talk moves GPT-OSS's projections as much as sustained attack, so nothing survives the criterion: its instability is a standing representational property, not a reaction to hostility.

**Multiplicity check.** Applying Benjamini–Hochberg FDR (q=0.05) to permutation p-values across all 44 model × axis tests: 12 of the 14 composite-significant axes survive (the two that don't owe their composite pass to near-zero control slopes and are treated as tentative); 26 of 44 tests show FDR-significant conflict-vs-control *differences*. The composite verdicts are insensitive to the arbitrary 2× ratio choice (13 vs 17 axes at 3× / 1.5×).
