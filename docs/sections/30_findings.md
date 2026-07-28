## Key Findings

1. **Distinct, stable behavioral fingerprints.** Each model has a characteristic per-axis profile, reproducible across independent calibration sets (cross-set consistency 0.80–0.92 for eight of nine cohort models). No aligned assistant presents as warm by default; confidence is where organizations differ most (DeepSeek +0.46 vs Gemma 4 −0.32).

2. **Conflict drift splits by scale.** Under sustained adversarial pressure, the >20B models harden — colder, more confident, more formal (six of nine models show conflict-specific drift; all four >20B models are among them) — while smaller models de-escalate toward the empathetic pole with apologetic, conciliatory text. Recovery is partial everywhere (30–63% within four post-conflict turns).

3. **Neutral conversations drift too.** With 30 matched controls, control-condition slopes are often reliably nonzero — conversation length alone moves personality projections. Any drift claim without a matched-length control is confounded.

4. **The GPT-OSS anomaly is format dominance, not MoE or quantization.** Its degenerate axis geometry and weakest-in-cohort consistency (0.39) trace to a dominant response-format reaction to persona prompts; projecting out a low-rank format subspace restores held-out separation from 0.73 to 1.00 on its worst axis. Two larger MoE models sit squarely in the stable cohort, and axis directions are precision-invariant across bf16/4-bit pairs (mean cos 0.95).

5. **Instruction tuning reorganizes personality representations without erasing them.** Own-axis separability is near-ceiling in both base and instruct models (93–97% vs 100%); a base-fit axis transfers to instruct states at 97.3% — the direction pre-dates alignment. The real alignment effect is subtler: reduced representational legibility and base-model unreliability on stylistic-register axes specifically.

6. **Reasoning models think with a different temperament than they answer — in model-specific directions.** DeepSeek thinks warmer, GPT-OSS colder, VibeThinker with a +0.65 confidence surge; the closest regularity is hedged deliberation behind confident delivery (3 of 5 models).

7. **The same vectors measure and steer.** Adding calibrated axis vectors to hidden states shifts generation along the targeted axis while preserving content — measurement directions are causally meaningful (quantitative steerability comparison being recomputed; see [Steering](#steering-part-2)).
