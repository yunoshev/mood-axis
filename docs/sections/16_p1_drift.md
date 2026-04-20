### Conflict Drift

We test personality stability under stress: {{conflict_scenarios}} adversarial conflict scenarios, {{conflict_turns_conflict}} turns of escalating disagreement, then {{conflict_turns_recovery}} recovery turns.

{{table_drift}}

![Conflict Drift](docs/figures/drift.png)

- **GPT-OSS 20B** and **Gemma 2 9B** are the most resilient -- stay consistent under adversarial pressure
- **Mistral 7B** drifts the most -- proactive_reluctant axis shifts by {{drift_mistral_7b_proactive_reluctant_delta}} (withdraws significantly)
- **Recovery:** Most models return to baseline within 4 neutral turns. Mistral is slowest to recover.
- **Control conversations** (no conflict) show near-zero drift, confirming that observed shifts are caused by adversarial content, not conversation length.
