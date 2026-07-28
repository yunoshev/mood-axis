## Limitations

1. **English-only dataset.** All questions are LLM-generated English; axis transfer to other languages is untested.
2. **No human validation of axis labels.** Axes are operationally defined through hidden-state separability, not human annotation.
3. **Style instructions carry lexical markers.** Some pole instructions suggest concrete phrasings ("I could be wrong, but…"), so axis vectors may partly encode those markers; confident_cautious is most exposed.
4. **verbose_concise partly measures length** (r ≈ 0.68 with raw response length) — part of what it captures is a length detector, not a purely stylistic disposition.
5. **Calibration poles are not length-matched.** Persona instructions are free to change response format along with style; for GPT-OSS this format response dominates the contrast. The format-confound check diagnoses and largely repairs this post-hoc; a length-matched calibration design is future work.
6. **Single chat template, single decoding regime, single run per model** (temperature 0.7, fixed seed). Bootstrap CIs cover question sampling, not run-to-run variance.
7. **Conflict scenarios are scripted, not interactive.** Every model faces literally the same pressure sequence (guaranteeing comparability), but a scripted accuser that ignores the model's apology is not a realistic dialogue partner; matched-length scripted controls bound but don't remove this concern.
8. **Gemma 3 12B excluded for steering degeneration.** Under hidden-state steering it collapses into repetition loops (a known sensitivity of the Gemma 3 12B/27B generation); available mitigations would all break cross-model protocol comparability, so it is excluded from all reported results rather than measured on degenerate text. None of the other eight current-generation models shows this failure mode under the identical protocol.
9. **No behavioral validation of instruction-following was collected.** The base-vs-instruct results are representational; whether they translate into different compliance rates with personality-directed prompts in generated text remains open.
10. **One instruction phrasing per pole.** Cross-set consistency shows robustness to changing the *questions*; robustness to paraphrasing the *instructions* is untested.
