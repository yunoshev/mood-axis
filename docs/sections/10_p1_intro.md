---

## Part 1: Measuring LLM Personality

Ask the same question to five different language models and you'll get five different *tones* -- one hedges, another is blunt, a third is warm. These aren't random: they're consistent behavioral tendencies baked in by training data and alignment procedures.

Part 1 develops a method to **measure** these tendencies using hidden state representations. We define 7 bipolar personality axes (warm/cold, confident/cautious, etc.), calibrate direction vectors from contrastive instructions, and project neutral responses onto these axes to produce a quantitative personality fingerprint for each model.

We validate the method at three levels: held-out accuracy (do the axes discriminate?), test-retest stability (do they reproduce?), and base-vs-instruct comparison (where does personality come from?). The answer to the last question is surprising: RLHF doesn't suppress personality -- it teaches models to *resist* personality instructions while preserving the underlying signal in hidden states.
