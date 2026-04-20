---

## Part 2: Steering LLM Personality

Part 1 established that personality is linearly encoded in hidden states -- each model has a measurable fingerprint along calibrated axes. But are these representations merely *descriptive*, or are they *causal*?

Part 2 tests this directly: we add scaled axis vectors to hidden states during generation and measure whether model personality actually shifts. If warm/cold is a meaningful direction in representation space, then injecting the "warm" direction should make the model warmer.

The results reveal a striking 100x variation in steerability across models -- from Mistral 7B (highly responsive) to Qwen3 8B (essentially immovable) -- and an unexpected finding: the dead zones identified in Part 1 do not predict steerability at all. Measurement and control are orthogonal properties of the representation.
