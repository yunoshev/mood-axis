## Key Findings

1. **Distinct, stable behavioral fingerprints.** Each model has a unique personality profile that persists across sessions (ICC {{icc_range}}). Mistral is cold, Qwen is cautious, Gemma is neutral -- these aren't random; they're baked in by training.

2. **RLHF appears to train instruction resistance rather than suppression.** Base models follow personality instructions at {{llama_8b_base_instruction_acc}} accuracy; aligned models drop to {{llama_8b_instruction_acc}} (Llama). But the signal *persists* in hidden states -- decoding probes still detect it at >50% accuracy. Models learn to *ignore* personality instructions, not to erase the underlying representation.

3. **Instruction resistance is reversible.** Removing the refusal direction partially restores base-like behavior (33-50% recovery). Amplifying personality instructions breaks through suppression (40% -> 100% on verbose_concise). RLHF creates learned constraints, not architectural limitations.

4. **~40x variation in steerability.** {{steering_most_steerable}} ({{steering_most_steerable_effect}}) vs {{steering_least_steerable}} ({{steering_least_steerable_effect}}). Same method, same axes, vastly different susceptibility. External judge confirms: {{judge_mistral_7b_pairwise}} detection for Mistral, {{judge_gemma_9b_pairwise}} for Gemma.

5. **Dead zones show no significant correlation with steerability** (n=5, rho={{steering_dz_rho}}), though the sample is small. A model can exhibit neutral behavior yet be highly responsive to hidden-state intervention (Mistral), or show varied personality but resist steering entirely (Qwen3).

6. **Organization-specific alignment strategies.** Meta heavily suppresses personality (-24% instruction accuracy). Alibaba preserves it (+9%). Mistral is balanced (-3%). These likely reflect different alignment priorities.
