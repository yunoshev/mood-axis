### Instruction Resistance: How RLHF Affects Personality

**Core question:** Are personality differences due to signal loss (suppression) or learned resistance to being steered?

We apply identical methodology to both base and instruct models. Each receives the same calibration questions with explicit personality instructions ("Respond warmly" vs "Respond coldly"). We measure: does the model follow the instruction?

#### Instruction Following Accuracy

{{table_instruction_accuracy}}

**Per-axis breakdown for Llama 8B (most extreme case):**

{{table_instruction_accuracy_llama_breakdown}}

**Statistical details:**
- **Llama 8B:** Base 87% -> Instruct 63%, McNemar chi2=9.84, **p=0.002** (significant after Bonferroni correction)
- **Qwen 7B:** Base 88% -> Instruct 97%, p=0.16 (not significant)
- **Mistral 7B:** Base 84% -> Instruct 81%, p=0.62 (not significant)

**Interpretation:** Llama's base model achieves high accuracy following personality instructions -- the signal exists. The instruct model selectively ignores them. This is not uniform suppression (which would show ~50% on both), but instruction resistance. Organizations tune this differently: Meta suppresses, Alibaba preserves.

#### Refusal Direction Effect

To test whether instruction resistance is specifically due to RLHF, we compare Qwen 2.5 across three versions: base (no alignment), instruct (full alignment), and uncensored (alignment with refusal direction removed).

{{table_refusal_direction_effect}}

Removing the refusal direction partially restores base-like behavior (33-50% recovery across axes). This proves instruction resistance is **directional** (aligned with safety objectives), **component-specific** (safety training responsible), and **reversible** (learned constraints, not architectural changes).

#### Decoding Probes: Is the Signal Suppressed or Ignored?

{{decoding_probes_summary_table}}

All models show >75% accuracy on base versions (signal present) and >50% on instruct versions (signal not suppressed -- just ignored). The base-to-instruct drop (-23% Llama, -11% Qwen) mirrors instruction resistance measured above.

**Conclusion:** RLHF doesn't erase personality signal from hidden states. Models learn to *ignore* personality-directed instructions, not to *erase* the underlying signal.

#### Steering Override: Can Amplification Break Through?

If instruction resistance is learned (not architectural), amplifying personality instructions should increase compliance.

{{experiment_d_table_results}}

The verbose_concise axis -- most suppressed at -39% in baseline -- shows the largest override: 40% -> 100% at amplification level 1. This graded, selective override proves RLHF creates learned behavioral constraints, not architectural limitations.
