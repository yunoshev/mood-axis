"""Unified model configuration for Mood Axis.

Single source of truth for model identifiers, display names, and settings.
"""

from dataclasses import dataclass
from typing import Optional
import os


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    model_id: str           # HuggingFace model ID
    model_short: str        # Short identifier for filenames
    display_name: str       # Human-readable name for charts/tables
    color: str              # Hex color for visualizations
    requires_auth: bool     # Whether HF token is needed
    hidden_dim: int         # Hidden state dimension (for validation)
    style_overrides: Optional[str] = None
    is_base_model: bool = False
    instruct_counterpart: Optional[str] = None
    chat_template_kwargs: Optional[dict] = None
    quantize: Optional[str] = None       # MLX precision: None (native), "8bit", "4bit"
    mlx_model_id: Optional[str] = None   # Explicit MLX repo override (e.g. mlx-community 8-bit)
    # Real no-think fix (2026-07-23): chat_template_kwargs={"enable_thinking": False}
    # is a no-op for several reasoning models -- their chat template has no
    # enable_thinking conditional at all, so the model still emits a reasoning
    # segment regardless (byte-identical think_on/off generations; see
    # data/analysis/gptoss_probe_check.json). When set, this is spliced onto
    # the END of the rendered prompt (src/model/mlx_inference.py /
    # src/model/inference.py::_apply_chat_template) whenever chat_template_kwargs
    # requests enable_thinking=False -- a pre-"closed" empty think block (or,
    # for Harmony/GPT-OSS, a forced channel switch) so the model starts
    # generating past the reasoning segment instead of inside it.
    no_think_prefill: Optional[str] = None


MODELS = {
    # --- Article models (7-9B) ---
    "qwen_7b": ModelConfig(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        model_short="qwen_7b",
        display_name="Qwen 2.5 7B",
        color="#FF6B35",
        requires_auth=False,
        hidden_dim=3584,
    ),
    "mistral_7b": ModelConfig(
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        model_short="mistral_7b",
        display_name="Mistral 7B",
        color="#A855F7",
        requires_auth=False,
        hidden_dim=4096,
    ),
    "deepseek_7b": ModelConfig(
        model_id="deepseek-ai/deepseek-llm-7b-chat",
        model_short="deepseek_7b",
        display_name="DeepSeek 7B",
        color="#00D9A5",
        requires_auth=False,
        hidden_dim=4096,
    ),
    "llama_8b": ModelConfig(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        model_short="llama_8b",
        display_name="Llama 3.1 8B",
        color="#3B82F6",
        requires_auth=True,
        hidden_dim=4096,
    ),
    "gemma_9b": ModelConfig(
        model_id="google/gemma-2-9b-it",
        model_short="gemma_9b",
        display_name="Gemma 2 9B",
        color="#4285F4",
        requires_auth=True,
        hidden_dim=3584,
    ),

    # --- V6 models ---
    "qwen3_8b": ModelConfig(
        model_id="Qwen/Qwen3-8B",
        model_short="qwen3_8b",
        display_name="Qwen3 8B",
        color="#FF4500",
        requires_auth=False,
        hidden_dim=4096,
        chat_template_kwargs={"enable_thinking": False},
    ),
    "phi4": ModelConfig(
        model_id="microsoft/phi-4",
        model_short="phi4",
        display_name="Phi-4 14B",
        color="#7B68EE",
        requires_auth=False,
        hidden_dim=5120,
    ),

    # --- Thinking models ---
    "deepseek_r1_14b": ModelConfig(
        model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        model_short="deepseek_r1_14b",
        display_name="DeepSeek-R1 14B",
        color="#00D9A5",
        requires_auth=False,
        hidden_dim=5120,
        # no_think_prefill=True: GLOBAL activation (smoke-verified 2026-07-23,
        # data/smoke/no_think_deepseek_r1_14b.json) -- the whole main pipeline
        # runs with thinking really off. Pre-recompute data (calibrated with
        # thinking silently ON) is preserved in git history.
        chat_template_kwargs={"enable_thinking": False,
                              "no_think_prefill": True},
        # Verified 2026-07-23 against the live HF chat_template: it has no
        # enable_thinking conditional and always renders
        # "<｜Assistant｜><think>\n" for the generation prompt -- the flag above
        # is a no-op. _apply_chat_template's existing bare-"<think>" strip
        # removes that auto-opened tag; this prefill re-closes it empty.
        no_think_prefill="<think>\n\n</think>\n\n",
    ),
    "glm_z1_9b": ModelConfig(
        model_id="zai-org/GLM-Z1-9B-0414",
        model_short="glm_z1_9b",
        display_name="GLM-Z1 9B",
        color="#FF8C00",
        requires_auth=False,
        hidden_dim=4096,
        # no_think_prefill=True: GLOBAL activation (see deepseek_r1_14b note).
        chat_template_kwargs={"enable_thinking": False,
                              "no_think_prefill": True},
        # Verified 2026-07-23 against the live HF chat_template: it has no
        # enable_thinking conditional; add_generation_prompt always renders
        # "<|assistant|>\n<think>" (same no-op pattern as DeepSeek-R1 above).
        no_think_prefill="<think>\n\n</think>\n\n",
    ),

    # --- Other ---
    "gemma3_12b": ModelConfig(
        model_id="google/gemma-3-12b-it",
        model_short="gemma3_12b",
        display_name="Gemma 3 12B",
        color="#34A853",
        requires_auth=True,
        hidden_dim=3840,
    ),

    # --- Gemma 3n (AltUp architecture) ---
    "gemma3n_e2b": ModelConfig(
        model_id="google/gemma-3n-E2B-it",
        model_short="gemma3n_e2b",
        display_name="Gemma 3n E2B",
        color="#0F9D58",
        requires_auth=True,
        hidden_dim=2048,
    ),
    "gemma3n_e4b": ModelConfig(
        model_id="google/gemma-3n-E4B-it",
        model_short="gemma3n_e4b",
        display_name="Gemma 3n E4B",
        color="#137333",
        requires_auth=True,
        hidden_dim=2048,
    ),

    # --- Gemma 4 (April 2026, multimodal) ---
    "gemma4_e4b": ModelConfig(
        model_id="google/gemma-4-E4B-it",
        model_short="gemma4_e4b",
        display_name="Gemma 4 E4B",
        color="#FBBC05",
        requires_auth=True,
        hidden_dim=2560,
    ),

    # --- Diverse architectures ---
    "olmo2_13b": ModelConfig(
        model_id="allenai/OLMo-2-1124-13B-Instruct",
        model_short="olmo2_13b",
        display_name="OLMo-2 13B",
        color="#6A0DAD",
        requires_auth=False,
        hidden_dim=5120,
    ),
    "command_r7b": ModelConfig(
        model_id="CohereLabs/c4ai-command-r7b-12-2024",
        model_short="command_r7b",
        display_name="Command-R 7B",
        color="#39A7FF",
        requires_auth=True,
        hidden_dim=4096,
    ),
    "internlm3_8b": ModelConfig(
        model_id="internlm/internlm3-8b-instruct",
        model_short="internlm3_8b",
        display_name="InternLM3 8B",
        color="#FF1493",
        requires_auth=False,
        hidden_dim=4096,
    ),

    # --- Ministral 3 ---
    "ministral3_8b": ModelConfig(
        model_id="mistralai/Ministral-3-8B-Instruct-2512",
        model_short="ministral3_8b",
        display_name="Ministral 3 8B",
        color="#7C3AED",
        requires_auth=False,
        hidden_dim=4096,
    ),
    "ministral3_8b_base": ModelConfig(
        model_id="mistralai/Ministral-3-8B-Base-2512",
        model_short="ministral3_8b_base",
        display_name="Ministral 3 8B (base)",
        color="#7C3AED",
        requires_auth=False,
        hidden_dim=4096,
        is_base_model=True,
        instruct_counterpart="ministral3_8b",
    ),

    "ministral3_8b_reasoning": ModelConfig(
        model_id="mistralai/Ministral-3-8B-Reasoning-2512",
        model_short="ministral3_8b_reasoning",
        display_name="Ministral 3 8B Reasoning",
        color="#9D5CF5",
        requires_auth=False,
        hidden_dim=4096,
        # no_think_prefill=True: GLOBAL activation (see deepseek_r1_14b note).
        chat_template_kwargs={"no_think_prefill": True},
        # Verified 2026-07-23 against the live HF chat_template.jinja: there is
        # no enable_thinking (or any) template kwarg at all -- thinking is
        # instructed purely via a default system-prompt string that always
        # tells the model to draft "[THINK]...[/THINK]" before answering, and
        # the template has no add_generation_prompt tag to strip (Mistral
        # tekken-style: prompt just ends at "[/INST]"). This prefill splices a
        # closed empty think block directly after "[/INST]".
        no_think_prefill="[THINK]\n\n[/THINK]\n\n",
    ),
    "ministral3_14b": ModelConfig(
        model_id="mistralai/Ministral-3-14B-Instruct-2512",
        model_short="ministral3_14b",
        display_name="Ministral 3 14B",
        color="#6D28D9",
        requires_auth=False,
        hidden_dim=5120,
    ),
    "mistral_small_3.2_24b": ModelConfig(
        model_id="mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        model_short="mistral_small_3.2_24b",
        quantize="4bit",  # >20B policy 2026-07-24: deployment-realistic 4-bit (mlx-community)
        display_name="Mistral Small 3.2 24B",
        color="#5B21B6",
        requires_auth=False,
        hidden_dim=5120,
    ),

    # --- Qwen 3.6 (2026-04) ---
    "qwen3.6_27b": ModelConfig(
        model_id="Qwen/Qwen3.6-27B",
        model_short="qwen3.6_27b",
        quantize="4bit",  # >20B policy 2026-07-24 + mitigates MLX perf disease (memory pressure)
        display_name="Qwen3.6 27B",
        color="#D97706",
        requires_auth=False,
        hidden_dim=5120,
        chat_template_kwargs={"enable_thinking": False},
    ),
    "qwen3.6_35b_a3b": ModelConfig(
        model_id="Qwen/Qwen3.6-35B-A3B",
        model_short="qwen3.6_35b_a3b",
        quantize="4bit",  # >20B policy 2026-07-24 + mitigates MLX perf disease (0.5 tok/s in bf16)
        display_name="Qwen3.6 35B-A3B (MoE)",
        color="#B45309",
        requires_auth=False,
        hidden_dim=2048,
        chat_template_kwargs={"enable_thinking": False},
    ),

    # --- Gemma 4 MoE (2026-04; requires mlx_lm gemma4 load test) ---
    "gemma4_26b_a4b": ModelConfig(
        model_id="google/gemma-4-26B-A4B-it",
        model_short="gemma4_26b_a4b",
        quantize="4bit",  # >20B policy 2026-07-24; bf16 full run in git = paired comparison point
        mlx_model_id="mlx-community/gemma-4-26b-a4b-it-4bit",  # lowercase repo name (auto-resolve would guess wrong case)
        display_name="Gemma 4 26B-A4B (MoE)",
        color="#0F9D58",
        requires_auth=True,
        hidden_dim=2816,
        # Smoke 2026-07-14: model spontaneously emits a "thought" channel even
        # with enable_thinking undefined — verify no <|channel>thought markers
        # in calibration canary output; may need output-side stripping.
        chat_template_kwargs={"enable_thinking": False},
    ),

    # --- VibeThinker (WeiboAI, MIT; tiny reasoning fine-tune of Qwen2.5-3B) ---
    "vibethinker_3b": ModelConfig(
        model_id="WeiboAI/VibeThinker-3B",
        model_short="vibethinker_3b",
        display_name="VibeThinker 3B",
        color="#EC4899",
        requires_auth=False,
        hidden_dim=2048,
        # no_think_prefill=True: GLOBAL activation (see deepseek_r1_14b note).
        chat_template_kwargs={"no_think_prefill": True},
        # Verified 2026-07-23 against the live HF chat_template: it's a plain
        # Qwen2.5-style template (no enable_thinking conditional, no
        # auto-opened "<think>" tag either) -- the model spontaneously opens
        # its own "<think>" block from fine-tuning, independent of the prompt
        # (mostly no-op: gptoss_probe_check.json shows 28/30 identical
        # think_on/off generations). This prefill forces a closed empty block.
        no_think_prefill="<think>\n\n</think>\n\n",
    ),

    # --- Blocked on MLX support (see PLAN_V2 wave 3; do NOT queue until smoke passes) ---
    # gemma4_unified arch: no bf16 text-only MLX conversion yet (checked 2026-07-14)
    "gemma4_12b": ModelConfig(
        model_id="google/gemma-4-12B-it",
        model_short="gemma4_12b",
        display_name="Gemma 4 12B",
        color="#34A853",
        requires_auth=True,
        hidden_dim=3840,
    ),
    # "zaya" arch: not supported by mlx_lm at all (checked 2026-07-14)
    "zaya1_8b": ModelConfig(
        model_id="Zyphra/ZAYA1-8B",
        model_short="zaya1_8b",
        display_name="ZAYA1 8B (MoE)",
        color="#DC2626",
        requires_auth=False,
        hidden_dim=2048,
    ),

    # --- MoE ---
    "llama4_scout": ModelConfig(
        model_id="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        model_short="llama4_scout",
        display_name="Llama 4 Scout",
        color="#1877F2",
        requires_auth=True,
        hidden_dim=5120,
    ),
    "gpt_oss_20b": ModelConfig(
        model_id="openai/gpt-oss-20b",
        model_short="gpt_oss_20b",
        display_name="GPT-OSS 20B",
        color="#10A37F",
        requires_auth=False,
        hidden_dim=2880,
        # no_think_prefill=True: GLOBAL activation (smoke-verified 2026-07-23,
        # data/smoke/no_think_gpt_oss_20b.json; see deepseek_r1_14b note).
        chat_template_kwargs={"no_think_prefill": True},
        # GPT-OSS's Harmony template has no enable_thinking (or any) toggle at
        # all -- it exposes "reasoning_effort" instead, and always emits an
        # analysis-channel segment regardless. Verified 2026-07-23 against the
        # live HF chat_template.jinja: add_generation_prompt renders only
        # "<|start|>assistant" (model picks analysis vs. final itself). This
        # is NOT a closed think/no-think pair -- it force-jumps straight into
        # the final channel, skipping the analysis segment outright.
        no_think_prefill="<|channel|>final<|message|>",
    ),
    "mixtral_8x7b": ModelConfig(
        model_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        model_short="mixtral_8x7b",
        display_name="Mixtral 8x7B",
        color="#D946EF",
        requires_auth=False,
        hidden_dim=4096,
    ),

    # --- Qwen 3.5 scaling chain ---
    "qwen3.5_0.8b": ModelConfig(
        model_id="Qwen/Qwen3.5-0.8B",
        model_short="qwen3.5_0.8b",
        display_name="Qwen3.5 0.8B",
        color="#E63946",
        requires_auth=False,
        hidden_dim=1024,
        chat_template_kwargs={"enable_thinking": False},
    ),
    "qwen3.5_2b": ModelConfig(
        model_id="Qwen/Qwen3.5-2B",
        model_short="qwen3.5_2b",
        display_name="Qwen3.5 2B",
        color="#E76F51",
        requires_auth=False,
        hidden_dim=2048,
        chat_template_kwargs={"enable_thinking": False},
    ),
    "qwen3.5_4b": ModelConfig(
        model_id="Qwen/Qwen3.5-4B",
        model_short="qwen3.5_4b",
        display_name="Qwen3.5 4B",
        color="#F4A261",
        requires_auth=False,
        hidden_dim=2560,
        chat_template_kwargs={"enable_thinking": False},
    ),
    "qwen3.5_9b": ModelConfig(
        model_id="Qwen/Qwen3.5-9B",
        model_short="qwen3.5_9b",
        display_name="Qwen3.5 9B",
        color="#E9C46A",
        requires_auth=False,
        hidden_dim=4096,
        chat_template_kwargs={"enable_thinking": False},
    ),
    # --- 4-bit twins of two main-cast models (quantization-robustness check,
    # 2026-07-24): same weights, mlx-community 4-bit. Justifies measuring >20B
    # models in 4-bit only: if axes/fingerprints match bf16 here, precision is
    # not a confound. Data lands under the _4bit model_short suffix.
    "qwen3.5_9b_4bit": ModelConfig(
        model_id="Qwen/Qwen3.5-9B",
        model_short="qwen3.5_9b_4bit",
        display_name="Qwen3.5 9B (4-bit)",
        color="#E9C46A",
        requires_auth=False,
        hidden_dim=4096,
        chat_template_kwargs={"enable_thinking": False},
        quantize="4bit",
    ),
    "gemma3_12b_4bit": ModelConfig(
        model_id="google/gemma-3-12b-it",
        model_short="gemma3_12b_4bit",
        display_name="Gemma 3 12B (4-bit)",
        color="#34A853",
        requires_auth=True,
        hidden_dim=3840,
        quantize="4bit",
    ),
    "phi4_4bit": ModelConfig(
        model_id="microsoft/phi-4",
        model_short="phi4_4bit",
        display_name="Phi-4 14B (4-bit)",
        color="#7B68EE",
        requires_auth=False,
        hidden_dim=5120,
        quantize="4bit",
    ),
    "deepseek_r1_14b_4bit": ModelConfig(
        model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        model_short="deepseek_r1_14b_4bit",
        display_name="DeepSeek-R1 14B (4-bit)",
        color="#00D9A5",
        requires_auth=False,
        hidden_dim=5120,
        # Must match the bf16 twin's no-think regime or the pair is invalid.
        chat_template_kwargs={"enable_thinking": False,
                              "no_think_prefill": True},
        no_think_prefill="<think>\n\n</think>\n\n",
        quantize="4bit",
    ),
    "ministral3_8b_4bit": ModelConfig(
        model_id="mistralai/Ministral-3-8B-Instruct-2512",
        model_short="ministral3_8b_4bit",
        display_name="Ministral 3 8B (4-bit)",
        color="#7C3AED",
        requires_auth=False,
        hidden_dim=4096,
        quantize="4bit",
    ),

    # --- Qwen 2.5 scaling chain ---
    "qwen2.5_0.5b": ModelConfig(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        model_short="qwen2.5_0.5b",
        display_name="Qwen 2.5 0.5B",
        color="#FFB347",
        requires_auth=False,
        hidden_dim=896,
    ),
    "qwen_1.5b": ModelConfig(
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        model_short="qwen_1.5b",
        display_name="Qwen 2.5 1.5B",
        color="#FF9F1C",
        requires_auth=False,
        hidden_dim=1536,
    ),
    "qwen2.5_3b": ModelConfig(
        model_id="Qwen/Qwen2.5-3B-Instruct",
        model_short="qwen2.5_3b",
        display_name="Qwen 2.5 3B",
        color="#FF7F50",
        requires_auth=False,
        hidden_dim=2048,
    ),
    # qwen_7b is in "Article models" section above

    # --- Qwen 3 scaling chain ---
    "qwen3_0.6b": ModelConfig(
        model_id="Qwen/Qwen3-0.6B",
        model_short="qwen3_0.6b",
        display_name="Qwen3 0.6B",
        color="#DC143C",
        requires_auth=False,
        hidden_dim=1024,
        chat_template_kwargs={"enable_thinking": False},
    ),
    "qwen3_1.7b": ModelConfig(
        model_id="Qwen/Qwen3-1.7B",
        model_short="qwen3_1.7b",
        display_name="Qwen3 1.7B",
        color="#CD5C5C",
        requires_auth=False,
        hidden_dim=2048,
        chat_template_kwargs={"enable_thinking": False},
    ),
    "qwen3_4b": ModelConfig(
        model_id="Qwen/Qwen3-4B",
        model_short="qwen3_4b",
        display_name="Qwen3 4B",
        color="#B22222",
        requires_auth=False,
        hidden_dim=2560,
        chat_template_kwargs={"enable_thinking": False},
    ),
    # qwen3_8b is in "V6 models" section above
    "smollm_1.7b": ModelConfig(
        model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        model_short="smollm_1.7b",
        display_name="SmolLM2 1.7B",
        color="#2EC4B6",
        requires_auth=False,
        hidden_dim=2048,
    ),
    "llama_1b": ModelConfig(
        model_id="meta-llama/Llama-3.2-1B-Instruct",
        model_short="llama_1b",
        display_name="Llama 3.2 1B",
        color="#5B9BD5",
        requires_auth=True,
        hidden_dim=2048,
    ),

    # --- Future models ---
    "yi_9b": ModelConfig(
        model_id="01-ai/Yi-1.5-9B-Chat",
        model_short="yi_9b",
        display_name="Yi-1.5 9B",
        color="#00B4D8",
        requires_auth=False,
        hidden_dim=4096,
    ),
    "falcon_h1_7b": ModelConfig(
        model_id="tiiuae/Falcon-H1-7B-Instruct",
        model_short="falcon_h1_7b",
        display_name="Falcon-H1 7B",
        color="#C77DFF",
        requires_auth=False,
        hidden_dim=3072,
    ),
    "granite_8b": ModelConfig(
        model_id="ibm-granite/granite-3.3-8b-instruct",
        model_short="granite_8b",
        display_name="Granite 3.3 8B",
        color="#648FFF",
        requires_auth=False,
        hidden_dim=4096,
    ),
    "exaone_7b": ModelConfig(
        model_id="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
        model_short="exaone_7b",
        display_name="EXAONE 3.5 7.8B",
        color="#DC267F",
        requires_auth=False,
        hidden_dim=4096,
    ),
    "smollm3_3b": ModelConfig(
        model_id="HuggingFaceTB/SmolLM3-3B",
        model_short="smollm3_3b",
        display_name="SmolLM3 3B",
        color="#785EF0",
        requires_auth=False,
        hidden_dim=2048,
        chat_template_kwargs={"enable_thinking": False},
    ),
    "nemotron_h_8b": ModelConfig(
        model_id="nvidia/Nemotron-H-8B-Reasoning-128K",
        model_short="nemotron_h_8b",
        display_name="Nemotron-H 8B",
        color="#76B900",
        requires_auth=True,
        hidden_dim=4096,
        chat_template_kwargs={"enable_thinking": False},
    ),

    # --- Falcon-H1 scaling chain ---
    "falcon_h1_0.5b": ModelConfig(
        model_id="tiiuae/Falcon-H1-0.5B-Instruct",
        model_short="falcon_h1_0.5b",
        display_name="Falcon-H1 0.5B",
        color="#C77DFF",
        requires_auth=False,
        hidden_dim=1024,
    ),
    "falcon_h1_1.5b": ModelConfig(
        model_id="tiiuae/Falcon-H1-1.5B-Instruct",
        model_short="falcon_h1_1.5b",
        display_name="Falcon-H1 1.5B",
        color="#C77DFF",
        requires_auth=False,
        hidden_dim=2048,
    ),
    "falcon_h1_3b": ModelConfig(
        model_id="tiiuae/Falcon-H1-3B-Instruct",
        model_short="falcon_h1_3b",
        display_name="Falcon-H1 3B",
        color="#C77DFF",
        requires_auth=False,
        hidden_dim=2560,
    ),

    # --- Granite scaling ---
    "granite_2b": ModelConfig(
        model_id="ibm-granite/granite-3.3-2b-instruct",
        model_short="granite_2b",
        display_name="Granite 3.3 2B",
        color="#648FFF",
        requires_auth=False,
        hidden_dim=2048,
    ),

    # --- EXAONE scaling ---
    "exaone_2b": ModelConfig(
        model_id="LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
        model_short="exaone_2b",
        display_name="EXAONE 3.5 2.4B",
        color="#DC267F",
        requires_auth=False,
        hidden_dim=2560,
    ),

    # --- Base models ---
    "gemma_9b_base": ModelConfig(
        model_id="google/gemma-2-9b",
        model_short="gemma_9b_base",
        display_name="Gemma 2 9B (base)",
        color="#4285F4",
        requires_auth=True,
        hidden_dim=3584,
        is_base_model=True,
        instruct_counterpart="gemma_9b",
    ),
    "llama_8b_base": ModelConfig(
        model_id="meta-llama/Llama-3.1-8B",
        model_short="llama_8b_base",
        display_name="Llama 3.1 8B (base)",
        color="#3B82F6",
        requires_auth=True,
        hidden_dim=4096,
        is_base_model=True,
        instruct_counterpart="llama_8b",
    ),
    "qwen_7b_base": ModelConfig(
        model_id="Qwen/Qwen2.5-7B",
        model_short="qwen_7b_base",
        display_name="Qwen 2.5 7B (base)",
        color="#FF6B35",
        requires_auth=False,
        hidden_dim=3584,
        is_base_model=True,
        instruct_counterpart="qwen_7b",
    ),
    "mistral_7b_base": ModelConfig(
        model_id="mistralai/Mistral-7B-v0.3",
        model_short="mistral_7b_base",
        display_name="Mistral 7B (base)",
        color="#A855F7",
        requires_auth=False,
        hidden_dim=4096,
        is_base_model=True,
        instruct_counterpart="mistral_7b",
    ),
    "deepseek_7b_base": ModelConfig(
        model_id="deepseek-ai/deepseek-llm-7b-base",
        model_short="deepseek_7b_base",
        display_name="DeepSeek 7B (base)",
        color="#00D9A5",
        requires_auth=False,
        hidden_dim=4096,
        is_base_model=True,
        instruct_counterpart="deepseek_7b",
    ),
}

MODEL_SETS = {
    # Main text (4 models, current-gen, 4 orgs)
    # gemma3_12b removed: degenerates (repetition loops) under activation steering
    # with no protocol-consistent fix — see limitations. Gemma 2 9B stays in legacy.
    "paper": [
        "qwen3.5_9b",       # Alibaba — hybrid SSM, newest
        "phi4",              # Microsoft — 14B
        "deepseek_r1_14b",   # DeepSeek — thinking distill
        "ministral3_8b",     # Mistral — new Mistral
    ],
    # Separate section
    "paper_extra": ["gpt_oss_20b"],
    # Legacy (appendix)
    "legacy": [
        "qwen_7b", "llama_8b", "gemma_9b", "mistral_7b", "qwen3_8b",
    ],
    "base": ["qwen_7b_base", "llama_8b_base", "gemma_9b_base", "mistral_7b_base",
             "ministral3_8b_base"],
    "small": ["qwen_1.5b", "smollm_1.7b", "llama_1b"],
    "qwen2.5_chain": ["qwen2.5_0.5b", "qwen_1.5b", "qwen2.5_3b", "qwen_7b"],
    "qwen3_chain": ["qwen3_0.6b", "qwen3_1.7b", "qwen3_4b", "qwen3_8b"],
    "qwen3.5_chain": ["qwen3.5_0.8b", "qwen3.5_2b", "qwen3.5_4b", "qwen3.5_9b"],
    "gemma3n_chain": ["gemma3n_e2b", "gemma3n_e4b"],
    "diverse": ["glm_z1_9b", "olmo2_13b", "command_r7b", "internlm3_8b"],
    "future": ["falcon_h1_7b", "yi_9b", "granite_8b", "exaone_7b", "smollm3_3b", "nemotron_h_8b"],
    "falcon_h1_chain": ["falcon_h1_0.5b", "falcon_h1_1.5b", "falcon_h1_3b", "falcon_h1_7b"],
    # New 2026 additions (wave 3): scale, MoE, reasoning pair, trendy small
    "wave3_new": ["qwen3.6_27b", "qwen3.6_35b_a3b", "ministral3_8b_reasoning",
                  "ministral3_14b", "mistral_small_3.2_24b", "gemma4_26b_a4b",
                  "vibethinker_3b"],
    # Wanted but blocked on MLX support — re-run 98_smoke_test.py periodically
    "wave3_blocked": ["gemma4_12b", "zaya1_8b"],
    "quick": ["qwen_7b"],
    "all": list(MODELS.keys()),
}

ACTIVE_MODEL_SET = os.environ.get("MOOD_AXIS_MODEL_SET", "paper")

MODEL_IDS = {k: v.model_id for k, v in MODELS.items()}
MODEL_NAMES = {k: v.display_name for k, v in MODELS.items()}
MODEL_COLORS = {k: v.color for k, v in MODELS.items()}


def get_model_config(model_short: str) -> ModelConfig:
    if model_short not in MODELS:
        raise ValueError(f"Unknown model: {model_short}. Available: {list(MODELS.keys())}")
    return MODELS[model_short]


def get_model_by_id(model_id: str) -> Optional[ModelConfig]:
    for config in MODELS.values():
        if config.model_id == model_id:
            return config
    return None


def get_short_name_from_id(model_id: str) -> str:
    config = get_model_by_id(model_id)
    if config:
        return config.model_short
    return model_id.split("/")[-1].lower().replace("-", "_").replace(".", "_")


def get_active_models() -> list[str]:
    model_set = os.environ.get("MOOD_AXIS_MODEL_SET", ACTIVE_MODEL_SET)
    if model_set not in MODEL_SETS:
        raise ValueError(f"Unknown model set: {model_set}. Available: {list(MODEL_SETS.keys())}")
    return MODEL_SETS[model_set]


# Re-export for backwards compat
from config.settings import MOOD_AXES as AXES, AXIS_LABELS
