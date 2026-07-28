## Models

### Main cast (four organizations)

| Key | Model | Params | Organization | Notes |
|-----|-------|--------|--------------|-------|
| `qwen3.5_9b` | Qwen3.5 9B Instruct | ~9B | Alibaba | hybrid SSM architecture |
| `phi4` | Phi-4 | ~14B | Microsoft | — |
| `deepseek_r1_14b` | DeepSeek-R1-Distill-Qwen 14B | ~14B | DeepSeek | reasoning distillation |
| `ministral3_8b` | Ministral 3 8B Instruct | ~8B | Mistral | — |

### Analyzed alongside

| Key | Model | Params | Notes |
|-----|-------|--------|-------|
| `gpt_oss_20b` | GPT-OSS 20B | ~20B (MoE) | OpenAI; separate case study (format confound) |
| `qwen3.6_27b` | Qwen3.6 27B | ~27B | 4-bit |
| `mistral_small_3.2_24b` | Mistral Small 3.2 24B | ~24B | 4-bit |
| `qwen3.6_35b_a3b` | Qwen3.6-35B-A3B | ~35B (MoE) | 4-bit |
| `gemma4_26b_a4b` | Gemma 4 26B-A4B | ~26B (MoE) | 4-bit |

Together these nine models form the drift/consistency cohort. Each main-cast model also has a 4-bit twin (`*_4bit`) for the quantization-robustness check.

Gemma 3 12B was part of an earlier version of the cohort but is **excluded from all reported results**: under activation steering it degenerates into repetition loops, and no mitigation compatible with the fixed cross-model protocol resolved it (see Limitations).

### Legacy base/instruct pairs (Part 1 alignment analyses)

`llama_8b`/`llama_8b_base`, `qwen_7b`/`qwen_7b_base`, `mistral_7b`/`mistral_7b_base`, `gemma_9b`/`gemma_9b_base`, plus `qwen_7b_uncensored` (refusal-ablated) — used for the base-vs-instruct separability, cross-calibration, and refusal-ablation results.

The full registry (63 model keys, including Qwen scaling chains and the diverse appendix cohort) lives in `config/models.py`.
