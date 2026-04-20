## Models Tested

### Featured Models

| Short Name | HuggingFace ID | Size | Organization | Notes |
|-----------|---------------|------|:------------|:------|
| `qwen3_5_9b` | Qwen/Qwen3.5-9B-Instruct | 9B | Alibaba | Hybrid SSM |
| `deepseek_r1_14b` | deepseek-ai/DeepSeek-R1-Distill-Qwen-14B | 14B | DeepSeek | Reasoning distillation |
| `gemma3_12b` | google/gemma-3-12b-it | 12B | Google | |
| `phi4` | microsoft/phi-4 | 14B | Microsoft | |
| `llama_8b` | meta-llama/Llama-3.1-8B-Instruct | 8B | Meta | |
| `gpt_oss_20b` | openai/gpt-oss-20b | 20B | OpenAI | MoE |

### Legacy Models

| Short Name | HuggingFace ID | Size | Hidden Dim | Auth |
|-----------|---------------|------|:---------:|:----:|
| `qwen_7b` | Qwen/Qwen2.5-7B-Instruct | 7B | 3584 | No |
| `mistral_7b` | mistralai/Mistral-7B-Instruct-v0.3 | 7B | 4096 | No |
| `gemma_9b` | google/gemma-2-9b-it | 9B | 3584 | Yes |
| `qwen3_8b` | Qwen/Qwen3-8B | 8B | 4096 | No |

Additional models with pre-computed data: DeepSeek 7B, Yi 9B, plus 5 base (pre-RLHF) variants, 3 small models (1-1.7B), and diverse models (Granite 8B, GLM-Z1 9B, Command-R 7B, InternLM3 8B, OLMo2 13B, Falcon-H1 7B, ExaOne 7B, Yi 9B, SmolLM3 3B). See `config/models.py` for the full registry.

## How to Add Your Own Model

1. Add a `ModelConfig` entry in `config/models.py` with the HuggingFace ID and hidden dimension
2. Run the pipeline: `make pipeline MODEL=your_model_key`
3. Re-run analysis: `make analysis`

Requires ~16 GB VRAM for 7-9B models. The pipeline handles chat template detection, tokenizer quirks, and hidden state extraction automatically.
