<details>
<summary>Reproduce: External Judge Validation</summary>

**Prerequisites:** Steering results in `data/steering/`. Requires `ANTHROPIC_API_KEY` environment variable (Claude Sonnet API calls).

**Input:**
- `data/steering/{model}_basic.json` -- steered responses (from steering_basic.py)

**Run:**
```bash
# Prepare judge batches (CPU, ~5 seconds)
python scripts/judge_prepare.py

# Run external judge validation (API calls, ~5 min, ~$2 cost)
export ANTHROPIC_API_KEY=your_key_here
python scripts/steering_judge.py

# Analyze judge results (CPU, ~5 seconds)
python scripts/judge_analyze.py
```

**Output:**
- `data/judge/batch_pairwise.json` -- pairwise forced-choice results
- `data/judge/batch_likert.json` -- Likert dose-response ratings
- `data/steering/judge_results.json` -- aggregated judge accuracy, Spearman rho per model

**Validation:**
- Steerable models (Mistral): pairwise accuracy > 90%, Likert rho > 0.8
- Resistant models (Gemma, Qwen3): pairwise accuracy near 50% (chance), Likert rho near 0

**Note:** This is the only step requiring an API key. All other steps are fully local. Pre-computed judge results are included in the repo.

</details>
