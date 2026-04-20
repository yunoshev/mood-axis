<details>
<summary>Reproduce: Conflict Drift</summary>

**Prerequisites:** Calibrated axes for the target model. GPU required.

**Input:**
- `data/axes/{model}_axes.npz` -- calibrated axis vectors
- `config/conflict_scenarios.py` -- 50 adversarial scenarios across 10 categories
- `config/settings.py` -- drift parameters (turns, max_new_tokens)

**Run:**
```bash
# Run conflict drift protocol (~30-60 min per model)
python scripts/05_drift.py --model qwen_7b

# Or via Makefile:
make drift MODEL=qwen_7b
```

**Output:**
- `data/drift/qwen_7b.json` -- per-scenario, per-turn personality projections across all 7 axes
  - Includes: 50 conflict conversations (12 conflict + 4 recovery turns each)
  - Includes: 10 neutral control conversations
  - Per-turn: raw projections, normalized scores, running averages

**Validation:**
- Control conversations: mean drift < 0.1 per axis (no natural drift from length alone)
- Conflict conversations: at least some axes show significant drift (bootstrap CI excludes zero AND > 2x control slope)
- Recovery turns: most models return toward baseline within 4 neutral turns

**Pre-computed:** Drift data for all 5 main models + GPT-OSS 20B is included.

</details>
