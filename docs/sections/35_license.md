## API Key Requirement

The external judge scripts (`scripts/steering_judge.py`, `scripts/judge_analyze.py`) call the Anthropic API:

```bash
export ANTHROPIC_API_KEY=your_key_here
python3 scripts/steering_judge.py
```

All other scripts are fully local and require no API keys.

## Citation

```bibtex
@misc{yunoshev2026moodaxis,
  title={Mood Axis: Measuring and Steering LLM Personality via Hidden States},
  author={Yunoshev, Andrey},
  year={2026},
  url={https://github.com/yunoshev/mood-axis}
}
```

## License

MIT
