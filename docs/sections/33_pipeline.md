## Pipeline Reference

| Step | Script | GPU | Input | Output |
|------|--------|:---:|-------|--------|
| 1. Calibrate | `scripts/01_calibrate.py` | Yes | config/prompts.py | data/axes/{model}_axes.npz |
| 2. Baseline | `scripts/02_baseline.py` | Yes | axis vectors | data/baselines/{model}.json |
| 3. Benchmark | `scripts/03_benchmark.py` | Yes | axis vectors | data/benchmarks/{model}.json |
| 4. Stability | `scripts/04_stability.py` | Yes | stability_prompts.py | data/stability/{model}.json |
| 5. Drift | `scripts/05_drift.py` | Yes | axes, conflict scenarios | data/drift/{model}.json |
| 6. Analysis | `scripts/06_analysis.py` | No | all baseline + drift | data/analysis/*.json |
| 7. Experiments | `scripts/07_experiments.py` | Yes | axis vectors | data/experiments/ |
| S1. Steering | `scripts/steering_basic.py` | Yes | axis vectors | data/steering/{model}_basic.json |
| S2. Judge | `scripts/steering_judge.py` | No | steering results | data/steering/judge_results.json |
| S3. Analyze | `scripts/steering_analyze.py` | No | steering + judge | data/analysis/steering_summary.json |

Steps 1-5 run per-model. Step 6 reads all models and produces cross-model analysis. Steering scripts (S1-S3) run after the main pipeline.

### Configuration

| File | Contents |
|------|----------|
| `config/settings.py` | Axis definitions, generation parameters, paths |
| `config/models.py` | Model registry (HuggingFace IDs, display names, hidden dimensions) |
| `config/prompts.py` | {{total_unique_questions}} unique calibration/eval/baseline questions |
| `config/conflict_scenarios.py` | {{conflict_scenarios}} adversarial scenarios for drift |
| `config/stability_prompts.py` | Independent question sets B and C for stability |

### Build System

README.md is generated from sections + data (same pattern as the paper):

```bash
python build_data.py        # data/ -> paper_data.json
python build_figures.py      # data/ -> docs/figures/*.png
python build_readme.py       # docs/sections/ + paper_data.json -> README.md

# Or all at once:
make rebuild-readme
```

**Never edit README.md directly** -- edit `docs/sections/*.md` and `paper_data.json`, then rebuild.
