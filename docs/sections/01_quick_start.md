## Quick Start

```bash
git clone {{github_url}}.git
cd mood-axis
pip install -r requirements.txt

# Part 1: Measure a model's personality (GPU required, ~15 min per model)
make pipeline MODEL=qwen_7b

# Or run individual steps
make calibrate MODEL=qwen_7b    # Build axis vectors from contrastive prompts
make baseline MODEL=qwen_7b     # Measure default personality profile
make drift MODEL=qwen_7b        # Test personality under adversarial conflict

# Part 2: Steer personality via hidden states (GPU required, ~60 min per model)
python scripts/steering_basic.py --model qwen_7b

# Cross-model analysis (CPU only, reads all available data)
make analysis

# Run tests
make test
```

Pre-computed data for all models is included in `data/` -- you can skip straight to analysis.
