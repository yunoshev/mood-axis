<details>
<summary><b>Reproduce: axis geometry</b></summary>

**Input:** archived calibration hidden states (`data/axes/<model>_hidden_states.npz`, written by `01_calibrate.py`).

**Run:**
```bash
python3 scripts/06d_axis_geometry.py            # all default models
#   --models qwen3.5_9b,gpt_oss_20b             subset
#   --n-permutations 200 --seed 42              permutation null settings
```

**Output:** `data/analysis/axis_geometry.json` — pairwise axis cosines, pooled-projection PCA, concept-cone effective rank, pairwise coherence, anti-directed pair fractions, permutation z-scores.

**Pre-computed:** `v2/data/analysis/axis_geometry.json`.

</details>
