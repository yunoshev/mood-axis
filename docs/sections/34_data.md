## Data Files

Two data trees with distinct roles:

**`v2/data/` — the pinned, curated dataset (single source of truth).** Every number in this README, every figure, and every value on the [Interactive Explorer](https://yunoshev.github.io/mood-axis/) is derived from these files — the site is rebuilt by `python3 v2/build_site_data.py`, which reads *only* `v2/data/` and writes `docs/site_data.json` + `docs/data/` (regenerated automatically by CI on every push that touches `v2/data/`). Nothing on the site or in the README is hand-entered.

| Directory | Contents |
|-----------|----------|
| `v2/data/models.json` | model registry for the site (names, orgs, colors, twins) |
| `v2/data/axes_meta/` | calibration quality per model (val accuracy, d′ per axis) |
| `v2/data/baselines/` | 30-question fingerprints + example responses |
| `v2/data/stability/` | cross-set consistency, rank-ICC, Procrustes |
| `v2/data/drift/` | per-turn drift trajectories, 50 scenarios + 30 controls per model (LFS) |
| `v2/data/steering/` | steering dose-response generations (LFS) |
| `v2/data/steering_sweep/` | layer-sweep summaries |
| `v2/data/analysis/` | all cross-model statistics (drift_stats, axis_geometry, format_confound, quant_robustness, …) |

**`data/` — the working directory the pipeline scripts write to.** Running any script from [Pipeline Reference](#pipeline-reference) reproduces outputs here; compare them against the pinned copies in `v2/data/`. (It also still contains some files from earlier iterations of this project; when in doubt, trust `v2/data/`.)

Large JSONs (drift trajectories, steering generations) are stored in **Git LFS** — `git lfs pull` after cloning if your clone skipped them.
