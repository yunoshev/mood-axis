<details>
<summary>Reproduce: Cross-Model Analysis (Fingerprints, PCA, Dead Zones)</summary>

**Prerequisites:** Baseline data for at least 2 models in `data/baselines/`. No GPU needed.

**Input:**
- `data/baselines/*.json` -- personality profiles (from Step 2)
- `data/axes/*_meta.json` -- calibration metadata
- `data/drift/*.json` -- drift results (optional, for drift summary)

**Run:**
```bash
# CPU only, reads all available model data (~10 seconds)
python scripts/06_analysis.py

# Or via Makefile:
make analysis
```

**Output:**
- `data/analysis/fingerprints.json` -- cross-model personality comparison
- `data/analysis/pca_dimensionality.json` -- PCA analysis per model
- `data/analysis/dead_zones.json` -- dead zone detection (low-variance axes)
- `data/analysis/correlations.json` -- cross-axis and surface metric correlations

**Validation:**
- `fingerprints.json` should contain entries for each model in `data/baselines/`
- PCA effective dimensions typically 2.5-4.5 per model
- Dead zones: axes with IQR < 0.3 in calibration data

</details>
