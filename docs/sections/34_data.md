## Data Files

All pre-computed results are in `data/`:

| Directory | Contents | Size |
|-----------|----------|------|
| `axes/` | Calibrated axis vectors (.npz) + metadata | ~2 MB |
| `baselines/` | Default personality profiles ({{n_models}} models) | 672 KB |
| `benchmarks/` | Systematic manipulation validation | 232 KB |
| `calibration/` | Calibration question datasets | 44 KB |
| `drift/` | Conflict drift results | 10 MB |
| `stability/` | ICC stability (3 sets per model) | 1.7 MB |
| `steering/` | Steering results + judge validation | 16 MB |
| `analysis/` | Cross-model analysis (fingerprints, PCA, dead zones, correlations) | 68 KB |
| `judge/` | External judge batches and results | 376 KB |

Total: ~36 MB. Raw hidden states (`*_hidden_states.npz`, ~850 MB) excluded -- regenerate with `make calibrate`.
