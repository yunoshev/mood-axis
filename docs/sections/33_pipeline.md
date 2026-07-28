## Pipeline Reference

| Step | Script | GPU | Output |
|------|--------|:---:|--------|
| 1. Calibrate axes | `scripts/01_calibrate.py` | ✓ | `data/axes/<m>_axes.npz` + meta + hidden states |
| 2. Baseline fingerprint | `scripts/02_baseline.py` | ✓ | `data/baselines/<m>.json` |
| 3. Directional benchmark | `scripts/03_benchmark.py` | ✓ | `data/benchmarks/<m>.json` |
| 4. Stability sets A/B/C | `scripts/04_stability.py` | ✓ | `data/stability/<m>.json` |
| 5. Conflict drift | `scripts/05_drift.py` | ✓ | `data/drift/<m>.json` |
| 6. Cross-model stats | `scripts/06[d-o]_*.py` | — | `data/analysis/*.json` |
| 7. Figures | `scripts/07_figures.py` | — | `paper/figures/fig_*.png` |
| S. Layer sweep + steering | `scripts/steering_layer_sweep.py`, `scripts/steering_basic.py` | ✓ | `data/steering_sweep/`, `data/steering/` |

CPU analysis scripts (step 6):

| Script | Computes |
|--------|----------|
| `06b_layer_sweep.py` | per-layer validation accuracy / d′ sweep |
| `06d_axis_geometry.py` | axis cosines, pooled PCA, concept-cone rank, permutation nulls |
| `06e_drift_stats.py` | drift slopes, bootstrap CIs, composite significance, recovery |
| `06f_probe_stats.py` | base-vs-instruct item-level separability + McNemar |
| `06g_cross_calibration.py` | base↔instruct axis transfer |
| `06h_consistency_null.py` | permutation null for cross-set consistency |
| `06j_quant_robustness.py` | bf16 vs 4-bit axis invariance |
| `06k_format_confound.py` | format-confound check (cohort-wide) |
| `06l_think_stats.py` | think-vs-answer temperament contrasts |
| `06m_partialled_drift.py` | drift robustness on format-partialled axes |
| `06n_drift_multiplicity.py` | permutation tests + BH-FDR over all drift tests |
| `06o_layer_pooling_sensitivity.py` | per-layer / last-token pooling sensitivity |

Common make targets:

```bash
make pipeline MODEL=qwen3.5_9b   # steps 1-5 for one model
make analysis                    # step 6 (all models, CPU)
make figures                     # step 7
make readme                      # rebuild README.md from docs/sections/
make check MODEL=qwen3.5_9b      # preflight checks, no GPU
```
