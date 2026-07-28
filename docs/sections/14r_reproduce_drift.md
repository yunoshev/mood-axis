<details>
<summary><b>Reproduce: conflict drift</b></summary>

**Input:** calibrated axes + `config/conflict_scenarios.py` (50 scenarios) + `config/drift_prompts.py` (30 neutral controls).

**Run:**
```bash
# GPU: run the 50 conflict scenarios + 30 controls (checkpointed per scenario, resumable)
python3 scripts/05_drift.py --model qwen3.5_9b --backend mlx
#   --scenarios 5     quick smoke run
#   --reproject       recompute projections from saved hidden states (no generation)

# CPU: slopes, bootstrap CIs, composite significance, recovery
python3 scripts/06e_drift_stats.py

# CPU: permutation tests + BH-FDR across all 44 model x axis tests
python3 scripts/06n_drift_multiplicity.py

# CPU (robustness): re-run the full significance machinery on format-partialled axes
python3 scripts/06m_partialled_drift.py
```

**Output:**
- `data/drift/<model>.json` — per-turn projections for every scenario and control
- `data/analysis/drift_stats.json` — slopes, CIs, composite verdicts, recovery ratios
- `data/analysis/drift_multiplicity.json` — permutation p-values, q_fdr per test
- `data/analysis/format_partialled_drift.json` — drift verdicts after format-subspace removal

**Validation:** `drift_stats.json` reports 9 models × 5 axes; every composite-significant axis in the table above has `significant_drift: true`.

**Pre-computed:** `v2/data/drift/<model>.json` (per-turn trajectories, LFS) + `v2/data/analysis/drift_stats.json` — the exact inputs behind the table, the figure, and the drift explorer on the site.

</details>
