<details>
<summary><b>Reproduce: format-confound check</b></summary>

**Input:** archived calibration hidden states + response texts (written by `01_calibrate.py`).

**Run:**
```bash
python3 scripts/06k_format_confound.py          # full cohort
#   --models gpt_oss_20b,deepseek_r1_14b        subset
python3 scripts/06m_partialled_drift.py         # drift robustness on partialled axes
```

**Output:**
- `data/analysis/format_confound.json` — per-model, per-axis: cos(axis, verbose_concise), projection–length correlations, raw→partialled held-out accuracy and cross-set agreement
- `data/analysis/format_partialled_drift.json` — raw vs partialled drift verdicts

**Validation:** GPT-OSS empathetic_analytical: `val_pair_acc` 0.73 → 1.00 after partialling; DeepSeek/Gemma 4 axes change by ≤ 0.03.

**Pre-computed:** `v2/data/analysis/format_confound.json`, `v2/data/analysis/format_partialled_drift.json`.

</details>
