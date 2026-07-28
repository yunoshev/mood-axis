# Mood Axis Pipeline
# Usage: make pipeline MODEL=qwen3.5_9b

MODEL ?= qwen3.5_9b

# ============================================================
# Per-model GPU steps (dependency order documented)
# ============================================================

# Step 1: calibrate (no deps)
calibrate:
	python3 scripts/01_calibrate.py --model $(MODEL)

# Step 2: baseline (depends: calibration)
baseline: | calibrate
	python3 scripts/02_baseline.py --model $(MODEL)

# Step 3: benchmark (depends: calibration)
benchmark: | calibrate
	python3 scripts/03_benchmark.py --model $(MODEL)

# Step 4: stability (no deps — independent calibration sets)
stability:
	python3 scripts/04_stability.py --model $(MODEL) calibrate
	python3 scripts/04_stability.py --model $(MODEL) analyze

# Step 5: drift (depends: calibration)
drift: | calibrate
	python3 scripts/05_drift.py --model $(MODEL)

# All GPU steps for one model
pipeline: calibrate baseline benchmark drift stability

# ============================================================
# CPU analysis (reads data/ for all models; no GPU needed)
# ============================================================

analysis:
	python3 scripts/06e_drift_stats.py
	python3 scripts/06n_drift_multiplicity.py
	python3 scripts/06d_axis_geometry.py
	python3 scripts/06h_consistency_null.py
	python3 scripts/06j_quant_robustness.py
	python3 scripts/06k_format_confound.py
	python3 scripts/06m_partialled_drift.py

figures:
	python3 scripts/07_figures.py

# ============================================================
# README + site build (CPU)
# ============================================================

# Assemble README.md from docs/sections/
readme:
	python3 build_readme.py

# Rebuild the interactive site's data from v2/data/ (single source of truth)
site:
	python3 v2/build_site_data.py

# ============================================================
# Utils
# ============================================================

test:
	python3 -m pytest tests/ -v

check:
	python3 scripts/01_calibrate.py --model $(MODEL) --check
	python3 scripts/02_baseline.py --model $(MODEL) --check
	python3 scripts/04_stability.py --model $(MODEL) --check
	python3 scripts/05_drift.py --model $(MODEL) --check

.PHONY: calibrate baseline benchmark stability drift pipeline \
	analysis figures readme site test check
