### Axis Stability

We evaluate axis reproducibility across 3 independent calibration sets (A, B, C, each with 210 unique questions) by computing ICC(2,k) -- average pairwise consistency across k=3 raters (question sets). Interpretation thresholds (Cicchetti, 1994): 0.75-1.0 Excellent, 0.60-0.74 Good, 0.40-0.59 Fair.

{{table_stability}}

![Axis Stability](docs/figures/stability.png)

All models achieve ICC {{icc_range}}. Rank ICC (ordering consistency, ignores scale) exceeds 0.84 for all models (0.93+ for 4/5) -- the relative ordering of axes is highly consistent even if absolute magnitudes shift slightly between calibration sets.

Bootstrap CI (1000 resamples) confirms all ICC estimates exceed 0.65 at p<0.01.
