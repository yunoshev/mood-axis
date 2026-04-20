### Dead Zones (RLHF Suppression)

Some model-axis combinations are "dead" -- the model produces near-identical hidden states regardless of the style instruction, collapsing the axis to a narrow range. This is a signature of RLHF suppression.

{{table_dead_zones}}

Base (pre-RLHF) models confirm that dead zones are introduced by alignment:

{{table_dead_zones_with_base}}

Gemma 2 9B instruct has 3 dead zones, but its base version has **0** -- RLHF suppresses personality expression on warm_cold, empathetic_analytical, and formal_casual axes.

#### Surface Metrics Orthogonality

Do mood axes measure genuine semantic properties, or are they artifacts of surface linguistic variation?

{{surface_metrics_summary_table}}

Most correlations |r| < 0.6. Interpretable exceptions: verbose_concise with response length (r=0.68, expected), confident_cautious with caution markers (r=0.58, expected). Axes measure semantic personality, not surface features.
