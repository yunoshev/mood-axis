### PCA Dimensionality

How "one-dimensional" is a model's personality? We run PCA on baseline projections across {{baseline_questions}} questions x axes:

{{table_pca}}

![PCA Dimensionality](docs/figures/pca.png)

PC1 range across all {{n_models}} models: {{pc1_range}}, effective dimensions: {{effective_dimension_range}}.

Base models do not show higher dimensionality than instruct models -- RLHF constrains behavior utilization but does not restructure the representation geometry.

<details>
<summary>Full PCA table (all models including base)</summary>

{{table_pca_all}}

</details>
