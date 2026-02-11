# Entropy Analysis Results

Date: 2026-02-11

## Experiment 1: Generation Entropy per Model

Source: `top_k_logprobs` from baseline hidden states (30 neutral questions, no system prompt).
Entropy computed from top-50 next-token probability distribution.

| Model | Mean entropy (nats) | Std | Median | n_tokens | Behavioral eff. dim |
|-------|:---:|:---:|:---:|:---:|:---:|
| Gemma 2 9B | **0.173** (min) | 0.045 | 0.165 | 10238 | 1.28 (min) |
| Phi-4 14B | 0.208 | 0.046 | 0.201 | 8713 | N/A |
| Llama 3.1 8B | 0.209 | 0.046 | 0.203 | 11415 | 2.41 |
| Mistral 7B v0.3 | 0.247 | 0.048 | 0.246 | 8990 | 2.78 |
| Yi 1.5 9B | 0.271 | 0.057 | 0.284 | 9485 | 1.85 |
| DeepSeek 7B | 0.289 | 0.052 | 0.286 | 5775 | 3.66 (max) |
| Qwen 2.5 7B | **0.319** (max) | 0.057 | 0.319 | 10654 | 1.91 |

**Note:** Qwen3-8B skipped — NPZ file corrupted.

### Key observations

1. **Entropy-dimensionality correlation**: Models with lower generation entropy tend toward more compressed behavioral space. Gemma (lowest entropy 0.173) has the highest behavioral collapse (eff dim 1.28). DeepSeek (high entropy 0.289) has the highest behavioral independence (eff dim 3.66).

2. **Exception**: Qwen 2.5 7B has highest entropy (0.319) but moderate eff dim (1.91) — breaks the simple correlation. Yi (0.271) has lower eff dim (1.85) than Mistral (0.247, eff dim 2.78).

3. **Alignment signature**: The three most "constrained" models behaviorally (Gemma, Llama, Phi-4) cluster at the low-entropy end (0.173–0.209). The two most "independent" (DeepSeek, Qwen) are at the high end (0.289–0.319).

4. **Phi-4 pattern**: Low entropy (0.208) + extreme cautious/reluctant profile — consistent with a strong conservative alignment prior that makes the model both predictable and behaviorally constrained.

## Experiment 2: Layer-wise Activation Entropy Profiles

Source: `per_layer_states` from baseline hidden states. Entropy via histogram binning (Freedman-Diaconis, 50-200 bins) of activation values per layer.

| Model | Layers | Hidden dim | Entropy range |
|-------|:---:|:---:|:---:|
| Qwen 2.5 7B | 29 | 3584 | [1.22, 1.73] |
| Mistral 7B v0.3 | 33 | 4096 | [1.62, 2.69] |
| DeepSeek 7B | 31 | 4096 | [0.37, 2.37] |
| Llama 3.1 8B | 33 | 4096 | [1.81, 3.13] |
| Gemma 2 9B | 43 | 3584 | [1.08, 2.28] |
| Yi 1.5 9B | 49 | 4096 | [0.63, 2.94] |
| Phi-4 14B | 41 | 5120 | [1.30, 2.64] |

### Key observations

1. **Distinct signatures**: Each model has a visually distinct entropy-by-layer curve. This supports the "geometric fingerprint" hypothesis — models can be identified by their activation entropy profile.

2. **DeepSeek anomaly**: Dramatic entropy dip in early layers (down to 0.37) — unique among all models. May reflect its architecture or training.

3. **Yi pattern**: Also shows a characteristic dip, but later and less extreme than DeepSeek.

4. **Qwen family comparison**: Could not compare Qwen 2.5 7B vs Qwen3-8B (corrupted NPZ). This is the most interesting comparison for testing whether model families share entropy signatures.

5. **Entropy range varies 3x**: From Qwen's narrow [1.22, 1.73] to Llama's wide [1.81, 3.13]. The dynamic range itself may be informative.

## Experiment 3: Entropy vs Behavioral Axes

Source: per-question generation entropy correlated with per-question axis projections from baseline JSON.

### 3a. Per-question correlations (Spearman, n=30 per model)

Consistent patterns across multiple models:

| Axis | Pattern | Models with p<0.05 |
|------|---------|-------------------|
| warm_cold | Higher entropy → colder | Mistral (r=−0.53), Yi (r=−0.39), Qwen (r=−0.38). Exception: DeepSeek (r=+0.39, opposite) |
| formal_casual | Higher entropy → more formal | Mistral (r=+0.58), Yi (r=+0.38). Makes sense: formal text has richer vocabulary → less predictable |
| confident_cautious | Higher entropy → more cautious | Llama (r=−0.44). Direct link: model uncertainty in next-token prediction aligns with measured cautiousness |
| patient_irritated | Higher entropy → less patient | Llama (r=−0.43) |
| proactive_reluctant | Higher entropy → less proactive | Mistral (r=−0.53) |

**Key finding**: warm_cold and formal_casual show the most consistent entropy correlations across models. This suggests a genuine link between generation uncertainty and behavioral style — not just a single-model artifact.

**Llama is special**: the only model where entropy correlates significantly with confident_cautious (r=−0.44). Llama also has 4/7 dead zones — its constrained behavioral space may force uncertainty into the few remaining free axes.

### 3b. Mean entropy vs effective dimensionality

Spearman correlation could not be computed (n=6, DeepSeek proactive_reluctant is constant → nan). Visual trend on scatter plot suggests positive association (higher entropy ≈ higher eff dim) but too few points for significance.

### 3c. Dead zone vs healthy models

| Model | Mean entropy | Dead zones |
|-------|:---:|:---:|
| Gemma 2 9B | 0.173 | 0/7 |
| Llama 3.1 8B | 0.209 | 4/7 |
| Mistral 7B | 0.247 | 1/7 |
| Yi 1.5 9B | 0.271 | 0/7 |
| DeepSeek 7B | 0.289 | 1/7 |
| Qwen 2.5 7B | 0.319 | 0/7 |

No simple correlation between entropy and dead zone count. Llama (4 dead zones) has low entropy, but Gemma (0 dead zones) has even lower. Dead zones appear orthogonal to generation entropy — they measure different properties.

## Experiment 4: Calibration Entropy by Pole

Source: `top_k_logprobs` from calibration hidden states (420 responses = 7 axes × 2 poles × 30 questions). Available for: Yi 1.5 9B, Gemma 2 9B (base). Phi-4 and Qwen3-8B NPZ files corrupted.

**Note:** patient_irritated and proactive_reluctant returned NaN for both models (likely token_offsets issue at end of file).

### Yi 1.5 9B

| Axis | Pos pole entropy | Neg pole entropy | Delta | p |
|------|:---:|:---:|:---:|:---:|
| formal_casual | 0.361 | 0.306 | **+0.055** | **<0.001** |
| confident_cautious | 0.288 | 0.243 | +0.045 | 0.110 |
| warm_cold | 0.307 | 0.334 | −0.027 | 0.333 |
| empathetic_analytical | 0.316 | 0.334 | −0.018 | 0.438 |
| verbose_concise | 0.266 | 0.276 | −0.010 | 0.717 |

### Gemma 2 9B (base)

| Axis | Pos pole entropy | Neg pole entropy | Delta | p |
|------|:---:|:---:|:---:|:---:|
| confident_cautious | 0.599 | 0.351 | **+0.248** | **<0.001** |
| formal_casual | 0.535 | 0.380 | **+0.156** | **<0.001** |
| warm_cold | 0.369 | 0.390 | −0.020 | 0.728 |
| verbose_concise | 0.418 | 0.412 | +0.006 | 0.959 |
| empathetic_analytical | 0.292 | 0.309 | −0.018 | 0.429 |
| patient_irritated | 0.244 | 0.215 | +0.029 | 0.307 |

### Key findings

1. **formal_casual is the strongest entropy axis**: "be formal" generates with significantly higher entropy than "be casual" in BOTH models (Yi Δ=+0.055, Gemma base Δ=+0.156). Formal text is literally less predictable at the token level.

2. **confident_cautious in Gemma base**: massive entropy difference (Δ=+0.248, p<0.001). "Be confident" → high entropy, "be cautious" → low entropy. Counter-intuitive: confident generation is LESS predictable. Possible explanation: confident responses make stronger claims with more diverse vocabulary, while cautious responses hedge with repetitive patterns ("it might", "perhaps", "it's possible").

3. **warm_cold shows no entropy difference** in either model. Warmth/coldness operates independently of generation uncertainty — it modulates style without changing how predictable the next token is.

4. **verbose_concise shows no entropy difference** — surprising given the length confound. Being verbose doesn't make generation more or less uncertain per token, it just generates more tokens.

## Experiment 5: Layer × Axis Entropy Correlation Heatmaps

Source: for each model, compute activation entropy at every layer for each of 30 baseline questions, then Spearman-correlate with axis projections. Result: (n_axes × n_layers) correlation matrix per model.

### Per-model heatmaps

Each model has a unique "behavioral entropy signature" — the pattern of where in the network activation entropy correlates with behavioral axes.

### Peak correlation layer per model × axis

| Model | warm/cold | formal/casual | verbose/concise | empath/analyt | confid/cautious | patient/irrit | proactive/reluct |
|-------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Qwen 2.5 7B | L27 −0.38* | L12 +0.42* | L28 −0.77* | L26 −0.39* | L28 +0.75* | L18 −0.48* | L14 +0.80* |
| Mistral 7B | L16 +0.45* | L26 −0.48* | L04 +0.51* | L07 −0.34 | L21 −0.51* | L32 −0.48* | L13 +0.44* |
| DeepSeek 7B | L10 +0.57* | L02 −0.37* | L16 −0.46* | L29 +0.38* | L13 +0.35 | L16 −0.36 | L00 +0.00 |
| Llama 3.1 8B | L11 +0.40* | L00 −0.18 | L31 −0.60* | L32 −0.22 | L31 +0.28 | L06 −0.47* | L31 −0.56* |
| Gemma 2 9B | L42 −0.58* | L17 +0.62* | L00 +0.77* | L00 −0.65* | L00 +0.59* | L39 −0.69* | L00 +0.82* |
| Yi 1.5 9B | L40 +0.33 | L40 −0.47* | L04 +0.65* | L27 +0.45* | L03 +0.58* | L42 −0.59* | L00 +0.74* |

### Key findings

1. **Each model has a unique entropy-behavioral fingerprint.** The pattern of which layers correlate with which axes differs dramatically across models. No two models share the same signature.

2. **Gemma 2 9B is the most structured.** Clear split: early layers (0–30%) correlate with verbose, empathetic, confident, proactive (|r| = 0.59–0.82). Late layers (70–100%) correlate with warm_cold and patient (|r| = 0.58–0.69). Behavioral information is spatially organized in the network.

3. **Qwen 2.5 7B concentrates signal in final layers.** Verbose (L28, r=−0.77), confident (L28, r=+0.75), proactive (L14, r=+0.80). Almost all behavioral signal "resolves" at the end of the network.

4. **DeepSeek 7B has early-layer signal.** warm_cold peaks at L10 (r=+0.57), formal at L02 (r=−0.37). Signal fades in later layers. Consistent with the anomalous entropy dip in early layers from Experiment 2.

5. **Llama 3.1 8B has weak signal everywhere** except last layers (L31–32) for verbose and proactive. Consistent with dead zones — little behavioral variation for entropy to correlate with.

6. **Mistral 7B distributes signal evenly** across the network. No sharp concentration — the most "distributed" model.

7. **Yi 1.5 9B resembles Gemma** in early-layer signal (verbose L04 r=+0.65, confident L03 r=+0.58, proactive L00 r=+0.74) but differs in late layers.

### Combined visualizations

Three cross-model comparison plots:

**a) Panel plot** (`combined_layer_axis_panels.png`): 7 panels (one per axis), all 6 models overlaid as lines on normalized layer position (0=first layer, 1=last layer). Shows where each model's signal lives for each axis.

**b) Binned heatmap** (`combined_binned_heatmap.png`): All models normalized to 10 layer bins (0–10%, 10–20%, ..., 90–100%). Side-by-side comparison reveals:
- Gemma: strong early red (proactive, verbose) → late blue (patient, warm). Most structured.
- Qwen: mirror pattern — early blue, late red.
- DeepSeek: early signal only, fades after 30%.
- Llama: pale everywhere — weakest signal.
- Mistral: even distribution, no sharp transitions.
- Yi: strong early red, late blue. Similar to Gemma but less extreme.

**c) Peak positions** (`combined_peak_positions.png`): Bar chart showing at what relative layer position each axis reaches peak |r|, annotated with Spearman r values. Reveals that the same axis (e.g., verbose_concise) peaks at completely different network depths depending on the model.

## Plots

- `exp1_generation_entropy.png` — bar chart + box plot of generation entropy
- `exp2_layer_entropy.png` — layer entropy curves (absolute + normalized position)
- `exp3_entropy_vs_behavior.png` — scatter: entropy vs confidence, verbosity, eff dim
- `exp4_calibration_entropy.png` — bar chart: calibration entropy by pole (Yi, Gemma base)
- `layer_axis_heatmaps.png` — individual heatmaps per model (layer × axis)
- `combined_layer_axis_panels.png` — 7 panels, all models overlaid
- `combined_binned_heatmap.png` — binned heatmap, all models side by side
- `combined_peak_positions.png` — peak layer positions per model × axis

## TODO

- [ ] Fix/re-download Qwen3-8B and Phi-4 NPZ for family comparison and calibration
- [ ] Fix patient_irritated/proactive_reluctant NaN in calibration (token_offsets issue)
- [ ] Compute formal Spearman for entropy vs eff dim (fix DeepSeek constant axis)
- [ ] Consider: spectral entropy of activation covariance, entropy rate across tokens
- [ ] Test whether formal_casual entropy gap persists in instruct models (currently only have Yi instruct + Gemma base calibration data)
- [ ] Explore whether Gemma/Yi early-layer similarity reflects shared architectural choices or training data
