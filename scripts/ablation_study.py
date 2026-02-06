"""Ablation study for hidden state extraction hyperparameters.

Tests ~150 configurations of layer selection and token aggregation strategies
to empirically justify the production parameters (last 4 layers, weights
[0.1, 0.2, 0.3, 0.4], token decay 0.9).

Two-phase design:
  collect  -- GPU, runs inference once and saves all per-layer hidden states
  analyze  -- CPU, sweeps configs and produces tables/figures

Usage:
  python scripts/ablation_study.py collect --model qwen_7b
  python scripts/ablation_study.py analyze --model qwen_7b
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# --- path setup -------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.models import MODELS, get_model_config
from config.prompts import CALIBRATION_QUESTIONS, STYLE_INSTRUCTIONS, get_calibration_prompt
from config.settings import CALIBRATION_MAX_NEW_TOKENS
from src.calibration.dataset import CalibrationSample, get_messages_for_sample

# All 8 axes (not just the 4 active MOOD_AXES)
ALL_AXES = list(CALIBRATION_QUESTIONS.keys())

# Token aggregation strategy names
TOKEN_STRATEGIES = ["last_token", "mean", "decay_0.9", "decay_0.8"]

# Train/val split per (axis, pole): first 15 train, last 5 val
TRAIN_PER_POLE = 15
VAL_PER_POLE = 5


# ======================================================================
# Phase 1: collect
# ======================================================================

def _build_samples() -> list[CalibrationSample]:
    """Build calibration samples for all 8 axes x 2 poles x 20 questions = 320."""
    samples = []
    for axis in ALL_AXES:
        questions = CALIBRATION_QUESTIONS[axis][:20]
        for pole in ["positive", "negative"]:
            for question in questions:
                prompt = get_calibration_prompt(axis, pole, question)
                samples.append(CalibrationSample(
                    axis=axis,
                    pole=pole,
                    question=question,
                    system_prompt=prompt["system"],
                    user_prompt=prompt["user"],
                ))
    return samples


def _aggregate_tokens(token_states: torch.Tensor, strategy: str) -> torch.Tensor:
    """Aggregate token-level hidden states into a single vector.

    Args:
        token_states: (num_tokens, hidden_dim)
        strategy: one of TOKEN_STRATEGIES

    Returns:
        (hidden_dim,) tensor
    """
    num_tokens = token_states.shape[0]

    if strategy == "last_token":
        return token_states[-1]

    if strategy == "mean":
        return token_states.mean(dim=0)

    # Exponential decay: more recent tokens get higher weight
    if strategy.startswith("decay_"):
        decay = float(strategy.split("_")[1])
        weights = torch.tensor(
            [decay ** (num_tokens - 1 - i) for i in range(num_tokens)],
            device=token_states.device,
            dtype=token_states.dtype,
        )
        weights = weights / weights.sum()
        return (token_states * weights.unsqueeze(1)).sum(dim=0)

    raise ValueError(f"Unknown strategy: {strategy}")


@torch.no_grad()
def _generate_and_extract_all_layers(
    model,
    tokenizer,
    messages: list[dict[str, str]],
    max_new_tokens: int = CALIBRATION_MAX_NEW_TOKENS,
) -> dict[str, np.ndarray]:
    """Generate a response and extract per-layer, per-strategy hidden states.

    Returns:
        Dict with keys like "states_{strategy}" each of shape (num_layers, hidden_dim)
        as float16 numpy arrays, plus "_text" with the generated text.
    """
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    prompt_length = input_ids.shape[1]

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        output_hidden_states=True,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.pad_token_id,
    )

    generated_ids = outputs.sequences[0, prompt_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    num_generated = len(outputs.hidden_states)
    if num_generated == 0:
        raise ValueError("No tokens were generated")

    num_layers = len(outputs.hidden_states[0])

    # Collect per-layer token states: (num_layers, num_generated, hidden_dim)
    all_layer_token_states = []
    for layer_idx in range(num_layers):
        layer_tokens = []
        for token_idx in range(num_generated):
            layer_tokens.append(outputs.hidden_states[token_idx][layer_idx][0, 0, :])  # (hidden_dim,)
        all_layer_token_states.append(torch.stack(layer_tokens, dim=0))  # (num_generated, hidden_dim)

    # Apply each token strategy per layer
    result = {"_text": generated_text}
    for strategy in TOKEN_STRATEGIES:
        per_layer = []
        for layer_idx in range(num_layers):
            agg = _aggregate_tokens(all_layer_token_states[layer_idx], strategy)
            per_layer.append(agg)
        stacked = torch.stack(per_layer, dim=0)  # (num_layers, hidden_dim)
        result[f"states_{strategy}"] = stacked.cpu().float().numpy()

    return result


def run_collect(model_short: str):
    """Phase 1: collect raw hidden states for all samples."""
    config = get_model_config(model_short)
    out_dir = PROJECT_ROOT / "data" / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_short}_raw_states.npz"

    print(f"=== Ablation Study: COLLECT for {config.display_name} ===")
    print(f"Output: {out_path}")

    # Build samples
    samples = _build_samples()
    print(f"Total samples: {len(samples)} (8 axes x 2 poles x 20 questions)")

    # Load model (lazy import -- only needed for collect phase)
    from src.model.loader import load_model
    model, tokenizer = load_model(config.model_id)

    # Storage lists
    sample_axes = []
    sample_poles = []
    all_states = {strategy: [] for strategy in TOKEN_STRATEGIES}

    t0 = time.time()
    for i, sample in enumerate(tqdm(samples, desc="Inference")):
        messages = get_messages_for_sample(sample)
        result = _generate_and_extract_all_layers(model, tokenizer, messages)

        sample_axes.append(sample.axis)
        sample_poles.append(sample.pole)
        for strategy in TOKEN_STRATEGIES:
            all_states[strategy].append(result[f"states_{strategy}"])

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(samples) - i - 1) / rate
            print(f"  [{i+1}/{len(samples)}] {rate:.1f} samples/s, ~{remaining/60:.0f} min remaining")

    elapsed = time.time() - t0
    print(f"Inference done in {elapsed/60:.1f} min")

    # Get metadata from first result
    num_layers = all_states[TOKEN_STRATEGIES[0]][0].shape[0]
    hidden_dim = all_states[TOKEN_STRATEGIES[0]][0].shape[1]

    # Save
    save_dict = {
        "_model_short": np.array(model_short),
        "_model_id": np.array(config.model_id),
        "_num_layers": np.array(num_layers),
        "_hidden_dim": np.array(hidden_dim),
        "_axes": np.array(ALL_AXES),
        "_sample_axes": np.array(sample_axes),
        "_sample_poles": np.array(sample_poles),
    }
    for strategy in TOKEN_STRATEGIES:
        save_dict[f"states_{strategy}"] = np.stack(all_states[strategy], axis=0)

    np.savez_compressed(out_path, **save_dict)
    file_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Saved {out_path} ({file_mb:.0f} MB)")
    print(f"Shape: ({len(samples)}, {num_layers}, {hidden_dim}) x {len(TOKEN_STRATEGIES)} strategies")


# ======================================================================
# Phase 2: analyze
# ======================================================================

def _load_ablation_data(model_short: str) -> dict:
    """Load collected ablation data from NPZ."""
    path = PROJECT_ROOT / "data" / "ablation" / f"{model_short}_raw_states.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"No ablation data found at {path}. "
            f"Run: python scripts/ablation_study.py collect --model {model_short}"
        )
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _generate_configs(num_layers: int) -> list[dict]:
    """Generate all configurations to test.

    Returns list of dicts with keys:
        name: human-readable config name
        strategy: token aggregation strategy
        layers: list of layer indices to use (0-indexed from first layer)
        layer_weights: weights for each layer (or None for equal)
    """
    configs = []

    for strategy in TOKEN_STRATEGIES:
        # --- Single layer configs ---
        for layer_idx in range(num_layers):
            configs.append({
                "name": f"L{layer_idx}/{strategy}",
                "strategy": strategy,
                "layers": [layer_idx],
                "layer_weights": None,
            })

        # --- Multi-layer equal weight ---
        for n_last in [2, 3, 4, 6, 8]:
            if n_last > num_layers:
                continue
            layer_indices = list(range(num_layers - n_last, num_layers))
            configs.append({
                "name": f"last{n_last}_equal/{strategy}",
                "strategy": strategy,
                "layers": layer_indices,
                "layer_weights": None,  # equal
            })

        # --- Multi-layer weighted (last 4 only, various weight schemes) ---
        if num_layers >= 4:
            last4 = list(range(num_layers - 4, num_layers))

            # Production weights
            configs.append({
                "name": f"last4_prod/{strategy}",
                "strategy": strategy,
                "layers": last4,
                "layer_weights": [0.1, 0.2, 0.3, 0.4],
            })

            # Inverse (shallower layers weighted more)
            configs.append({
                "name": f"last4_inverse/{strategy}",
                "strategy": strategy,
                "layers": last4,
                "layer_weights": [0.4, 0.3, 0.2, 0.1],
            })

            # Quadratic
            configs.append({
                "name": f"last4_quad/{strategy}",
                "strategy": strategy,
                "layers": last4,
                "layer_weights": [1, 4, 9, 16],  # will be normalized
            })

    return configs


def _aggregate_layers(states: np.ndarray, layer_indices: list[int],
                      layer_weights: list[float] | None) -> np.ndarray:
    """Aggregate across selected layers, skipping NaN layers.

    Args:
        states: (num_layers, hidden_dim) for one sample
        layer_indices: which layers to use
        layer_weights: weights per layer (None = equal)

    Returns:
        (hidden_dim,) aggregated vector in float32
    """
    selected = states[layer_indices].astype(np.float32)
    # Identify valid (non-NaN) layers
    valid_mask = ~np.any(np.isnan(selected), axis=1)  # (n_layers,)
    if not valid_mask.any():
        return np.full(selected.shape[1], np.nan, dtype=np.float32)
    selected = selected[valid_mask]
    if layer_weights is None:
        return selected.mean(axis=0)
    w = np.array(layer_weights, dtype=np.float32)[valid_mask]
    w = w / w.sum()
    return (selected * w[:, np.newaxis]).sum(axis=0)


def _compute_validation_metrics(axis_vector, val_pos_states, val_neg_states):
    """Compute d-prime and accuracy on validation set."""
    pos_proj = [np.dot(s, axis_vector) for s in val_pos_states]
    neg_proj = [np.dot(s, axis_vector) for s in val_neg_states]

    from src.calibration.axis_computer import compute_dprime
    dprime = compute_dprime(pos_proj, neg_proj)

    correct = sum(1 for p in pos_proj for n in neg_proj if p > n)
    total = len(pos_proj) * len(neg_proj)
    accuracy = correct / total if total > 0 else 0

    return {"dprime": dprime, "accuracy": accuracy}


def _evaluate_config(config: dict, data: dict) -> dict:
    """Evaluate a single configuration across all 8 axes.

    Returns dict with per-axis d-prime and accuracy, plus means.
    """
    strategy = config["strategy"]
    states_key = f"states_{strategy}"
    all_states = data[states_key]  # (320, num_layers, hidden_dim)
    sample_axes = data["_sample_axes"]
    sample_poles = data["_sample_poles"]

    per_axis = {}
    for axis in ALL_AXES:
        # Get indices for this axis
        axis_mask = sample_axes == axis
        pos_mask = axis_mask & (sample_poles == "positive")
        neg_mask = axis_mask & (sample_poles == "negative")

        pos_indices = np.where(pos_mask)[0]
        neg_indices = np.where(neg_mask)[0]

        # Aggregate layers for each sample (skip NaN samples)
        pos_aggregated = []
        for i in pos_indices:
            agg = _aggregate_layers(all_states[i], config["layers"], config["layer_weights"])
            if not np.any(np.isnan(agg)):
                pos_aggregated.append(agg)
        neg_aggregated = []
        for i in neg_indices:
            agg = _aggregate_layers(all_states[i], config["layers"], config["layer_weights"])
            if not np.any(np.isnan(agg)):
                neg_aggregated.append(agg)

        # Train/val split: use available samples (may be < 20 after NaN filter)
        n_pos = len(pos_aggregated)
        n_neg = len(neg_aggregated)
        train_n_pos = min(TRAIN_PER_POLE, n_pos - 1)  # leave at least 1 for val
        train_n_neg = min(TRAIN_PER_POLE, n_neg - 1)
        train_pos = pos_aggregated[:train_n_pos]
        val_pos = pos_aggregated[train_n_pos:train_n_pos + VAL_PER_POLE]
        train_neg = neg_aggregated[:train_n_neg]
        val_neg = neg_aggregated[train_n_neg:train_n_neg + VAL_PER_POLE]

        if len(train_pos) < 2 or len(train_neg) < 2 or not val_pos or not val_neg:
            per_axis[axis] = {"dprime": 0.0, "accuracy": 0.0}
            continue

        # Compute axis vector from train
        from src.calibration.axis_computer import compute_axis_vector
        axis_vector = compute_axis_vector(train_pos, train_neg)

        # Evaluate on val
        metrics = _compute_validation_metrics(axis_vector, val_pos, val_neg)
        per_axis[axis] = metrics

    # Compute means across axes (treat NaN as 0)
    dprimes = [v["dprime"] for v in per_axis.values()]
    accs = [v["accuracy"] for v in per_axis.values()]
    mean_dprime = float(np.nanmean(dprimes)) if any(np.isfinite(d) for d in dprimes) else 0.0
    mean_accuracy = float(np.nanmean(accs)) if any(np.isfinite(a) for a in accs) else 0.0

    return {
        "config_name": config["name"],
        "strategy": config["strategy"],
        "layers": config["layers"],
        "layer_weights": config["layer_weights"],
        "per_axis": per_axis,
        "mean_dprime": float(mean_dprime),
        "mean_accuracy": float(mean_accuracy),
    }


def _is_production_config(config_name: str) -> bool:
    """Check if this is the current production configuration."""
    return config_name == "last4_prod/decay_0.9"


def _sort_results(results: list[dict]) -> list[dict]:
    """Sort results by mean d-prime descending, NaN-safe."""
    return sorted(results, key=lambda r: r["mean_dprime"] if np.isfinite(r["mean_dprime"]) else -999, reverse=True)


def _print_top_configs(results: list[dict], top_n: int = 20):
    """Print top configs table to console."""
    sorted_results = _sort_results(results)

    print(f"\n{'='*80}")
    print(f"TOP {top_n} CONFIGURATIONS by mean d-prime (validation set)")
    print(f"{'='*80}")
    print(f"{'Rank':>4}  {'Config':<30}  {'d-prime':>8}  {'Accuracy':>8}  {'Note':<10}")
    print(f"{'-'*4}  {'-'*30}  {'-'*8}  {'-'*8}  {'-'*10}")

    prod_rank = None
    for i, r in enumerate(sorted_results):
        is_prod = _is_production_config(r["config_name"])
        if is_prod:
            prod_rank = i + 1
        if i < top_n or is_prod:
            marker = "<<< PROD" if is_prod else ""
            print(f"{i+1:>4}  {r['config_name']:<30}  {r['mean_dprime']:>8.3f}  {r['mean_accuracy']:>7.1%}  {marker}")
            if i == top_n - 1 and prod_rank is None:
                print(f"  ...")

    if prod_rank is not None:
        print(f"\nProduction config rank: {prod_rank}/{len(sorted_results)}")
    else:
        print("\nProduction config (last4_prod/decay_0.9) not found in results")

    # Print best config per strategy
    print(f"\n{'='*80}")
    print("BEST CONFIG PER TOKEN STRATEGY")
    print(f"{'='*80}")
    for strategy in TOKEN_STRATEGIES:
        strategy_results = [r for r in sorted_results if r["strategy"] == strategy]
        if strategy_results:
            best = strategy_results[0]
            print(f"  {strategy:<15}  {best['config_name']:<30}  d'={best['mean_dprime']:.3f}  acc={best['mean_accuracy']:.1%}")


def _save_results(results: list[dict], model_short: str) -> Path:
    """Save results to JSON."""
    out_dir = PROJECT_ROOT / "data" / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_short}_results.json"

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable = json.loads(json.dumps(results, default=convert))

    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"Results saved to {out_path}")
    return out_path


def _generate_figures(results: list[dict], data: dict, model_short: str):
    """Generate visualization figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping figures")
        return

    out_dir = PROJECT_ROOT / "data" / "ablation" / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)

    num_layers = int(data["_num_layers"])
    display_name = get_model_config(model_short).display_name

    # --- Figure 1: d-prime heatmap (layer x axis) for best token strategy ---
    _plot_layer_heatmap(results, num_layers, display_name, model_short, out_dir)

    # --- Figure 2: Top configs bar chart ---
    _plot_top_configs_bar(results, display_name, model_short, out_dir)

    # --- Figure 3: Strategy comparison (d-prime vs layer) ---
    _plot_strategy_comparison(results, num_layers, display_name, model_short, out_dir)


def _plot_layer_heatmap(results, num_layers, display_name, model_short, out_dir):
    """Heatmap of d-prime per (single layer, axis) using the best token strategy."""
    import matplotlib.pyplot as plt

    # Find best overall strategy
    strategy_means = {}
    for strategy in TOKEN_STRATEGIES:
        strat_results = [r for r in results if r["strategy"] == strategy
                         and len(r["layers"]) == 1]
        if strat_results:
            strategy_means[strategy] = np.mean([r["mean_dprime"] for r in strat_results])
    best_strategy = max(strategy_means, key=strategy_means.get)

    # Build heatmap data
    heatmap = np.zeros((num_layers, len(ALL_AXES)))
    for r in results:
        if r["strategy"] == best_strategy and len(r["layers"]) == 1:
            layer_idx = r["layers"][0]
            for j, axis in enumerate(ALL_AXES):
                heatmap[layer_idx, j] = r["per_axis"][axis]["dprime"]

    fig, ax = plt.subplots(figsize=(12, max(6, num_layers * 0.25)))
    im = ax.imshow(heatmap, aspect="auto", cmap="RdYlGn", interpolation="nearest")
    ax.set_xlabel("Axis")
    ax.set_ylabel("Layer index")
    ax.set_xticks(range(len(ALL_AXES)))
    ax.set_xticklabels([a.replace("_", "\n") for a in ALL_AXES], fontsize=8)

    # Show only every Nth layer on y-axis for readability
    step = max(1, num_layers // 20)
    ax.set_yticks(range(0, num_layers, step))

    ax.set_title(f"{display_name}: d-prime by layer (strategy={best_strategy})")
    plt.colorbar(im, ax=ax, label="d-prime")
    fig.tight_layout()
    path = out_dir / f"{model_short}_layer_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def _plot_top_configs_bar(results, display_name, model_short, out_dir):
    """Bar chart of top 20 configs with production config marked."""
    import matplotlib.pyplot as plt

    sorted_results = _sort_results(results)[:20]
    names = [r["config_name"] for r in sorted_results]
    dprimes = [r["mean_dprime"] for r in sorted_results]
    colors = ["#e74c3c" if _is_production_config(n) else "#3498db" for n in names]

    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = range(len(names))
    ax.barh(y_pos, dprimes, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean d-prime (validation)")
    ax.set_title(f"{display_name}: Top 20 Configurations")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e74c3c", label="Production config"),
        Patch(facecolor="#3498db", label="Other configs"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    fig.tight_layout()
    path = out_dir / f"{model_short}_top_configs.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def _plot_strategy_comparison(results, num_layers, display_name, model_short, out_dir):
    """Line plot: d-prime vs layer index, one line per token strategy."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    colors_map = {
        "last_token": "#e74c3c",
        "mean": "#3498db",
        "decay_0.9": "#2ecc71",
        "decay_0.8": "#f39c12",
    }

    for strategy in TOKEN_STRATEGIES:
        layer_dprimes = []
        for layer_idx in range(num_layers):
            matching = [r for r in results
                        if r["strategy"] == strategy
                        and r["layers"] == [layer_idx]]
            if matching:
                layer_dprimes.append(matching[0]["mean_dprime"])
            else:
                layer_dprimes.append(0)

        ax.plot(range(num_layers), layer_dprimes,
                label=strategy, color=colors_map.get(strategy, "gray"),
                alpha=0.8, linewidth=1.5)

    ax.set_xlabel("Layer index")
    ax.set_ylabel("Mean d-prime (validation)")
    ax.set_title(f"{display_name}: d-prime by Layer and Token Strategy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / f"{model_short}_strategy_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def run_analyze(model_short: str):
    """Phase 2: analyze collected hidden states."""
    print(f"=== Ablation Study: ANALYZE for {model_short} ===")

    data = _load_ablation_data(model_short)
    num_layers = int(data["_num_layers"])
    hidden_dim = int(data["_hidden_dim"])
    model_id = str(data["_model_id"])
    print(f"Model: {model_id}")
    print(f"Layers: {num_layers}, Hidden dim: {hidden_dim}")
    print(f"Samples: {len(data['_sample_axes'])}")
    print(f"Axes: {list(data['_axes'])}")

    # Generate configs
    configs = _generate_configs(num_layers)
    print(f"Configurations to test: {len(configs)}")

    # Evaluate each config
    results = []
    for config in tqdm(configs, desc="Evaluating configs"):
        result = _evaluate_config(config, data)
        results.append(result)

    # Print results
    _print_top_configs(results)

    # Per-axis breakdown for production config
    prod_results = [r for r in results if _is_production_config(r["config_name"])]
    if prod_results:
        prod = prod_results[0]
        print(f"\n{'='*80}")
        print("PRODUCTION CONFIG DETAIL (last4_prod/decay_0.9)")
        print(f"{'='*80}")
        print(f"{'Axis':<25}  {'d-prime':>8}  {'Accuracy':>8}")
        print(f"{'-'*25}  {'-'*8}  {'-'*8}")
        for axis in ALL_AXES:
            m = prod["per_axis"][axis]
            print(f"{axis:<25}  {m['dprime']:>8.3f}  {m['accuracy']:>7.1%}")
        print(f"{'MEAN':<25}  {prod['mean_dprime']:>8.3f}  {prod['mean_accuracy']:>7.1%}")

    # Save results
    _save_results(results, model_short)

    # Generate figures
    _generate_figures(results, data, model_short)


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ablation study for hidden state extraction hyperparameters",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # collect
    p_collect = subparsers.add_parser("collect", help="Run inference and save raw hidden states (GPU)")
    p_collect.add_argument("--model", required=True, choices=list(MODELS.keys()),
                           help="Model short name")

    # analyze
    p_analyze = subparsers.add_parser("analyze", help="Analyze collected states (CPU)")
    p_analyze.add_argument("--model", required=True, choices=list(MODELS.keys()),
                           help="Model short name")

    args = parser.parse_args()

    if args.command == "collect":
        run_collect(args.model)
    elif args.command == "analyze":
        run_analyze(args.model)


if __name__ == "__main__":
    main()
