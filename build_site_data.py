#!/usr/bin/env python3
"""Build docs/site_data.json for the interactive visualization site.

Reads from data/ (baselines, stability, steering, drift) and produces
a structured JSON for the Chart.js-powered GitHub Pages site.

Usage:
    python build_site_data.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = ROOT / "docs" / "site_data.json"

MODELS = {
    # --- Paper models (main text, 5 orgs) ---
    "qwen3.5_9b":     {"name": "Qwen3.5 9B",         "org": "Alibaba",    "params": "9.0B",  "color": "#E9C46A"},
    "gemma3_12b":      {"name": "Gemma 3 12B",         "org": "Google",     "params": "12B",   "color": "#22D3EE"},
    "phi4":            {"name": "Phi-4 14B",            "org": "Microsoft",  "params": "14B",   "color": "#7B68EE"},
    "deepseek_r1_14b": {"name": "DeepSeek-R1 14B",     "org": "DeepSeek",   "params": "14B",   "color": "#F43F5E"},
    "ministral3_8b":   {"name": "Ministral 3 8B",      "org": "Mistral AI", "params": "8.5B",  "color": "#7C3AED"},
    # --- Paper extra ---
    "gpt_oss_20b":     {"name": "GPT-OSS 20B",         "org": "OpenAI",     "params": "20B",   "color": "#10A37F"},
    # --- Legacy (appendix) ---
    "qwen_7b":         {"name": "Qwen 2.5 7B",         "org": "Alibaba",    "params": "7.6B",  "color": "#FF6B35"},
    "llama_8b":        {"name": "Llama 3.1 8B",        "org": "Meta",       "params": "8.0B",  "color": "#3B82F6"},
    "gemma_9b":        {"name": "Gemma 2 9B",          "org": "Google",     "params": "9.2B",  "color": "#4285F4"},
    "mistral_7b":      {"name": "Mistral 7B v0.3",     "org": "Mistral AI", "params": "7.2B",  "color": "#A855F7"},
    "qwen3_8b":        {"name": "Qwen3 8B",            "org": "Alibaba",    "params": "8.2B",  "color": "#FF4500"},
    # --- Base models ---
    "qwen_7b_base":    {"name": "Qwen 2.5 7B (base)",  "org": "Alibaba",    "params": "7.6B",  "color": "#FF6B35", "is_base": True, "instruct": "qwen_7b"},
    "llama_8b_base":   {"name": "Llama 3.1 8B (base)", "org": "Meta",       "params": "8.0B",  "color": "#3B82F6", "is_base": True, "instruct": "llama_8b"},
    "gemma_9b_base":   {"name": "Gemma 2 9B (base)",   "org": "Google",     "params": "9.2B",  "color": "#4285F4", "is_base": True, "instruct": "gemma_9b"},
    "mistral_7b_base": {"name": "Mistral 7B (base)",   "org": "Mistral AI", "params": "7.2B",  "color": "#A855F7", "is_base": True, "instruct": "mistral_7b"},
    # --- Diverse architectures ---
    "exaone_7b":       {"name": "EXAONE 3.5 7.8B",    "org": "LG AI",      "params": "7.8B",  "color": "#DC267F"},
    "falcon_h1_7b":    {"name": "Falcon-H1 7B",       "org": "TII",        "params": "7.6B",  "color": "#C77DFF"},
    "glm_z1_9b":       {"name": "GLM-Z1 9B",          "org": "Zhipu AI",   "params": "9.0B",  "color": "#FF8C00"},
    "granite_8b":      {"name": "Granite 3.3 8B",     "org": "IBM",        "params": "8.2B",  "color": "#648FFF"},
    "internlm3_8b":    {"name": "InternLM3 8B",       "org": "Shanghai AI", "params": "8.0B", "color": "#FF1493"},
    "nemotron_h_8b":   {"name": "Nemotron-H 8B",      "org": "NVIDIA",     "params": "8.0B",  "color": "#76B900"},
    "olmo2_13b":       {"name": "OLMo-2 13B",         "org": "Allen AI",   "params": "13B",   "color": "#6A0DAD"},
    "yi_9b":           {"name": "Yi-1.5 9B",          "org": "01.AI",      "params": "9.0B",  "color": "#00B4D8"},
    "smollm3_3b":      {"name": "SmolLM3 3B",         "org": "HuggingFace", "params": "3.0B", "color": "#785EF0"},
    # --- Qwen 2.5 scaling chain ---
    "qwen2.5_0.5b":    {"name": "Qwen 2.5 0.5B",     "org": "Alibaba",    "params": "0.5B",  "color": "#FFB347"},
    "qwen_1.5b":       {"name": "Qwen 2.5 1.5B",     "org": "Alibaba",    "params": "1.5B",  "color": "#FF9F1C"},
    "qwen2.5_3b":      {"name": "Qwen 2.5 3B",       "org": "Alibaba",    "params": "3.0B",  "color": "#FF7F50"},
    # --- Qwen 3 scaling chain ---
    "qwen3_0.6b":      {"name": "Qwen3 0.6B",        "org": "Alibaba",    "params": "0.6B",  "color": "#DC143C"},
    "qwen3_1.7b":      {"name": "Qwen3 1.7B",        "org": "Alibaba",    "params": "1.7B",  "color": "#CD5C5C"},
    "qwen3_4b":        {"name": "Qwen3 4B",           "org": "Alibaba",    "params": "4.0B",  "color": "#B22222"},
    # --- Qwen 3.5 scaling chain ---
    "qwen3.5_0.8b":    {"name": "Qwen3.5 0.8B",      "org": "Alibaba",    "params": "0.8B",  "color": "#E63946"},
    "qwen3.5_2b":      {"name": "Qwen3.5 2B",        "org": "Alibaba",    "params": "2.0B",  "color": "#E76F51"},
    "qwen3.5_4b":      {"name": "Qwen3.5 4B",        "org": "Alibaba",    "params": "4.0B",  "color": "#F4A261"},
    # --- Falcon-H1 scaling chain ---
    "falcon_h1_0.5b":  {"name": "Falcon-H1 0.5B",    "org": "TII",        "params": "0.5B",  "color": "#C77DFF"},
    "falcon_h1_1.5b":  {"name": "Falcon-H1 1.5B",    "org": "TII",        "params": "1.5B",  "color": "#C77DFF"},
    # --- Other small ---
    "smollm_1.7b":     {"name": "SmolLM2 1.7B",      "org": "HuggingFace", "params": "1.7B", "color": "#2EC4B6"},
    "granite_2b":      {"name": "Granite 3.3 2B",    "org": "IBM",        "params": "2.5B",  "color": "#648FFF"},
}

AXES = [
    "warm_cold", "confident_cautious", "empathetic_analytical",
    "formal_casual", "verbose_concise",
]

AXIS_LABELS = {
    "warm_cold": ["Warm", "Cold"],
    "confident_cautious": ["Confident", "Cautious"],
    "empathetic_analytical": ["Empathetic", "Analytical"],
    "formal_casual": ["Formal", "Casual"],
    "verbose_concise": ["Verbose", "Concise"],
}

MODEL_ORDER = list(MODELS.keys())

GROUPS = {
    "paper": {"label": "Paper Models (main text)", "models": [
        "qwen3.5_9b", "gemma3_12b", "phi4", "deepseek_r1_14b", "ministral3_8b",
    ]},
    "paper_extra": {"label": "Paper Extra", "models": ["gpt_oss_20b"]},
    "legacy": {"label": "Legacy Instruct", "models": [
        "qwen_7b", "llama_8b", "gemma_9b", "mistral_7b", "qwen3_8b",
    ]},
    "base": {"label": "Base Models", "models": [
        "qwen_7b_base", "llama_8b_base", "gemma_9b_base", "mistral_7b_base",
    ]},
    "diverse": {"label": "Diverse Architectures", "models": [
        "exaone_7b", "falcon_h1_7b", "glm_z1_9b", "granite_8b",
        "internlm3_8b", "nemotron_h_8b", "olmo2_13b", "yi_9b", "smollm3_3b",
    ]},
    "qwen2.5_chain": {"label": "Qwen 2.5 Scaling", "models": [
        "qwen2.5_0.5b", "qwen_1.5b", "qwen2.5_3b", "qwen_7b",
    ]},
    "qwen3_chain": {"label": "Qwen 3 Scaling", "models": [
        "qwen3_0.6b", "qwen3_1.7b", "qwen3_4b", "qwen3_8b",
    ]},
    "qwen3.5_chain": {"label": "Qwen 3.5 Scaling", "models": [
        "qwen3.5_0.8b", "qwen3.5_2b", "qwen3.5_4b", "qwen3.5_9b",
    ]},
    "falcon_h1_chain": {"label": "Falcon-H1 Scaling", "models": [
        "falcon_h1_0.5b", "falcon_h1_1.5b", "falcon_h1_7b",
    ]},
    "small": {"label": "Other Small", "models": [
        "smollm_1.7b", "granite_2b",
    ]},
}


def load(path):
    if path.exists():
        return json.load(open(path))
    return {}


def build():
    result = {
        "axes": AXES,
        "axis_labels": AXIS_LABELS,
        "model_order": MODEL_ORDER,
        "groups": GROUPS,
        "models": {},
    }

    for model_key, meta in MODELS.items():
        m = dict(meta)

        # --- Fingerprint from baselines ---
        bl = load(DATA / "baselines" / f"{model_key}.json")
        if bl:
            m["fingerprint"] = {}
            m["fingerprint_std"] = {}
            for axis, vals in bl.get("axes", {}).items():
                if axis in AXES:
                    m["fingerprint"][axis] = round(vals["mean"], 3)
                    m["fingerprint_std"][axis] = round(vals["std"], 3)

        # --- Stability ---
        # Use cosine ICC summary (computed from set A/B/C axes)
        cosine_summary = load(DATA / "stability" / "cosine_icc_summary.json")
        if model_key in cosine_summary:
            cs = cosine_summary[model_key]
            per_axis = {ax: v for ax, v in cs.get("per_axis", {}).items() if ax in AXES}
            m["stability"] = {
                "icc": cs["mean_cosine"],
                "icc_per_axis": per_axis,
                "cosine": per_axis,
            }
        # Fallback: old format
        if "stability" not in m:
            stab = load(DATA / "stability" / f"{model_key}.json")
            if stab:
                icc = stab.get("icc", {})
                m["stability"] = {
                    "icc": icc.get("mean_icc"),
                    "icc_per_axis": icc.get("per_axis", {}),
                    "cosine": {ax: round(s["mean_cosine"], 3) for ax, s in stab.get("per_axis_summary", {}).items() if ax in AXES},
                }

        # --- Steering ---
        steer = load(DATA / "steering" / f"{model_key}_basic.json")
        if steer:
            axes_data = steer.get("axes", {})
            steering = {}
            total_effect = 0
            for axis, ad in axes_data.items():
                if axis not in AXES:
                    continue
                slope = ad.get("slope", 0)
                alpha_means_raw = ad.get("alpha_means", {})
                alpha_means = {}
                for ak, av in alpha_means_raw.items():
                    if isinstance(av, dict):
                        alpha_means[ak] = av.get(axis, 0.0)
                    else:
                        alpha_means[ak] = av
                if alpha_means:
                    vals = list(alpha_means.values())
                    effect = max(vals) - min(vals)
                else:
                    effect = abs(slope * 10)

                leakage_vals = ad.get("leakage", {})
                leak_ratios = []
                for l in leakage_vals.values():
                    if isinstance(l, dict):
                        leak_ratios.append(abs(l.get("ratio", 0)))
                    elif isinstance(l, (int, float)):
                        leak_ratios.append(abs(l))
                mean_leak = sum(leak_ratios) / max(len(leak_ratios), 1)

                steering[axis] = {
                    "slope": round(slope, 5),
                    "effect": round(effect, 3),
                    "leakage": round(mean_leak, 2),
                    "alpha_means": {k: round(v, 3) for k, v in alpha_means.items()},
                }
                total_effect += effect
            m["steering"] = steering
            m["steering_total"] = round(total_effect, 3)

        # --- Sweep (optimal steering layer) ---
        sweep = load(DATA / "steering_sweep" / f"{model_key}_sweep.json")
        if sweep:
            m["sweep"] = {
                "best_layer": sweep["best_layer"],
                "best_layer_pct": sweep["best_layer_pct"],
                "best_effect": round(sweep["best_effect"], 4),
                "n_layers": sweep.get("n_layers", 0),
            }

        # --- Drift ---
        dr = load(DATA / "drift" / f"{model_key}.json")
        if dr:
            drift = {}
            # Support both old (drift_summary) and new (conflict_drift) formats
            drift_data = dr.get("conflict_drift", dr.get("drift_summary", {}))
            for axis, stats in drift_data.items():
                if axis not in AXES:
                    continue
                # New format: mean_conflict_slope/delta. Old: mean_slope/delta.
                slope = stats.get("mean_conflict_slope", stats.get("mean_slope", 0))
                delta = stats.get("mean_conflict_delta", stats.get("mean_delta", 0))
                drift[axis] = {
                    "slope": round(slope, 4),
                    "delta": round(delta, 3),
                }
            if drift:
                m["drift"] = drift
                m["drift_mean"] = round(
                    sum(abs(d["delta"]) for d in drift.values()) / max(len(drift), 1), 3
                )

        # Only include models with data
        if any(k in m for k in ("fingerprint", "stability", "steering", "drift")):
            result["models"][model_key] = m

    return result


def main():
    data = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(data, f, indent=2)

    models = data["models"]
    n_fp = sum(1 for m in models.values() if "fingerprint" in m)
    n_st = sum(1 for m in models.values() if "stability" in m)
    n_sr = sum(1 for m in models.values() if "steering" in m)
    n_dr = sum(1 for m in models.values() if "drift" in m)
    print(f"Wrote {OUT}")
    print(f"  {len(models)} models: {n_fp} fingerprints, {n_st} stability, {n_sr} steering, {n_dr} drift")


if __name__ == "__main__":
    main()
