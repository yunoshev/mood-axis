#!/usr/bin/env python3
"""Build paper_data.json from data/**/*.json.

Reads all computed data files and produces a flat JSON with {{key}} values
that can be injected into README sections and other documents.

Usage:
    python build_data.py
    python build_data.py --check   # show keys without writing
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_FILE = PROJECT_ROOT / "paper_data.json"

DISPLAY = {
    "qwen_7b": "Qwen 2.5 7B", "llama_8b": "Llama 3.1 8B",
    "mistral_7b": "Mistral 7B", "gemma_9b": "Gemma 2 9B",
    "qwen3_8b": "Qwen3 8B", "deepseek_7b": "DeepSeek 7B",
    "yi_9b": "Yi 1.5 9B", "gpt_oss_20b": "GPT-OSS 20B",
    "qwen_1.5b": "Qwen 2.5 1.5B", "smollm_1.7b": "SmolLM2 1.7B",
    "llama_1b": "Llama 3.2 1B",
    "gemma_9b_base": "Gemma 2 9B (base)", "llama_8b_base": "Llama 3.1 8B (base)",
    "qwen_7b_base": "Qwen 2.5 7B (base)", "mistral_7b_base": "Mistral 7B (base)",
    "yi_9b_base": "Yi 1.5 9B (base)", "deepseek_7b_base": "DeepSeek 7B (base)",
}

AXES_5 = ["warm_cold", "confident_cautious", "empathetic_analytical", "formal_casual", "verbose_concise"]
AXES_7 = AXES_5 + ["patient_irritated", "proactive_reluctant"]

PAPER_MODELS = ["qwen_7b", "llama_8b", "mistral_7b", "gemma_9b", "qwen3_8b"]


def load_json(path: Path) -> dict:
    if path.exists():
        return json.load(open(path))
    return {}


class PaperData:
    def __init__(self):
        self.data = {}
        self.sources = {}

    def set(self, key: str, value, source: str):
        self.data[key] = value
        self.sources[key] = source

    def set_many(self, pairs: dict, source: str):
        for k, v in pairs.items():
            self.set(k, v, source)

    def has(self, key: str) -> bool:
        return key in self.data

    def result(self) -> dict:
        result = dict(self.data)
        result["_meta"] = {
            "script": "build_data.py",
            "timestamp": datetime.now().isoformat(),
            "n_keys": len(self.data),
            "sources": dict(self.sources),
        }
        return result


def fmt(v, decimals=3):
    return f"{v:.{decimals}f}"


def build_data() -> dict:
    pd = PaperData()

    # --- Static config ---
    pd.set_many({
        "n_axes": "7",
        "n_axes_steering": "5",
        "temperature": "0.7",
        "top_p": "0.9",
        "hidden_layers_to_use": "4",
        "layer_weights": "[0.1, 0.2, 0.3, 0.4]",
        "token_weight_decay": "0.9",
        "max_new_tokens_baseline": "384",
        "max_new_tokens_calibration": "200",
        "total_calibration_questions": "210",
        "eval_questions": "70",
        "baseline_questions": "30",
        "total_unique_questions": "310",
        "grand_total_questions": "940",
        "signal_concentration_final_layer": "95%+",
        "github_url": "https://github.com/yunoshev/mood-axis",
        "conflict_scenarios": "50",
        "conflict_turns": "16",
        "conflict_turns_conflict": "12",
        "conflict_turns_recovery": "4",
        "control_conversations": "10",
        "steering_alphas": "{-5, -2, -1, 0, +1, +2, +5}",
        "steering_questions_per_combo": "30",
        "n_paper_models": str(len(PAPER_MODELS)),
    }, source="config")

    # --- Baselines & fingerprints ---
    fingerprints = load_json(DATA_DIR / "analysis" / "fingerprints.json")
    baselines_dir = DATA_DIR / "baselines"
    instruct_models, base_models, all_models = [], [], []

    for f in sorted(baselines_dir.glob("*.json")) if baselines_dir.exists() else []:
        model_key = f.stem
        bl = load_json(f)
        axes = bl.get("axes", {})
        all_models.append(model_key)
        if "_base" in model_key:
            base_models.append(model_key)
        else:
            instruct_models.append(model_key)
        for axis in AXES_7:
            if axis in axes:
                pd.set(f"baseline_{model_key}_{axis}_mean", fmt(axes[axis]["mean"]), "baseline")
                pd.set(f"baseline_{model_key}_{axis}_std", fmt(axes[axis]["std"]), "baseline")

    pd.set_many({
        "n_models": str(len(all_models)),
        "n_models_instruct": str(len(instruct_models)),
        "n_models_base": str(len(base_models)),
        "model_size_range": "1B\u201320B",
    }, source="baseline")

    # Fingerprints table (paper models)
    fp_rows = []
    for mk in PAPER_MODELS:
        fp = fingerprints.get(mk, {})
        if not fp:
            continue
        name = DISPLAY.get(mk, mk)
        vals = " | ".join(f"{fp.get(a, 0):+.2f}" for a in AXES_5)
        fp_rows.append(f"| **{name}** | {vals} |")
    if fp_rows:
        hdr = "| Model | warm/cold | confident/cautious | empathetic/analytical | formal/casual | verbose/concise |"
        sep = "|-------|:---------:|:------------------:|:--------------------:|:-------------:|:---------------:|"
        pd.set("table_fingerprints", hdr + "\n" + sep + "\n" + "\n".join(fp_rows), "baseline")

    # --- PCA dimensionality ---
    pca = load_json(DATA_DIR / "analysis" / "pca_dimensionality.json")
    pc1_values, eff_dims = [], []
    pca_rows = []
    for mk in sorted(pca.keys()):
        if mk.startswith("_") or "pc1_variance" not in pca[mk]:
            continue
        pc1 = pca[mk]["pc1_variance"]
        effdim = pca[mk]["effective_dimensionality"]
        pc1_values.append(pc1)
        eff_dims.append(effdim)
        pd.set(f"pca_{mk}_pc1", f"{pc1*100:.1f}", "pca")
        pd.set(f"pca_{mk}_effdim", f"{effdim:.2f}", "pca")
        pca_rows.append((mk, pc1, effdim))

    if pc1_values:
        pd.set_many({
            "pca_pc1_min": f"{min(pc1_values)*100:.1f}",
            "pca_pc1_max": f"{max(pc1_values)*100:.1f}",
            "pca_effdim_min": f"{min(eff_dims):.1f}",
            "pca_effdim_max": f"{max(eff_dims):.1f}",
            "pc1_range": f"{min(pc1_values)*100:.1f}\u2013{max(pc1_values)*100:.1f}%",
            "effective_dimension_range": f"{min(eff_dims):.1f}\u2013{max(eff_dims):.1f}",
        }, source="pca")

    # PCA table (paper models only)
    pca_table_rows = []
    for mk in PAPER_MODELS:
        d = pca.get(mk, {})
        if "pc1_variance" not in d:
            continue
        name = DISPLAY.get(mk, mk)
        pca_table_rows.append(f"| {name} | {d['pc1_variance']*100:.1f}% | {d['effective_dimensionality']:.2f} |")
    if pca_table_rows:
        hdr = "| Model | PC1 (%) | Eff. Dim |\n|-------|:------:|:--------:|"
        pd.set("table_pca", hdr + "\n" + "\n".join(pca_table_rows), "pca")

    # PCA table (all models)
    pca_all_rows = []
    for mk, pc1, effdim in sorted(pca_rows, key=lambda x: -x[1]):
        name = DISPLAY.get(mk, mk)
        pca_all_rows.append(f"| {name} | {pc1*100:.1f}% | {effdim:.2f} |")
    if pca_all_rows:
        hdr = "| Model | PC1 (%) | Eff. Dim |\n|-------|:------:|:--------:|"
        pd.set("table_pca_all", hdr + "\n" + "\n".join(pca_all_rows), "pca")

    # --- Dead zones ---
    dead = load_json(DATA_DIR / "analysis" / "dead_zones.json")
    dead_rows = []
    for mk in PAPER_MODELS:
        dz = dead.get(mk, {})
        if "n_dead" not in dz:
            continue
        name = DISPLAY.get(mk, mk)
        n = dz["n_dead"]
        axes_str = ", ".join(a.replace("_", "/") for a in dz["dead_axes"]) if dz["dead_axes"] else "\u2014"
        pd.set(f"dead_{mk}_n", str(n), "dead_zones")
        pd.set(f"dead_{mk}_axes", axes_str, "dead_zones")
        dead_rows.append(f"| {name} | {n} | {axes_str} |")
    if dead_rows:
        hdr = "| Model | Dead Zones | Suppressed Axes |\n|-------|:----------:|-----------------|"
        pd.set("table_dead_zones", hdr + "\n" + "\n".join(dead_rows), "dead_zones")

    # Dead zones with base models
    dead_base_rows = []
    for mk in ["gemma_9b", "gemma_9b_base", "llama_8b", "llama_8b_base", "mistral_7b", "mistral_7b_base",
                "qwen_7b", "qwen_7b_base", "yi_9b", "yi_9b_base"]:
        dz = dead.get(mk, {})
        if "n_dead" not in dz:
            continue
        name = DISPLAY.get(mk, mk)
        n = dz["n_dead"]
        axes_str = ", ".join(a.replace("_", "/") for a in dz["dead_axes"]) if dz["dead_axes"] else "\u2014"
        dead_base_rows.append(f"| {name} | {n} | {axes_str} |")
    if dead_base_rows:
        hdr = "| Model | Dead Zones | Suppressed Axes |\n|-------|:----------:|-----------------|"
        pd.set("table_dead_zones_with_base", hdr + "\n" + "\n".join(dead_base_rows), "dead_zones")

    # --- Drift ---
    drift = load_json(DATA_DIR / "analysis" / "drift_summary.json")
    model_drifts = {}
    for mk, drift_data in drift.items():
        if mk.startswith("_") or not isinstance(drift_data, dict):
            continue
        deltas = []
        for axis, stats in drift_data.items():
            if isinstance(stats, dict) and "mean_delta" in stats:
                deltas.append(abs(stats["mean_delta"]))
                pd.set(f"drift_{mk}_{axis}_delta", f"{stats['mean_delta']:+.3f}", "drift")
                pd.set(f"drift_{mk}_{axis}_slope", f"{stats['mean_slope']:.4f}", "drift")
        if deltas:
            model_drifts[mk] = float(np.mean(deltas))
            pd.set(f"drift_{mk}_mean", f"{model_drifts[mk]:.3f}", "drift")

    drift_rows = []
    for mk, d in sorted(model_drifts.items(), key=lambda x: x[1]):
        if not isinstance(d, (int, float)):
            continue
        name = DISPLAY.get(mk, mk)
        drift_rows.append(f"| {name} | {d:.3f} |")
    if drift_rows:
        hdr = "| Model | Mean |abs(delta)| |\n|-------|:-----------:|"
        pd.set("table_drift", hdr + "\n" + "\n".join(drift_rows), "drift")

    # --- Stability ---
    stability_dir = DATA_DIR / "stability"
    model_icc = {}
    model_rank_icc = {}
    if stability_dir.exists():
        for f in stability_dir.glob("*.json"):
            if f.stem in ("summary", "_meta") or f.stem.startswith("_"):
                continue
            mk = f.stem
            stab = load_json(f)
            # ICC
            icc = stab.get("icc", {})
            if "mean_icc" in icc:
                model_icc[mk] = icc["mean_icc"]
                pd.set(f"icc_{mk}", f"{icc['mean_icc']:.3f}", "stability")
            # Rank ICC
            rank_icc = stab.get("rank_icc", {})
            if "mean_rank_icc" in rank_icc:
                model_rank_icc[mk] = rank_icc["mean_rank_icc"]
                pd.set(f"rank_icc_{mk}", f"{rank_icc['mean_rank_icc']:.3f}", "stability")
            # Procrustes
            proc = stab.get("procrustes", {})
            if "mean_congruence" in proc:
                pd.set(f"procrustes_{mk}", f"{proc['mean_congruence']:.3f}", "stability")

    if model_icc:
        # Use paper models only for range (small models have lower ICC)
        paper_icc = {k: v for k, v in model_icc.items() if k in PAPER_MODELS}
        icc_vals = paper_icc.values() if paper_icc else model_icc.values()
        pd.set("icc_range", f"{min(icc_vals):.2f}\u2013{max(icc_vals):.2f}", "stability")
        icc_rows = []
        for mk in PAPER_MODELS:
            if mk not in model_icc:
                continue
            name = DISPLAY.get(mk, mk)
            icc_val = model_icc[mk]
            rank_val = model_rank_icc.get(mk)
            rank_str = f"{rank_val:.3f}" if rank_val else "\u2014"
            icc_rows.append(f"| {name} | {icc_val:.3f} | {rank_str} |")
        if icc_rows:
            hdr = "| Model | ICC | Rank ICC |\n|-------|:---:|:--------:|"
            pd.set("table_stability", hdr + "\n" + "\n".join(icc_rows), "stability")

    # --- Steering ---
    steering = load_json(DATA_DIR / "analysis" / "steering_summary.json")
    judge = load_json(DATA_DIR / "steering" / "judge_results.json")
    per_model = steering.get("per_model", {})
    rankings = steering.get("rankings", [])
    findings = steering.get("findings", {})
    heatmap_data = steering.get("axis_heatmap", {})

    for mk, mdata in per_model.items():
        pd.set(f"steering_{mk}_total", f"{mdata['total_effect']:.2f}", "steering")
        pd.set(f"steering_{mk}_cluster", mdata["cluster"], "steering")
        pd.set(f"steering_{mk}_leakage", f"{mdata['mean_leakage']:.2f}", "steering")
        for axis, adat in mdata.get("axes", {}).items():
            pd.set(f"steering_{mk}_{axis}_slope", f"{adat['slope']:.4f}", "steering")
            pd.set(f"steering_{mk}_{axis}_effect", f"{adat['total_effect']:.3f}", "steering")

    # Judge keys
    for mk, jdata in judge.items():
        if isinstance(jdata, dict) and "pairwise_accuracy" in jdata:
            pd.set(f"judge_{mk}_pairwise", f"{jdata['pairwise_accuracy']*100:.0f}%", "judge")
            if jdata.get("likert_rho_alpha") is not None:
                pd.set(f"judge_{mk}_rho", f"{jdata['likert_rho_alpha']:.2f}", "judge")

    if findings:
        most = findings.get("most_steerable", {})
        least = findings.get("least_steerable", {})
        pd.set_many({
            "steering_most_steerable": most.get("display_name", ""),
            "steering_most_steerable_effect": f"{most.get('total_effect', 0):.2f}",
            "steering_least_steerable": least.get("display_name", ""),
            "steering_least_steerable_effect": f"{least.get('total_effect', 0):.2f}",
        }, source="steering")

        dz_corr = findings.get("dead_zone_correlation", {})
        if dz_corr:
            pd.set("steering_dz_rho", f"{dz_corr.get('spearman_rho', 0):.2f}", "steering")
            pd.set("steering_dz_p", f"{dz_corr.get('p_value', 0):.2f}", "steering")

    # Steering overview table
    if rankings:
        rows = []
        for r in rankings:
            name = r.get("display_name", r["model"])
            rows.append(f"| {name} | {r['total_effect']:.2f} | {r['cluster']} |")
        hdr = "| Model | Total Effect | Cluster |\n|-------|:-----------:|:-------:|"
        pd.set("table_steering_overview", hdr + "\n" + "\n".join(rows), "steering")

    # Steering heatmap
    if heatmap_data and rankings:
        axes_order = sorted(next(iter(heatmap_data.values())).keys())
        hdr = "| Model | " + " | ".join(a.replace("_", "/") for a in axes_order) + " |"
        sep = "|-------|" + "|".join(":------:" for _ in axes_order) + "|"
        hm_rows = []
        for r in rankings:
            mk = r["model"]
            name = r.get("display_name", mk)
            vals = " | ".join(f"{heatmap_data[mk].get(a, 0):.3f}" for a in axes_order)
            hm_rows.append(f"| {name} | {vals} |")
        pd.set("table_steering_heatmap", hdr + "\n" + sep + "\n" + "\n".join(hm_rows), "steering")

    # Steering + judge combined table
    combined_rows = []
    for r in rankings:
        mk = r["model"]
        name = r.get("display_name", mk)
        jd = judge.get(mk, {})
        pw = f"{jd['pairwise_accuracy']*100:.0f}%" if isinstance(jd, dict) and "pairwise_accuracy" in jd else "\u2014"
        rho = f"{jd['likert_rho_alpha']:.2f}" if isinstance(jd, dict) and jd.get("likert_rho_alpha") is not None else "\u2014"
        combined_rows.append(f"| {name} | {r['total_effect']:.2f} | {r['cluster']} | {pw} | {rho} |")
    if combined_rows:
        hdr = "| Model | Total Effect | Cluster | Judge Pairwise | Judge Likert \u03c1 |\n|-------|:-----------:|:-------:|:--------------:|:-------------:|"
        pd.set("table_steering_full", hdr + "\n" + "\n".join(combined_rows), "steering")

    # Judge table
    judge_rows = []
    for mk in ["mistral_7b", "llama_8b", "qwen_7b", "gemma_9b", "qwen3_8b"]:
        jd = judge.get(mk, {})
        if not isinstance(jd, dict) or "pairwise_accuracy" not in jd:
            continue
        name = DISPLAY.get(mk, mk)
        pw = f"{jd['pairwise_accuracy']*100:.0f}%"
        rho = f"{jd.get('likert_rho_alpha', 0):.2f}" if jd.get("likert_rho_alpha") is not None else "\u2014"
        judge_rows.append(f"| {name} | {pw} | {rho} |")
    if judge_rows:
        hdr = "| Model | Pairwise Acc. | Likert \u03c1 |\n|-------|:------------:|:---------:|"
        pd.set("table_judge", hdr + "\n" + "\n".join(judge_rows), "judge")

    # --- Reference values (instruction accuracy, experiment D, decoding probes, surface metrics) ---
    ref_file = PROJECT_ROOT / "paper_data_reference.json"
    if ref_file.exists():
        ref = json.load(open(ref_file))
        n_from_ref = 0
        for k, v in ref.items():
            if k.startswith("_"):
                continue
            if not pd.has(k):
                pd.set(k, v, "reference")
                n_from_ref += 1
        if n_from_ref:
            print(f"  Loaded {n_from_ref} reference values", flush=True)

    return pd.result()


def main():
    parser = argparse.ArgumentParser(description="Build paper_data.json from data/")
    parser.add_argument("--check", action="store_true", help="Show keys without writing")
    args = parser.parse_args()

    data = build_data()

    if args.check:
        sources = data.get("_meta", {}).get("sources", {})
        by_source = {}
        for k, src in sources.items():
            by_source.setdefault(src, []).append(k)
        for src in sorted(by_source.keys()):
            keys = sorted(by_source[src])
            print(f"\n  [{src}] ({len(keys)} keys):")
            for k in keys[:5]:
                val = str(data.get(k, ""))
                if len(val) > 60:
                    val = val[:60] + "..."
                print(f"    {k} = {val}")
            if len(keys) > 5:
                print(f"    ... and {len(keys) - 5} more")
        return

    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)

    n_keys = len([k for k in data if not k.startswith("_")])
    sources = data.get("_meta", {}).get("sources", {})
    by_source = {}
    for k, src in sources.items():
        by_source.setdefault(src, []).append(k)
    print(f"Wrote {OUTPUT_FILE} ({n_keys} data keys)", flush=True)
    for src in sorted(by_source.keys()):
        print(f"  {src}: {len(by_source[src])} keys", flush=True)


if __name__ == "__main__":
    main()
