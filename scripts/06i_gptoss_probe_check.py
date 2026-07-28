#!/usr/bin/env python3
"""Step 06i: instrumental-hypothesis check for the gpt_oss_20b geometric outlier.

Question: is gpt_oss_20b's axis-geometry anomaly (consistency 0.36, effective
rank 0.65, per-axis residual PC1 34-42%, confident_cautious indistinguishable
from the cross-set-null; see data/analysis/axis_geometry.json and
data/analysis/consistency_null.json) an artifact of HOW hidden states are
aggregated for this model, rather than a real personality-geometry property?

Standard aggregation (src/model/mlx_inference.py::_aggregate_decay, used for
every calibration sample that feeds data/axes/<model>_axes.npz): last 4
layers, LAYER_WEIGHTS=[0.1,0.2,0.3,0.4], decay-weighted mean over ALL
generated tokens with TOKEN_WEIGHT_DECAY=0.9 (effective half-life ~6.6
tokens, so in practice dominated by the last few dozen tokens of whatever the
model generated). Hypothesis: for reasoning models this mixes analysis-channel
(reasoning) tokens with final-channel (answer) tokens, adding noise that a
answer-only aggregate would not have.

Data used (CPU only, no model load): data/thinking/<model>.json +
data/thinking/<model>_trajectories.npz, produced by scripts/07_thinking.py
(D3 experiment). IMPORTANT DATA LIMITATION (see _meta.limitations in the
output): these files store per-token SCALAR PROJECTIONS onto the *existing*
(already-calibrated, possibly-noisy) axis vectors -- not raw per-token hidden
state VECTORS. So this script can test whether isolating the answer segment
changes the *coherence of the projected signal*, but it CANNOT re-derive a
new axis DIRECTION from answer-only tokens, and cannot compute an a-vs-c
direction cosine. That would require raw hidden states, which are not
persisted anywhere for the D3 experiment or for calibration (only aggregated
per-pair diffs survive in data/axes/<model>_hidden_states.npz, and those were
never segmented in the first place). Re-deriving a true answer-only axis
direction needs a GPU rerun: re-generate the calibration corpus (or at least
a validation subsample) for gpt_oss_20b with segment_think=True and rebuild
axis vectors from answer-segment-only hidden states.

Writes: data/analysis/gptoss_probe_check.json

Usage:
    python3 scripts/06i_gptoss_probe_check.py
    python3 scripts/06i_gptoss_probe_check.py --check
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.prompts import BASELINE_QUESTIONS  # noqa: E402

THINKING_DIR = PROJECT_ROOT / "data" / "thinking"
MODELS_DIR = PROJECT_ROOT / "config"
OUTPUT_DIR = PROJECT_ROOT / "data" / "analysis"

AXES = ["warm_cold", "confident_cautious", "empathetic_analytical",
        "formal_casual", "verbose_concise"]

MODELS = ["gpt_oss_20b", "deepseek_r1_14b"]

# Side-check only (not full-analyzed): does think_off vs think_on efficacy
# vary across the OTHER thinking models in data/thinking/, to tell whether
# the no-op we find for gpt_oss/deepseek is universal or family-specific?
SIDE_CHECK_MODELS = ["qwen3.5_9b", "qwen3_8b", "glm_z1_9b",
                     "ministral3_8b", "ministral3_8b_reasoning", "vibethinker_3b"]


def preflight_check():
    print(f"Thinking dir: {THINKING_DIR}", flush=True)
    for m in MODELS:
        j = THINKING_DIR / f"{m}.json"
        n = THINKING_DIR / f"{m}_trajectories.npz"
        print(f"  {m}: json={'OK' if j.exists() else 'MISSING'} "
              f"npz={'OK' if n.exists() else 'MISSING'}", flush=True)
    print(f"Axes: {AXES}", flush=True)
    print(f"Output: {OUTPUT_DIR / 'gptoss_probe_check.json'}", flush=True)
    print("Preflight OK", flush=True)


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def participation_ratio(matrix_5xN):
    """SVD-based effective rank of a (n_axes, n_samples) matrix, centered per
    row. Mirrors scripts/06d_axis_geometry.py's concept_cone.effective_rank
    definition: (sum(sigma))^2 / sum(sigma^2), normalized by n_axes."""
    X = matrix_5xN - matrix_5xN.mean(axis=1, keepdims=True)
    try:
        sigma = np.linalg.svd(X, compute_uv=False)
    except np.linalg.LinAlgError:
        return None, None
    if sigma.sum() < 1e-12:
        return None, None
    eff_rank = float((sigma.sum() ** 2) / (sigma ** 2).sum())
    var = sigma ** 2
    pc1_share = float(var[0] / var.sum())
    return eff_rank, pc1_share


def load_model_data(model_key):
    j = THINKING_DIR / f"{model_key}.json"
    n = THINKING_DIR / f"{model_key}_trajectories.npz"
    data = json.load(open(j))
    npz = np.load(n)
    return data, npz


def structural_summary(data, npz, model_key):
    recs = [r for r in data["records"] if r["mode"] == "think_on"]
    n = len(recs)
    ranges = [r["think_token_range"] for r in recs]
    n_none = sum(1 for r in ranges if r is None)
    n_truncated = sum(1 for r in recs if r["truncated"])
    # "whole generation is think, no answer segment found" case: answer_start
    # == n_tokens is what segment_think_tokens returns when the close marker
    # is missing (truncated thought) -- reconstruct from range vs n_tokens.
    n_whole_think = 0
    think_frac = []
    n_tokens_list = []
    for r in recs:
        tr = r["think_token_range"]
        nt = r["n_tokens"]
        n_tokens_list.append(nt)
        if tr is None:
            continue
        ts, te = tr
        if te >= nt:
            n_whole_think += 1
        think_frac.append((te - ts) / nt if nt > 0 else None)
    think_frac = [x for x in think_frac if x is not None]
    return {
        "n_think_on_records": n,
        "n_think_token_range_none": n_none,
        "n_truncated": n_truncated,
        "n_whole_generation_is_think": n_whole_think,
        "n_tokens_mean": float(np.mean(n_tokens_list)) if n_tokens_list else None,
        "n_tokens_min": int(min(n_tokens_list)) if n_tokens_list else None,
        "n_tokens_max": int(max(n_tokens_list)) if n_tokens_list else None,
        "think_fraction_of_tokens_mean": float(np.mean(think_frac)) if think_frac else None,
        "think_fraction_of_tokens_median": float(np.median(think_frac)) if think_frac else None,
        "npz_n_keys": len(npz.files),
        "npz_axes_present": sorted(set(k.split("_", 1)[1] for k in npz.files)),
        "npz_n_questions": len(set(int(k.split("_")[0][1:]) for k in npz.files)),
    }


def thinking_disable_efficacy(data, model_key):
    """Does chat_template_kwargs={'enable_thinking': False} (mode='think_off')
    actually change anything for this model, or is it a no-op (same prompt,
    same seed=42 -> byte-identical deterministic generation as think_on,
    just truncated to the smaller think_off token budget)? Proxy: per-question
    n_tokens equality between the two modes. If ALL (or nearly all) questions
    have identical n_tokens, the two conditions are the same underlying
    generation and 'think_off full' is not an independent measurement."""
    off = {r["question"]: r["n_tokens"] for r in data["records"] if r["mode"] == "think_off"}
    on = {r["question"]: r["n_tokens"] for r in data["records"] if r["mode"] == "think_on"}
    common = [q for q in off if q in on]
    n_identical = sum(1 for q in common if off[q] == on[q])
    return {
        "n_questions": len(common),
        "n_identical_token_count": n_identical,
        "frac_identical": n_identical / len(common) if common else None,
        "interpretation": (
            "n_identical_token_count == n_questions means think_off and "
            "think_on produced token-count-identical generations for every "
            "question -- consistent with chat_template_kwargs "
            "enable_thinking=False being a no-op for this model's chat "
            "template (same prompt tokens, same seed=42 -> deterministic "
            "sampler reproduces the same continuation; think_off is simply "
            "the same generation truncated to its smaller max_new_tokens "
            "budget, which never actually binds here since n_tokens < 4096 "
            "for all records). If so, 'think_off full' is NOT an "
            "independently-sampled think-disabled baseline; it duplicates "
            "think_on and comparisons against it are close to tautological."
        ),
    }


def segment_correlations(data, model_key):
    """Per-axis, across the 30 baseline questions: Pearson correlation
    between (answer, think_off_full), (think, think_off_full), (answer,
    think). Uses the decay-aggregated projections already stored per record
    (computed with the STANDARD, pre-existing axis vector -- see module
    docstring for what this can and cannot show)."""
    by_q_off = {}
    by_q_on = {}
    for r in data["records"]:
        qidx = BASELINE_QUESTIONS.index(r["question"])
        if r["mode"] == "think_off":
            by_q_off[qidx] = r["projections"]
        else:
            by_q_on[qidx] = r["projections"]

    out = {}
    for axis in AXES:
        offs, ans, thk, ans_u, thk_u, qs = [], [], [], [], [], []
        for qidx in sorted(set(by_q_off) & set(by_q_on)):
            po = by_q_off[qidx].get(axis, {})
            pn = by_q_on[qidx].get(axis, {})
            if po.get("full") is None or pn.get("answer") is None or pn.get("think") is None:
                continue
            offs.append(po["full"])
            ans.append(pn["answer"])
            thk.append(pn["think"])
            ans_u.append(pn.get("answer_uniform"))
            thk_u.append(pn.get("think_uniform"))
            qs.append(qidx)
        out[axis] = {
            "n": len(qs),
            "corr_answer_vs_thinkoff_full": pearson(ans, offs),
            "corr_think_vs_thinkoff_full": pearson(thk, offs),
            "corr_answer_vs_think": pearson(ans, thk),
            "corr_answeruniform_vs_thinkoff_full": pearson(ans_u, offs) if all(v is not None for v in ans_u) else None,
            "mean_think_answer_delta": float(np.mean(np.array(thk) - np.array(ans))) if ans and thk else None,
            "std_answer": float(np.std(ans)) if ans else None,
            "std_think": float(np.std(thk)) if thk else None,
            "std_thinkoff_full": float(np.std(offs)) if offs else None,
        }
    return out, by_q_off, by_q_on


def cross_axis_structure(by_q_off, by_q_on):
    """Build 5(axes) x N(questions) matrices for three conditions and compute
    PCA effective-rank / PC1-share on each, mirroring 06d_axis_geometry.py's
    concept_cone metric (participation ratio), but applied to the D3
    30-baseline-question axis-score matrix per condition instead of the
    65-pair calibration diff matrix."""
    common_off = sorted(by_q_off.keys())
    common_on = sorted(by_q_on.keys())
    common = sorted(set(common_off) & set(common_on))

    def build(matrix_source, field, seg=None):
        rows = []
        for axis in AXES:
            col = []
            for qidx in common:
                p = matrix_source[qidx].get(axis, {})
                v = p.get(field)
                col.append(v)
            rows.append(col)
        arr = np.array(rows, dtype=np.float64)
        if np.any(np.isnan(arr)) or arr.size == 0:
            return None
        return arr

    conditions = {
        "think_off_full": build(by_q_off, "full"),
        "think_on_answer": build(by_q_on, "answer"),
        "think_on_think": build(by_q_on, "think"),
    }
    out = {}
    for name, mat in conditions.items():
        if mat is None:
            out[name] = None
            continue
        eff_rank, pc1_share = participation_ratio(mat)
        out[name] = {
            "n_samples": mat.shape[1],
            "effective_rank": eff_rank,
            "effective_rank_normalized": eff_rank / len(AXES) if eff_rank is not None else None,
            "pc1_variance_share": pc1_share,
        }
    return out


def token_level_stats(npz, data, model_key):
    """Per-axis within-segment token variance (answer vs think) directly from
    the per-token trajectory NPZ, using each record's think_token_range to
    slice. Cross-checks that slicing matches the stored answer_uniform /
    think_uniform scalars (sanity check on our own slicing logic)."""
    recs = {r["question"]: r for r in data["records"] if r["mode"] == "think_on"}
    out = {axis: {"within_answer_std": [], "within_think_std": [],
                  "uniform_check_answer_maxdiff": [], "uniform_check_think_maxdiff": []}
           for axis in AXES}
    for question, r in recs.items():
        qidx = BASELINE_QUESTIONS.index(question)
        tr = r["think_token_range"]
        if tr is None:
            continue
        ts, te = tr
        for axis in AXES:
            key = f"q{qidx}_{axis}"
            if key not in npz.files:
                continue
            traj = npz[key].astype(np.float64)
            think_seg = traj[ts:te]
            answer_seg = traj[te:]
            if len(think_seg) > 1:
                out[axis]["within_think_std"].append(float(np.std(think_seg)))
                stored = r["projections"].get(axis, {}).get("think_uniform")
                if stored is not None:
                    out[axis]["uniform_check_think_maxdiff"].append(
                        abs(float(np.mean(think_seg)) - stored))
            if len(answer_seg) > 1:
                out[axis]["within_answer_std"].append(float(np.std(answer_seg)))
                stored = r["projections"].get(axis, {}).get("answer_uniform")
                if stored is not None:
                    out[axis]["uniform_check_answer_maxdiff"].append(
                        abs(float(np.mean(answer_seg)) - stored))
    summary = {}
    for axis, d in out.items():
        summary[axis] = {
            "mean_within_answer_std": float(np.mean(d["within_answer_std"])) if d["within_answer_std"] else None,
            "mean_within_think_std": float(np.mean(d["within_think_std"])) if d["within_think_std"] else None,
            "max_uniform_check_diff_answer": float(np.max(d["uniform_check_answer_maxdiff"])) if d["uniform_check_answer_maxdiff"] else None,
            "max_uniform_check_diff_think": float(np.max(d["uniform_check_think_maxdiff"])) if d["uniform_check_think_maxdiff"] else None,
        }
    return summary


def check_calibration_thinking_config():
    """Structural fact (not derived from D3 data): does the model's
    ModelConfig explicitly request enable_thinking=False for the main
    calibration corpus? Read directly from config/models.py source text
    (no torch/mlx import needed)."""
    src = (PROJECT_ROOT / "config" / "models.py").read_text()
    out = {}
    for model_key in MODELS:
        marker = f'"{model_key}": ModelConfig('
        idx = src.find(marker)
        if idx < 0:
            out[model_key] = {"found": False}
            continue
        # ModelConfig(...) block ends at the matching top-level "),\n" -- take
        # a generous slice up to the next "\"" + next top-level key start, or
        # just search for chat_template_kwargs within the next ~600 chars.
        block = src[idx:idx + 800]
        end = block.find("\n    \"", 20)  # next dict entry starts
        if end > 0:
            block = block[:end]
        has_kwarg = "chat_template_kwargs" in block
        out[model_key] = {
            "found": True,
            "has_chat_template_kwargs": has_kwarg,
            "block_excerpt": block.strip(),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if args.check:
        preflight_check()
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for model_key in MODELS:
        print(f"Analyzing {model_key} ...", flush=True)
        data, npz = load_model_data(model_key)
        struct = structural_summary(data, npz, model_key)
        think_off_efficacy = thinking_disable_efficacy(data, model_key)
        corr, by_q_off, by_q_on = segment_correlations(data, model_key)
        cross = cross_axis_structure(by_q_off, by_q_on)
        tok = token_level_stats(npz, data, model_key)
        results[model_key] = {
            "structural": struct,
            "think_off_vs_think_on_efficacy": think_off_efficacy,
            "segment_correlations": corr,
            "cross_axis_structure": cross,
            "token_level_within_segment_std": tok,
        }
        print(f"  {model_key}: {struct['n_think_on_records']} think_on records, "
              f"{struct['n_think_token_range_none']} unresolved, "
              f"think_fraction_median={struct['think_fraction_of_tokens_median']}",
              flush=True)

    calib_cfg = check_calibration_thinking_config()

    side_check = {}
    for model_key in SIDE_CHECK_MODELS:
        j = THINKING_DIR / f"{model_key}.json"
        if not j.exists():
            side_check[model_key] = {"available": False}
            continue
        data = json.load(open(j))
        side_check[model_key] = {
            "available": True,
            **thinking_disable_efficacy(data, model_key),
        }
    print("\nSide-check: think_off vs think_on token-count efficacy across "
          "all available thinking models:", flush=True)
    for m in MODELS + SIDE_CHECK_MODELS:
        d = results.get(m, {}).get("think_off_vs_think_on_efficacy") or side_check.get(m)
        if d and d.get("available", True):
            print(f"  {m}: {d['n_identical_token_count']}/{d['n_questions']} "
                  f"identical token counts", flush=True)
        else:
            print(f"  {m}: no data", flush=True)

    output = {
        "_meta": {
            "script": "scripts/06i_gptoss_probe_check.py",
            "timestamp": datetime.now().isoformat(),
            "hypothesis": (
                "gpt_oss_20b's axis-geometry outlier status (consistency "
                "0.36, effective_rank_normalized ~0.65, per-axis "
                "pc1_variance_share 34-42%, confident_cautious indistinguishable "
                "from cross-set null; see data/analysis/axis_geometry.json, "
                "data/analysis/consistency_null.json) is an artifact of the "
                "standard hidden-state aggregation (last-4-layers weighted "
                "decay over ALL generated tokens) mixing reasoning-channel "
                "and answer-channel tokens for this reasoning model."
            ),
            "data_source": (
                "data/thinking/<model>.json + <model>_trajectories.npz, "
                "produced by scripts/07_thinking.py (D3 experiment, 30 "
                "BASELINE_QUESTIONS, think_off/think_on conditions, "
                "segment boundaries from "
                "src/model/mlx_inference.py::segment_think_tokens using "
                "Harmony '<|channel|>analysis<|message|>' / "
                "'<|channel|>final<|message|>' markers for gpt_oss_20b and "
                "'<think>'/'</think>' for deepseek_r1_14b)."
            ),
            "limitations": (
                "data/thinking/* stores per-token SCALAR PROJECTIONS onto the "
                "ALREADY-CALIBRATED (frozen) axis vector, not raw per-token "
                "hidden-state VECTORS (see mlx_inference.py: token_states is "
                "computed in memory and only `token_states @ axis_vector` is "
                "persisted to the trajectories NPZ). Consequences: (1) this "
                "script CANNOT re-derive a new axis DIRECTION from "
                "answer-only or think-only tokens, so no a-vs-c direction "
                "cosine is computable here -- every number below is a "
                "downstream PROJECTION effect, filtered through whatever "
                "noise is already baked into the existing axis vector. "
                "(2) The 'think_off' condition is a SEPARATE generation "
                "(different sampled text) from 'think_on', not the same "
                "response with a segment removed -- so answer-vs-think_off "
                "correlations mix segment effects with generation-to-"
                "generation sampling variance. (3) The calibration corpus "
                "itself (which produced the actual axis vectors judged "
                "anomalous) is not segmented at all and its raw hidden "
                "states are not on disk (data/axes/<model>_hidden_states.npz "
                "only has aggregated per-pair diffs). A real test of the "
                "hypothesis -- does re-deriving the axis DIRECTION from "
                "answer-segment-only calibration hidden states raise "
                "gpt_oss_20b's cross-set consistency / effective rank -- "
                "requires a GPU rerun: regenerate the calibration corpus (or "
                "a validation subsample, e.g. the 3 stability sets) for "
                "gpt_oss_20b with segment_think=True, rebuild axis vectors "
                "from answer-only hidden states, and recompute "
                "scripts/06d_axis_geometry.py + scripts/06h_consistency_null.py "
                "on the result."
            ),
            "calibration_thinking_config": calib_cfg,
        },
        "models": results,
        "side_check_think_disable_efficacy_all_thinking_models": side_check,
    }

    out_path = OUTPUT_DIR / "gptoss_probe_check.json"
    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2)
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
