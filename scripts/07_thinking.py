#!/usr/bin/env python3
"""Step 07 (D3): Does a model think with the same temperament it answers with?

Three conditions per thinking model (protocol approved 2026-07-14, PLAN_V2 Phase 5):
  1. think_off        — thinking disabled (comparable to the main corpus)
  2. think_on_answer  — thinking enabled, projections of the ANSWER segment
  3. think_on_think   — thinking enabled, projections of the THINK segment

Pre-registered hypothesis: the think segment is systematically colder/more
analytical than the answer segment ("performative warmth").

Produces: data/thinking/<model>.json
Depends on: 01_calibrate axes NPZ. MLX backend only (segmenting needs
mlx_inference.segment_think_tokens).

Real no-think fix (2026-07-23): chat_template_kwargs={"enable_thinking": False}
turned out to be a no-op for deepseek_r1_14b, glm_z1_9b, vibethinker_3b, and
ministral3_8b_reasoning — their chat templates have no enable_thinking
conditional at all, so think_off's "disabled" generation was byte-identical
to think_on (see data/analysis/gptoss_probe_check.json); gpt_oss_20b has no
enable_thinking-style kwarg in its Harmony template to begin with. The
think_off condition below (via `_think_kwargs(..., enable=False)`) now also
transparently resolves each model's config.models.ModelConfig.no_think_prefill
(src/model/mlx_inference.py::_resolve_no_think_prefill) and splices it onto
the rendered prompt — a pre-"closed" empty think block for the <think>/[THINK]
-style models, or a forced jump straight to the Harmony final channel for
gpt_oss_20b — so think_off generations now actually skip the reasoning
segment instead of duplicating think_on. No code change was needed in this
script for the fix to take effect.

Usage:
    python scripts/07_thinking.py --model deepseek_r1_14b \
        --axes warm_cold confident_cautious empathetic_analytical
    python scripts/07_thinking.py --model ministral3_8b_reasoning --check
"""

import argparse
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parent.parent))

try:
    from fw.progress import log_progress
except ImportError:
    import json as _pj

    def log_progress(done, total, elapsed_s, n_elapsed=None, extra=None):
        n = n_elapsed if n_elapsed is not None else done
        rate = (elapsed_s / n) if (n and elapsed_s > 0) else None
        eta = rate * (total - done) if rate is not None else None
        payload = {"done": done, "total": total, "elapsed_s": round(elapsed_s, 1)}
        if eta is not None:
            payload["eta_s"] = round(eta, 1)
        if extra:
            payload.update(extra)
        print("#PROGRESS " + _pj.dumps(payload), flush=True)

from config.models import MODELS
from config.settings import AXES_DIR, MOOD_AXES, TEMPERATURE, TOP_P
from config.prompts import BASELINE_QUESTIONS
from src.calibration.axis_computer import load_axis_vectors

OUTPUT_DIR = PROJECT_ROOT / "data" / "thinking"
THINK_MAX_NEW_TOKENS = 16384  # R1-class models think long; monitor truncation
DEFAULT_AXES = ["warm_cold", "confident_cautious", "empathetic_analytical"]
CONDITIONS = ("think_off", "think_on_answer", "think_on_think")


def _think_kwargs(base_kwargs, enable):
    kw = dict(base_kwargs or {})
    kw["enable_thinking"] = enable
    if enable:
        # Model configs now carry no_think_prefill=True globally (the main
        # pipeline runs with thinking really off) — think_on must strip it,
        # or the prefill would suppress thinking in BOTH conditions.
        kw.pop("no_think_prefill", None)
    else:
        # Explicit opt-in for the prefill-based real no-think (see
        # _resolve_no_think_prefill in src/model/*inference.py): enable_thinking
        # alone is a template no-op for most reasoning models.
        kw["no_think_prefill"] = True
    return kw


def project(state, axis_vec, center, scale):
    v = float(np.dot(state, axis_vec))
    return float(np.clip((v - center) / scale, -1.0, 1.0))


def run(model_key, axes, force=False, time_budget_min=0, time_budget_exit_code=75):
    cfg = MODELS[model_key]
    axes_file = AXES_DIR / f"{model_key}_axes.npz"
    assert axes_file.exists(), f"Missing axes NPZ: {axes_file} (run 01_calibrate)"
    axis_vectors = load_axis_vectors(axes_file)
    npz = np.load(axes_file)
    norm = {a: (float(npz[f"{a}_center"]), float(npz[f"{a}_scale"]))
            for a in axes if a in axis_vectors}

    from src.model.mlx_inference import load_mlx_model, get_full_result_for_prompt
    print(f"Loading {cfg.model_id} (MLX)", flush=True)
    ctx = load_mlx_model(cfg.model_id, quantize=cfg.quantize,
                        mlx_model_id=cfg.mlx_model_id)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / f"{model_key}.json"
    partial_file = OUTPUT_DIR / f"{model_key}_partial.json"

    records = []
    if partial_file.exists() and not force:
        records = json.load(open(partial_file)).get("records", [])
        print(f"Resuming: {len(records)} records", flush=True)
    done = {(r["question"], r["mode"]) for r in records}

    # think_off and think_on are separate generations; segments come from think_on
    modes = ["think_off", "think_on"]
    total = len(BASELINE_QUESTIONS) * len(modes)
    t0 = time.monotonic()
    n_new = 0
    for question in BASELINE_QUESTIONS:
        for mode in modes:
            if (question, mode) in done:
                continue
            enable = mode == "think_on"
            result = get_full_result_for_prompt(
                ctx, [{"role": "user", "content": question}],
                max_new_tokens=THINK_MAX_NEW_TOKENS if enable else 4096,
                chat_template_kwargs=_think_kwargs(cfg.chat_template_kwargs, enable),
                save_extra=False,
                segment_think=enable,
            )
            rec = {
                "question": question,
                "mode": mode,
                "n_tokens": result.n_generated_tokens,
                "truncated": result.n_generated_tokens >= (
                    THINK_MAX_NEW_TOKENS if enable else 4096),
                "think_token_range": list(result.think_token_range)
                    if result.think_token_range else None,
                "projections": {},
            }
            for a in norm:
                c, s = norm[a]
                if enable:
                    rec["projections"][a] = {
                        "answer": project(result.answer_state, axis_vectors[a], c, s)
                            if result.answer_state is not None else None,
                        "think": project(result.think_state, axis_vectors[a], c, s)
                            if result.think_state is not None else None,
                        # uniform segment means — robustness check: decay
                        # aggregates weight only the segment tail
                        "answer_uniform": project(result.answer_state_uniform, axis_vectors[a], c, s)
                            if result.answer_state_uniform is not None else None,
                        "think_uniform": project(result.think_state_uniform, axis_vectors[a], c, s)
                            if result.think_state_uniform is not None else None,
                    }
                else:
                    rec["projections"][a] = {
                        "full": project(result.hidden_state, axis_vectors[a], c, s),
                    }
            # Per-token projection trajectories (think_on only) -> sidecar NPZ
            if enable and result.token_states is not None:
                q_idx = BASELINE_QUESTIONS.index(question)
                traj = {}
                for a in norm:
                    c, s = norm[a]
                    v = result.token_states @ axis_vectors[a]
                    traj[f"q{q_idx}_{a}"] = np.clip(
                        (v - c) / s, -1.0, 1.0).astype(np.float16)
                traj_file = OUTPUT_DIR / f"{model_key}_trajectories.npz"
                old = dict(np.load(traj_file)) if traj_file.exists() else {}
                old.update(traj)
                np.savez_compressed(traj_file, **old)
            records.append(rec)
            n_new += 1
            json.dump({"records": records}, open(partial_file, "w"))
            log_progress(len(records), total, time.monotonic() - t0, n_elapsed=n_new)
            if time_budget_min and (time.monotonic() - t0) > time_budget_min * 60:
                print(f"Time budget {time_budget_min} min reached at "
                      f"{len(records)}/{total}; clean exit "
                      f"{time_budget_exit_code}", flush=True)
                sys.exit(time_budget_exit_code)

    # Summary: per axis, mean projection per condition + paired deltas
    summary = {}
    for a in norm:
        off = [r["projections"][a]["full"] for r in records
               if r["mode"] == "think_off" and a in r["projections"]]
        ans = [r["projections"][a].get("answer") for r in records
               if r["mode"] == "think_on" and a in r["projections"]]
        thk = [r["projections"][a].get("think") for r in records
               if r["mode"] == "think_on" and a in r["projections"]]
        ans_u = [r["projections"][a].get("answer_uniform") for r in records
                 if r["mode"] == "think_on" and a in r["projections"]]
        thk_u = [r["projections"][a].get("think_uniform") for r in records
                 if r["mode"] == "think_on" and a in r["projections"]]
        ans = [x for x in ans if x is not None]
        thk = [x for x in thk if x is not None]
        ans_u = [x for x in ans_u if x is not None]
        thk_u = [x for x in thk_u if x is not None]
        summary[a] = {
            "think_answer_delta_uniform": float(np.mean(thk_u) - np.mean(ans_u))
                if thk_u and ans_u else None,
            "think_off_mean": float(np.mean(off)) if off else None,
            "answer_mean": float(np.mean(ans)) if ans else None,
            "think_mean": float(np.mean(thk)) if thk else None,
            "reflection_delta": float(np.mean(ans) - np.mean(off)) if ans and off else None,
            "think_answer_delta": float(np.mean(thk) - np.mean(ans)) if thk and ans else None,
            "n_think_records": len(thk),
        }

    out = {
        "_meta": {
            "script": "scripts/07_thinking.py",
            "model": model_key,
            "model_id": cfg.model_id,
            "timestamp": datetime.now().isoformat(),
            "axes": sorted(norm),
            "n_questions": len(BASELINE_QUESTIONS),
            "think_max_new_tokens": THINK_MAX_NEW_TOKENS,
            "generation_settings": {"temperature": TEMPERATURE, "top_p": TOP_P},
            "hypothesis": "think segment colder/more analytical than answer "
                          "(performative warmth); pre-registered 2026-07-14",
        },
        "summary": summary,
        "records": records,
    }
    json.dump(out, open(out_file, "w"), indent=1)
    print(f"Saved {out_file}", flush=True)
    for a, s in summary.items():
        print(f"  {a}: off={s['think_off_mean']} answer={s['answer_mean']} "
              f"think={s['think_mean']} Δthink-answer={s['think_answer_delta']}", flush=True)
    del ctx
    gc.collect()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--axes", nargs="+", default=DEFAULT_AXES,
                    choices=list(MOOD_AXES))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--check", action="store_true", help="preflight only")
    ap.add_argument("--time-budget-min", type=int, default=0)
    ap.add_argument("--time-budget-exit-code", type=int, default=75)
    args = ap.parse_args()
    assert args.model in MODELS, f"Unknown model {args.model}"
    if args.check:
        axes_file = AXES_DIR / f"{args.model}_axes.npz"
        print(f"Model: {MODELS[args.model].model_id}")
        print(f"Axes NPZ: {axes_file} (exists: {axes_file.exists()})")
        print("Preflight OK", flush=True)
        return
    run(args.model, args.axes, force=args.force,
        time_budget_min=args.time_budget_min,
        time_budget_exit_code=args.time_budget_exit_code)


if __name__ == "__main__":
    main()
