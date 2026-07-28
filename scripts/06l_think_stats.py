#!/usr/bin/env python3
"""Step 06l: D3 statistics — does a model think with the temperament it
answers with? (CPU only.)

Input: data/thinking/<model>.json produced by scripts/07_thinking.py under
the real no-think protocol (Section 3.9). Per baseline question we have
paired projections for three conditions: think_off (full response, reasoning
structurally disabled), think_on_answer (answer segment) and think_on_think
(think segment) — the same questions in every condition, so all contrasts
below are paired by question.

Per (model, axis) this script reports, with a paired bootstrap 95% CI over
questions (10k resamples, seed 42):
  * think_minus_answer          — decay-weighted segment aggregates
  * think_minus_answer_uniform  — uniform segment means (robustness: decay
                                  aggregates overweight the segment tail)
  * answer_minus_off            — the "reflection effect": does generating a
                                  reasoning segment change the temperament of
                                  the ANSWER itself?
Pre-registered hypothesis (2026-07-14): the think segment is colder / more
analytical than the answer ("performative warmth") — i.e. negative
think_minus_answer on warm_cold and empathetic_analytical.

Writes: data/analysis/think_stats.json

Usage:
    python3 -u scripts/06l_think_stats.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

THINK_DIR = PROJECT_ROOT / "data" / "thinking"
OUT_FILE = PROJECT_ROOT / "data" / "analysis" / "think_stats.json"

MODELS = ["deepseek_r1_14b", "gpt_oss_20b", "glm_z1_9b", "vibethinker_3b",
          "ministral3_8b_reasoning"]
AXES = ["warm_cold", "confident_cautious", "empathetic_analytical"]
N_BOOT = 10_000


def boot_ci(deltas, rng):
    deltas = np.asarray(deltas, dtype=np.float64)
    idx = rng.integers(0, len(deltas), size=(N_BOOT, len(deltas)))
    means = deltas[idx].mean(axis=1)
    return (float(deltas.mean()),
            float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)))


def stat(pairs, rng):
    if len(pairs) < 5:
        return None
    m, lo, hi = boot_ci(pairs, rng)
    return {"mean": m, "ci_lo": lo, "ci_hi": hi,
            "significant": bool(lo > 0 or hi < 0), "n": len(pairs)}


def analyze(model):
    f = THINK_DIR / f"{model}.json"
    if not f.exists():
        return None
    d = json.load(open(f))
    recs = d["records"]
    off = {r["question"]: r for r in recs if r["mode"] == "think_off"}
    on = {r["question"]: r for r in recs if r["mode"] == "think_on"}
    questions = sorted(set(off) & set(on))
    rng = np.random.default_rng(42)
    out_axes = {}
    for a in AXES:
        ta, ta_u, ao = [], [], []
        for q in questions:
            po, pn = off[q]["projections"].get(a), on[q]["projections"].get(a)
            if not po or not pn:
                continue
            if pn.get("think") is not None and pn.get("answer") is not None:
                ta.append(pn["think"] - pn["answer"])
            if (pn.get("think_uniform") is not None
                    and pn.get("answer_uniform") is not None):
                ta_u.append(pn["think_uniform"] - pn["answer_uniform"])
            if pn.get("answer") is not None and po.get("full") is not None:
                ao.append(pn["answer"] - po["full"])
        out_axes[a] = {
            "think_minus_answer": stat(ta, rng),
            "think_minus_answer_uniform": stat(ta_u, rng),
            "answer_minus_off": stat(ao, rng),
        }
    n_think = sum(1 for q in questions
                  if on[q].get("think_token_range") is not None)
    trunc = sum(1 for r in recs if r.get("truncated"))
    return {"axes": out_axes, "n_questions": len(questions),
            "n_with_think_segment": n_think, "n_truncated": trunc}


def main():
    results = {}
    for m in MODELS:
        r = analyze(m)
        if r is None:
            print(f"{m}: missing, skipped", flush=True)
            continue
        results[m] = r
        print(f"== {m} (n={r['n_questions']}, think segments "
              f"{r['n_with_think_segment']}, truncated {r['n_truncated']})",
              flush=True)
        for a, x in r["axes"].items():
            def fmt(s):
                if s is None:
                    return "     —      "
                star = "*" if s["significant"] else " "
                return (f"{s['mean']:+.3f} [{s['ci_lo']:+.3f},"
                        f"{s['ci_hi']:+.3f}]{star}")
            print(f"   {a:24} T-A {fmt(x['think_minus_answer'])}  "
                  f"unif {fmt(x['think_minus_answer_uniform'])}  "
                  f"A-off {fmt(x['answer_minus_off'])}", flush=True)

    out = {
        "_meta": {
            "script": "scripts/06l_think_stats.py",
            "timestamp": datetime.now().isoformat(),
            "n_bootstrap": N_BOOT,
            "protocol": "paired by question; no-think prefill per Section 3.9",
            "hypothesis": "think segment colder/more analytical than answer "
                          "(pre-registered 2026-07-14)",
        },
        "models": results,
    }
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT_FILE, "w"), indent=1)
    print(f"Saved {OUT_FILE}", flush=True)


if __name__ == "__main__":
    main()
