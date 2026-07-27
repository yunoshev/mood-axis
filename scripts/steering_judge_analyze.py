#!/usr/bin/env python3
"""Score blind-judge verdicts for steering validation (v2).

Judging protocol (no API, fully reproducible):
  1. `steering_judge_prep.py` builds blind batches (data/judge_steering/batches/)
     and answer keys (*_key.json). SEED 42 fixes pair order and A/B swaps.
  2. Each batch is judged by an LLM judge that reads ONLY the batch file
     (never the key) and writes data/judge_steering/verdicts/<m>_<axis>.json:
         {"pairwise": [{"idx": 0, "choice": "A"}, ...],
          "likert":   [{"id": 0, "rating": 3}, ...]}
     We use Claude subagents as judges; any capable LLM can reproduce this
     step with the same files and instructions embedded in each batch.
  3. This script scores verdicts against keys:
         pairwise accuracy  — did the judge pick the higher-alpha text as
                              expressing the positive pole more?
         Spearman rho(alpha, rating) — Likert dose-response.
     Optional second-rater verdicts in verdicts_rater2/ produce inter-rater
     agreement (pairwise % agreement, rating rho, |diff|<=1 share).

Output: data/judge_steering/judge_results.json + console table.

Usage:
    python3 scripts/steering_judge_analyze.py
"""

import json
from pathlib import Path

from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).parent.parent
JUDGE_DIR = PROJECT_ROOT / "data" / "judge_steering"


def validate(batch: dict, verdict: dict) -> list:
    """Structural checks: full coverage, no dupes, values in domain."""
    errs = []
    want = {it["idx"] for it in batch["pairwise"]}
    got = [it["idx"] for it in verdict.get("pairwise", [])]
    if len(got) != len(set(got)):
        errs.append("pairwise duplicates")
    if set(got) != want:
        errs.append("pairwise idx mismatch")
    if any(it.get("choice") not in ("A", "B") for it in verdict.get("pairwise", [])):
        errs.append("pairwise choice out of domain")
    want_l = {it["id"] for it in batch["likert"]}
    got_l = [it["id"] for it in verdict.get("likert", [])]
    if len(got_l) != len(set(got_l)):
        errs.append("likert duplicates")
    if set(got_l) != want_l:
        errs.append("likert id mismatch")
    if any(not (isinstance(it.get("rating"), int) and 1 <= it["rating"] <= 5)
           for it in verdict.get("likert", [])):
        errs.append("likert rating out of domain")
    return errs


def score_one(name: str) -> dict:
    batch = json.load(open(JUDGE_DIR / "batches" / f"{name}.json"))
    key = json.load(open(JUDGE_DIR / "batches" / f"{name}_key.json"))
    verdict = json.load(open(JUDGE_DIR / "verdicts" / f"{name}.json"))
    errs = validate(batch, verdict)

    kmap = {k["idx"]: k["positive_is"] for k in key["pairwise"]}
    pw = [it["choice"] == kmap[it["idx"]]
          for it in verdict["pairwise"] if it["idx"] in kmap]
    amap = {k["id"]: k["alpha"] for k in key["likert"]}
    pairs = [(amap[it["id"]], it["rating"])
             for it in verdict["likert"] if it["id"] in amap]
    rho, p = spearmanr([a for a, _ in pairs], [r for _, r in pairs])

    return {
        "model": key["model"], "axis": key["axis"],
        "pairwise_accuracy": round(sum(pw) / len(pw), 4), "pairwise_n": len(pw),
        "likert_spearman_rho": round(float(rho), 4),
        "likert_p": float(f"{p:.3e}"), "likert_n": len(pairs),
        "alpha_lo": key["pairwise"][0]["alpha_lo"],
        "alpha_hi": key["pairwise"][0]["alpha_hi"],
        "structural_errors": errs,
    }


def score_interrater(name: str) -> dict:
    v1 = json.load(open(JUDGE_DIR / "verdicts" / f"{name}.json"))
    v2 = json.load(open(JUDGE_DIR / "verdicts_rater2" / f"{name}.json"))
    c1 = {i["idx"]: i["choice"] for i in v1["pairwise"]}
    c2 = {i["idx"]: i["choice"] for i in v2["pairwise"]}
    common = sorted(set(c1) & set(c2))
    r1 = {i["id"]: i["rating"] for i in v1["likert"]}
    r2 = {i["id"]: i["rating"] for i in v2["likert"]}
    ids = sorted(set(r1) & set(r2))
    rho, _ = spearmanr([r1[i] for i in ids], [r2[i] for i in ids])
    return {
        "batch": name,
        "pairwise_agreement": round(sum(c1[k] == c2[k] for k in common) / len(common), 4),
        "likert_rho": round(float(rho), 4),
        "likert_within1": round(sum(abs(r1[i] - r2[i]) <= 1 for i in ids) / len(ids), 4),
    }


def main():
    results, problems = [], []
    for vp in sorted((JUDGE_DIR / "verdicts").glob("*.json")):
        r = score_one(vp.stem)
        results.append(r)
        if r["structural_errors"]:
            problems.append((vp.stem, r["structural_errors"]))
        print(f"{r['model']:24s} {r['axis']:24s} acc={r['pairwise_accuracy']:.2f} "
              f"rho={r['likert_spearman_rho']:+.3f} (n={r['likert_n']})", flush=True)

    interrater = []
    r2dir = JUDGE_DIR / "verdicts_rater2"
    if r2dir.exists():
        for vp in sorted(r2dir.glob("*.json")):
            if (JUDGE_DIR / "verdicts" / vp.name).exists():
                ir = score_interrater(vp.stem)
                interrater.append(ir)
                print(f"inter-rater {ir['batch']}: agree={ir['pairwise_agreement']:.2f} "
                      f"rho={ir['likert_rho']:.3f} within1={ir['likert_within1']:.2f}", flush=True)

    accs = [r["pairwise_accuracy"] for r in results]
    rhos = [r["likert_spearman_rho"] for r in results]
    summary = {
        "n_batches": len(results),
        "mean_pairwise_accuracy": round(sum(accs) / len(accs), 4) if accs else None,
        "mean_likert_rho": round(sum(rhos) / len(rhos), 4) if rhos else None,
        "results": results, "interrater": interrater,
    }
    out = JUDGE_DIR / "judge_results.json"
    json.dump(summary, open(out, "w"), indent=1)
    print(f"\n{len(results)} batches -> {out}", flush=True)
    for name, errs in problems:
        print(f"  STRUCTURAL PROBLEM {name}: {errs}", flush=True)


if __name__ == "__main__":
    main()
