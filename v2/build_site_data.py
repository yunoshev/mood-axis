#!/usr/bin/env python3
"""Build the interactive site's data files from v2/data/ (single source of truth).

Reads ONLY committed files under v2/data/ — the same data the paper numbers
come from. Produces:

    docs/site_data.json       compact core rendered inline by the page
    docs/data/drift_<m>.json  per-unit drift trajectories (fetched on demand)
    docs/data/playground.json steering texts for the playground

Every number on the site is derived here; nothing is hand-edited.

Usage:
    python3 v2/build_site_data.py
"""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # public repo root
V2 = ROOT / "v2" / "data"
DOCS = ROOT / "docs"
DOCS_DATA = DOCS / "data"

AXES = ["warm_cold", "confident_cautious", "empathetic_analytical",
        "formal_casual", "verbose_concise"]
AXIS_LABELS = {
    "warm_cold": ["Warm", "Cold"],
    "confident_cautious": ["Confident", "Cautious"],
    "empathetic_analytical": ["Empathetic", "Analytical"],
    "formal_casual": ["Formal", "Casual"],
    "verbose_concise": ["Verbose", "Concise"],
}

MAX_TURNS = 16
PLAYGROUND_QUESTIONS = 3          # questions per axis in the playground
PLAYGROUND_TEXT_LIMIT = 700       # chars per steered text
EXAMPLE_TEXT_LIMIT = 380          # chars per baseline example


def load(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def r3(x):
    return None if x is None else round(float(x), 3)


def r4(x):
    return None if x is None else round(float(x), 4)


def _git(*args):
    try:
        out = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
        return out or None
    except Exception:
        return None


def data_version():
    """Commit + date of the last change to v2/data — deterministic, so CI
    rebuilds are idempotent when the data hasn't changed."""
    commit = _git("log", "-1", "--format=%h", "--", "v2/data") or "local"
    date = _git("log", "-1", "--format=%cs", "--", "v2/data") or "uncommitted"
    return commit, date


def unit_traj(turns, axis):
    vals = []
    for t in turns[:MAX_TURNS]:
        v = (t.get("values") or {}).get(axis)
        vals.append(r3(v) if v is not None else None)
    return vals


def mean_traj(units, axis):
    """Per-turn mean over units (ignoring missing turns)."""
    out = []
    for i in range(MAX_TURNS):
        vals = []
        for u in units:
            turns = u.get("turns") or []
            if i < len(turns):
                v = (turns[i].get("values") or {}).get(axis)
                if v is not None:
                    vals.append(v)
        out.append(r3(sum(vals) / len(vals)) if vals else None)
    return out


def unit_deltas(units, axis):
    """Final-minus-first projection per unit (for dot strips)."""
    out = []
    for u in units:
        turns = u.get("turns") or []
        vals = [(t.get("values") or {}).get(axis) for t in turns]
        vals = [v for v in vals if v is not None]
        if len(vals) >= 2:
            out.append(r3(vals[-1] - vals[0]))
    return out


def build():
    models_reg = load(V2 / "models.json") or {}
    drift_stats = (load(V2 / "analysis" / "drift_stats.json") or {}).get("models", {})
    multiplicity = load(V2 / "analysis" / "drift_multiplicity.json") or {}
    mult_by_key = {(t["model"], t["axis"]): t for t in multiplicity.get("tests", [])}
    quant = load(V2 / "analysis" / "quant_robustness.json") or {}
    fmt_part = (load(V2 / "analysis" / "format_partialled_drift.json") or {}).get("models", {})
    layer_pool = (load(V2 / "analysis" / "layer_pooling_sensitivity.json") or {}).get("models", {})

    commit, date = data_version()
    site = {
        "_meta": {
            "site_version": "v2",
            "generated_at": date,
            "data_commit": commit,
            "builder": "v2/build_site_data.py",
        },
        "axes": AXES,
        "axis_labels": AXIS_LABELS,
        "models": {},
        "analysis": {},
    }

    DOCS_DATA.mkdir(parents=True, exist_ok=True)
    playground = {"models": {}}

    for key, reg in models_reg.items():
        m = {k: reg.get(k) for k in ("name", "org", "params", "group", "color", "model_id", "twin_of", "quantize")}

        # --- Fingerprint + example responses (02_baseline) ---
        bl = load(V2 / "baselines" / f"{key}.json")
        if bl:
            m["fingerprint"] = {
                ax: {"mean": r3(v["mean"]), "std": r3(v["std"])}
                for ax, v in bl.get("axes", {}).items() if ax in AXES
            }
            m["fingerprint_n"] = (bl.get("_meta") or {}).get("n_questions")
            m["examples"] = [
                {"q": r["question"], "a": r["response"][:EXAMPLE_TEXT_LIMIT]}
                for r in (bl.get("responses") or [])[:2]
            ]

        # --- Calibration quality (01_calibrate meta) ---
        am = load(V2 / "axes_meta" / f"{key}_axes_meta.json")
        if am:
            m["calibration"] = {
                ax: {"val_acc": r3(am[ax].get("val_accuracy")),
                     "val_dprime": r3(am[ax].get("val_dprime"))}
                for ax in AXES if ax in am
            }

        # --- Stability (04_stability) ---
        st = load(V2 / "stability" / f"{key}.json")
        if st:
            icc = st.get("icc") or {}
            ricc = st.get("rank_icc") or {}
            m["stability"] = {
                "icc": icc.get("mean_icc"),
                "icc_per_axis": icc.get("per_axis", {}),
                "rank_icc": ricc.get("mean_rank_icc"),
                "rank_icc_per_axis": ricc.get("per_axis", {}),
                "cosine_per_axis": {
                    ax: r3(s.get("mean_cosine"))
                    for ax, s in (st.get("per_axis_summary") or {}).items() if ax in AXES
                },
                "procrustes": r3((st.get("procrustes") or {}).get("mean_congruence")),
            }

        # --- Drift (05_drift + 06e stats + 06n FDR) ---
        dr = load(V2 / "drift" / f"{key}.json")
        if dr:
            conflicts = dr.get("conflict_scenarios") or []
            controls = dr.get("neutral_controls") or []
            stats = (drift_stats.get(key) or {}).get("axes", {})
            daxes = {}
            detail = {"axes": {}}
            for ax in AXES:
                s = stats.get(ax) or {}
                mt = mult_by_key.get((key, ax)) or {}
                rec = {
                    "conflict_slope": r4((s.get("conflict_slope") or {}).get("mean")),
                    "conflict_ci": [r4(x) for x in (s.get("conflict_slope") or {}).get("ci_95", [None, None])],
                    "control_slope": r4((s.get("control_slope") or {}).get("mean")),
                    "control_ci": [r4(x) for x in (s.get("control_slope") or {}).get("ci_95", [None, None])],
                    "significant": s.get("significant_drift"),
                    "p_perm": r4((s.get("permutation_test") or {}).get("p_value")),
                    "q_fdr": r4(mt.get("q_fdr")),
                    "fdr_significant": mt.get("fdr_significant"),
                    "ratio": r3(mt.get("ratio")),
                    "conflict_abs_delta": r3((s.get("magnitude") or {}).get("conflict_mean_abs_delta_turn1_to_last")),
                    "control_abs_delta": r3((s.get("magnitude") or {}).get("control_mean_abs_delta_turn1_to_last")),
                    "recovery_ratio": r3((s.get("recovery") or {}).get("mean_recovery_ratio")),
                    "mean_conflict_traj": mean_traj(conflicts, ax),
                    "mean_control_traj": mean_traj(controls, ax),
                    "conflict_deltas": unit_deltas(conflicts, ax),
                    "control_deltas": unit_deltas(controls, ax),
                }
                daxes[ax] = rec
                detail["axes"][ax] = {
                    "conflict": [
                        {"id": c.get("scenario_id"), "cat": c.get("category"),
                         "vals": unit_traj(c.get("turns") or [], ax)}
                        for c in conflicts
                    ],
                    "control": [
                        {"id": c.get("name"), "vals": unit_traj(c.get("turns") or [], ax)}
                        for c in controls
                    ],
                }
            m["drift"] = {
                "n_conflict": len(conflicts),
                "n_controls": len(controls),
                "axes": daxes,
                "detail_file": f"data/drift_{key}.json",
            }
            with open(DOCS_DATA / f"drift_{key}.json", "w") as f:
                json.dump(detail, f, separators=(",", ":"))

        # --- Steering dose-response (steering_basic v2) ---
        sb = load(V2 / "steering" / f"{key}_basic.json")
        if sb:
            meta = sb.get("_meta") or {}
            pli = meta.get("peak_layer_info") or {}
            saxes = {}
            pg_axes = {}
            for ax, rec in (sb.get("axes") or {}).items():
                if ax not in AXES:
                    continue
                er = rec.get("effective_range") or {}
                ci = rec.get("slope_ci") or {}
                alpha_means = {}
                for ak, av in (rec.get("alpha_means") or {}).items():
                    alpha_means[ak] = r3(av.get(ax) if isinstance(av, dict) else av)
                leak = rec.get("leakage") or {}
                leak_vals = [abs(v) for v in leak.values() if isinstance(v, (int, float))]
                saxes[ax] = {
                    "alpha_means": alpha_means,
                    "slope": r4(rec.get("slope")),
                    "slope_ci": [r4(ci.get("ci_low")), r4(ci.get("ci_high"))],
                    "total_effect": r3(er.get("total_effect")),
                    "leakage_mean": r3(sum(leak_vals) / len(leak_vals)) if leak_vals else None,
                }
                # playground texts: fixed questions across alphas
                results = rec.get("results") or {}
                if results:
                    alphas_sorted = sorted(results.keys(), key=float)
                    first = results[alphas_sorted[0]]
                    qidx = [r["question_idx"] for r in first[:PLAYGROUND_QUESTIONS]]
                    questions = [r["question"] for r in first[:PLAYGROUND_QUESTIONS]]
                    texts, projs = {}, {}
                    for ak in alphas_sorted:
                        by_q = {r["question_idx"]: r for r in results[ak]}
                        texts[ak] = [
                            (by_q.get(qi, {}).get("steered_text") or "")[:PLAYGROUND_TEXT_LIMIT]
                            for qi in qidx
                        ]
                        projs[ak] = [
                            r3((by_q.get(qi, {}).get("projections") or {}).get(ax))
                            for qi in qidx
                        ]
                    pg_axes[ax] = {
                        "alphas": [float(a) for a in alphas_sorted],
                        "alpha_keys": alphas_sorted,
                        "questions": questions,
                        "texts": texts,
                        "projections": projs,
                    }
            m["steering"] = {
                "layer": meta.get("target_layer"),
                "n_layers": meta.get("n_layers"),
                "layer_pct": pli.get("best_layer_pct"),
                "layer_source": pli.get("source"),
                "alphas": meta.get("alphas"),
                "n_questions": meta.get("n_questions"),
                "axes": saxes,
            }
            if pg_axes:
                playground["models"][key] = {
                    "name": reg.get("name", key),
                    "layer": meta.get("target_layer"),
                    "axes": pg_axes,
                }

        # --- Layer sweep summary ---
        sw = load(V2 / "steering_sweep" / f"{key}_sweep.json")
        if sw:
            m["sweep"] = {
                "best_layer": sw.get("best_layer"),
                "best_layer_pct": sw.get("best_layer_pct"),
                "best_effect": r4(sw.get("best_effect")),
                "n_layers": sw.get("n_layers"),
                "axis": sw.get("axis"),
                "layers": [
                    {"layer": L.get("layer"), "pct": L.get("layer_pct"), "effect": r4(L.get("effect"))}
                    for L in (sw.get("layers") or [])
                ],
            }

        site["models"][key] = m

    # ------------------------------------------------------------------
    # Cross-model analysis panels
    # ------------------------------------------------------------------
    if quant:
        def _pack(group):
            out = {}
            for mk, rec in (quant.get(group) or {}).items():
                out[mk] = {
                    "mean_axis_cos": r3(rec.get("mean_axis_cos")),
                    "min_axis_cos": r3(rec.get("min_axis_cos")),
                    "min_axis": rec.get("min_axis"),
                    "axes": {ax: r3(v.get("axis_cos")) for ax, v in (rec.get("axes") or {}).items()},
                }
            return out
        site["analysis"]["quant"] = {
            "summary": {
                "n_models": (quant.get("summary") or {}).get("n_models"),
                "pooled_mean": r3((quant.get("summary") or {}).get("pooled_mean_axis_cos")),
                "pooled_min": r3((quant.get("summary") or {}).get("pooled_min_axis_cos")),
            },
            "twins": _pack("twins"),
            "giants": _pack("giants"),
        }

    if fmt_part:
        fp = {}
        for mk, rec in fmt_part.items():
            fp[mk] = {
                "n_controls_kept": rec.get("n_controls_kept"),
                "n_conflict_kept": rec.get("n_conflict_kept"),
                "warning": rec.get("warning"),
                "axes": {
                    ax: {
                        "raw_sig": a.get("raw_significant"),
                        "part_sig": a.get("part_significant"),
                        "raw_slope": r4(a.get("raw_slope")),
                        "part_slope": r4(a.get("part_slope")),
                        "cos": r3(a.get("cos_raw_partialled")),
                    }
                    for ax, a in (rec.get("axes") or {}).items()
                },
            }
        site["analysis"]["format_partialled"] = fp

    if layer_pool:
        lp = {}
        for mk, axes_rec in layer_pool.items():
            lp[mk] = {}
            for ax, rec in axes_rec.items():
                if ax not in AXES:
                    continue
                per_layer = rec.get("per_layer") or []
                accs = [(L.get("layer"), L.get("val_pair_acc")) for L in per_layer]
                n_layers = rec.get("n_layers") or len(accs)
                upper = [a for (l, a) in accs if l is not None and a is not None and l >= n_layers / 2]
                lp[mk][ax] = {
                    "n_layers": n_layers,
                    "n_acc_ge_090": sum(1 for (_, a) in accs if a is not None and a >= 0.9),
                    "upper_half_min_acc": r3(min(upper)) if upper else None,
                    "acc_per_layer": [r3(a) for (_, a) in accs],
                }
        site["analysis"]["layer_pooling"] = lp

    if multiplicity:
        site["analysis"]["multiplicity"] = {
            "n_tests": len(multiplicity.get("tests", [])),
            "threshold_counts": multiplicity.get("threshold_counts"),
        }

    # ------------------------------------------------------------------
    with open(DOCS / "site_data.json", "w") as f:
        json.dump(site, f, separators=(",", ":"))
    with open(DOCS_DATA / "playground.json", "w") as f:
        json.dump(playground, f, separators=(",", ":"))

    n = site["models"]
    print(f"Wrote docs/site_data.json ({(DOCS / 'site_data.json').stat().st_size // 1024} KB)")
    print(f"  models: {len(n)}  fingerprints: {sum(1 for m in n.values() if 'fingerprint' in m)}"
          f"  stability: {sum(1 for m in n.values() if 'stability' in m)}"
          f"  drift: {sum(1 for m in n.values() if 'drift' in m)}"
          f"  steering: {sum(1 for m in n.values() if 'steering' in m)}")
    print(f"Wrote docs/data/playground.json ({(DOCS_DATA / 'playground.json').stat().st_size // 1024} KB), "
          f"{len(playground['models'])} models")
    print(f"Drift detail files: {len(list(DOCS_DATA.glob('drift_*.json')))}")


if __name__ == "__main__":
    build()
