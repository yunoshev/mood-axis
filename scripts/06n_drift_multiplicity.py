#!/usr/bin/env python3
"""06n: Multiple-comparison and threshold-sensitivity audit of drift results
(reviewer defense, pt 6).

Three questions over data/analysis/drift_stats.json:
  1. BH-FDR (q=0.05) across ALL model x axis permutation p-values — which
     axes survive conflict-vs-control testing with multiplicity control?
  2. How does the significant-axes table change if the ad hoc 2x ratio
     threshold is moved to 1.5x / 3x?
  3. Where do the two evidence levels disagree (composite criterion says
     YES, permutation p is weak) — the honest-reporting list for §5.2.

CPU-only, no new data. Output: data/analysis/drift_multiplicity.json
"""
import json
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent.parent
STATS = PAPER_DIR / "data" / "analysis" / "drift_stats.json"
OUT = PAPER_DIR / "data" / "analysis" / "drift_multiplicity.json"

Q_FDR = 0.05
THRESHOLDS = [1.5, 2.0, 3.0]


def bh_fdr(pvals):
    """Benjamini-Hochberg: returns dict p -> adjusted p (q-value), monotone."""
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    n = len(pvals)
    adj = [0.0] * n
    prev = 1.0
    for rank_from_end, i in enumerate(reversed(idx)):
        rank = n - rank_from_end
        q = min(prev, pvals[i] * n / rank)
        adj[i] = q
        prev = q
    return adj


def main():
    stats = json.loads(STATS.read_text())
    models = stats.get("models", stats)

    tests = []
    for m, md in models.items():
        for ax, a in md.get("axes", {}).items():
            p = (a.get("permutation_test") or {}).get("p_value")
            cs = (a.get("conflict_slope") or {}).get("mean")
            ctl = (a.get("control_slope") or {}).get("mean")
            ci = a.get("criteria", {}).get("ci_excludes_zero")
            if p is None or cs is None:
                continue
            tests.append({
                "model": m, "axis": ax, "p_perm": float(p),
                "conflict_slope": float(cs),
                "control_slope": float(ctl) if ctl is not None else None,
                "ci_excludes_zero": bool(ci),
                "significant_composite": bool(a.get("significant_drift")),
            })

    adj = bh_fdr([t["p_perm"] for t in tests])
    for t, q in zip(tests, adj):
        t["q_fdr"] = float(q)
        t["fdr_significant"] = q <= Q_FDR

    # Threshold sensitivity for the ratio criterion (CI part held fixed).
    for t in tests:
        ctl = abs(t["control_slope"]) if t["control_slope"] is not None else 0.0
        t["ratio"] = abs(t["conflict_slope"]) / ctl if ctl > 0 else float("inf")
        for thr in THRESHOLDS:
            t[f"sig_at_{thr}x"] = t["ci_excludes_zero"] and t["ratio"] > thr

    n_total = len(tests)
    comp = [t for t in tests if t["significant_composite"]]
    fdr_ok = [t for t in tests if t["fdr_significant"]]
    both = [t for t in comp if t["fdr_significant"]]
    weak = [t for t in comp if not t["fdr_significant"]]

    print(f"{n_total} tests ({len(models)} models x axes)", flush=True)
    print(f"composite-significant: {len(comp)}; BH-FDR(q={Q_FDR}) significant: "
          f"{len(fdr_ok)}; both: {len(both)}", flush=True)

    print("\n-- Composite YES, confirmed by FDR-controlled permutation test:",
          flush=True)
    for t in both:
        print(f"  {t['model']:24s} {t['axis']:24s} p={t['p_perm']:.4f} "
              f"q={t['q_fdr']:.4f}", flush=True)

    print("\n-- Composite YES, NOT confirmed (honest-reporting list for 5.2):",
          flush=True)
    for t in weak:
        print(f"  {t['model']:24s} {t['axis']:24s} p={t['p_perm']:.4f} "
              f"q={t['q_fdr']:.4f}", flush=True)

    print("\n-- Threshold sensitivity of the composite criterion:", flush=True)
    hdr = "  ".join(f"{thr}x:{sum(t[f'sig_at_{thr}x'] for t in tests):2d}"
                    for thr in THRESHOLDS)
    print(f"  significant-axis count at {hdr}", flush=True)
    for t in tests:
        states = [t[f"sig_at_{thr}x"] for thr in THRESHOLDS]
        if len(set(states)) > 1:
            det = " ".join(f"{thr}x={'Y' if t[f'sig_at_{thr}x'] else 'n'}"
                           for thr in THRESHOLDS)
            print(f"  sensitive: {t['model']:24s} {t['axis']:24s} "
                  f"ratio={t['ratio']:.2f}  {det}", flush=True)

    OUT.write_text(json.dumps({
        "_meta": {"q_fdr": Q_FDR, "thresholds": THRESHOLDS,
                  "n_tests": n_total,
                  "note": "permutation test = conflict-vs-control mean slope "
                          "diff, limited by n=10 controls; composite = "
                          "bootstrap CI + 2x ratio"},
        "tests": tests,
    }, indent=2))
    print(f"\nSaved {OUT}", flush=True)


if __name__ == "__main__":
    main()
